import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import SignerAdapter, TranslationLog, User
from app.db.session import get_db
from app.schemas.schemas import AdapterOut
from app.services.letter_fewshot import (
    finalize_adapter_delete,
    restore_adapter_delete,
    stage_adapter_delete,
)

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "created_at": current_user.created_at,
        "is_verified": current_user.is_verified,
    }


@router.get("/me/adapters", response_model=list[AdapterOut])
def list_my_adapters(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    adapters = (
        db.query(SignerAdapter)
        .filter(SignerAdapter.owner_id == current_user.id)
        .order_by(SignerAdapter.created_at.desc())
        .all()
    )

    result = []
    for adapter in adapters:
        if not adapter.payload_json and not Path(adapter.weights_path).name.startswith("letter_adapter_"):
            continue
        letters: list[str] = []
        shots: dict[str, int] = {}
        if adapter.payload_json:
            try:
                payload = json.loads(adapter.payload_json)
                prototypes = payload.get("prototypes") if isinstance(payload, dict) else {}
                raw_shots = payload.get("shots") if isinstance(payload, dict) else {}
                if isinstance(prototypes, dict):
                    letters = [str(letter) for letter in prototypes]
                if isinstance(raw_shots, dict):
                    shots = {str(letter): int(value) for letter, value in raw_shots.items()}
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        result.append({
            "id": adapter.id,
            "owner_id": adapter.owner_id,
            "calibration_seconds": adapter.calibration_seconds,
            "param_count": adapter.param_count,
            "accuracy_gain_pct": adapter.accuracy_gain_pct,
            "letters": letters,
            "shots": shots,
            "created_at": adapter.created_at,
        })
    return result


@router.delete("/me/adapters/{adapter_id}")
def delete_my_adapter(
    adapter_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    adapter = (
        db.query(SignerAdapter)
        .filter(
            SignerAdapter.id == adapter_id,
            SignerAdapter.owner_id == current_user.id,
        )
        .first()
    )

    if adapter is None:
        raise HTTPException(status_code=404, detail="Adapter not found")

    weights_path = adapter.weights_path

    try:
        tombstone = (
            stage_adapter_delete(weights_path)
            if Path(weights_path).name.startswith("letter_adapter_") and Path(weights_path).exists()
            else None
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Adapter deletion could not be staged safely",
        ) from exc

    try:
        db.query(TranslationLog).filter(TranslationLog.adapter_id == adapter.id).update(
            {TranslationLog.adapter_id: None},
            synchronize_session=False,
        )
        db.delete(adapter)
        db.commit()
    except Exception as exc:
        db.rollback()
        if tombstone is not None:
            restore_adapter_delete(tombstone, weights_path)
        raise HTTPException(
            status_code=500,
            detail="Adapter deletion could not be completed safely",
        ) from exc

    try:
        finalize_adapter_delete(tombstone)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Adapter record deleted, but stored file cleanup is pending",
        ) from exc

    return {"deleted": True, "adapter_id": adapter_id}
