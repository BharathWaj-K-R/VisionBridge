from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import SignerAdapter, User
from app.db.session import get_db
from app.schemas.schemas import AdapterOut
from app.services.calibration_service import (
    finalize_staged_adapter_weight_delete,
    restore_staged_adapter_weight,
    stage_adapter_weight_delete,
)

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "created_at": current_user.created_at,
    }


@router.get("/me/adapters", response_model=list[AdapterOut])
def list_my_adapters(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return (
        db.query(SignerAdapter)
        .filter(SignerAdapter.owner_id == current_user.id)
        .order_by(SignerAdapter.created_at.desc())
        .all()
    )


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
    if not adapter:
        raise HTTPException(status_code=404, detail="Adapter not found")

    weights_path = adapter.weights_path
    try:
        tombstone = stage_adapter_weight_delete(weights_path)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Adapter deletion could not be staged safely",
        ) from exc

    db.delete(adapter)
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        if tombstone is not None:
            restore_staged_adapter_weight(tombstone, weights_path)
        raise HTTPException(
            status_code=500,
            detail="Adapter deletion could not be completed safely",
        ) from exc

    try:
        finalize_staged_adapter_weight_delete(tombstone)
    except OSError as exc:
        # The DB row is already gone. Keep the response successful but leave
        # the staged file for operational cleanup rather than resurrecting DB state.
        raise HTTPException(
            status_code=500,
            detail="Adapter record deleted, but stored weight cleanup is pending",
        ) from exc

    return {"deleted": True, "adapter_id": adapter_id}
