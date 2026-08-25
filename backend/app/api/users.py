from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import SignerAdapter, User
from app.db.session import get_db
from app.schemas.schemas import AdapterOut

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {"id": current_user.id, "username": current_user.username, "created_at": current_user.created_at}


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
        .filter(SignerAdapter.id == adapter_id, SignerAdapter.owner_id == current_user.id)
        .first()
    )
    if not adapter:
        raise HTTPException(status_code=404, detail="Adapter not found")
    db.delete(adapter)
    db.commit()
    return {"deleted": True, "adapter_id": adapter_id}
