from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import CommunicationWord, QuickAccessSlot, User
from app.db.session import get_db
from app.schemas.schemas import CommunicationWordCreate, CommunicationWordOut, QuickAccessPayload

router = APIRouter(prefix="/communication", tags=["communication"])


def _word(item: CommunicationWord) -> dict:
    return {
        "id": item.id,
        "phrase": item.phrase,
        "category": item.category,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


@router.get("/words", response_model=list[CommunicationWordOut])
def list_custom_words(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    items = (
        db.query(CommunicationWord)
        .filter(CommunicationWord.user_id == current_user.id)
        .order_by(CommunicationWord.category.asc(), CommunicationWord.phrase.asc())
        .all()
    )
    return [_word(item) for item in items]


@router.post("/words", response_model=CommunicationWordOut, status_code=201)
def create_custom_word(
    payload: CommunicationWordCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    phrase = payload.phrase.strip()
    category = payload.category.strip()
    exists = (
        db.query(CommunicationWord)
        .filter(
            CommunicationWord.user_id == current_user.id,
            CommunicationWord.phrase.ilike(phrase),
        )
        .first()
    )
    if exists:
        raise HTTPException(status_code=409, detail="That custom phrase already exists.")
    item = CommunicationWord(user_id=current_user.id, phrase=phrase, category=category)
    db.add(item)
    db.commit()
    db.refresh(item)
    return _word(item)


@router.put("/words/{word_id}", response_model=CommunicationWordOut)
def update_custom_word(
    word_id: int,
    payload: CommunicationWordCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = (
        db.query(CommunicationWord)
        .filter(CommunicationWord.id == word_id, CommunicationWord.user_id == current_user.id)
        .first()
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Custom phrase not found.")

    phrase = payload.phrase.strip()
    category = payload.category.strip()
    conflict = (
        db.query(CommunicationWord)
        .filter(
            CommunicationWord.user_id == current_user.id,
            CommunicationWord.id != word_id,
            CommunicationWord.phrase.ilike(phrase),
        )
        .first()
    )
    if conflict:
        raise HTTPException(status_code=409, detail="That custom phrase already exists.")

    item.phrase = phrase
    item.category = category
    db.commit()
    db.refresh(item)
    return _word(item)


@router.delete("/words/{word_id}")
def delete_custom_word(
    word_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = (
        db.query(CommunicationWord)
        .filter(CommunicationWord.id == word_id, CommunicationWord.user_id == current_user.id)
        .first()
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Custom phrase not found.")
    db.delete(item)
    db.commit()
    return {"deleted": True, "word_id": word_id}


@router.get("/quick-access")
def get_quick_access(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = (
        db.query(QuickAccessSlot)
        .filter(QuickAccessSlot.user_id == current_user.id)
        .all()
    )
    slots = [None] * 10
    for row in rows:
        if 0 <= row.slot < 10:
            slots[row.slot] = row.phrase
    return {"slots": slots}


@router.put("/quick-access")
def save_quick_access(
    payload: QuickAccessPayload,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    existing = {
        row.slot: row
        for row in db.query(QuickAccessSlot).filter(QuickAccessSlot.user_id == current_user.id).all()
    }

    for slot, raw_phrase in enumerate(payload.slots):
        phrase = raw_phrase.strip() if isinstance(raw_phrase, str) else None
        row = existing.get(slot)
        if row is None:
            row = QuickAccessSlot(user_id=current_user.id, slot=slot)
            db.add(row)
        row.phrase = phrase

    db.commit()
    return {"slots": payload.slots}
