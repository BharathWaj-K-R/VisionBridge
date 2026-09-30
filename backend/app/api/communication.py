import datetime as dt
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import CommunicationUsage, CommunicationWord, PersonalizationProfile, QuickAccessSlot, User
from app.db.session import get_db
from app.schemas.schemas import (
    CommunicationUsageCreate,
    CommunicationWordCreate,
    CommunicationWordOut,
    PersonalizationProfileCreate,
    PersonalizationProfileOut,
    PersonalizationProfileUpdate,
    QuickAccessPayload,
)

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


def _profile(item: PersonalizationProfile) -> dict:
    try:
        config = json.loads(item.config_json or "{}")
    except (TypeError, json.JSONDecodeError):
        config = {}
    return {
        "id": item.id,
        "name": item.name,
        "config": config,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


@router.get("/profiles", response_model=list[PersonalizationProfileOut])
def list_profiles(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    items = (
        db.query(PersonalizationProfile)
        .filter(PersonalizationProfile.user_id == current_user.id)
        .order_by(PersonalizationProfile.created_at.asc())
        .all()
    )
    return [_profile(item) for item in items]


@router.post("/profiles", response_model=PersonalizationProfileOut, status_code=201)
def create_profile(
    payload: PersonalizationProfileCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    name = payload.name.strip()
    exists = (
        db.query(PersonalizationProfile)
        .filter(
            PersonalizationProfile.user_id == current_user.id,
            PersonalizationProfile.name.ilike(name),
        )
        .first()
    )
    if exists:
        raise HTTPException(status_code=409, detail="A profile with that name already exists.")

    item = PersonalizationProfile(
        user_id=current_user.id,
        name=name,
        config_json=payload.config.model_dump_json(),
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return _profile(item)


@router.put("/profiles/{profile_id}", response_model=PersonalizationProfileOut)
def update_profile(
    profile_id: int,
    payload: PersonalizationProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = (
        db.query(PersonalizationProfile)
        .filter(
            PersonalizationProfile.id == profile_id,
            PersonalizationProfile.user_id == current_user.id,
        )
        .first()
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Profile not found.")

    if payload.name is not None:
        name = payload.name.strip()
        conflict = (
            db.query(PersonalizationProfile)
            .filter(
                PersonalizationProfile.user_id == current_user.id,
                PersonalizationProfile.id != profile_id,
                PersonalizationProfile.name.ilike(name),
            )
            .first()
        )
        if conflict:
            raise HTTPException(status_code=409, detail="A profile with that name already exists.")
        item.name = name

    item.config_json = payload.config.model_dump_json()
    item.updated_at = dt.datetime.now(dt.timezone.utc)
    db.commit()
    db.refresh(item)
    return _profile(item)


@router.delete("/profiles/{profile_id}")
def delete_profile(
    profile_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = (
        db.query(PersonalizationProfile)
        .filter(
            PersonalizationProfile.id == profile_id,
            PersonalizationProfile.user_id == current_user.id,
        )
        .first()
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Profile not found.")

    remaining = (
        db.query(PersonalizationProfile)
        .filter(
            PersonalizationProfile.user_id == current_user.id,
            PersonalizationProfile.id != profile_id,
        )
        .count()
    )
    if remaining == 0:
        raise HTTPException(status_code=409, detail="Keep at least one profile. Create another profile before deleting the last one.")

    db.query(CommunicationUsage).filter(
        CommunicationUsage.user_id == current_user.id,
        CommunicationUsage.profile_id == item.id,
    ).delete(synchronize_session=False)

    db.delete(item)
    db.commit()
    return {"deleted": True, "profile_id": profile_id}


@router.post("/usage")
def record_usage(
    payload: CommunicationUsageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    phrase = payload.phrase.strip()
    profile_id = payload.profileId
    if profile_id is not None:
        owns_profile = (
            db.query(PersonalizationProfile)
            .filter(
                PersonalizationProfile.id == profile_id,
                PersonalizationProfile.user_id == current_user.id,
            )
            .first()
        )
        if owns_profile is None:
            raise HTTPException(status_code=404, detail="Profile not found.")

    item_query = db.query(CommunicationUsage).filter(
        CommunicationUsage.user_id == current_user.id,
        CommunicationUsage.phrase.ilike(phrase),
    )
    if profile_id is None:
        item_query = item_query.filter(CommunicationUsage.profile_id.is_(None))
    else:
        item_query = item_query.filter(CommunicationUsage.profile_id == profile_id)
    item = item_query.first()
    now = dt.datetime.now(dt.timezone.utc)
    if item is None:
        item = CommunicationUsage(
            user_id=current_user.id,
            profile_id=profile_id,
            phrase=phrase,
            usage_count=1,
            last_used_at=now,
        )
        db.add(item)
    else:
        item.usage_count += 1
        item.last_used_at = now
    db.commit()
    return {"phrase": item.phrase, "usage_count": item.usage_count}


@router.get("/most-used")
def most_used(
    profileId: int | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    usage_query = db.query(CommunicationUsage).filter(CommunicationUsage.user_id == current_user.id)
    if profileId is not None:
        owns_profile = (
            db.query(PersonalizationProfile)
            .filter(
                PersonalizationProfile.id == profileId,
                PersonalizationProfile.user_id == current_user.id,
            )
            .first()
        )
        if owns_profile is None:
            raise HTTPException(status_code=404, detail="Profile not found.")
        usage_query = usage_query.filter(CommunicationUsage.profile_id == profileId)
    rows = (
        usage_query
        .order_by(CommunicationUsage.usage_count.desc(), CommunicationUsage.last_used_at.desc())
        .limit(12)
        .all()
    )
    return [
        {
            "phrase": row.phrase,
            "usage_count": row.usage_count,
            "last_used_at": row.last_used_at,
        }
        for row in rows
    ]
