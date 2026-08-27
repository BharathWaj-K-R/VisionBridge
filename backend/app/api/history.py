import datetime as dt
import io
import csv

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import TranslationLog, User
from app.db.session import get_db

router = APIRouter(prefix="/history", tags=["history"])


def _serialize(item: TranslationLog) -> dict:
    return {
        "id": item.id,
        "predicted_text": item.predicted_text,
        "confidence": item.confidence,
        "latency_ms": item.latency_ms,
        "used_adapter": bool(item.used_adapter),
        "created_at": item.created_at,
    }


def _start_for_range(value: str) -> dt.datetime | None:
    now = dt.datetime.utcnow()
    days = {"7d": 7, "30d": 30, "90d": 90}.get(value)
    return now - dt.timedelta(days=days) if days else None


@router.get("")
def list_history(
    search: str | None = Query(default=None, max_length=200),
    range: str = Query(default="7d", pattern=r"^(7d|30d|90d|all)$"),
    sort: str = Query(default="newest", pattern=r"^(newest|confidence|length)$"),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(TranslationLog).filter(TranslationLog.user_id == current_user.id)
    start = _start_for_range(range)
    if start is not None:
        query = query.filter(TranslationLog.created_at >= start)
    if search:
        query = query.filter(TranslationLog.predicted_text.ilike(f"%{search}%"))

    total_count = query.count()
    if sort == "confidence":
        query = query.order_by(TranslationLog.confidence.desc().nullslast(), TranslationLog.created_at.desc())
    elif sort == "length":
        query = query.order_by(func.length(TranslationLog.predicted_text).desc(), TranslationLog.created_at.desc())
    else:
        query = query.order_by(TranslationLog.created_at.desc())

    items = query.limit(limit).all()
    return {"items": [_serialize(item) for item in items], "count": total_count}


@router.get("/export.csv")
def export_history_csv(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    items = (
        db.query(TranslationLog)
        .filter(TranslationLog.user_id == current_user.id)
        .order_by(TranslationLog.created_at.desc())
        .all()
    )
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "created_at", "predicted_text", "confidence", "latency_ms", "used_adapter"])
    for item in items:
        writer.writerow([
            item.id,
            item.created_at.isoformat() if item.created_at else "",
            item.predicted_text,
            item.confidence,
            item.latency_ms,
            bool(item.used_adapter),
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=visionbridge-history.csv"},
    )
