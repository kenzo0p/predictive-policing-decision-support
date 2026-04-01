from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List


def mongo_available() -> bool:
    try:
        import pymongo  # noqa: F401

        return True
    except Exception:
        return False


def mongo_configured() -> bool:
    return bool(os.getenv("MONGO_URI")) and bool(os.getenv("MONGO_DB_NAME", "crime_analytics"))


def _get_collection():
    if not mongo_available() or not mongo_configured():
        return None

    from pymongo import MongoClient

    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("MONGO_DB_NAME", "crime_analytics")]
    return db["predictions"]


def save_prediction(payload: Dict[str, Any], output: Dict[str, Any]) -> bool:
    collection = _get_collection()
    if collection is None:
        return False

    doc = {
        "input": payload,
        "output": output,
        "created_at": datetime.now(timezone.utc),
    }
    collection.insert_one(doc)
    return True


def recent_predictions(limit: int = 20) -> List[Dict[str, Any]]:
    collection = _get_collection()
    if collection is None:
        return []

    docs = (
        collection.find({}, {"_id": 0})
        .sort("created_at", -1)
        .limit(limit)
    )
    return list(docs)
