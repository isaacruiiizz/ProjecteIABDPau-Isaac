from datetime import datetime, timezone

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase


async def create_match(db: AsyncIOMotorDatabase, doc: dict) -> str:
    result = await db["matches"].insert_one(doc)
    return str(result.inserted_id)


async def update_match_config(db: AsyncIOMotorDatabase, match_id: str, config: dict) -> dict | None:
    return await db["matches"].find_one_and_update(
        {"_id": ObjectId(match_id)},
        {"$set": {**config, "updated_at": datetime.now(timezone.utc)}},
        return_document=True,
    )
