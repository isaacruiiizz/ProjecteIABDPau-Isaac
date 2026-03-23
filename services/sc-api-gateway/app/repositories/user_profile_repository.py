from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase


async def get_profile_by_user_id(
    app_db: AsyncIOMotorDatabase, user_id: str
) -> dict | None:
    return await app_db["user_profiles"].find_one({"user_id": ObjectId(user_id)})
