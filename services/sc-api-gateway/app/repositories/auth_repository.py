from datetime import datetime

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase


async def get_user_by_email(auth_db: AsyncIOMotorDatabase, email: str) -> dict | None:
    return await auth_db["users"].find_one({"email": email})


async def get_user_by_id(auth_db: AsyncIOMotorDatabase, user_id: str) -> dict | None:
    return await auth_db["users"].find_one({"_id": ObjectId(user_id)})


async def update_user_password(
    auth_db: AsyncIOMotorDatabase, user_id: str, password_hash: str
) -> None:
    await auth_db["users"].update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"password_hash": password_hash}, "$unset": {"force_reset": ""}},
    )


async def create_refresh_token(
    auth_db: AsyncIOMotorDatabase,
    user_id: str,
    token_hash: str,
    expires_at: datetime,
) -> None:
    await auth_db["refresh_tokens"].insert_one(
        {
            "user_id": ObjectId(user_id),
            "token_hash": token_hash,
            "expires_at": expires_at,
        }
    )


async def get_refresh_token_by_hash(
    auth_db: AsyncIOMotorDatabase, token_hash: str
) -> dict | None:
    return await auth_db["refresh_tokens"].find_one({"token_hash": token_hash})


async def delete_refresh_token(
    auth_db: AsyncIOMotorDatabase, token_hash: str
) -> None:
    await auth_db["refresh_tokens"].delete_one({"token_hash": token_hash})


async def delete_all_user_refresh_tokens(
    auth_db: AsyncIOMotorDatabase, user_id: str
) -> None:
    await auth_db["refresh_tokens"].delete_many({"user_id": ObjectId(user_id)})
