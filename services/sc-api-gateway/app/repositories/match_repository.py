from motor.motor_asyncio import AsyncIOMotorDatabase


async def create_match(db: AsyncIOMotorDatabase, doc: dict) -> str:
    result = await db["matches"].insert_one(doc)
    return str(result.inserted_id)
