from datetime import datetime
from bson import ObjectId
from pydantic import BaseModel, Field
from typing import Dict, Optional
from pymongo import ASCENDING, DESCENDING



class Metadata(BaseModel):
    similar_chunks: Optional[list[Dict]] = None
    timestamp: Optional[datetime] = None


class ChatHistorySchema(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    user_id: str
    chat_id: str
    question: str
    answer: str
    metadata: Metadata

    class Config:
        arbitrary_types_allowed = True


    @classmethod
    async def get_indexes(cls):
        return [
            ("user_id", ASCENDING),
            ("chat_id", ASCENDING),
            ("metadata.timestamp", DESCENDING)
        ]