from pydantic import BaseModel
from typing import List


class VectorStoreMetadata(BaseModel):
    file_id: str
    file_name: str
    file_url: str
    chunk_order: int


class VectorStoreSchema(BaseModel):
    chunk_id: str
    text: str
    embedding: List[float]
    metadata: VectorStoreMetadata