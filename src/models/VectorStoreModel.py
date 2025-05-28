from datetime import datetime
import uuid
from models.BaseDataModel import BaseDataModel
from models.schemas.VectorStoreSchema import VectorStoreMetadata, VectorStoreSchema
import logging
from typing import Any, Dict, List, Optional
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from qdrant_client.http.exceptions import UnexpectedResponse
import numpy as np
from models.enums.VectorStoreEnum import VectorStoreEnum


class VectorStoreModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client)
        self.logger = logging.getLogger(__name__)
        self.collection_name = VectorStoreEnum.VECTOR_STORE_COLLECTION.value
        self.qdrant_client = self.db_client


    @classmethod
    async def create_instance(cls, db_client: object):
        try:
            instance = cls(db_client)
            await instance.init_collection()
            return instance
        except Exception as e:
            logging.error(f"Error creating VectorStoreModel instance: {str(e)}")
            raise

    async def init_collection(self):
        try:
            # Check if collection exists
            collections = await self.qdrant_client.get_collections()
            if not any(c.name == self.collection_name for c in collections.collections):
                await self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.settings.EMBEDDING_SIZE,
                        distance=Distance.COSINE
                    )
                )

                self.logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                self.logger.info(f"Collection {self.collection_name} already exists")
        except UnexpectedResponse as e:
            self.logger.error(f"Qdrant API error while initializing collection: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in init_collection: {str(e)}")
            raise


    async def save_chunks(self, documents_with_embeddings: List[Dict[str, Any]]) -> bool:
        try:

            points = []
            for chunk in documents_with_embeddings:

                metadata = chunk["metadata"]
                metadata["file_id"] = metadata["file_id"]
                metadata["file_name"] = metadata["file_name"]
                metadata["file_url"] = metadata["file_url"]
                metadata["course_id"] = metadata["course_id"]
                metadata["chunk_order"] = metadata["chunk_order"]
                metadata["current_date"] = datetime.now().strftime("%Y-%m-%d")
                metadata["text"] = chunk["text"]
                embedding = chunk["embedding"]

                point_id = str(uuid.uuid4())
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=metadata
                )
                points.append(point)

            # Insert point into collection
            await self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return True
        except UnexpectedResponse as e:
            self.logger.error(f"Qdrant API error while inserting chunk: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while inserting chunk: {str(e)}")
            raise


    async def search_similar_chunks(self, 
                                  query_vector: List[float], 
                                  limit: int = 5,
                                  score_threshold: float = 0.7) -> List[VectorStoreSchema]:
        try:
            # Search for similar vector0s
            search_result = await self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            return [
                {
                    "text": point.payload["text"],
                    "metadata": point.payload,
                    "score": point.score
                } for point in search_result]
        except UnexpectedResponse as e:
            self.logger.error(f"Qdrant API error while searching chunks: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while searching chunks: {str(e)}")
            raise


    async def get_chunk_by_id(self, chunk_id: str) -> Optional[VectorStoreSchema]:
        try:
            point = self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id]
            )
            
            if not point:
                return None
                
            return VectorStoreSchema.create_from_qdrant_point(point[0])
        except UnexpectedResponse as e:
            self.logger.error(f"Qdrant API error while retrieving chunk: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while retrieving chunk: {str(e)}")
            raise
