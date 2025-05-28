from src.models.BaseDataModel import BaseDataModel
from src.models.schemas.ChatHistorySchema import ChatHistorySchema
import logging
from pymongo.errors import PyMongoError
from typing import List
from src.models.enums.ChatHistoryEnum import ChatHistoryEnum

class ChatHistoryModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client)
        self.collection_name = ChatHistoryEnum.CHAT_HISTORY_COLLECTION.value
        self.collection = self.db_client[self.collection_name]
        self.logger = logging.getLogger(__name__)


    @classmethod
    async def create_instance(cls, db_client: object):
        try:
            instance = cls(db_client)
            await instance.init_collection()
            return instance
        except Exception as e:
            logging.error(f"Error creating ChatHistoryModel instance: {str(e)}")
            raise


    async def init_collection(self):
        try:

            all_collections = await self.db_client.list_collection_names()
            if self.collection_name not in all_collections:
                await self.db_client.create_collection(self.collection_name)
                indexes = await ChatHistorySchema.get_indexes()
                for index in indexes:
                    await self.collection.create_index(index)
                self.logger.info(f"Collection {self.collection_name} initialized successfully")
            else:
                self.logger.info(f"Collection {self.collection_name} already exists")
        
        except PyMongoError as e:
            self.logger.error(f"Error initializing collection: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in init_collection: {str(e)}")
            raise

    
    async def save_chat_history(self, chat_history: ChatHistorySchema) -> bool:
        try:

            formatted_chat = await self.format_chat_history(chat_history)
            await self.collection.insert_one(formatted_chat)
            return True
        
        except PyMongoError as e:
            self.logger.error(f"Database error while inserting chat history: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while inserting chat history: {str(e)}")
            raise


    async def format_chat_history(self, chat_history: ChatHistorySchema) -> dict:
        try:

            return {
                "user_id": chat_history.user_id,
                "chat_id": chat_history.chat_id,
                "question": chat_history.question,
                "answer": chat_history.answer,
                "metadata": {
                    "similar_chunks": chat_history.metadata.similar_chunks,
                    "timestamp": chat_history.metadata.timestamp
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error formatting chat history: {str(e)}")
            raise


    async def get_chat_history(self, user_id: str, limit: int = 10) -> List[dict]:
        try:

            chat_history = self.collection.find(
                {"user_id": user_id}
            ).sort("metadata.timestamp", -1).limit(limit)

            return await chat_history.to_list(length=limit)
        
        except PyMongoError as e:
            self.logger.error(f"Database error while fetching chat history: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while fetching chat history: {str(e)}")
            raise
    

    async def get_chat_history_by_chat_id(self, company_id: str, chat_id: str, limit: int = 10) -> List[dict]:
        try:
        
            chat_history = await self.collection.find(
                {"company_id": company_id, "chat_id": chat_id}
            ).sort("metadata.timestamp", -1).limit(limit)
            return await chat_history.to_list(length=limit)
        
        except PyMongoError as e:
            self.logger.error(f"Database error while fetching chat history by chat_id: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while fetching chat history by chat_id: {str(e)}")
            raise
    



