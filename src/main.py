from fastapi import FastAPI
from models.ChatHistoryModel import ChatHistoryModel
from models.VectorStoreModel import VectorStoreModel
from modules.llm.LLMProviderFactory import LLMProviderFactory
from routes.base import base_router
from routes.file import file_router
from routes.chat import chat_router
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import AsyncQdrantClient
from helpers.config import get_settings
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.warning("Starting Fusion-Ed")
    settings = get_settings()

    app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    app.mongo_client = app.mongo_conn[settings.MONGODB_DATABASE]
    app.qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    app.vector_store = await VectorStoreModel.create_instance(app.qdrant_client)
    app.chat_history_model = await ChatHistoryModel.create_instance(app.mongo_client)

    llm_factory = LLMProviderFactory()
    app.llm = await llm_factory.create(provider=settings.LLM_PROVIDER)

    try:    
        yield
    finally:
        logger.info("Shutting down Fusion-Ed")
        await app.mongo_conn.close()
        await app.qdrant_client.close()



app = FastAPI(lifespan=lifespan)
app.include_router(base_router)
app.include_router(file_router)
app.include_router(chat_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8070)