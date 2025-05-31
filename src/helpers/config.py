from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str

    MONGODB_URL: str
    MONGODB_DATABASE: str
    MONGODB_COLLECTION: str    

    EMBEDDING_MODEL: str
    EMBEDDING_API_KEY: str
    EMBEDDING_SIZE: int

    QDRANT_COLLECTION_NAME: str
    QDRANT_URL: str
    QDRANT_API_KEY: str
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int

    LLM_PROVIDER: str
    LLM_API_KEY: str
    LLM_MODEL_ID: str
    LLM_MAX_TOKENS: int
    LLM_TEMPERATURE: float
    LLM_API_URL: str
    GROQ_API_KEY: str
    OPENROUTER_API_KEY: str
    AZURE_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_API_VERSION: str

    # class Config:
    #     env_file = ".env"


def get_settings():
    return Settings()
