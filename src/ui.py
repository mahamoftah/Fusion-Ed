import streamlit as st
import requests
import json
from typing import List, Dict
import os
import uuid
from pathlib import Path
import uvicorn
import threading
import time
from fastapi import FastAPI
from src.routes.base import base_router
from src.routes.file import file_router
from src.routes.chat import chat_router
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import AsyncQdrantClient
from src.helpers.config import get_settings
from src.models.ChatHistoryModel import ChatHistoryModel
from src.models.VectorStoreModel import VectorStoreModel
from src.modules.llm.LLMProviderFactory import LLMProviderFactory
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
API_BASE_URL = "http://localhost:8010/api/v1"
DATA_DIR = Path("data")


def ensure_data_dir():
    """Ensure the data directory exists."""
    DATA_DIR.mkdir(exist_ok=True)

def get_data_files() -> List[str]:
    """Get list of files in the data directory."""
    ensure_data_dir()
    return [f.name for f in DATA_DIR.iterdir() if f.is_file()]

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    if "current_llm" not in st.session_state:
        st.session_state.current_llm = "OpenAI"
    if "server_started" not in st.session_state:
        st.session_state.server_started = False

def start_fastapi_server():
    """Start the FastAPI server in a separate thread."""
    app = FastAPI()
    
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

    config = uvicorn.Config(app, host="127.0.0.1", port=8010, log_level="info")
    server = uvicorn.Server(config)
    server.run()

def update_llm_config():
    """Update LLM configuration in the backend."""
    try:
        provider = st.session_state.current_llm
        config = LLM_PROVIDERS[provider]
        
        # Create the configuration update request
        config_request = {
            "provider": provider.upper(),
            "model_id": config["model_id"],
            "max_tokens": config["max_tokens"],
            "temperature": config["temperature"]
        }
        
        # Send the request to update LLM configuration
        response = requests.post(
            f"{API_BASE_URL}/llm/update",
            json=config_request
        )
        
        if response.status_code == 200:
            st.success(f"Successfully switched to {provider}")
        else:
            st.error(f"Error updating LLM configuration: {response.text}")
    except Exception as e:
        st.error(f"Error updating LLM configuration: {str(e)}")

def send_message(message: str) -> str:
    """Send a message to the chat endpoint and get the response."""
    try:
        # Create the chat request payload according to the schema
        chat_request = {
            "user_id": st.session_state.user_id,
            "chat_id": "",  # Empty string as requested
            "question": message
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat/answer",
            json=chat_request
        )
        
        if response.status_code == 200:
            return response.json()["answer"]
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.set_page_config(page_title="Fusion-Ed Chat Interface", layout="wide")
    st.title("Chat with Fusion-Ed")
    
    initialize_session_state()
    
    # Start FastAPI server if not already started
    if not st.session_state.server_started:
        server_thread = threading.Thread(target=start_fastapi_server, daemon=True)
        server_thread.start()
        st.session_state.server_started = True
        time.sleep(2)  # Give the server time to start
    
    # Sidebar for LLM configuration
    st.sidebar.header("LLM Configuration")
    st.sidebar.write("Coming Soon!")

    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            response = send_message(prompt)
            st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()