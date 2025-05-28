import asyncio
import os
import sys
import socket
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
import json
from typing import List, Dict
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
DATA_DIR = Path("data")

def get_api_url(port: int) -> str:
    """Get the API URL for the given port."""
    return f"http://localhost:{port}/api/v1"

def send_message(message: str, port: int) -> str:
    """Send a message to the chat endpoint and get the response."""
    try:
        api_url = get_api_url(port)
        # Create the chat request payload according to the schema
        chat_request = {
            "user_id": st.session_state.user_id,
            "chat_id": "",  # Empty string as requested
            "question": message
        }
        
        response = requests.post(
            f"{api_url}/chat/answer",
            json=chat_request
        )
        
        if response.status_code == 200:
            return response.json()["answer"]
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def run_streamlit(port: int):
    """Run the Streamlit UI."""
    st.set_page_config(page_title="Fusion-Ed Chat Interface", layout="wide")
    st.title("Chat with Fusion-Ed")
    
    initialize_session_state()
    
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
            response = send_message(prompt, port)
            st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.warning("Starting Fusion-Ed")
    settings = get_settings()
    
    # Initialize connections as None
    app.mongo_conn = None
    app.qdrant_client = None
    
    try:
        # Initialize MongoDB connection with more permissive SSL configuration
        try:
            # Set environment variables for SSL
            os.environ['PYTHONHTTPSVERIFY'] = '0'
            
            mongo_client_options = {
                "tls": True,
                "tlsAllowInvalidCertificates": True,
                "tlsAllowInvalidHostnames": True,
                "tlsInsecure": True,  # More permissive SSL
                "serverSelectionTimeoutMS": 30000,  # Increased timeout
                "connectTimeoutMS": 30000,  # Increased timeout
                "socketTimeoutMS": 30000,  # Increased timeout
                "retryWrites": True,
                "w": "majority",
                "ssl_cert_reqs": "CERT_NONE",  # Disable certificate verification
                "ssl": True,
                "ssl_ca_certs": None  # Don't use CA certificates
            }
            
            # Log the MongoDB URL (without password) for debugging
            safe_url = settings.MONGODB_URL.replace(settings.MONGODB_URL.split('@')[0], '***')
            logger.info(f"Attempting to connect to MongoDB at: {safe_url}")
            
            app.mongo_conn = AsyncIOMotorClient(
                settings.MONGODB_URL,
                **mongo_client_options
            )
            
            # Test the connection with retries
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    await app.mongo_conn.admin.command('ping')
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise
                    logger.warning(f"MongoDB connection attempt {retry_count} failed: {str(e)}")
                    await asyncio.sleep(2)  # Wait before retrying
            
            app.mongo_client = app.mongo_conn[settings.MONGODB_DATABASE]
            logger.info("MongoDB connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            logger.error("Please check your MongoDB connection string and network settings")
            raise
        
        # Initialize Qdrant client
        try:
            app.qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
            app.vector_store = await VectorStoreModel.create_instance(app.qdrant_client)
            app.chat_history_model = await ChatHistoryModel.create_instance(app.mongo_client)
            logger.info("Qdrant client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise

        # Initialize LLM
        try:
            llm_factory = LLMProviderFactory()
            app.llm = await llm_factory.create(provider=settings.LLM_PROVIDER)
            logger.info("LLM initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
        
        yield
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise
    finally:
        logger.info("Shutting down Fusion-Ed")
        try:
            if app.mongo_conn is not None:
                app.mongo_conn.close()
                logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {str(e)}")
            
        try:
            if app.qdrant_client is not None:
                await app.qdrant_client.close()
                logger.info("Qdrant client closed")
        except Exception as e:
            logger.error(f"Error closing Qdrant client: {str(e)}")

app = FastAPI(lifespan=lifespan)
app.include_router(base_router)
app.include_router(file_router)
app.include_router(chat_router)

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

def is_port_in_use(port: int, host: str = '127.0.0.1') -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True

def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    port = start_port
    for _ in range(max_attempts):
        if not is_port_in_use(port):
            return port
        port += 1
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

if __name__ == "__main__":
    try:
        # Try to find an available port
        port = find_available_port()
        if port != 8000:
            logger.warning(f"Port 8000 is in use, using port {port} instead")
        
        logger.info(f"Starting API server on port {port}")
        
        # Start the FastAPI server in a separate thread
        server_thread = threading.Thread(
            target=lambda: uvicorn.run(app, host="127.0.0.1", port=port, log_level="info"),
            daemon=True
        )
        server_thread.start()
        
        # Give the server time to start
        time.sleep(2)
        
        # Run the Streamlit UI with the correct port
        run_streamlit(port)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        sys.exit(1)