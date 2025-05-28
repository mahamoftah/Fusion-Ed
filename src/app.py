import os
import sys
import asyncio
import streamlit as st
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import AsyncQdrantClient
import logging
import uuid
import nest_asyncio
from functools import partial

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # Insert at beginning to ensure it's found first

from src.models.ChatHistoryModel import ChatHistoryModel
from src.models.VectorStoreModel import VectorStoreModel
from src.modules.llm.LLMProviderFactory import LLMProviderFactory
from src.helpers.config import get_settings
from src.routes.chat import answer
import certifi

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable nested event loops
nest_asyncio.apply()

# Global variables
llm = None
chat_history_model = None
vector_store = None
mongo_conn = None
qdrant_client = None

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        raise ex

async def initialize_app():
    global llm, chat_history_model, vector_store, mongo_conn, qdrant_client
    
    logger.warning("Starting Fusion-Ed")
    settings = get_settings()

    try:
        mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL, tlsCAFile=certifi.where())
        mongo_client = mongo_conn[settings.MONGODB_DATABASE]
        qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        vector_store = await VectorStoreModel.create_instance(qdrant_client)
        chat_history_model = await ChatHistoryModel.create_instance(mongo_client)

        llm_factory = LLMProviderFactory()
        llm = await llm_factory.create(provider=settings.LLM_PROVIDER)
        
        logger.info("Fusion-Ed initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Fusion-Ed: {e}")
        raise e

async def cleanup():
    global mongo_conn, qdrant_client
    logger.info("Cleaning up Fusion-Ed resources")
    if mongo_conn:
        mongo_conn.close()
    if qdrant_client:
        await qdrant_client.close()

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "user_id" not in st.session_state:
        # Generate a unique user ID for this session
        st.session_state.user_id = str(uuid.uuid4())
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = ""

async def send_message(message: str) -> str:
    """Send a message to the chat endpoint and get the response."""
    try:
        response = await answer(message, st.session_state.user_id, st.session_state.chat_id, llm, chat_history_model, vector_store)
        return response.answer
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        return f"Error: {str(e)}"

def run_async(coro):
    """Run an async function in the current event loop."""
    loop = get_or_create_eventloop()
    return loop.run_until_complete(coro)

def main():
    st.set_page_config(page_title="Fusion-Ed Chat Interface", layout="wide")
    st.title("Chat with Fusion-Ed")
    
    initialize_session_state()
    
    # Sidebar for LLM configuration
    st.sidebar.header("LLM Configuration")
    st.sidebar.write(f"Coming Soon!")
    
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
            response = run_async(send_message(prompt))
            st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    try:
        # Initialize the application
        run_async(initialize_app())
        
        # Run the main Streamlit app
        main()
    finally:
        # Cleanup
        run_async(cleanup())