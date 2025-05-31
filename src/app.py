import os
import sys
import asyncio
import streamlit as st
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import AsyncQdrantClient
import logging
import uuid
import nest_asyncio
import certifi
import backoff

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from src.models.ChatHistoryModel import ChatHistoryModel
from src.models.VectorStoreModel import VectorStoreModel
from src.modules.llm.LLMProviderFactory import LLMProviderFactory
from src.helpers.config import get_settings
from src.routes.chat import answer
from src.helpers.config import get_settings
settings = get_settings()

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

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
async def connect_to_mongodb():
    """Connect to MongoDB with retry logic."""
    try:
        client = AsyncIOMotorClient(settings.MONGODB_URL, tlsCAFile=certifi.where())
        # Test the connection
        await client.admin.command('ping')
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
async def connect_to_qdrant():
    """Connect to Qdrant with retry logic."""
    try:
        client = AsyncQdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        # Test the connection
        await client.get_collections()
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise

async def initialize_app():
    global llm, chat_history_model, vector_store, mongo_conn, qdrant_client
    
    logger.info("Starting Fusion-Ed initialization")
    settings = get_settings()

    try:
        # Connect to MongoDB with retry
        mongo_conn = await connect_to_mongodb()
        mongo_client = mongo_conn[settings.MONGODB_DATABASE]
        
        # Connect to Qdrant with retry
        qdrant_client = await connect_to_qdrant()
        
        # Initialize models
        vector_store = await VectorStoreModel.create_instance(qdrant_client)
        chat_history_model = await ChatHistoryModel.create_instance(mongo_client)

        # Initialize LLM
        llm_factory = LLMProviderFactory()
        llm = await llm_factory.create(provider=settings.LLM_PROVIDER)
        
        logger.info("Fusion-Ed initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Fusion-Ed: {e}")
        # Clean up any partial initialization
        await cleanup()
        raise

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

async def main():
    st.set_page_config(page_title="Fusion-Ed Chat Interface", layout="wide")
    st.title("Chat with Fusion-Ed")
    
    initialize_session_state()
    
    # Sidebar for LLM configuration
    st.sidebar.header("LLM Configuration")
    
    # Model selection dropdown
    model_options = ["Gemini", "Qwen 8B", "LLAMA 8B", "LLAMA 70B", "Gemma 9B", "Azure GPT-4o-mini"]
    selected_model = st.sidebar.selectbox("Select Model", model_options, index=0)

    global llm
    # Model configuration
    if selected_model == "Gemini":
        llm = await LLMProviderFactory().create(
            provider="GOOGLE",
            api_key=settings.LLM_API_KEY,
            model_id="gemini-1.5-flash"
        )

    elif selected_model == "Qwen 8B":
        llm = await LLMProviderFactory().create(
            provider="OPENROUTER",
            api_key=settings.OPENROUTER_API_KEY,
            model_id="qwen/qwen3-8b",
            base_url="https://openrouter.ai/api/v1"
        )
    elif selected_model == "LLAMA 8B":
        llm = await LLMProviderFactory().create(
            provider="GROQ",
            api_key=settings.GROQ_API_KEY,
            model_id="llama3-8b-8192"
        )
    elif selected_model == "LLAMA 70B":
        llm = await LLMProviderFactory().create(
            provider="GROQ",
            api_key=settings.GROQ_API_KEY,
            model_id="llama3-70b-8192"
        )
    elif selected_model == "Gemma 9B":
        llm = await LLMProviderFactory().create(
            provider="GROQ",
            api_key=settings.GROQ_API_KEY,
            model_id="gemma2-9b-it"
        )
    elif selected_model == "Azure GPT-4o-mini":
        llm = await LLMProviderFactory().create(
            provider="AZUREOPENAI",
            api_key=settings.AZURE_OPENAI_API_KEY,
            model_id="gpt-4o-mini"
        )
    

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
            with st.spinner("Thinking..."):
                response = run_async(send_message(prompt))
                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    try:
        # Initialize the application
        run_async(initialize_app())
        
        # Run the main Streamlit app
        run_async(main())
    except:
        # Cleanup will only happen when the application is actually shutting down
        run_async(cleanup())