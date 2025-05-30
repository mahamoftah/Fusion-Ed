import os
import sys
import streamlit as st
import requests
import json
from typing import List, Dict
import uuid
from pathlib import Path

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Constants
API_BASE_URL = "http://localhost:8010/api/v1"  # Update this if your FastAPI server runs on a different port
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
        # Generate a unique user ID for this session
        st.session_state.user_id = str(uuid.uuid4())
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = ""

def upload_file(file):
    """Upload a file to the backend."""
    try:
        ensure_data_dir()
        
        # Save file to data directory
        file_path = DATA_DIR / file.name
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        
        # Create the file request payload according to the schema
        file_request = {
            "files": [{
                "file_id": str(uuid.uuid4()),
                "file_url": str(file_path),  # Send the file path
                "file_name": file.name,
                "course_id": "default"
            }]
        }
        
        # Send the request with file metadata
        response = requests.post(
            f"{API_BASE_URL}/files/upload",
            json=file_request
        )
        
        if response.status_code == 200:
            st.success(f"Successfully uploaded {file.name}")
            st.session_state.uploaded_files.append(file.name)
        else:
            st.error(f"Error uploading {file.name}: {response.text}")
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")

def send_message(message: str) -> str:
    """Send a message to the chat endpoint and get the response."""
    try:
        # Create the chat request payload according to the schema
        chat_request = {
            "user_id": st.session_state.user_id,
            "chat_id": st.session_state.chat_id,
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

    st.sidebar.header("LLM Configuration")
    
    # Sidebar for file upload
    # with st.sidebar:
    #     st.header("Document Upload")
    #     uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx"])
    #     if uploaded_file:
    #         upload_file(uploaded_file)
        
    #     st.header("Uploaded Files")
    #     for file in st.session_state.uploaded_files:
    #         st.write(f"📄 {file}")
    
    
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
                response = send_message(prompt)
                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

    