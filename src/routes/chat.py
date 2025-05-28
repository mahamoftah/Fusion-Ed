from fastapi import APIRouter, Depends, Request
from controllers.ChatController import ChatController
from controllers.DataExtractionController import DataExtractionController
from controllers.RagController import RagController
from helpers.config import Settings, get_settings
from routes.schemas.chat import ChatHistory, ChatHistoryRequest, ChatHistoryResponse, ChatRequest, ChatResponse
from modules.llm.LLMProviderFactory import LLMProviderFactory
import logging

logger = logging.getLogger(__name__)

chat_router = APIRouter(
    prefix="/api/v1/chat",
    tags=["chat"]
)

@chat_router.post("/answer",response_model=ChatResponse)
async def upload_file(request: Request, 
                      chat_request: ChatRequest,
                      settings: Settings = Depends(get_settings)):
    
    chat_controller = ChatController(llm=request.app.llm, chat_history_model=request.app.chat_history_model, vector_store=request.app.vector_store)
    response = await chat_controller.generate_response(chat_request.question, chat_request.user_id, chat_request.chat_id)
    logger.info(f"Response: {response}")

    return ChatResponse(
        answer=response
    )

@chat_router.get("/history",response_model=ChatHistoryResponse)
async def get_chat_history(request: Request,
                          chat_request: ChatHistoryRequest,
                          settings: Settings = Depends(get_settings)):
    
    chat_controller = ChatController(llm=request.app.llm, chat_history_model=request.app.chat_history_model, vector_store=request.app.vector_store)
    response = await chat_controller.get_chat_history(chat_request.user_id)

    return ChatHistoryResponse(
        history=[ChatHistory(
            query=chat.get("question", ""),
            response=chat.get("answer", ""),
            timestamp=chat.get("timestamp", "")
        ) for chat in response]
    )

@chat_router.post("/llm/update")
async def update_llm_config(request: Request, config: dict):
    try:
        llm_factory = LLMProviderFactory()
        new_llm = await llm_factory.create(
            provider=config["provider"],
            model_id=config["model_id"],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"]
        )
        
        if new_llm:
            request.app.llm = new_llm
            return {"status": "success", "message": f"Successfully switched to {config['provider']}"}
        else:
            return {"status": "error", "message": "Failed to create new LLM instance"}
    except Exception as e:
        logger.error(f"Error updating LLM configuration: {str(e)}")
        return {"status": "error", "message": str(e)}