from fastapi import APIRouter, Depends, Request
from src.controllers.ChatController import ChatController
from src.controllers.QueryTranslationController import QueryTranslationController
from src.helpers.config import Settings, get_settings
from src.routes.schemas.chat import ChatHistory, ChatHistoryRequest, ChatHistoryResponse, ChatRequest, ChatResponse
from src.modules.llm.LLMProviderFactory import LLMProviderFactory
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

    query_translator = QueryTranslationController(llm=request.app.llm)  
    chat_controller = ChatController(llm=request.app.llm, chat_history_model=request.app.chat_history_model, vector_store=request.app.vector_store, query_translator=query_translator)
    response = await chat_controller.generate_response(chat_request.question, chat_request.user_id, chat_request.chat_id)
    logger.info(f"Response: {response}")

    return ChatResponse(
        answer=response
    )


# async def answer(question: str, user_id: str, chat_id: str, llm, chat_history_model, vector_store):
    
#     query_translator = QueryTranslationController(llm=llm)
#     chat_controller = ChatController(llm=llm, chat_history_model=chat_history_model, vector_store=vector_store, query_translator=query_translator)
#     response = await chat_controller.generate_response(question, user_id, chat_id)
#     logger.info(f"Response: {response}")

#     return ChatResponse(
#         answer=response
#     )


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

