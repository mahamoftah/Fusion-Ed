from typing import List, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    user_id: str
    chat_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str

class ChatHistoryRequest(BaseModel):
    user_id: str
    chat_id: str
    timestamp: Optional[str] = None
    limit: Optional[int] = 10

class ChatHistory(BaseModel):
    query: Optional[str] = None
    response: Optional[str] = None
    timestamp: Optional[str] = None

class ChatHistoryResponse(BaseModel):
    history: Optional[List[Optional[ChatHistory]]] = None


