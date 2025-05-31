from enum import Enum

class LLMEnums(Enum):
    AZUREOPENAI = "AZUREOPENAI"
    GOOGLE = "GOOGLE"
    GROQ = "GROQ"
    OPENROUTER = "OPENROUTER"
    DEEPSEEK = "DEEPSEEK"
    OPENAI = "OPENAI"

class OpenAIEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class GroqEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    
class DocumentTypeEnum(Enum):
    DOCUMENT = "document"
    QUERY = "query"