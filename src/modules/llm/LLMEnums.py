from enum import Enum

class LLMEnums(Enum):
    OPENAI = "OPENAI"
    GOOGLE = "GOOGLE"
    GROQ = "GROQ"
    DEEPSEEK = "DEEPSEEK"

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