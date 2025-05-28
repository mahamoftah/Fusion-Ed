

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from modules.BaseModule import BaseModule
from modules.llm.LLMEnums import *
from modules.llm.providers.BaseProvider import BaseProvider


class LLMProviderFactory(BaseModule):
    def __init__(self):
        super().__init__()

    async def create(self, provider: str, api_key: str = None, model_id: str = None, max_tokens: int = None, temperature: float = None, base_url: str = None):

        if provider == LLMEnums.OPENAI.value or provider == LLMEnums.DEEPSEEK.value:
            client = ChatOpenAI(
                api_key = api_key or self.settings.LLM_API_KEY,
                model = model_id or self.settings.LLM_MODEL_ID,
                max_tokens = max_tokens or self.settings.LLM_MAX_TOKENS,
                temperature = temperature or self.settings.LLM_TEMPERATURE,
                base_url = base_url or self.settings.LLM_API_URL
            )
            return BaseProvider(client)

        if provider == LLMEnums.GOOGLE.value:
            client = ChatGoogleGenerativeAI(
                api_key=api_key or self.settings.LLM_API_KEY,
                model=model_id or self.settings.LLM_MODEL_ID,
                max_tokens=max_tokens or self.settings.LLM_MAX_TOKENS,
                temperature=temperature or self.settings.LLM_TEMPERATURE,
            )
            return BaseProvider(client)

        if provider == LLMEnums.GROQ.value:
            client = ChatGroq(
                api_key=api_key or self.settings.LLM_API_KEY,
                model=model_id or self.settings.LLM_MODEL_ID,
                max_tokens=max_tokens or self.settings.LLM_MAX_TOKENS,
                temperature=temperature or self.settings.LLM_TEMPERATURE,
            )
            return BaseProvider(client)

        self.logger.error(f"Invalid provider: {provider}")
        return None