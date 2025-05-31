

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from src.modules.BaseModule import BaseModule
from src.modules.llm.LLMEnums import *
from src.modules.llm.providers.BaseProvider import BaseProvider


class LLMProviderFactory(BaseModule):
    def __init__(self):
        super().__init__()

    async def create(self, provider: str, api_key: str = None, model_id: str = None, max_tokens: int = None, temperature: float = None, base_url: str = None):

        if provider == LLMEnums.AZUREOPENAI.value:
            client = AzureChatOpenAI(
                api_key = api_key or self.settings.AZURE_OPENAI_API_KEY,
                azure_deployment = model_id or self.settings.LLM_MODEL_ID,
                max_tokens = max_tokens or self.settings.LLM_MAX_TOKENS,
                temperature = temperature or self.settings.LLM_TEMPERATURE,
                azure_endpoint = base_url or self.settings.AZURE_ENDPOINT,
                openai_api_version = self.settings.AZURE_OPENAI_API_VERSION
            )
            return BaseProvider(client)
        
        if provider == LLMEnums.DEEPSEEK.value or provider == LLMEnums.OPENROUTER.value:
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