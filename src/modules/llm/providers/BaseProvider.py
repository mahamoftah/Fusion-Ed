import logging
from modules.BaseModule import BaseModule


# Google, OpenAI, Groq, DeepSeek, Qwen
class BaseProvider(BaseModule):
    """Provider for Google Generative AI (Gemini/PaLM models).
    Example:
        >>> provider = GoogleGenerativeAIProvider(api_key, model)
    """
    def __init__(self, llm_client):

        super().__init__()
        self.logger = logging.getLogger(__name__)

        try:
            self.client = llm_client
        except Exception as e:
            self.logger.error(f"Failed to initialize ChatGoogleGenerativeAI client: {e}")
            self.client = None

    
    async def generate_response(self, messages: list[dict[str, str]], structured_response: bool=False, response_model=None):
        """Generates a response from the model given a list of messages."""
        if not self.client:
            self.logger.error("Client is not initialized.")
            return None
        
        try:
            if structured_response:
                client = self.client.with_structured_output(response_model)
                response = await client.ainvoke(messages)
                return response
                
            response = await self.client.ainvoke(messages)
            return response.content
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return None