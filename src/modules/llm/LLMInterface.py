from abc import ABC, abstractmethod

class LLMInterface(ABC):

    @abstractmethod
    async def generate_response(self, messages: list[dict[str, str]], structured_response: bool=False, Response=None):
        pass