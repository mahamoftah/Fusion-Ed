from typing import List
from src.modules.BaseModule import BaseModule
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

class Embedding(BaseModule):
    def __init__(self):
        super().__init__()
        # self.embedding_model = OpenAIEmbeddings(
        #     model=self.settings.EMBEDDING_MODEL,
        #     api_key=self.settings.EMBEDDING_API_KEY,
        # )
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=self.settings.EMBEDDING_MODEL,
            google_api_key=self.settings.EMBEDDING_API_KEY,
        )

    
    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return await self.embedding_model.aembed_documents(documents)

    async def embed_query(self, query: str):
        return await self.embedding_model.aembed_query(query)