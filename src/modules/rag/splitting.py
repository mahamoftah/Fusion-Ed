from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional
from langchain.schema import Document
from modules.BaseModule import BaseModule

class RecursiveSplitter(BaseModule):
    def __init__(self):
        super().__init__()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP
        )

    async def split_documents(self, documents: List[str], metadatas: Optional[List[dict]] = None) -> List[Document]:
        return self.text_splitter.create_documents(documents, metadatas)