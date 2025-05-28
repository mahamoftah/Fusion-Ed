from fastapi import HTTPException, status
from src.controllers.BaseController import BaseController
from src.modules.rag.embedding import Embedding
from src.modules.rag.splitting import RecursiveSplitter


class RagController(BaseController):
    def __init__(self, vector_store):
        super().__init__()
        self.vector_store = vector_store
        self.text_splitter = RecursiveSplitter()
        self.embedding_model = Embedding()

    async def text_splits_embeddings(self, contents, metadata):
        # Split documents into chunks
        split_documents = await self.text_splitter.split_documents(contents, metadata)
        
        # Add chunk order to metadata
        updated_split_documents = []
        for doc_index, (original_doc, original_metadata) in enumerate(zip(contents, metadata)):
                chunked_docs = [
                    (chunk_index, doc) for chunk_index, doc in enumerate(split_documents) 
                    if doc.metadata['file_id'] == original_metadata['file_id']
                ]
                for chunk_index, chunked_doc in chunked_docs:
                    chunked_doc.metadata['chunk_order'] = chunk_index + 1  
                    updated_split_documents.append(chunked_doc)
            
            # Extract updated page contents and metadata
        page_contents = [doc.page_content for doc in updated_split_documents]
        metadatas = [doc.metadata for doc in updated_split_documents]
            

        embeddings = await self.embedding_model.embed_documents(page_contents)

        # Prepare documents for insertion
        documents_with_embeddings = [
            {"text": text, "embedding": embedding, "metadata": metadata}
            for text, embedding, metadata in zip(page_contents, embeddings, metadatas or [{}]*len(page_contents))
        ]
        return documents_with_embeddings
    

    async def save_embeddings_to_vectordb(self, documents_with_embeddings):
            
        try:
            batch_size = 5

            for i in range(0, len(documents_with_embeddings), batch_size):
                batch = documents_with_embeddings[i:i + batch_size]
                try:
                    await self.vector_store.save_chunks(batch)
                    self.logger.info(f"Successfully added batch {i // batch_size + 1}")
                except Exception as e:
                    self.logger.error(f"Failed to add batch {i // batch_size + 1} to vector store: {str(e)}")
            # Add documents to the vector database            
            self.logger.info("Documents successfully added to the vector database.")

        except Exception as e:
            self.logger.error(f"Error saving embeddings to vector database: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error saving embeddings to vector database: {e}")
