from fastapi import APIRouter, Depends, Request, HTTPException, status
from src.controllers.DataExtractionController import DataExtractionController
from src.controllers.RagController import RagController
from src.routes.schemas.base import HealthCheckResponse
from src.helpers.config import Settings, get_settings
from src.routes.schemas.file import FileRequest, FileResponse
import logging
import os

logger = logging.getLogger(__name__)

file_router = APIRouter(
    prefix="/api/v1/files",
    tags=["files"]
)

@file_router.post("/upload",response_model=FileResponse)
async def upload_file(request: Request, 
                      file_request: FileRequest,
                      settings: Settings = Depends(get_settings)):

    try:
        file_urls = [file.file_url for file in file_request.files]
        data_extraction_controller = DataExtractionController(file_urls)
        file_contents = await data_extraction_controller.load()

        contents = []
        indexes = []
        failed_files = []
        
        for content in file_contents:
            if content["success"]:
                contents.append(content["content"])
                indexes.append(content["index"])
            else:
                failed_files.append({
                    "file_url": file_request.files[content["index"]].file_url,
                    "file_name": file_request.files[content["index"]].file_name,
                    "message": content.get("message", "Failed to process file")
                })
        
        if not contents:
            return FileResponse(
                success=False,
                data={
                    "total_processed": len(file_request.files),
                    "successful": 0,
                    "files": []
                },
                error_messages=[file["message"] for file in failed_files],
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        metadata = [
            {
                "file_id": file_request.files[index].file_id,
                "file_url": file_request.files[index].file_url,
                "file_name": file_request.files[index].file_name,
                "course_id": file_request.files[index].course_id
            }
            for index in indexes
        ]

        rag_controller = RagController(request.app.vector_store)
        documents_with_embeddings = await rag_controller.text_splits_embeddings(contents, metadata)
        await rag_controller.save_embeddings_to_vectordb(documents_with_embeddings)
        
        successful_files = [
            {
                "file_id": file_request.files[index].file_id,
                "file_url": file_request.files[index].file_url,
                "file_name": file_request.files[index].file_name,
                "course_id": file_request.files[index].course_id,
                "success": True
            }
            for index in indexes
        ]
        
        return FileResponse(
            success=True,
            data={
                "total_processed": len(file_request.files),
                "successful": len(successful_files),
                "files": successful_files
            },
            error_messages=[f"{file['file_name']} {file['message']}" for file in failed_files],
            status_code=status.HTTP_200_OK
        )
    
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{str(e)}: Unexpected Error during the File Uploading"
        )
