from typing import List
from pydantic import BaseModel


class File(BaseModel):
    file_id: str
    file_url: str
    file_name: str
    course_id: str

class FileRequest(BaseModel):
    files: List[File]


class FileResponse(BaseModel):
    success: bool
    data: dict
    error_messages: List[str]
    status_code: int

