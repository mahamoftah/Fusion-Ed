from fastapi import APIRouter, Depends
from src.routes.schemas.base import HealthCheckResponse
from src.helpers.config import Settings, get_settings
import logging

logger = logging.getLogger(__name__)

base_router = APIRouter(
    prefix="/api/v1",
    tags=["base"]
)

@base_router.get("/")
async def health_check(settings: Settings = Depends(get_settings)):
    return HealthCheckResponse(
        status="Up and Running",
        app_name=settings.APP_NAME,
        app_version=settings.APP_VERSION
    )
