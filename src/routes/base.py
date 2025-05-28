from fastapi import APIRouter, Depends
from routes.schemas.base import HealthCheckResponse
from helpers.config import Settings, get_settings
import logging

logger = logging.getLogger(__name__)

base_router = APIRouter(
    prefix="/api/v1",
    tags=["base"]
)

@base_router.get("/health", response_model=HealthCheckResponse)
async def health_check(settings: Settings = Depends(get_settings)):
    return HealthCheckResponse(
        status="healthy",
        version=settings.APP_VERSION
    )
