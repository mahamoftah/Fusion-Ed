from helpers.config import get_settings
import logging

class BaseModule:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)