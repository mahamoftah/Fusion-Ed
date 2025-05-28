from helpers.config import get_settings
import logging
class BaseController:

    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        