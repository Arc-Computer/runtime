"""
Configuration management for Arc Runtime
"""

import os
import logging
from typing import Optional


class Config:
    """Arc Runtime configuration"""
    
    DEFAULT_ENDPOINT = "grpc://localhost:50051"
    DEFAULT_CACHE_DIR = "~/.arc/cache"
    DEFAULT_LOG_LEVEL = "INFO"
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        log_level: Optional[str] = None,
    ):
        # Load from environment or use defaults
        self.endpoint = endpoint or os.environ.get("ARC_ENDPOINT", self.DEFAULT_ENDPOINT)
        self.api_key = api_key or os.environ.get("ARC_API_KEY")
        self.cache_dir = os.path.expanduser(
            cache_dir or os.environ.get("ARC_CACHE_DIR", self.DEFAULT_CACHE_DIR)
        )
        self.log_level = log_level or os.environ.get("ARC_LOG_LEVEL", self.DEFAULT_LOG_LEVEL)
        
        # Configure logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging based on log level"""
        numeric_level = getattr(logging, self.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            numeric_level = logging.INFO
            
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )