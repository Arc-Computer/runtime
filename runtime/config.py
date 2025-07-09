"""
Configuration management for Arc Runtime
"""

import logging
import os
from dataclasses import dataclass
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
        self.endpoint = endpoint or os.environ.get(
            "ARC_ENDPOINT", self.DEFAULT_ENDPOINT
        )
        self.api_key = api_key or os.environ.get("ARC_API_KEY")
        self.cache_dir = os.path.expanduser(
            cache_dir or os.environ.get("ARC_CACHE_DIR", self.DEFAULT_CACHE_DIR)
        )
        self.log_level = log_level or os.environ.get(
            "ARC_LOG_LEVEL", self.DEFAULT_LOG_LEVEL
        )

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


@dataclass
class TelemetryConfig:
    """Configuration for telemetry client with Kong Konnect support"""
    
    endpoint: str = "localhost:50051"
    api_key: Optional[str] = None
    use_kong_gateway: bool = False
    kong_gateway_url: Optional[str] = None
    use_tls: bool = False
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None
    tls_ca_cert_path: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'TelemetryConfig':
        """Create TelemetryConfig from environment variables"""
        config = cls(
            endpoint=os.getenv('ARC_TELEMETRY_ENDPOINT', 'localhost:50051'),
            api_key=os.getenv('ARC_API_KEY'),
            use_kong_gateway=os.getenv('ARC_USE_KONG_GATEWAY', 'false').lower() == 'true',
            kong_gateway_url=os.getenv('ARC_KONG_GATEWAY_URL'),
            use_tls=os.getenv('ARC_USE_TLS', 'false').lower() == 'true',
            tls_cert_path=os.getenv('ARC_TLS_CERT_PATH'),
            tls_key_path=os.getenv('ARC_TLS_KEY_PATH'),
            tls_ca_cert_path=os.getenv('ARC_TLS_CA_CERT_PATH')
        )
        
        # Validate configuration
        if config.use_kong_gateway and not config.kong_gateway_url:
            raise ValueError("kong_gateway_url is required when use_kong_gateway is True")
        
        return config
