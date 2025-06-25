"""
Arc Runtime - Lightweight AI failure prevention system
"""

__version__ = "0.1.0"

import os
import sys
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Check if Arc is disabled
if os.environ.get("ARC_DISABLE", "").lower() in ("1", "true", "yes"):
    logger.info("Arc Runtime disabled via ARC_DISABLE environment variable")
else:
    from runtime.arc import Arc
    
    # Auto-initialize with default settings
    _default_arc = Arc()

# Public API
__all__ = ["Arc"]