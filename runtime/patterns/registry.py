"""
Pattern and fix registry for Arc Runtime
"""

import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

from runtime.patterns.matcher import PatternMatcher


class PatternRegistry:
    """
    Registry for patterns and their associated fixes
    """

    def __init__(self):
        self.patterns: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
        self.matcher = PatternMatcher()
        self._lock = threading.RLock()

        # Register default patterns for MVP
        self._register_default_patterns()

    def _register_default_patterns(self):
        """Register hardcoded patterns for MVP demo"""
        # Pattern: GPT-4 with temperature > 0.9
        self.register(
            pattern={"model": "gpt-4.1", "temperature": {">": 0.9}},
            fix={"temperature": 0.7},
        )

        # Also match gpt-4 for backward compatibility
        self.register(
            pattern={"model": "gpt-4", "temperature": {">": 0.9}},
            fix={"temperature": 0.7},
        )

    def register(self, pattern: Dict[str, Any], fix: Dict[str, Any]) -> str:
        """
        Register a new pattern and fix

        Args:
            pattern: Pattern to match
            fix: Fix to apply when pattern matches

        Returns:
            Pattern ID
        """
        with self._lock:
            pattern_id = str(uuid.uuid4())
            self.patterns[pattern_id] = (pattern, fix)
            return pattern_id

    def unregister(self, pattern_id: str) -> bool:
        """
        Remove a pattern from the registry

        Args:
            pattern_id: ID of pattern to remove

        Returns:
            True if pattern was removed, False if not found
        """
        with self._lock:
            if pattern_id in self.patterns:
                del self.patterns[pattern_id]
                return True
            return False

    def match(self, request_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find first matching pattern and return its fix

        Args:
            request_params: Request parameters to match against

        Returns:
            Fix to apply if pattern matches, None otherwise
        """
        with self._lock:
            for pattern_id, (pattern, fix) in self.patterns.items():
                if self.matcher.match(request_params, pattern):
                    return fix
            return None

    def list_patterns(self) -> List[Dict[str, Any]]:
        """
        List all registered patterns

        Returns:
            List of pattern info dictionaries
        """
        with self._lock:
            return [
                {"id": pattern_id, "pattern": pattern, "fix": fix}
                for pattern_id, (pattern, fix) in self.patterns.items()
            ]
