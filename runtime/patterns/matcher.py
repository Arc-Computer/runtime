"""
Pattern matching engine for Arc Runtime
"""

import threading
from typing import Any, Dict, Optional


class PatternMatcher:
    """
    Simple pattern matching engine for MVP

    Future optimizations:
    - Bloom filters for negative caching
    - Hash-based exact match lookup
    - Trie structure for prefix matching
    """

    def __init__(self):
        self._lock = threading.RLock()

    def match(self, request_params: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
        """
        Check if request parameters match a pattern

        Args:
            request_params: Request parameters to check
            pattern: Pattern to match against

        Returns:
            True if pattern matches, False otherwise
        """
        with self._lock:
            for key, pattern_value in pattern.items():
                if key not in request_params:
                    return False

                request_value = request_params[key]

                # Handle comparison operators
                if isinstance(pattern_value, dict):
                    if not self._match_comparison(request_value, pattern_value):
                        return False
                else:
                    # Exact match
                    if request_value != pattern_value:
                        return False

            return True

    def _match_comparison(self, value: Any, comparison: Dict[str, Any]) -> bool:
        """
        Match value against comparison operators

        Supported operators:
        - ">": greater than
        - ">=": greater than or equal
        - "<": less than
        - "<=": less than or equal
        - "!=": not equal
        - "in": value in list
        - "not_in": value not in list
        """
        for op, comp_value in comparison.items():
            if op == ">" and not (value > comp_value):
                return False
            elif op == ">=" and not (value >= comp_value):
                return False
            elif op == "<" and not (value < comp_value):
                return False
            elif op == "<=" and not (value <= comp_value):
                return False
            elif op == "!=" and not (value != comp_value):
                return False
            elif op == "in" and value not in comp_value:
                return False
            elif op == "not_in" and value in comp_value:
                return False

        return True
