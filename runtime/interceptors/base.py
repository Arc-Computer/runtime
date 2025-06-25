"""
Base interceptor class for all LLM providers
"""

import time
import uuid
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable

logger = logging.getLogger(__name__)


class BaseInterceptor(ABC):
    """
    Abstract base class for LLM provider interceptors
    """
    
    def __init__(self, pattern_registry, telemetry_client):
        self.pattern_registry = pattern_registry
        self.telemetry_client = telemetry_client
        self._original_methods = {}
    
    @abstractmethod
    def patch(self):
        """Apply monkey patches to intercept API calls"""
        pass
    
    @abstractmethod
    def wrap_client(self, client):
        """Wrap a client instance for explicit protection"""
        pass
    
    def _intercept_request(
        self,
        provider: str,
        method: str,
        original_func: Callable,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """
        Common interception logic for all providers
        
        Args:
            provider: Provider name (e.g., "openai")
            method: Method name (e.g., "chat.completions.create")
            original_func: Original method to call
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Response from the original method (potentially with modified args)
        """
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())
        
        # Extract request parameters
        request_params = self._extract_params(args, kwargs)
        
        # Start OTel trace
        with self.telemetry_client.trace_llm_request(provider, method, request_params) as span:
            # Measure Arc interception overhead
            interception_start = time.perf_counter()
            
            # Check for pattern match
            pattern_match = self.pattern_registry.match(request_params)
            fix_applied = None
            
            if pattern_match:
                # Apply fix
                fix_applied = pattern_match
                kwargs = self._apply_fix(kwargs, fix_applied)
                logger.info(
                    f"Arc Runtime: Intercepted {provider} request {request_id} - "
                    f"Applied fix: {fix_applied}"
                )
            
            # Record Arc intervention
            interception_latency_ms = (time.perf_counter() - interception_start) * 1000
            if span:
                self.telemetry_client.record_arc_intervention(
                    span, 
                    pattern_match is not None, 
                    fix_applied,
                    interception_latency_ms
                )
            
            # Call original method
            try:
                response = original_func(*args, **kwargs)
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                # Record response details
                if span:
                    self.telemetry_client.record_llm_response(span, response, latency_ms)
                
                # Update basic metrics
                self.telemetry_client.metrics.increment("arc_requests_intercepted_total")
                if pattern_match:
                    self.telemetry_client.metrics.increment("arc_pattern_matches_total")
                if fix_applied:
                    self.telemetry_client.metrics.increment("arc_fixes_applied_total")
                self.telemetry_client.metrics.record_histogram("arc_interception_latency_ms", interception_latency_ms)
                
                # Legacy telemetry for backward compatibility
                self._log_telemetry(
                    request_id=request_id,
                    provider=provider,
                    method=method,
                    request_params=request_params,
                    pattern_matched=pattern_match is not None,
                    fix_applied=fix_applied,
                    latency_ms=latency_ms,
                    success=True,
                )
                
                return response
                
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                if span:
                    span.record_exception(e)
                    # Set error status if OTel is available
                    try:
                        from opentelemetry import trace
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    except ImportError:
                        pass
                
                # Legacy telemetry for failure
                self._log_telemetry(
                    request_id=request_id,
                    provider=provider,
                    method=method,
                    request_params=request_params,
                    pattern_matched=pattern_match is not None,
                    fix_applied=fix_applied,
                    latency_ms=latency_ms,
                    success=False,
                    error=str(e),
                )
                
                raise
    
    def _extract_params(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract relevant parameters from method arguments"""
        # Default implementation - subclasses can override
        return kwargs.copy()
    
    def _apply_fix(self, kwargs: dict, fix: dict) -> dict:
        """Apply fix to request parameters"""
        # Create a copy to avoid modifying original
        modified_kwargs = kwargs.copy()
        
        for key, value in fix.items():
            modified_kwargs[key] = value
            
        return modified_kwargs
    
    def _log_telemetry(
        self,
        request_id: str,
        provider: str,
        method: str,
        request_params: dict,
        pattern_matched: bool,
        fix_applied: Optional[dict],
        latency_ms: float,
        success: bool,
        error: Optional[str] = None,
    ):
        """Log telemetry event"""
        event = {
            "timestamp": time.time(),
            "request_id": request_id,
            "provider": provider,
            "method": method,
            "original_params": request_params,
            "pattern_matched": pattern_matched,
            "fix_applied": fix_applied,
            "latency_ms": latency_ms,
            "success": success,
        }
        
        if error:
            event["error"] = error
            
        self.telemetry_client.record(event)