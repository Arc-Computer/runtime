"""
OpenAI-specific interceptor implementation
"""

import sys
import logging
import functools
import threading
from typing import Any

import wrapt

from runtime.interceptors.base import BaseInterceptor

logger = logging.getLogger(__name__)


class OpenAIInterceptor(BaseInterceptor):
    """
    Interceptor for OpenAI Python SDK
    """
    
    def __init__(self, pattern_registry, telemetry_client):
        super().__init__(pattern_registry, telemetry_client)
        self._patch_lock = threading.Lock()
        self._patched = False
    
    def patch(self):
        """Apply monkey patches to OpenAI client"""
        with self._patch_lock:
            if self._patched:
                return
                
            # Check if openai is already imported
            if "openai" in sys.modules:
                self._patch_existing()
            
            # Install import hook for future imports
            self._install_import_hook()
            
            self._patched = True
    
    def _patch_existing(self):
        """Patch already-imported OpenAI module"""
        try:
            import openai
            
            # Patch synchronous client
            if hasattr(openai, "OpenAI"):
                self._patch_client_class(openai.OpenAI)
            
            # Patch async client
            if hasattr(openai, "AsyncOpenAI"):
                self._patch_client_class(openai.AsyncOpenAI)
                
            logger.debug("Successfully patched existing OpenAI module")
            
        except Exception as e:
            logger.warning(f"Failed to patch existing OpenAI module: {e}")
    
    def _install_import_hook(self):
        """Install import hook to patch OpenAI on import"""
        
        class OpenAIImportHook:
            def __init__(self, interceptor):
                self.interceptor = interceptor
            
            def find_module(self, fullname, path=None):
                if fullname == "openai":
                    return self
                return None
            
            def load_module(self, fullname):
                # Import the module normally
                if fullname in sys.modules:
                    return sys.modules[fullname]
                
                # Use importlib to import the module
                import importlib
                module = importlib.import_module(fullname)
                
                # Patch it
                self.interceptor._patch_existing()
                
                return module
        
        # Add to meta_path
        sys.meta_path.insert(0, OpenAIImportHook(self))
    
    def _patch_client_class(self, client_class):
        """Patch a specific client class (OpenAI or AsyncOpenAI)"""
        original_init = client_class.__init__
        interceptor = self
        
        @functools.wraps(original_init)
        def patched_init(self, *args, **kwargs):
            # Call original init
            original_init(self, *args, **kwargs)
            
            # Patch the chat.completions.create method on this instance
            if hasattr(self, "chat") and hasattr(self.chat, "completions"):
                interceptor._patch_completions(self.chat.completions)
        
        client_class.__init__ = patched_init
    
    def _patch_completions(self, completions_resource):
        """Patch the completions resource create method"""
        if hasattr(completions_resource, "_patched_by_arc"):
            return  # Already patched
            
        original_create = completions_resource.create
        interceptor = self
        
        @wrapt.synchronized
        @functools.wraps(original_create)
        def patched_create(*args, **kwargs):
            return interceptor._intercept_request(
                provider="openai",
                method="chat.completions.create",
                original_func=original_create,
                args=args,
                kwargs=kwargs,
            )
        
        completions_resource.create = patched_create
        completions_resource._patched_by_arc = True
    
    def wrap_client(self, client):
        """Explicitly wrap an OpenAI client instance"""
        # Check if it's an OpenAI client
        client_module = type(client).__module__
        if not client_module.startswith("openai"):
            return client
            
        # Patch the completions resource
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            self._patch_completions(client.chat.completions)
            
        return client
    
    def _extract_params(self, args: tuple, kwargs: dict) -> dict:
        """Extract relevant parameters from OpenAI API call"""
        # OpenAI client typically uses kwargs
        params = kwargs.copy()
        
        # Extract key fields for pattern matching
        relevant_fields = ["model", "temperature", "max_tokens", "top_p", "messages"]
        
        return {k: v for k, v in params.items() if k in relevant_fields}