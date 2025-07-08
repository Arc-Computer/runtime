"""
OpenAI-specific interceptor implementation
"""

import functools
import logging
import sys
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

            # Always try to patch, whether openai is imported or not
            try:
                import openai

                self._patch_existing()
                logger.debug("Patched OpenAI after import")
            except ImportError:
                # OpenAI not yet imported, install import hook
                self._install_import_hook()
                logger.debug("Installed OpenAI import hook")

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
        import importlib.abc
        import importlib.machinery

        class OpenAIImportHook(importlib.abc.MetaPathFinder):
            def __init__(self, interceptor):
                self.interceptor = interceptor
                self._loading = False

            def find_spec(self, fullname, path, target=None):
                if fullname == "openai" and not self._loading:
                    # Return a spec that will trigger our exec_module
                    from importlib.machinery import ModuleSpec

                    return ModuleSpec(fullname, self)
                return None

            def create_module(self, spec):
                # Return None to use default module creation
                return None

            def exec_module(self, module):
                # Prevent recursion
                self._loading = True
                try:
                    # Let the real import happen
                    import importlib

                    spec = importlib.util.find_spec("openai")
                    if spec and spec.loader:
                        spec.loader.exec_module(module)

                    # Now patch it
                    self.interceptor._patch_existing()
                    logger.debug("Patched OpenAI via import hook")
                finally:
                    self._loading = False

        # Remove any existing hooks for OpenAI
        sys.meta_path = [
            h
            for h in sys.meta_path
            if not (
                hasattr(h, "__class__") and h.__class__.__name__ == "OpenAIImportHook"
            )
        ]

        # Add our hook
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
                logger.debug(
                    f"Patched completions for {client_class.__name__} instance"
                )

        client_class.__init__ = patched_init
        logger.debug(f"Patched {client_class.__name__} __init__ method")

    def _patch_completions(self, completions_resource):
        """Patch the completions resource create method"""
        if hasattr(completions_resource, "_patched_by_arc"):
            return  # Already patched

        original_create = completions_resource.create
        interceptor = self

        @wrapt.synchronized
        @functools.wraps(original_create)
        def patched_create(*args, **kwargs):
            # Extract agent metadata from extra_headers if present
            agent_metadata = None
            if "extra_headers" in kwargs and isinstance(kwargs["extra_headers"], dict):
                agent_metadata = {
                    "agent_name": kwargs["extra_headers"].get("X-Agent-Name")
                }

            return interceptor._intercept_request(
                provider="openai",
                method="chat.completions.create",
                original_func=original_create,
                args=args,
                kwargs=kwargs,
                agent_metadata=agent_metadata,
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
