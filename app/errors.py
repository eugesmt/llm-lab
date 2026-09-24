"""Normalized application-level errors.

Every error that can reach an HTTP client is expressed here with its status
code; provider adapters translate SDK-specific exceptions into these types so
the rest of the application never sees a raw traceback or SDK error.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for expected, user-facing errors."""

    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ModelNotFoundError(AppError):
    status_code = 404


class UnsupportedReasoningEffortError(AppError):
    status_code = 422


class ProviderNotConfiguredError(AppError):
    status_code = 503


class ProviderAuthError(AppError):
    status_code = 502


class ProviderRateLimitError(AppError):
    status_code = 429


class ProviderTimeoutError(AppError):
    status_code = 504


class ProviderUnavailableError(AppError):
    status_code = 502


class EmptyResponseError(AppError):
    status_code = 502


class ProviderRequestError(AppError):
    """Provider rejected the request or returned an unexpected payload."""

    status_code = 502


class MCPUnavailableError(AppError):
    status_code = 502


class MCPToolError(AppError):
    status_code = 502


class MCPAgentLoopError(AppError):
    status_code = 502
