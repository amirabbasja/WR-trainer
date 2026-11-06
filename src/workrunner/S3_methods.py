"""Neutral S3 operations and the explicit S4-compatible wire boundary."""

from typing import Any

from .fixed_constants import constant_ascii_matches, constant_text


COMPUTE_REGISTER = "compute.register"
COMPUTE_SESSION = "compute.session"
WORK_COMPLETE = "work.complete"
WORK_UPDATE = "work.update"
COMPUTE_RECONNECT = "compute.reconnect"

_REQUEST_METHODS = {
    COMPUTE_REGISTER: 'training.subscribe',
    COMPUTE_SESSION: 'training.authorize',
    WORK_COMPLETE: 'training.submit',
}
_NOTIFICATION_METHODS = (
    ('training.notify', WORK_UPDATE),
    (b'#LLdY$<v9|!o<kQ$->0b', COMPUTE_RECONNECT),
    ('training.reconnect', COMPUTE_RECONNECT),
)


def s4_request(message: dict[str, Any]) -> dict[str, Any]:
    """Translate supported operations without changing IDs, payloads or callers."""
    method = message.get("method")
    if isinstance(method, str) and method in _REQUEST_METHODS:
        return {**message, "method": constant_text(_REQUEST_METHODS[method])}
    return message


def s4_notification_method(method: Any) -> str | None:
    """Classify wire notifications; internal operation names are not wire aliases."""
    if isinstance(method, str):
        for encoded, operation in _NOTIFICATION_METHODS:
            if constant_ascii_matches(encoded, method):
                return operation
    return None
