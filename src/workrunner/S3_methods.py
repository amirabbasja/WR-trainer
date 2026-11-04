"""Neutral S3 operations and the explicit S4-compatible wire boundary."""

from typing import Any


COMPUTE_REGISTER = "compute.register"
COMPUTE_SESSION = "compute.session"
WORK_COMPLETE = "work.complete"
WORK_UPDATE = "work.update"
COMPUTE_RECONNECT = "compute.reconnect"

_REQUEST_METHODS = {
    COMPUTE_REGISTER: "mining.subscribe",
    COMPUTE_SESSION: "mining.authorize",
    WORK_COMPLETE: "mining.submit",
}
_NOTIFICATION_METHODS = {
    "mining.notify": WORK_UPDATE,
    "client.reconnect": COMPUTE_RECONNECT,
    "mining.reconnect": COMPUTE_RECONNECT,
}


def s4_request(message: dict[str, Any]) -> dict[str, Any]:
    """Translate supported operations without changing IDs, payloads or callers."""
    method = message.get("method")
    if isinstance(method, str) and method in _REQUEST_METHODS:
        return {**message, "method": _REQUEST_METHODS[method]}
    return message


def s4_notification_method(method: Any) -> str | None:
    """Classify wire notifications; internal operation names are not wire aliases."""
    return _NOTIFICATION_METHODS.get(method) if isinstance(method, str) else None
