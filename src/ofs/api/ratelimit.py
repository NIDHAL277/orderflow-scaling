"""Token-bucket rate limit per client address (pure ASGI middleware)."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class TokenBucket:
    """Token bucket per client key: ``per_minute`` requests, refilled continuously."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self.clock = clock
        self.buckets: dict[str, tuple[float, float]] = {}

    def allow(self, key: str) -> bool:
        """Take one token for ``key``; False when the bucket is empty."""
        now = self.clock()
        tokens, last = self.buckets.get(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.rate)
        allowed = tokens >= 1.0
        self.buckets[key] = (tokens - 1.0 if allowed else tokens, now)
        if len(self.buckets) > 10_000:  # bound memory under scans
            self.buckets.clear()
        return allowed


class RateLimitMiddleware:
    """ASGI middleware answering 429 when a client exceeds its rate."""

    def __init__(self, app: ASGIApp, per_minute: int = 120) -> None:
        self.app = app
        self.bucket = TokenBucket(per_minute)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass the request on, or answer 429 when the client has no token left."""
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        forwarded = headers.get(b"x-forwarded-for", b"").decode().split(",")[0].strip()
        client = forwarded or (scope.get("client") or ("unknown", 0))[0]
        if self.bucket.allow(str(client)):
            await self.app(scope, receive, send)
            return
        body = b'{"detail":"rate limit exceeded"}'
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [(b"content-type", b"application/json"), (b"retry-after", b"60")],
            }
        )
        await send({"type": "http.response.body", "body": body})
