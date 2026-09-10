from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from fastapi.middleware.cors import CORSMiddleware


class ConditionalCORSMiddleware:
    """
    Wraps the global CORSMiddleware so it only ever applies to
    non-widget paths. /widget/* traffic bypasses it completely and is
    handled solely by WidgetCorsMiddleware + the per-company checks in
    widget.py — no double-processing, no ordering guesswork, one
    source of truth for widget CORS headers.
    """

    def __init__(self, app, **cors_kwargs):
        self.cors_app = CORSMiddleware(app, **cors_kwargs)
        self.plain_app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/widget/"):
            await self.plain_app(scope, receive, send)
        else:
            await self.cors_app(scope, receive, send)


class WidgetCorsMiddleware(BaseHTTPMiddleware):
    """
    Handles CORS for /widget/* only. Per-company origin checking can't
    happen here — at this point we don't know which company a request
    belongs to (that requires a DB lookup done inside the route). This
    middleware answers preflight OPTIONS requests, and on the way out,
    strips any CORS headers the global CORSMiddleware may have added
    (it unconditionally adds Access-Control-Allow-Credentials when
    configured, regardless of origin) so widget.py's own per-company
    decision is the single source of truth for this path.
    """

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith("/widget/"):
            return await call_next(request)

        origin = request.headers.get("origin", "")

        if request.method == "OPTIONS":
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, X-API-Key",
                    "Access-Control-Max-Age": "600",
                    "Vary": "Origin",
                },
            )

        response = await call_next(request)
        
        return response