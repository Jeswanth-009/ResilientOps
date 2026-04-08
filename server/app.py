import os
import uvicorn
from env import ResilientOpsAction, ResilientOpsEnv, ResilientOpsObservation

# Hook into OpenEnv's core server framework
try:
    from openenv.core.env_server.http_server import create_app

    _shared_env = ResilientOpsEnv()

    def _env_factory() -> ResilientOpsEnv:
        # Keep one in-memory simulator so /reset, /step, and /state share episode state.
        return _shared_env

    app = create_app(
        _env_factory,
        ResilientOpsAction,
        ResilientOpsObservation,
        env_name="resilientops",
        max_concurrent_envs=1,
    )
except ImportError:
    # Safe fallback just in case the framework version differs
    from fastapi import FastAPI
    app = FastAPI()


def _has_get_route(path: str) -> bool:
    """Check whether the app already exposes a GET route for a path."""
    for route in getattr(app, "routes", []):
        methods = getattr(route, "methods", set())
        if getattr(route, "path", None) == path and "GET" in methods:
            return True
    return False


# Add friendly defaults only if the OpenEnv app did not define them.
if hasattr(app, "get"):
    if not _has_get_route("/"):
        @app.get("/", include_in_schema=False)
        async def root() -> dict[str, str]:
            return {
                "status": "ok",
                "message": "ResilientOps server is running",
                "docs": "/docs",
                "health": "/health",
            }

    if not _has_get_route("/health"):
        @app.get("/health", include_in_schema=False)
        async def health() -> dict[str, str]:
            return {"status": "ok"}

def main():
    # Hugging Face Spaces strictly requires port 7860
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()