import os
from importlib import import_module
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from env import ResilientOpsAction, ResilientOpsEnv, ResilientOpsObservation

_shared_env = ResilientOpsEnv()


def _env_factory() -> ResilientOpsEnv:
    # Keep one in-memory simulator so /reset, /step, and /state share episode state.
    return _shared_env


def _load_openenv_factory():
    """Try known OpenEnv create_app locations across package versions."""
    candidates = (
        "openenv.core.env_server.http_server",
        "openenv.core.env_server.http_server.create_app",
    )
    for module_name in candidates:
        try:
            module = import_module(module_name)
            create_app = getattr(module, "create_app", None)
            if create_app is not None:
                return create_app
        except Exception:
            continue
    return None


def _build_fallback_app() -> FastAPI:
    """Fallback FastAPI app that still exposes OpenEnv-like endpoints."""
    fallback = FastAPI(title="ResilientOps", version="0.2.0")

    async def _reset_impl(payload: Optional[dict[str, Any]] = None) -> ResilientOpsObservation:
        body = payload or {}
        task_id = str(body.get("task_id", "minor-delay"))
        seed = body.get("seed")
        episode_id = body.get("episode_id")
        try:
            return await _shared_env.reset_async(task_id=task_id, seed=seed, episode_id=episode_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def _step_impl(payload: dict[str, Any]) -> ResilientOpsObservation:
        try:
            action = ResilientOpsAction(**payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc

        try:
            return await _shared_env.step_async(action)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @fallback.post("/reset", response_model=ResilientOpsObservation)
    async def reset(payload: Optional[dict[str, Any]] = None) -> ResilientOpsObservation:
        return await _reset_impl(payload)

    @fallback.post("/step", response_model=ResilientOpsObservation)
    async def step(payload: dict[str, Any]) -> ResilientOpsObservation:
        return await _step_impl(payload)

    @fallback.get("/state")
    async def state() -> dict[str, Any]:
        return (await _shared_env.state_async()).model_dump()

    # /web aliases are useful when deployments run behind a /web base path.
    @fallback.post("/web/reset", response_model=ResilientOpsObservation, include_in_schema=False)
    async def web_reset(payload: Optional[dict[str, Any]] = None) -> ResilientOpsObservation:
        return await _reset_impl(payload)

    @fallback.post("/web/step", response_model=ResilientOpsObservation, include_in_schema=False)
    async def web_step(payload: dict[str, Any]) -> ResilientOpsObservation:
        return await _step_impl(payload)

    @fallback.get("/web/state", include_in_schema=False)
    async def web_state() -> dict[str, Any]:
        return (await _shared_env.state_async()).model_dump()

    return fallback


_create_app = _load_openenv_factory()
if _create_app is None:
    app = _build_fallback_app()
else:
    try:
        app = _create_app(
            _env_factory,
            ResilientOpsAction,
            ResilientOpsObservation,
            env_name="resilientops",
            max_concurrent_envs=1,
        )
    except Exception:
        app = _build_fallback_app()


def _has_get_route(path: str) -> bool:
    """Check whether the app already exposes a GET route for a path."""
    for route in getattr(app, "routes", []):
        methods = getattr(route, "methods", set())
        if getattr(route, "path", None) == path and "GET" in methods:
            return True
    return False


def _has_post_route(path: str) -> bool:
    """Check whether the app already exposes a POST route for a path."""
    for route in getattr(app, "routes", []):
        methods = getattr(route, "methods", set())
        if getattr(route, "path", None) == path and "POST" in methods:
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
                "reset": "/reset",
                "step": "/step",
                "state": "/state",
            }

    if not _has_get_route("/health"):
        @app.get("/health", include_in_schema=False)
        async def health() -> dict[str, str]:
            return {"status": "ok"}

    # Mirror routes for environments expecting /web as base_path.
    if not _has_get_route("/web"):
        @app.get("/web", include_in_schema=False)
        async def web_root() -> dict[str, str]:
            return {
                "status": "ok",
                "message": "ResilientOps server is running",
                "docs": "/docs",
                "health": "/health",
                "reset": "/web/reset",
                "step": "/web/step",
                "state": "/web/state",
            }

    if not _has_get_route("/web/health"):
        @app.get("/web/health", include_in_schema=False)
        async def web_health() -> dict[str, str]:
            return {"status": "ok"}

if hasattr(app, "post"):
    if _has_post_route("/reset") and not _has_post_route("/web/reset"):
        @app.post("/web/reset", response_model=ResilientOpsObservation, include_in_schema=False)
        async def web_reset(payload: Optional[dict[str, Any]] = None) -> ResilientOpsObservation:
            body = payload or {}
            task_id = str(body.get("task_id", "minor-delay"))
            seed = body.get("seed")
            episode_id = body.get("episode_id")
            return await _shared_env.reset_async(task_id=task_id, seed=seed, episode_id=episode_id)

    if _has_post_route("/step") and not _has_post_route("/web/step"):
        @app.post("/web/step", response_model=ResilientOpsObservation, include_in_schema=False)
        async def web_step(payload: dict[str, Any]) -> ResilientOpsObservation:
            try:
                action = ResilientOpsAction(**payload)
            except ValidationError as exc:
                raise HTTPException(status_code=422, detail=exc.errors()) from exc
            return await _shared_env.step_async(action)

if hasattr(app, "get") and not _has_get_route("/web/state"):
    @app.get("/web/state", include_in_schema=False)
    async def web_state() -> dict[str, Any]:
        return (await _shared_env.state_async()).model_dump()


def main():
    # Hugging Face Spaces strictly requires port 7860
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()