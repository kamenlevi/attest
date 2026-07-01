"""FastAPI server: the bridge between the sleek frontend and the Attest engine.

Endpoints are thin — all the real work lives in AppState. Long calls (ask, index,
vision convert) run in a threadpool so the UI stays responsive.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..cli import _load_dotenv
from .state import AppState

_STATIC = Path(__file__).parent / "static"


class Ask(BaseModel):
    question: str


class IndexReq(BaseModel):
    path: str
    vision: bool = False


class ConvertReq(BaseModel):
    path: str
    vision: bool = False
    pages: str | None = None   # "40-46" / "41,45" (1-based), vision only
    out: str | None = None


class EvalReq(BaseModel):
    questions_path: str
    judge: bool = True
    model: str | None = None   # override the generator for this run only


class CompareReq(BaseModel):
    questions_path: str
    model_a: str
    model_b: str
    judge: bool = True


class Settings(BaseModel):
    patch: dict


def create_app() -> FastAPI:
    _load_dotenv()  # pick up ATTEST_* from .env before seeding config
    app = FastAPI(title="Attest")
    state = AppState()

    @app.get("/api/state")
    async def get_state():
        return state.public_state()

    @app.post("/api/ask")
    async def ask(req: Ask):
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(state.ask, req.question)

    @app.post("/api/index")
    async def index(req: IndexReq):
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(state.index_file, req.path, req.vision)

    @app.post("/api/convert")
    async def convert(req: ConvertReq):
        from starlette.concurrency import run_in_threadpool

        from ..cli import _parse_pages
        pages = _parse_pages(req.pages)
        return await run_in_threadpool(
            state.convert_file, req.path, req.vision, pages, req.out)

    @app.post("/api/eval")
    async def eval_run(req: EvalReq):
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(
            state.run_eval, req.questions_path, req.judge, req.model)

    @app.post("/api/compare")
    async def compare(req: CompareReq):
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(
            state.compare, req.questions_path, req.model_a, req.model_b, req.judge)

    @app.post("/api/settings")
    async def settings(req: Settings):
        patch = req.patch
        # never overwrite the stored key with the masked placeholder from the UI
        prov = patch.get("provider")
        if isinstance(prov, dict):
            k = prov.get("api_key")
            if k is not None and (k == "" or k.startswith("•")):
                prov.pop("api_key", None)
        return state.update_settings(patch)

    @app.get("/")
    async def root():
        return FileResponse(_STATIC / "index.html")

    app.mount("/", StaticFiles(directory=str(_STATIC)), name="static")
    return app
