"""FastAPI server: the bridge between the sleek frontend and the Attest engine.

Endpoints are thin — all the real work lives in AppState. Fast calls (ask,
settings) run synchronously in a threadpool; long calls (index, convert, eval,
compare) run as background JOBS: the POST returns a job id immediately and the
UI polls /api/jobs/{id} for progress ("embedded 512/970 chunks") and the result.

Security: when created with a `token` (the launcher generates one per run), every
/api request must carry it (X-Attest-Token header, or ?token= on the initial page
load). Without it, any webpage you visit could POST to this localhost server —
browsers allow cross-origin requests to 127.0.0.1.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..cli import _load_dotenv, _parse_pages
from .jobs import JobManager
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


def create_app(token: str | None = None) -> FastAPI:
    _load_dotenv()  # pick up ATTEST_* from .env before seeding config
    app = FastAPI(title="Attest")
    state = AppState()
    jobs = JobManager()

    if token:
        @app.middleware("http")
        async def _require_token(request: Request, call_next):
            if request.url.path.startswith("/api"):
                supplied = (request.headers.get("x-attest-token")
                            or request.query_params.get("token"))
                if supplied != token:
                    return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

    @app.get("/api/state")
    async def get_state():
        return state.public_state()

    @app.post("/api/ask")
    async def ask(req: Ask):
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(state.ask, req.question)

    # ---- long-running work becomes a job; poll /api/jobs/{id} -------------
    @app.post("/api/index")
    async def index(req: IndexReq):
        return {"job": jobs.start(
            "index", lambda report: state.index_file(req.path, req.vision, progress=report))}

    @app.post("/api/convert")
    async def convert(req: ConvertReq):
        pages = _parse_pages(req.pages)
        return {"job": jobs.start(
            "convert", lambda report: state.convert_file(
                req.path, req.vision, pages, req.out, progress=report))}

    @app.post("/api/eval")
    async def eval_run(req: EvalReq):
        return {"job": jobs.start(
            "eval", lambda report: state.run_eval(
                req.questions_path, req.judge, req.model, progress=report))}

    @app.post("/api/compare")
    async def compare(req: CompareReq):
        return {"job": jobs.start(
            "compare", lambda report: state.compare(
                req.questions_path, req.model_a, req.model_b, req.judge, progress=report))}

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str):
        return jobs.get(job_id)

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
