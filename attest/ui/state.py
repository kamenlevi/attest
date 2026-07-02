"""AppState: holds settings + the active index, and assembles the RAG pipeline
exactly as the UI toggles describe it. One place that turns config into behaviour.

Everything is lazy and cached: the embedding model, the index, AND the assembled
pipeline (BM25 index, reranker model, expander) load on first use and are reused
across questions — rebuilding them per question meant reloading the cross-encoder
model on every single ask. Caches are invalidated only when the settings they
depend on actually change.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..chunking import chunk_pages
from ..config import load_config, save_config
from ..eval import EvalItem, compute_metrics, grade_results, run_eval
from ..grounding import build_prompt, parse_response
from ..ingest import load_pages
from ..interfaces import Embedder, Generator
from ..judge import Judge
from ..store import IndexedStore, file_fingerprint
from ..verify import SupportChecker, verify_answer


class AppState:
    def __init__(self) -> None:
        self.config = load_config()
        self._embedder: Embedder | None = None
        self._store: IndexedStore | None = None
        self._store_path: str | None = None
        self._pipeline = None
        self._pipeline_key: str | None = None

    # ---- builders -------------------------------------------------------
    def _make_generator(self, model_key: str = "generator",
                        model_override: str | None = None) -> Generator:
        model = (model_override or self.config["models"].get(model_key)
                 or self.config["models"]["generator"])
        prov = self.config["provider"]
        if not model or not prov.get("base_url"):
            from ..backends.mock import MockGenerator
            return MockGenerator()
        from ..backends.openai_compat import OpenAICompatibleGenerator
        return OpenAICompatibleGenerator(model, prov["base_url"], prov.get("api_key") or None)

    def _has_provider(self) -> bool:
        return bool(self.config["provider"].get("base_url"))

    def embedder(self) -> Embedder:
        if self._embedder is None:
            if self.config["models"]["embedder"] == "mock":
                from ..backends.mock import MockEmbedder
                self._embedder = MockEmbedder()
            else:
                from ..backends.local_embed import LocalEmbedder
                self._embedder = LocalEmbedder()
        return self._embedder

    def store(self) -> IndexedStore | None:
        path = self.config.get("index_path")
        if not path or not (Path(path) / "vectors.npy").exists():
            return None
        if self._store is None or self._store_path != path:
            self._store = IndexedStore.load(path, self.embedder())
            self._store_path = path
        return self._store

    def retriever(self):
        """The assembled lexical→expand→rerank pipeline, cached across questions.

        Rebuilt only when the settings it depends on change — so asking ten
        questions loads the reranker model once, not ten times.
        """
        key = json.dumps({
            "index": self.config.get("index_path"),
            "pipeline": self.config["pipeline"],
            "models": self.config["models"],
            "base_url": self.config["provider"].get("base_url"),
        }, sort_keys=True)
        if self._pipeline is not None and self._pipeline_key == key:
            return self._pipeline
        base = self.store()
        if base is None:
            return None
        pipe = self.config["pipeline"]
        retriever = base
        if pipe.get("lexical"):
            from ..lexical import BM25Retriever
            from ..retrieval import FusedRetriever
            retriever = FusedRetriever([retriever, BM25Retriever(base.chunks())], pool=40)
        if pipe.get("expand"):
            from ..query import ExpandingRetriever, QueryExpander
            retriever = ExpandingRetriever(retriever, QueryExpander(self._make_generator()), pool=40)
        if pipe.get("rerank"):
            from ..backends.rerank import CrossEncoderReranker
            from ..retrieval import RerankingRetriever
            retriever = RerankingRetriever(retriever, CrossEncoderReranker(), pool=50)
        self._pipeline, self._pipeline_key = retriever, key
        return retriever

    # ---- actions --------------------------------------------------------
    def ask(self, question: str) -> dict:
        retriever = self.retriever()
        if retriever is None:
            return {"error": "No document indexed yet. Add one in the Library tab."}
        pipe = self.config["pipeline"]
        generator = self._make_generator()
        retrieved = retriever.search(question, k=int(pipe.get("k", 8)))
        answer = parse_response(generator.generate(build_prompt(question, retrieved)))

        # The verification ladder: citation validity is free; the support check
        # costs one judge call and only runs against a real provider.
        checker = None
        if (pipe.get("verify") and not answer.abstained and answer.citations
                and self._has_provider() and self.config["models"].get("judge")):
            checker = SupportChecker(self._make_generator("judge"))
        verification = verify_answer(question, answer, retrieved, checker)

        by_id = {r.chunk.index: r.chunk for r in retrieved}
        passages = [
            {"id": r.chunk.index, "source": Path(r.chunk.source).name if r.chunk.source else "",
             "page": r.chunk.page, "text": r.chunk.text[:600], "score": round(float(r.score), 3)}
            for r in retrieved
        ]
        citations = [
            {"id": c, "source": Path(by_id[c].source).name if by_id[c].source else "",
             "page": by_id[c].page}
            for c in verification.valid_citations if c in by_id
        ]
        result = {
            "abstained": answer.abstained,
            "answer": answer.text,
            "citations": answer.citations,
            "cited": citations,
            "verification": {
                "status": verification.status,
                "valid": verification.valid_citations,
                "invalid": verification.invalid_citations,
                "note": verification.note,
            },
            "passages": passages,
        }
        if answer.abstained and pipe.get("allow_uncited"):
            result["uncited"] = generator.generate(question)
        return result

    def index_file(self, path: str, vision: bool = False, progress=None) -> dict:
        """Index a document into the active index (creating one if needed).

        `progress` is the job system's report(message, current, total) callback;
        omitted for direct/synchronous calls.
        """
        report = progress or (lambda *a, **kw: None)
        src = Path(path).expanduser()
        if not src.exists():
            return {"error": f"File not found: {path}"}
        path = str(src.resolve())  # absolute, so it resolves from any working dir
        default_index = Path.home() / ".attest" / "library.idx"
        index_path = self.config.get("index_path") or str(default_index)
        report("loading the embedding model…")
        embedder = self.embedder()
        if (Path(index_path) / "vectors.npy").exists():
            store = IndexedStore.load(index_path, embedder)
        else:
            store = IndexedStore.build([], embedder)
        fp = file_fingerprint(path)
        if store.fingerprint_of(path) == fp:
            return {"ok": True, "skipped": True, "chunks": len(store), "sources": store.sources()}
        if vision:
            pages = self._vision_pages(
                path, on_page=lambda n, t: report(f"transcribing page {n}/{t}…", n, t))
        else:
            report("extracting text…")
            pages = load_pages(path)
        report("embedding…")
        store.add([(path, chunk_pages(pages))], embedder, fingerprints={path: fp},
                  progress=lambda done, total: report(f"embedded {done}/{total} chunks", done, total))
        store.save(index_path)
        self.config["index_path"] = index_path
        save_config(self.config)
        self._store = None  # force reload
        self._pipeline = None  # the BM25/expanded pipeline was built over the old chunks
        return {"ok": True, "chunks": len(store), "sources": store.sources()}

    def _vision_extractor(self):
        from ..backends.vision_extract import VisionExtractor
        prov = self.config["provider"]
        model = self.config["models"].get("vision") or "openai/gpt-4o-mini"
        return VisionExtractor(model, prov["base_url"], prov.get("api_key") or None)

    def _vision_pages(self, path: str, on_page=None) -> list[tuple[int, str]]:
        return self._vision_extractor().extract_pages(path, on_page=on_page)

    def convert_file(self, path: str, vision: bool = False,
                     pages: list[int] | None = None, out: str | None = None,
                     progress=None) -> dict:
        """Convert a document to clean text (or vision-transcribed Markdown+LaTeX)."""
        report = progress or (lambda *a, **kw: None)
        src = Path(path).expanduser()
        if not src.exists():
            return {"error": f"File not found: {path}"}
        try:
            if vision:
                if not self._has_provider():
                    return {"error": "Vision conversion needs a provider — set the "
                                     "Base URL in Settings."}
                text = self._vision_extractor().extract(
                    str(src.resolve()), pages=pages,
                    on_page=lambda n, t: report(f"transcribing page {n}/{t}…", n, t))
                suffix = ".md"
            else:
                from ..ingest import load_text
                report("extracting & cleaning…")
                text = load_text(str(src.resolve()))
                suffix = ".txt"
        except Exception as exc:  # noqa: BLE001 - surface as a friendly message
            return {"error": f"Conversion failed: {exc}"}
        dest = Path(out).expanduser() if out else src.with_suffix(suffix)
        if dest.resolve() == src.resolve():  # never silently overwrite the original
            dest = src.with_name(src.stem + ".clean" + suffix)
        dest.write_text(text, encoding="utf-8")
        return {"ok": True, "out": str(dest), "chars": len(text),
                "preview": text[:1200]}

    def run_eval(self, questions_path: str, judge: bool = True,
                 model: str | None = None, progress=None, label: str = "") -> dict:
        """Run a question set through the pipeline and return the trust report.

        `model` overrides the generator for this run only — that's what lets the
        Compare tab race two models over the same documents and questions.
        `label` prefixes progress messages (e.g. the model name during a compare).
        """
        report = progress or (lambda *a, **kw: None)
        tag = f"{label}: " if label else ""
        qp = Path(questions_path).expanduser()
        if not qp.exists():
            return {"error": f"Questions file not found: {questions_path}"}
        try:
            raw = json.loads(qp.read_text(encoding="utf-8"))
            items = [EvalItem(q["question"], bool(q.get("answerable", True)))
                     for q in raw]
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            return {"error": f"Bad questions file ({exc}). Expected a JSON list like "
                             '[{"question": "...", "answerable": true}, ...]'}
        retriever = self.retriever()
        if retriever is None:
            return {"error": "No document indexed yet. Add one in the Library tab."}
        generator = self._make_generator(model_override=model)
        results = run_eval(items, retriever, generator,
                           k=int(self.config["pipeline"].get("k", 8)),
                           progress=lambda i, n: report(f"{tag}answering {i}/{n}…", i, n))
        if judge and self._has_provider() and self.config["models"].get("judge"):
            results = grade_results(results, Judge(self._make_generator("judge")),
                                    progress=lambda i, n: report(f"{tag}grading {i}/{n}…", i, n))
        rows = [
            {"question": r.item.question, "answerable": r.item.answerable,
             "abstained": r.answer.abstained, "citations": r.answer.citations,
             "correct": r.correct,
             "answer": " ".join(r.answer.text.split())[:200]}
            for r in results
        ]
        return {"ok": True, "model": model or self.config["models"]["generator"] or "mock",
                "metrics": compute_metrics(results), "rows": rows}

    def compare(self, questions_path: str, model_a: str, model_b: str,
                judge: bool = True, progress=None) -> dict:
        """Same documents, same questions, same pipeline — two generators.

        This is the honest version of 'which model should I trust on my library':
        everything is held constant except the model, and each side gets the full
        trust report. (Quantized and fine-tuned variants plug into this same
        comparison once P2/P3 land — they're just another model name.)
        """
        a = self.run_eval(questions_path, judge=judge, model=model_a,
                          progress=progress, label=model_a)
        if "error" in a:
            return a
        b = self.run_eval(questions_path, judge=judge, model=model_b,
                          progress=progress, label=model_b)
        if "error" in b:
            return b
        return {"ok": True, "a": a, "b": b}

    def update_settings(self, patch: dict) -> dict:
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(self.config.get(key), dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
        save_config(self.config)
        # Invalidate only the caches the patch actually touches: a theme change
        # must not throw away a loaded embedding model.
        if "embedder" in patch.get("models", {}):
            self._embedder = None
            self._store = None
        if "index_path" in patch:
            self._store = None
        # The pipeline cache invalidates itself: retriever() keys on the settings
        # it depends on, so a relevant change rebuilds and a theme change doesn't.
        return self.public_state()

    def library(self) -> list[dict]:
        store = self.store()
        if not store:
            return []
        counts: dict[str, int] = {}
        pages: dict[str, int] = {}
        for c in store.chunks():
            counts[c.source] = counts.get(c.source, 0) + 1
            if c.page:
                pages[c.source] = max(pages.get(c.source, 0), c.page)
        return [{"name": Path(s).name, "chunks": counts.get(s, 0),
                 "pages": pages.get(s) or None} for s in store.sources()]

    def public_state(self) -> dict:
        """Settings safe to send to the frontend (api key masked)."""
        cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.config.items()}
        key = cfg["provider"].get("api_key") or ""
        cfg["provider"]["api_key_set"] = bool(key)
        cfg["provider"]["api_key"] = ("•" * 8 + key[-4:]) if key else ""
        return {"config": cfg, "library": self.library()}
