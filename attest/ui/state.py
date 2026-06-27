"""AppState: holds settings + the active index, and assembles the RAG pipeline
exactly as the UI toggles describe it. One place that turns config into behaviour.

Everything is lazy and cached: the embedding model and index load on first use,
and are rebuilt only when the relevant settings change — so toggling 'rerank' or
switching models doesn't reload the whole index.
"""

from __future__ import annotations

from pathlib import Path

from ..chunking import chunk_text
from ..config import load_config, save_config
from ..grounding import build_prompt, parse_response
from ..ingest import load_text
from ..interfaces import Embedder, Generator
from ..store import IndexedStore, file_fingerprint


class AppState:
    def __init__(self) -> None:
        self.config = load_config()
        self._embedder: Embedder | None = None
        self._store: IndexedStore | None = None
        self._store_path: str | None = None

    # ---- builders -------------------------------------------------------
    def _make_generator(self, model_key: str = "generator") -> Generator:
        model = self.config["models"].get(model_key) or self.config["models"]["generator"]
        prov = self.config["provider"]
        if not model or not prov.get("base_url"):
            from ..backends.mock import MockGenerator
            return MockGenerator()
        from ..backends.openai_compat import OpenAICompatibleGenerator
        return OpenAICompatibleGenerator(model, prov["base_url"], prov.get("api_key") or None)

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
        """Assemble lexical→expand→rerank around the active index per the toggles."""
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
        passages = [
            {"id": r.chunk.index, "source": Path(r.chunk.source).name if r.chunk.source else "",
             "text": r.chunk.text[:600], "score": round(float(r.score), 3)}
            for r in retrieved
        ]
        result = {
            "abstained": answer.abstained,
            "answer": answer.text,
            "citations": answer.citations,
            "passages": passages,
        }
        if answer.abstained and pipe.get("allow_uncited"):
            result["uncited"] = generator.generate(question)
        return result

    def index_file(self, path: str, vision: bool = False) -> dict:
        """Index a document into the active index (creating one if needed)."""
        src = Path(path).expanduser()
        if not src.exists():
            return {"error": f"File not found: {path}"}
        path = str(src.resolve())  # absolute, so it resolves from any working dir
        default_index = Path.home() / ".attest" / "library.idx"
        index_path = self.config.get("index_path") or str(default_index)
        embedder = self.embedder()
        if (Path(index_path) / "vectors.npy").exists():
            store = IndexedStore.load(index_path, embedder)
        else:
            store = IndexedStore.build([], embedder)
        fp = file_fingerprint(path)
        if store.fingerprint_of(path) == fp:
            return {"ok": True, "skipped": True, "chunks": len(store), "sources": store.sources()}
        if vision:
            text = self._vision_text(path)
        else:
            text = load_text(path)
        store.add([(path, chunk_text(text))], embedder, fingerprints={path: fp})
        store.save(index_path)
        self.config["index_path"] = index_path
        save_config(self.config)
        self._store = None  # force reload
        return {"ok": True, "chunks": len(store), "sources": store.sources()}

    def _vision_text(self, path: str) -> str:
        from ..backends.vision_extract import VisionExtractor
        prov = self.config["provider"]
        model = self.config["models"].get("vision") or "openai/gpt-4o-mini"
        return VisionExtractor(model, prov["base_url"], prov.get("api_key") or None).extract(path)

    def update_settings(self, patch: dict) -> dict:
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(self.config.get(key), dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
        save_config(self.config)
        # invalidate caches that depend on settings
        self._embedder = None
        self._store = None
        return self.public_state()

    def library(self) -> list[str]:
        store = self.store()
        return [Path(s).name for s in store.sources()] if store else []

    def public_state(self) -> dict:
        """Settings safe to send to the frontend (api key masked)."""
        cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.config.items()}
        key = cfg["provider"].get("api_key") or ""
        cfg["provider"]["api_key_set"] = bool(key)
        cfg["provider"]["api_key"] = ("•" * 8 + key[-4:]) if key else ""
        return {"config": cfg, "library": self.library()}
