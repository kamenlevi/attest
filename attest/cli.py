"""Command-line interface for the Attest engine.

Examples:
    # 1) No model at all — the cardboard-cutout demo (proves the machine works):
    attest demo

    # 2) Ask one question against a document, using a real model:
    attest ask --doc book.txt --question "What is X?" \
        --provider openai --model gpt-4o-mini \
        --base-url https://api.openai.com/v1 --api-key sk-...

    # 2b) Same, but a LOCAL model via Ollama on this machine (no key needed):
    attest ask --doc book.txt --question "What is X?" \
        --provider openai --model llama3.2 --base-url http://localhost:11434/v1

    # 3) Run a whole eval set and print the trust report (bluff rate etc.):
    attest eval --doc book.txt --questions questions.json --provider openai \
        --model gpt-4o-mini --base-url https://api.openai.com/v1 --api-key sk-...

`questions.json` is a list like:
    [{"question": "What is X?", "answerable": true},
     {"question": "Unrelated trap?", "answerable": false}]

Note: embeddings currently use the lexical mock embedder, so retrieval is
keyword-based for now. Real embeddings are the next upgrade. The *generator*,
however, is the real model you choose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .backends.mock import MockEmbedder, MockGenerator
from .backends.openai_compat import OpenAICompatibleGenerator
from .chunking import chunk_text
from .eval import EvalItem, compute_metrics, format_report, grade_results, run_eval
from .grounding import build_prompt, parse_response
from .ingest import load_text
from .interfaces import Embedder, Generator
from .judge import Judge
from .query import ExpandingRetriever, QueryExpander
from .retrieval import HybridRetriever, RerankingRetriever, Retriever
from .store import IndexedStore, file_fingerprint


def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines from a local .env into the environment (no deps).

    Existing environment variables win, so an explicit `export` always overrides
    the file. Silently does nothing if there's no .env.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _make_generator(args: argparse.Namespace) -> Generator:
    if args.provider == "mock":
        return MockGenerator()
    if args.provider == "openai":
        # Each flag falls back to an env var, so you can `export` once and stop
        # typing long commands: ATTEST_MODEL, ATTEST_BASE_URL, ATTEST_API_KEY.
        model = args.model or os.environ.get("ATTEST_MODEL")
        base_url = args.base_url or os.environ.get("ATTEST_BASE_URL")
        api_key = args.api_key or os.environ.get("ATTEST_API_KEY")
        if not model:
            sys.exit("--model (or ATTEST_MODEL) is required for --provider openai")
        if not base_url:
            sys.exit("--base-url (or ATTEST_BASE_URL) is required for --provider openai")
        return OpenAICompatibleGenerator(model=model, base_url=base_url, api_key=api_key)
    sys.exit(f"unknown provider: {args.provider}")


def _make_judge_generator(args: argparse.Namespace, fallback: Generator) -> Generator:
    """A separate (usually stronger) model for grading. A small generator is a poor
    grader — it misjudges algebraically-equivalent answers — so let the judge be a
    bigger model than the one being tested. Falls back to the generator itself."""
    if not getattr(args, "judge_model", None):
        return fallback
    base_url = args.base_url or os.environ.get("ATTEST_BASE_URL")
    api_key = args.api_key or os.environ.get("ATTEST_API_KEY")
    return OpenAICompatibleGenerator(model=args.judge_model, base_url=base_url, api_key=api_key)


def _make_embedder(name: str) -> Embedder:
    if name == "mock":
        return MockEmbedder()
    if name == "local":
        from .backends.local_embed import LocalEmbedder

        return LocalEmbedder()
    sys.exit(f"unknown embedder: {name}")


def _make_retriever(embedder_name: str):
    if embedder_name == "hybrid":
        from .backends.local_embed import LocalEmbedder

        return HybridRetriever([Retriever(MockEmbedder()), Retriever(LocalEmbedder())])
    return Retriever(_make_embedder(embedder_name))


def _build_retriever(doc_path: str, embedder_name: str):
    text = load_text(doc_path)
    chunks = chunk_text(text)
    retriever = _make_retriever(embedder_name)
    retriever.build(chunks)
    print(f"Loaded {doc_path}: {len(chunks)} chunks.", file=sys.stderr)
    return retriever


def _get_retriever(args: argparse.Namespace):
    """Use a prebuilt on-disk index if --index is given, else embed --doc now."""
    if getattr(args, "index", None):
        store = IndexedStore.load(args.index, _make_embedder(args.embedder))
        print(f"Loaded index {args.index}: {len(store)} chunks (no re-embedding).",
              file=sys.stderr)
        return store
    if not getattr(args, "doc", None):
        sys.exit("provide either --doc <file> or --index <dir>")
    return _build_retriever(args.doc, args.embedder)


def _add_provider_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", choices=["mock", "openai"], default="mock")
    p.add_argument("--model", default=None, help="or set ATTEST_MODEL")
    p.add_argument("--base-url", default=None, help="or set ATTEST_BASE_URL")
    p.add_argument("--api-key", default=None, help="or set ATTEST_API_KEY")
    p.add_argument("--embedder", choices=["mock", "local", "hybrid"], default="mock",
                   help="mock=keyword, local=semantic, hybrid=both fused (needs '.[embed-local]')")
    p.add_argument("--k", type=int, default=4, help="passages to retrieve")
    p.add_argument("--expand", action="store_true",
                   help="expand the question (HyDE + key terms) before searching — "
                        "one extra model call per query, better recall")
    p.add_argument("--pool", type=int, default=30,
                   help="candidates per expansion text before fusing (with --expand)")
    p.add_argument("--lexical", action="store_true",
                   help="fuse in BM25 keyword retrieval (catches exact terms/proper "
                        "nouns/labels that semantic search misses) — better recall")
    p.add_argument("--rerank", action="store_true",
                   help="re-score retrieved candidates with a cross-encoder and keep "
                        "the best (precision pass; needs '.[embed-local]')")
    p.add_argument("--rerank-pool", type=int, default=40,
                   help="candidates to re-score before keeping top-k (with --rerank)")


def _cmd_demo(_args: argparse.Namespace) -> None:
    from .demo import main as demo_main

    demo_main()


def _cmd_convert(args: argparse.Namespace) -> None:
    """Extract a document to clean .txt the model can read without trouble."""
    cleaned = load_text(args.doc, clean=not args.raw)
    out = Path(args.out) if args.out else Path(args.doc).with_suffix(".txt")
    out.write_text(cleaned, encoding="utf-8")
    print(f"Wrote {len(cleaned):,} chars -> {out}"
          f"{'' if args.raw else ' (cleaned)'}")


def _cmd_index(args: argparse.Namespace) -> None:
    embedder = _make_embedder(args.embedder)  # lazy: no model load until we embed
    out = Path(args.out)
    # Add to an existing index if one is already there, so a file that's already
    # been indexed is skipped (instant) and only new/changed files are embedded.
    if (out / "vectors.npy").exists():
        store = IndexedStore.load(out, embedder)
        print(f"Opened existing index: {len(store)} chunks, {len(store.sources())} file(s).",
              file=sys.stderr)
    else:
        store = IndexedStore.build([], embedder)
    # Decide skips from a cheap file fingerprint BEFORE extracting/chunking — so
    # re-indexing an unchanged 600-page PDF is instant, not a re-extraction.
    docs, fingerprints, skipped = [], {}, 0
    for path in args.docs:
        fp = file_fingerprint(path)
        if store.fingerprint_of(path) == fp:
            print(f"  {path}: unchanged — skipped", file=sys.stderr)
            skipped += 1
            continue
        chunks = chunk_text(load_text(path))
        docs.append((path, chunks))
        fingerprints[path] = fp
        print(f"  {path}: {len(chunks)} chunks", file=sys.stderr)
    stats = (store.add(docs, embedder, fingerprints=fingerprints) if docs
             else {"added": 0, "updated": 0, "skipped": 0})
    stats["skipped"] += skipped
    store.save(out)
    print(f"Index {out}: +{stats['added']} new, {stats['updated']} updated, "
          f"{stats['skipped']} unchanged (skipped). "
          f"Now {len(store)} chunks from {len(store.sources())} file(s).")


def _wrap_retriever(retriever, generator, args: argparse.Namespace):
    """Apply the optional retrieval stages in order: lexical fusion (recall) ->
    expand (recall) -> rerank (precision). Each is opt-in, so plain search still
    works unchanged."""
    if getattr(args, "lexical", False):
        from .lexical import BM25Retriever
        from .retrieval import FusedRetriever

        if hasattr(retriever, "chunks"):
            bm25 = BM25Retriever(retriever.chunks())
            retriever = FusedRetriever([retriever, bm25], pool=args.pool)
        else:
            print("--lexical needs an index/doc with chunks(); skipping", file=sys.stderr)
    if getattr(args, "expand", False):
        retriever = ExpandingRetriever(retriever, QueryExpander(generator), pool=args.pool)
    if getattr(args, "rerank", False):
        from .backends.rerank import CrossEncoderReranker

        retriever = RerankingRetriever(retriever, CrossEncoderReranker(), pool=args.rerank_pool)
    return retriever


def _cmd_ask(args: argparse.Namespace) -> None:
    generator = _make_generator(args)
    retriever = _wrap_retriever(_get_retriever(args), generator, args)
    retrieved = retriever.search(args.question, k=args.k)
    response = generator.generate(build_prompt(args.question, retrieved))
    answer = parse_response(response)
    print(response)
    if answer.abstained:
        print("\n[abstained — not in sources]")
        if args.allow_uncited:
            # User opted in to seeing the model's own (un-sourced) knowledge.
            uncited = generator.generate(args.question)
            print("\n[UNCITED — from the model's own knowledge, NOT your sources]")
            print(uncited)
    elif answer.citations:
        print(f"\n[cited passages: {answer.citations}]")
    else:
        print("\n[warning: answered with no citation]")


def _cmd_eval(args: argparse.Namespace) -> None:
    generator = _make_generator(args)
    retriever = _wrap_retriever(_get_retriever(args), generator, args)
    with open(args.questions, encoding="utf-8") as fh:
        raw = json.load(fh)
    items = [EvalItem(q["question"], bool(q["answerable"])) for q in raw]
    results = run_eval(items, retriever, generator, k=args.k)

    if args.judge:
        # Grade answered answerable questions for correctness. By default reuses the
        # generator; --judge-model points grading at a separate, stronger model.
        results = grade_results(results, Judge(_make_judge_generator(args, generator)))

    if args.verbose:
        for r in results:
            label = "answerable" if r.item.answerable else "trap      "
            if r.answer.abstained:
                verdict = "ABSTAINED"
            else:
                verdict = f"answered {r.answer.citations or '(no citation)'}"
            if r.correct is not None:
                verdict += "  [graded: CORRECT]" if r.correct else "  [graded: INCORRECT]"
            snippet = " ".join(r.answer.text.split())[:100]
            print(f"[{label}] {r.item.question}\n    -> {verdict}: {snippet}\n")

    print(format_report(compute_metrics(results)))


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()  # pick up .env (your key) before anything else
    parser = argparse.ArgumentParser(prog="attest", description="Local-model trust workbench.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("demo", help="run the no-model demo").set_defaults(func=_cmd_demo)

    cv = sub.add_parser("convert", help="extract a PDF/doc to clean .txt the model reads cleanly")
    cv.add_argument("doc", help="a .pdf/.txt file to convert")
    cv.add_argument("--out", default=None, help="output path (default: alongside, .txt)")
    cv.add_argument("--raw", action="store_true", help="skip cleaning; emit the raw extraction")
    cv.set_defaults(func=_cmd_convert)

    ix = sub.add_parser("index", help="embed file(s) once and save a reusable index")
    ix.add_argument("docs", nargs="+", help="one or more .txt/.pdf files to index")
    ix.add_argument("--out", required=True, help="directory to write the index to")
    ix.add_argument("--embedder", choices=["mock", "local"], default="local")
    ix.set_defaults(func=_cmd_index)

    ask = sub.add_parser("ask", help="ask one grounded question about a document/index")
    ask.add_argument("--doc", help="a file to embed now (or use --index)")
    ask.add_argument("--index", help="a prebuilt index directory (from `attest index`)")
    ask.add_argument("--question", required=True)
    ask.add_argument("--allow-uncited", action="store_true",
                     help="if abstained, also show the model's own un-sourced answer")
    _add_provider_args(ask)
    ask.set_defaults(func=_cmd_ask)

    ev = sub.add_parser("eval", help="run an eval set and print the trust report")
    ev.add_argument("--doc", help="a file to embed now (or use --index)")
    ev.add_argument("--index", help="a prebuilt index directory (from `attest index`)")
    ev.add_argument("--questions", required=True, help="JSON list of {question, answerable}")
    ev.add_argument("--verbose", action="store_true", help="show each question's answer")
    ev.add_argument("--judge", action="store_true",
                    help="grade answer correctness with an LLM judge (extra model calls)")
    ev.add_argument("--judge-model", default=None,
                    help="use a separate (stronger) model for grading, e.g. "
                         "openai/gpt-4o-mini — a small model is an unreliable judge")
    _add_provider_args(ev)
    ev.set_defaults(func=_cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
