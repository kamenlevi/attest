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
from .eval import EvalItem, compute_metrics, format_report, run_eval
from .grounding import build_prompt, parse_response
from .ingest import load_text
from .interfaces import Embedder, Generator
from .retrieval import Retriever


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


def _make_embedder(name: str) -> Embedder:
    if name == "mock":
        return MockEmbedder()
    if name == "local":
        from .backends.local_embed import LocalEmbedder

        return LocalEmbedder()
    sys.exit(f"unknown embedder: {name}")


def _build_retriever(doc_path: str, embedder_name: str) -> Retriever:
    text = load_text(doc_path)
    chunks = chunk_text(text)
    retriever = Retriever(_make_embedder(embedder_name))
    retriever.build(chunks)
    print(f"Loaded {doc_path}: {len(chunks)} chunks.", file=sys.stderr)
    return retriever


def _add_provider_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", choices=["mock", "openai"], default="mock")
    p.add_argument("--model", default=None, help="or set ATTEST_MODEL")
    p.add_argument("--base-url", default=None, help="or set ATTEST_BASE_URL")
    p.add_argument("--api-key", default=None, help="or set ATTEST_API_KEY")
    p.add_argument("--embedder", choices=["mock", "local"], default="mock",
                   help="'local' = real CPU embeddings (needs '.[embed-local]')")
    p.add_argument("--k", type=int, default=4, help="passages to retrieve")


def _cmd_demo(_args: argparse.Namespace) -> None:
    from .demo import main as demo_main

    demo_main()


def _cmd_ask(args: argparse.Namespace) -> None:
    retriever = _build_retriever(args.doc, args.embedder)
    generator = _make_generator(args)
    retrieved = retriever.search(args.question, k=args.k)
    response = generator.generate(build_prompt(args.question, retrieved))
    answer = parse_response(response)
    print(response)
    if answer.abstained:
        print("\n[abstained — not in sources]")
    elif answer.citations:
        print(f"\n[cited passages: {answer.citations}]")
    else:
        print("\n[warning: answered with no citation]")


def _cmd_eval(args: argparse.Namespace) -> None:
    retriever = _build_retriever(args.doc, args.embedder)
    generator = _make_generator(args)
    with open(args.questions, encoding="utf-8") as fh:
        raw = json.load(fh)
    items = [EvalItem(q["question"], bool(q["answerable"])) for q in raw]
    results = run_eval(items, retriever, generator, k=args.k)

    if args.verbose:
        for r in results:
            label = "answerable" if r.item.answerable else "trap      "
            if r.answer.abstained:
                verdict = "ABSTAINED"
            else:
                verdict = f"answered {r.answer.citations or '(no citation)'}"
            snippet = " ".join(r.answer.text.split())[:100]
            print(f"[{label}] {r.item.question}\n    -> {verdict}: {snippet}\n")

    print(format_report(compute_metrics(results)))


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()  # pick up .env (your key) before anything else
    parser = argparse.ArgumentParser(prog="attest", description="Local-model trust workbench.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("demo", help="run the no-model demo").set_defaults(func=_cmd_demo)

    ask = sub.add_parser("ask", help="ask one grounded question about a document")
    ask.add_argument("--doc", required=True)
    ask.add_argument("--question", required=True)
    _add_provider_args(ask)
    ask.set_defaults(func=_cmd_ask)

    ev = sub.add_parser("eval", help="run an eval set and print the trust report")
    ev.add_argument("--doc", required=True)
    ev.add_argument("--questions", required=True, help="JSON list of {question, answerable}")
    ev.add_argument("--verbose", action="store_true", help="show each question's answer")
    _add_provider_args(ev)
    ev.set_defaults(func=_cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
