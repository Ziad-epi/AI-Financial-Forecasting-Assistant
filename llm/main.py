from __future__ import annotations

import argparse
from pathlib import Path

from llm.rag_pipeline import ChunkingConfig, PipelineConfig, RAGPipeline


def run_evaluation(pipeline: RAGPipeline) -> None:
    queries = [
        "What did the Federal Reserve signal about interest rates?",
        "How did oil prices move and why?",
        "What did the U.S. bank report about credit losses?",
    ]

    print("\n[Evaluation]")
    for idx, query in enumerate(queries, start=1):
        print(f"\n{idx}. Query: {query}")
        answer = pipeline.answer(query, debug=True)
        print("Answer:")
        print(answer)


def interactive_loop(pipeline: RAGPipeline) -> None:
    print("\nRAG CLI. Commands: :help, :k <num>, :add <path>, :eval, :exit")
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input in {":exit", "exit", "quit"}:
            break
        if user_input == ":help":
            print("Commands:")
            print(":k <num>   Set top-k retrieval")
            print(":add <path> Add a new text file (paragraphs separated by blank lines)")
            print(":eval      Run manual evaluation queries")
            print(":exit      Quit")
            continue
        if user_input.startswith(":k "):
            try:
                new_k = int(user_input.split(maxsplit=1)[1])
                pipeline.config.k = new_k
                print(f"Top-k set to {new_k}")
            except Exception:
                print("Invalid k. Usage: :k 4")
            continue
        if user_input.startswith(":add "):
            path = user_input.split(maxsplit=1)[1]
            if not Path(path).exists():
                print(f"File not found: {path}")
                continue
            docs = pipeline.load_documents(path)
            added = pipeline.add_documents(docs, source_name=Path(path).name)
            print(f"Added {added} chunks from {path}")
            continue
        if user_input == ":eval":
            run_evaluation(pipeline)
            continue

        answer = pipeline.answer(user_input, debug=True)
        print("Answer:")
        print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial RAG Assistant")
    parser.add_argument("--data", default="llm/data/financial_news.txt", help="Path to data file")
    parser.add_argument("--k", type=int, default=4, help="Top-k retrieval")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    parser.add_argument("--generator", default="openai", help="openai or local")
    parser.add_argument("--reset", action="store_true", help="Reset vector store")
    parser.add_argument("--index", action="store_true", help="Index data before answering")
    parser.add_argument("--eval", action="store_true", help="Run manual evaluation queries")
    parser.add_argument("--query", help="Single query to run and exit")
    args = parser.parse_args()

    config = PipelineConfig(
        data_path=args.data,
        k=args.k,
        chunking=ChunkingConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap),
    )

    pipeline = RAGPipeline(config=config, generator_name=args.generator)

    if args.reset:
        pipeline.reset()

    if args.index:
        total = pipeline.index(reset=False)
        print(f"Indexed {total} chunks from {args.data}")

    if args.eval:
        run_evaluation(pipeline)
        return

    if args.query:
        answer = pipeline.answer(args.query, debug=True)
        print("Answer:")
        print(answer)
        return

    interactive_loop(pipeline)


if __name__ == "__main__":
    main()
