"""Step 2: ask questions against the ingested documents.

Usage:
    python ask.py                          # interactive mode
    python ask.py "Mức bồi thường tối đa là bao nhiêu?"
    python ask.py --debug "..."            # also show retrieved chunks and scores
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src import config, generator, retriever, vector_store

console = Console()


def show_sources(chunks, debug: bool) -> None:
    if not chunks:
        return
    console.print("\n[bold]Nguồn trích dẫn:[/bold]")
    for chunk in chunks:
        if chunk.distance is None:
            label = "ngữ cảnh liền kề"
        elif chunk.rerank_score is not None:
            label = f"liên quan {chunk.rerank_score:.0%}, tương đồng {chunk.score}"
        else:
            label = f"độ tương đồng {chunk.score}"
        console.print(f"  [cyan][{chunk.number}][/cyan] {chunk.citation} [dim]({label})[/dim]")
        if debug:
            preview = chunk.text[:400].replace("\n", " ")
            console.print(f"      [dim]{preview}...[/dim]")


def answer_once(question: str, debug: bool) -> None:
    with console.status("[dim]Đang tìm trong tài liệu...[/dim]"):
        chunks = retriever.retrieve(question)

    with console.status("[dim]Đang soạn câu trả lời...[/dim]"):
        answer = generator.generate_answer(question, chunks)

    console.print(Panel(Markdown(answer), title="Trả lời", border_style="green"))
    show_sources(chunks, debug)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hỏi đáp trên tài liệu đã nạp")
    parser.add_argument("question", nargs="*", help="Câu hỏi (bỏ trống để vào chế độ tương tác)")
    parser.add_argument("--debug", action="store_true", help="Hiện cả nội dung chunk được lấy")
    args = parser.parse_args()

    config.ensure_dirs()

    total = vector_store.count()
    if total == 0:
        console.print("[red]Database đang rỗng.[/red] Chạy [cyan]python ingest.py[/cyan] trước.")
        return

    if args.question:
        answer_once(" ".join(args.question), args.debug)
        return

    console.print(Panel(
        f"Đã sẵn sàng với [green]{total}[/green] chunk tài liệu.\n"
        "Gõ câu hỏi rồi Enter. Gõ [cyan]exit[/cyan] để thoát.",
        title="Hệ thống tra cứu tài liệu nội bộ",
        border_style="blue",
    ))

    while True:
        try:
            question = console.input("\n[bold blue]Câu hỏi >[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nTạm biệt.")
            return
        if not question:
            continue
        if question.lower() in {"exit", "quit", "thoat", "thoát"}:
            console.print("Tạm biệt.")
            return
        answer_once(question, args.debug)


if __name__ == "__main__":
    main()
