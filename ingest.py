"""Step 1: load every document in data/documents into the vector database.

Usage:
    python ingest.py            # add / update documents
    python ingest.py --reset    # wipe the database first
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.table import Table

from src import config, embedder, vector_store
from src.chunker import chunk_blocks
from src.loader import load_directory

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Nạp tài liệu vào vector database")
    parser.add_argument("--reset", action="store_true", help="Xoá toàn bộ dữ liệu cũ trước khi nạp")
    args = parser.parse_args()

    config.ensure_dirs()

    if args.reset:
        console.print("[yellow]Đang xoá collection cũ...[/yellow]")
        vector_store.reset_collection()

    console.print(f"[bold]Đọc tài liệu từ:[/bold] {config.DOCUMENTS_DIR}")
    blocks = load_directory(config.DOCUMENTS_DIR, config.SUPPORTED_EXTENSIONS)

    if not blocks:
        console.print(
            "[red]Không tìm thấy tài liệu nào.[/red] "
            f"Hãy copy file PDF/DOCX/TXT vào thư mục {config.DOCUMENTS_DIR}"
        )
        return

    sources = sorted({b.source for b in blocks})
    console.print(f"Đã đọc [green]{len(blocks)}[/green] trang từ [green]{len(sources)}[/green] file:")
    for source in sources:
        pages = sum(1 for b in blocks if b.source == source)
        console.print(f"  • {source} ({pages} trang có nội dung)")

    chunks = chunk_blocks(blocks, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not args.reset:
        # Re-ingesting a file must REPLACE its chunks, not pile new ones on the old.
        for source in sources:
            removed = vector_store.delete_source(source)
            if removed:
                console.print(f"  • đã xoá {removed} chunk cũ của {source}")
    console.print(f"Đã cắt thành [green]{len(chunks)}[/green] chunk "
                  f"(kích thước {config.CHUNK_SIZE}, overlap {config.CHUNK_OVERLAP})")

    console.print(f"[bold]Tạo embedding trên:[/bold] {embedder.describe_device()}")
    console.print("[dim]Lần chạy đầu tiên sẽ tải model ~2GB, hãy kiên nhẫn.[/dim]")

    started = time.time()
    # Embed heading + body (see Chunk.embedding_text); we still STORE the plain
    # body, so the model reads clean text and citations stay unchanged.
    vectors = embedder.embed_documents([c.embedding_text for c in chunks])
    elapsed = time.time() - started

    vector_store.add_chunks(chunks, vectors)

    table = Table(title="Kết quả nạp tài liệu", show_header=False)
    table.add_row("Số chunk đã lưu", str(len(chunks)))
    table.add_row("Tổng chunk trong DB", str(vector_store.count()))
    table.add_row("Thời gian embedding", f"{elapsed:.1f}s")
    table.add_row("Vị trí database", str(config.CHROMA_DIR))
    console.print(table)
    console.print("\n[bold green]Xong.[/bold green] Chạy [cyan]python ask.py[/cyan] để bắt đầu hỏi.")


if __name__ == "__main__":
    main()