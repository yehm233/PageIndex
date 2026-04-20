"""Batch-build PageIndex structures for all PDFs in a folder."""
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch build PageIndex structure JSON files for all PDFs in a directory.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("pdfs"),
        help="Directory containing PDF files (default: ./pdfs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing output structure JSON files (default: ./results)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/qwen-doc-turbo",
        help="Model passed to run_pageindex.py (default: openai/qwen-doc-turbo)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdf",
        help="Filename pattern for documents (default: *.pdf)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_dir = args.pdf_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not pdf_dir.exists() or not pdf_dir.is_dir():
        print(f"❌ PDF 目录不存在: {pdf_dir}")
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.glob(args.pattern))

    print(f"✅ 目录: {pdf_dir}")
    print(f"✅ 输出目录: {output_dir}")
    print(f"✅ 找到 {len(pdf_files)} 个 PDF 文件")

    if not pdf_files:
        print("⚠️ 没有匹配的 PDF，退出。")
        return 0

    print("开始批量构建索引...\n")

    failures = []
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"[{idx}/{len(pdf_files)}] 构建索引: {pdf_path.name}")
        cmd = [
            sys.executable,
            "run_pageindex.py",
            "--pdf_path",
            str(pdf_path),
            "--output-dir",
            str(output_dir),
            "--model",
            args.model,
        ]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures.append((pdf_path, result.returncode))
            print(f"  ❌ 失败: {pdf_path.name} (exit={result.returncode})")
        else:
            print(f"  ✅ 完成: {pdf_path.name}")

    if failures:
        print("\n⚠️ 批量构建完成，但有失败文件:")
        for file_path, code in failures:
            print(f"- {file_path} (exit={code})")
        return 2

    print("\n✅ 所有文档索引构建完成！")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
