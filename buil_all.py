import os
import glob
import subprocess

PDF_DIR = "examples/mark_pdf"
MODEL = "openai/qwen-doc-turbo"

pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))

print(f"✅ 找到 {len(pdf_files)} 个 PDF 文件")
print("开始批量构建索引...\n")

for idx, pdf_path in enumerate(pdf_files, 1):
    print(f"[{idx}/{len(pdf_files)}] 构建索引: {pdf_path}")
    cmd = [
        "python", "run_pageindex.py",
        "--pdf_path", pdf_path,
        "--model", MODEL
    ]
    subprocess.run(cmd)

print("\n✅ 所有文档索引构建完成！")