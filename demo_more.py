"""多文档联合问答示例（基于 PageIndex workspace）。"""
import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from agents import Agent, Runner, function_tool, set_tracing_disabled
from pageindex import PageIndexClient

load_dotenv()

if "OPENAI_API_BASE" in os.environ and "OPENAI_BASE_URL" not in os.environ:
    os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]

# 问答阶段默认走本地模型；未配置时回落到 OPENAI_API_KEY
if os.getenv("LOCAL_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("LOCAL_API_KEY")
if os.getenv("LOCAL_API_BASE"):
    os.environ["OPENAI_BASE_URL"] = os.getenv("LOCAL_API_BASE")

DEFAULT_WORKSPACE = Path(__file__).parent / "workspace"

AGENT_SYSTEM_PROMPT = """
你是一个专业的多文档问答助手。
必须先调用 list_documents 识别候选文档；
再按需对相关文档调用 get_document_structure；
最后只抓取必要的 get_page_content（小范围页码）后回答。
回答要求：
1) 仅基于工具返回内容；
2) 明确标注文档名和页码；
3) 若信息不足，明确说明缺失点；
4) 简洁、分点。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PageIndex 多文档联合问答")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="PageIndex workspace 目录（默认: ./workspace）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("LOCAL_MODEL", "qwen3.5-27b"),
        help="问答模型（默认: LOCAL_MODEL 或 qwen3.5-27b）",
    )
    return parser.parse_args()


def build_tools(client: PageIndexClient):
    @function_tool
    def list_documents() -> str:
        """列出 workspace 全部文档元信息（doc_id、doc_name、描述、页数）。"""
        docs = []
        for doc_id in sorted(client.documents.keys()):
            docs.append(json.loads(client.get_document(doc_id)))
        return json.dumps(docs, ensure_ascii=False)

    @function_tool
    def get_document_structure(doc_id: str) -> str:
        """获取某个文档的目录树（不含正文 text），用于定位相关章节与页码范围。"""
        return client.get_document_structure(doc_id)

    @function_tool
    def get_page_content(doc_id: str, pages: str) -> str:
        """
        获取某个文档指定页码内容。
        pages 格式示例："5-7"、"3,8"、"12"。
        """
        return client.get_page_content(doc_id, pages)

    return [list_documents, get_document_structure, get_page_content]


def query_all_docs(client: PageIndexClient, question: str, model: str) -> str:
    tools = build_tools(client)
    agent = Agent(
        name="MultiDocQA",
        instructions=AGENT_SYSTEM_PROMPT,
        tools=tools,
        model=model,
    )

    async def _run() -> str:
        result = await Runner.run(agent, question)
        return str(result.final_output or "")

    return asyncio.run(_run())


def main() -> int:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()

    if not workspace.exists():
        print(f"❌ workspace 不存在: {workspace}")
        print("请先执行索引构建，再进行问答。")
        return 1

    set_tracing_disabled(True)
    client = PageIndexClient(workspace=workspace)

    if not client.documents:
        print(f"⚠️ workspace 中没有文档: {workspace}")
        print("请先执行 build_all.py 或 run_pageindex.py 构建索引。")
        return 1

    print(f"✅ workspace: {workspace}")
    print(f"✅ 已加载 {len(client.documents)} 个文档")
    print(f"✅ 使用模型: {args.model}")

    while True:
        question = input("\n请输入问题（输入 quit 退出）：").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue

        print("\n思考中（跨文档检索）...\n")
        answer = query_all_docs(client, question, args.model)
        print("\n📝 答案：\n")
        print(answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
