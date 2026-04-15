"""
✅ 多文档联合问答 Agent
支持：批量索引 + 跨文档检索 + 自动回答
"""
import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
if "OPENAI_API_BASE" in os.environ:
    os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent, Runner, function_tool, set_tracing_disabled
from pageindex import PageIndexClient
import pageindex.utils as utils

# ========== 强制使用本地模型做问答 ==========
os.environ["OPENAI_API_KEY"] = os.getenv("LOCAL_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("LOCAL_API_BASE")

WORKSPACE = Path(__file__).parent / "workspace"

AGENT_SYSTEM_PROMPT = """
你是一个专业的文档问答助手，能自动查询多个文件，提取最准确的信息回答用户。
请只根据工具返回的内容回答，不要编造。
回答简洁、准确、分点。
"""

def load_all_documents():
    """加载 workspace 中所有文档"""
    client = PageIndexClient(workspace=WORKSPACE)
    docs = []
    for doc_id, doc_info in client.documents.items():
        doc_name = doc_info.get("doc_name", "未知文档")
        docs.append((doc_id, doc_name, client))
    return docs


def query_agent_all_docs(docs, prompt):
    """查询所有文档，自动汇总答案"""

    @function_tool
    def search_all_documents(query: str):
        """搜索所有文档，返回相关结构和内容"""
        all_results = []
        for doc_id, doc_name, client in docs:
            structure = json.loads(client.get_document_structure(doc_id))
            pages = client.documents[doc_id]["pages"]
            pages_content = "\n".join([p["content"][:800] for p in pages[:6]])
            info = {
                "文档名": doc_name,
                "目录结构": json.dumps(structure, ensure_ascii=False, indent=1),
                "前几页内容预览": pages_content
            }
            all_results.append(info)
        return json.dumps(all_results, ensure_ascii=False, indent=2)

    agent = Agent(
        name="MultiDocQA",
        instructions=AGENT_SYSTEM_PROMPT,
        tools=[search_all_documents],
        model=os.getenv("LOCAL_MODEL", "qwen3.5-27b"),
    )

    async def _run():
        result = await Runner.run(agent, prompt)
        return result.final_output

    return asyncio.run(_run())


if __name__ == "__main__":
    set_tracing_disabled(True)
    print("✅ 加载所有文档索引...")
    docs = load_all_documents()
    print(f"✅ 已加载 {len(docs)} 个文档")

    while True:
        question = input("\n请输入问题（输入 quit 退出）：")
        if question.lower() in ["quit", "exit", "q"]:
            break
        if not question.strip():
            continue

        print("\n思考中，查询所有文档...\n")
        answer = query_agent_all_docs(docs, question)
        print("\n📝 答案：\n", answer)