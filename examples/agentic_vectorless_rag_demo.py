"""
Agentic Vectorless RAG with PageIndex - Demo

A simple example of building a document QA agent with self-hosted PageIndex
and the OpenAI Agents SDK. Instead of vector similarity search and chunking,
PageIndex builds a hierarchical tree index and uses agentic LLM reasoning for
human-like, context-aware retrieval.

Agent tools:
  - get_document()           — document metadata (status, page count, etc.)
  - get_document_structure() — tree structure index of a document
  - get_page_content()       — retrieve text content of specific pages

Steps:
  1 — Index a PDF and view its tree structure index
  2 — View document metadata
  3 — Ask a question (agent reasons over the index and auto-calls tools)

Requirements: pip install openai-agents
"""
import os
import sys
import json
import asyncio
import concurrent.futures
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()
if "OPENAI_API_BASE" in os.environ and "OPENAI_BASE_URL" not in os.environ:
    os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.model_settings import ModelSettings
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from openai.types.responses import ResponseTextDeltaEvent, ResponseReasoningSummaryTextDeltaEvent

from pageindex import PageIndexClient
import pageindex.utils as utils

_EXAMPLES_DIR = Path(__file__).parent
PDF_PATH = _EXAMPLES_DIR / "documents" / "中国通服黔107号.pdf"
WORKSPACE = _EXAMPLES_DIR / "workspace"

AGENT_SYSTEM_PROMPT = """
You are PageIndex, a document QA assistant.
TOOL USE:
- Call get_document() first to confirm status and page/line count.
- Call get_document_structure() to identify relevant page ranges.
- Call get_page_content(pages="5-7") with tight ranges; never fetch the whole document.
- Before each tool call, output one short sentence explaining the reason.
Answer based only on tool output. Be concise.
"""


def query_agent(client: PageIndexClient, doc_id: str, prompt: str, verbose: bool = False) -> str:
    """Run a document QA agent using the OpenAI Agents SDK.

    Streams text output token-by-token and returns the full answer string.
    Tool calls are always printed; verbose=True also prints arguments and output previews.
    """

    @function_tool
    def get_document() -> str:
        """Get document metadata: status, page count, name, and description."""
        return client.get_document(doc_id)

    @function_tool
    def get_document_structure() -> str:
        """Get the document's full tree structure (without text) to find relevant sections."""
        return client.get_document_structure(doc_id)

    @function_tool
    def get_page_content(pages: str) -> str:
        """
        Get the text content of specific pages or line numbers.
        Use tight ranges: e.g. '5-7' for pages 5 to 7, '3,8' for pages 3 and 8, '12' for page 12.
        For Markdown documents, use line numbers from the structure's line_num field.
        """
        return client.get_page_content(doc_id, pages)

    agent = Agent(
        name="PageIndex",
        instructions=AGENT_SYSTEM_PROMPT,
        tools=[get_document, get_document_structure, get_page_content],
        model="Qwen3.5-27B",
        # model_settings=ModelSettings(reasoning={"effort": "low", "summary": "auto"}),  # Uncomment to enable reasoning
    )

    async def _run():
        streamed_run = Runner.run_streamed(agent, prompt)
        current_stream_kind = None
        async for event in streamed_run.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                if isinstance(event.data, ResponseReasoningSummaryTextDeltaEvent):
                    if current_stream_kind != "reasoning":
                        if current_stream_kind is not None:
                            print()
                        print("\n[reasoning]: ", end="", flush=True)
                    delta = event.data.delta
                    print(delta, end="", flush=True)
                    current_stream_kind = "reasoning"
                elif isinstance(event.data, ResponseTextDeltaEvent):
                    if current_stream_kind != "text":
                        if current_stream_kind is not None:
                            print()
                        print("\n[text]: ", end="", flush=True)
                    delta = event.data.delta
                    print(delta, end="", flush=True)
                    current_stream_kind = "text"
            elif isinstance(event, RunItemStreamEvent):
                item = event.item
                if item.type == "tool_call_item":
                    if current_stream_kind is not None:
                        print()
                    raw = item.raw_item
                    args = getattr(raw, "arguments", "{}")
                    args_str = f"({args})" if verbose else ""
                    print(f"\n[tool call]: {raw.name}{args_str}", flush=True)
                    current_stream_kind = None
                elif item.type == "tool_call_output_item" and verbose:
                    if current_stream_kind is not None:
                        print()
                    output = str(item.output)
                    preview = output[:200] + "..." if len(output) > 200 else output
                    print(f"\n[tool call output]: {preview}", flush=True)
                    current_stream_kind = None
        if current_stream_kind is not None:
            print()
        return "" if not streamed_run.final_output else str(streamed_run.final_output)

    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _run()).result()
    except RuntimeError:
        return asyncio.run(_run())


if __name__ == "__main__":

    set_tracing_disabled(True)

    if not PDF_PATH.exists():
        print(f"错误: 找不到文件 {PDF_PATH}")
        sys.exit(1)

    # Setup
    client = PageIndexClient(workspace=WORKSPACE)

    # Step 1: Index PDF and view tree structure
    print("=" * 60)
    print("Step 1: Index PDF and view tree structure")
    print("=" * 60)
    doc_id = next(
        (did for did, doc in client.documents.items() if doc.get('doc_name') == PDF_PATH.name),
        None,
    )
    
    json_path = Path("results") / "中国通服黔107号_structure.json"
    
    if doc_id:
        print(f"\nLoaded cached doc_id: {doc_id} from workspace")
    elif json_path.exists():
        import PyPDF2
        import uuid
        
        with open(json_path, "r", encoding="utf-8") as f:
            structure = json.load(f)['structure']
            
        pages = []
        with open(PDF_PATH, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(pdf_reader.pages, 1):
                pages.append({'page': i, 'content': page.extract_text() or ''})
                
        doc_id = str(uuid.uuid4())
        client.documents[doc_id] = {
            'id': doc_id,
            'type': 'pdf',
            'path': str(PDF_PATH.absolute()),
            'doc_name': PDF_PATH.name,
            'doc_description': '',
            'page_count': len(pages),
            'structure': structure,
            'pages': pages
        }
        print(f"\nLoaded pre-parsed structure from {json_path}. doc_id: {doc_id}")
    else:
        doc_id = client.index(PDF_PATH)
        print(f"\nIndexed. doc_id: {doc_id}")
    print("\nTree Structure (top-level sections):")
    structure = json.loads(client.get_document_structure(doc_id))
    utils.print_tree(structure)

    # Step 2: View document metadata
    print("\n" + "=" * 60)
    print("Step 2: View document metadata")
    print("=" * 60)
    doc_metadata = client.get_document(doc_id)
    print(f"\n{doc_metadata}")

    # Step 3: Agent Query
    print("\n" + "=" * 60)
    print("Step 3: Agent Query (auto tool-use)")
    print("=" * 60)
    initial_question = "请总结一下这份文档的主要内容。"
    print(f"\nQuestion: '{initial_question}'")
    query_agent(client, doc_id, initial_question, verbose=True)

    # Interactive Loop
    while True:
        try:
            question = input("\n请输入你要问的问题 (输入 'quit' 退出): ")
            if question.lower().strip() in ('quit', 'exit', 'q'):
                break
            if not question.strip():
                continue
            print(f"\nQuestion: '{question}'")
            query_agent(client, doc_id, question, verbose=True)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
