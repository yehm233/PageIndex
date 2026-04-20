"""多文档联合问答示例（基于 PageIndex workspace）。"""
import argparse
import asyncio
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import PyPDF2
from openai import NotFoundError

from agents import (
    Agent,
    AsyncOpenAI,
    Runner,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.items import ItemHelpers
from pageindex import PageIndexClient

load_dotenv()

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass
class LLMRuntimeConfig:
    provider: str
    model: str
    api_key: str
    base_url: str
    api_mode: str


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def configure_openai_compatible_backend(config: LLMRuntimeConfig) -> None:
    """
    根据运行时配置动态切换底层 API 与客户端。
    """
    if not config.api_key:
        return

    client_kwargs = {"api_key": config.api_key}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url

    set_default_openai_api(config.api_mode)
    set_default_openai_client(
        AsyncOpenAI(**client_kwargs),
        use_for_tracing=False,
    )

DEFAULT_WORKSPACE = Path(__file__).parent / "workspace"
DEFAULT_PDF_DIR = Path(__file__).parent / "pdfs"
DEFAULT_LEGACY_DIR = Path(__file__).parent / "results"

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


def normalize_model_name(model_name: str) -> str:
    """兼容用户常见写法，自动映射到 DashScope 可用模型名。"""
    raw = (model_name or "").strip()
    alias_map = {
        "Qwen3-Max": "qwen3-max",
        "QWEN3-MAX": "qwen3-max",
        "Qwen-Max": "qwen-max",
        "QWEN-MAX": "qwen-max",
    }
    if raw in alias_map:
        return alias_map[raw]

    lowered = raw.lower()
    if lowered in {"qwen3-max", "qwen-max", "qwen-plus", "qwen-turbo"}:
        return lowered
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PageIndex 多文档联合问答")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="PageIndex workspace 目录（默认: ./workspace）",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_PDF_DIR,
        help="原始 PDF 目录（用于从 legacy *_structure.json 迁移，默认: ./pdfs）",
    )
    parser.add_argument(
        "--legacy-dir",
        type=Path,
        default=DEFAULT_LEGACY_DIR,
        help="legacy 结构文件目录（*_structure.json，默认: ./results）",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["auto", "dashscope", "local", "openai"],
        default="auto",
        help="模型提供方配置来源（默认: auto）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="问答模型名（未传时按 provider 自动推断）",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key（优先级最高）",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI 兼容接口 base URL（优先级最高）",
    )
    parser.add_argument(
        "--api-mode",
        type=str,
        choices=["chat_completions", "responses"],
        default=None,
        help="Agents SDK 底层 API（默认按 provider 自动选择）",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出当前 endpoint 可用模型后退出",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="开启流式输出（边生成边打印）",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Agent 最大轮次数（默认: 30）",
    )
    return parser.parse_args()


def resolve_runtime_llm_config(args: argparse.Namespace) -> LLMRuntimeConfig:
    # OpenAI 兼容字段兜底
    openai_base_from_legacy = _first_non_empty(
        os.getenv("OPENAI_BASE_URL"),
        os.getenv("OPENAI_API_BASE"),
    )
    provider = args.provider

    if provider == "dashscope":
        api_key = _first_non_empty(args.api_key, os.getenv("DEMO_API_KEY"), os.getenv("ALIYUN_API_KEY"), os.getenv("OPENAI_API_KEY"))
        base_url = _first_non_empty(args.base_url, os.getenv("DEMO_API_BASE"), os.getenv("ALIYUN_API_BASE"), openai_base_from_legacy, DASHSCOPE_BASE_URL)
        model = _first_non_empty(args.model, os.getenv("DEMO_MODEL"), "qwen3-max")
        api_mode = _first_non_empty(args.api_mode, os.getenv("DEMO_API_MODE"), "chat_completions")
    elif provider == "local":
        api_key = _first_non_empty(args.api_key, os.getenv("LOCAL_API_KEY"), os.getenv("OPENAI_API_KEY"))
        base_url = _first_non_empty(args.base_url, os.getenv("LOCAL_API_BASE"), openai_base_from_legacy)
        model = _first_non_empty(args.model, os.getenv("LOCAL_MODEL"), os.getenv("DEMO_MODEL"), "Qwen3.5-27B")
        api_mode = _first_non_empty(args.api_mode, os.getenv("LOCAL_API_MODE"), "chat_completions")
    elif provider == "openai":
        api_key = _first_non_empty(args.api_key, os.getenv("OPENAI_API_KEY"))
        base_url = _first_non_empty(args.base_url, openai_base_from_legacy)
        model = _first_non_empty(args.model, os.getenv("OPENAI_MODEL"), "gpt-4.1-mini")
        api_mode = _first_non_empty(args.api_mode, os.getenv("OPENAI_API_MODE"), "responses")
    else:
        api_key = _first_non_empty(args.api_key, os.getenv("DEMO_API_KEY"), os.getenv("ALIYUN_API_KEY"), os.getenv("OPENAI_API_KEY"), os.getenv("LOCAL_API_KEY"))
        base_url = _first_non_empty(args.base_url, os.getenv("DEMO_API_BASE"), os.getenv("ALIYUN_API_BASE"), openai_base_from_legacy, os.getenv("LOCAL_API_BASE"), DASHSCOPE_BASE_URL)
        model = _first_non_empty(args.model, os.getenv("DEMO_MODEL"), os.getenv("LOCAL_MODEL"), "qwen3-max")
        if args.api_mode:
            api_mode = args.api_mode
        elif "dashscope.aliyuncs.com" in base_url or "http://" in base_url:
            api_mode = "chat_completions"
        else:
            api_mode = "responses"

    # DashScope 常用模型名规范化，避免大小写/连字符造成 404
    if "dashscope.aliyuncs.com" in base_url or provider == "dashscope":
        model = normalize_model_name(model)

    return LLMRuntimeConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        api_mode=api_mode,
    )


def list_available_models(config: LLMRuntimeConfig) -> int:
    if not config.api_key:
        print("❌ 缺少 API Key，无法列出模型")
        return 1
    try:
        from openai import OpenAI
        model_client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        models = model_client.models.list()
        print("✅ 可用模型列表：")
        for item in models.data:
            print(f"- {item.id}")
        return 0
    except Exception as e:
        print(f"❌ 拉取模型列表失败: {e}")
        return 1


def _normalize_name(name: str) -> str:
    s = name.strip()
    if s.lower().endswith(".pdf"):
        s = s[:-4]
    # Remove common punctuation/spaces to improve filename matching tolerance.
    s = re.sub(r"[ \t\r\n()（）\[\]【】《》“”\"'：:，,。._\-]", "", s)
    return s


def _resolve_pdf_path(structure_file: Path, payload: dict, pdf_candidates: list[Path], pdf_dir: Path) -> Path | None:
    by_name = {p.name: p for p in pdf_candidates}
    by_norm = {}
    for p in pdf_candidates:
        by_norm.setdefault(_normalize_name(p.name), []).append(p)

    raw_name = str(payload.get("doc_name", "")).strip()
    stem_name = structure_file.name[:-len("_structure.json")] if structure_file.name.endswith("_structure.json") else structure_file.stem
    trial_names = [raw_name, stem_name, f"{stem_name}.pdf"]

    for trial in trial_names:
        if not trial:
            continue
        path_trial = Path(trial)
        if path_trial.is_absolute() and path_trial.exists():
            return path_trial

        trial_name = path_trial.name
        if not trial_name.lower().endswith(".pdf"):
            trial_name = f"{trial_name}.pdf"

        direct = pdf_dir / trial_name
        if direct.exists():
            return direct
        if trial_name in by_name:
            return by_name[trial_name]

        norm_key = _normalize_name(trial_name)
        matches = by_norm.get(norm_key, [])
        if matches:
            return sorted(matches)[0]
    return None


def migrate_legacy_structure_jsons(client: PageIndexClient, legacy_dir: Path, pdf_dir: Path) -> tuple[int, list[tuple[Path, str]]]:
    if not legacy_dir.exists():
        return 0, []

    legacy_files = sorted(legacy_dir.glob("*_structure.json"))
    if not legacy_files:
        return 0, []

    pdf_candidates = sorted(pdf_dir.rglob("*.pdf")) if pdf_dir.exists() else []
    existing_doc_names = {
        doc.get("doc_name")
        for doc in client.documents.values()
        if doc.get("type") == "pdf" and doc.get("doc_name")
    }

    imported = 0
    skipped: list[tuple[Path, str]] = []

    for structure_file in legacy_files:
        try:
            with open(structure_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            skipped.append((structure_file, f"JSON 读取失败: {e}"))
            continue

        structure = payload.get("structure")
        if not isinstance(structure, list):
            skipped.append((structure_file, "缺少 structure 数组"))
            continue

        pdf_path = _resolve_pdf_path(structure_file, payload, pdf_candidates, pdf_dir)
        if not pdf_path:
            skipped.append((structure_file, f"未匹配到 PDF（请放入 {pdf_dir}）"))
            continue

        doc_name = pdf_path.name
        if doc_name in existing_doc_names:
            continue

        try:
            pages = []
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages, 1):
                    pages.append({"page": i, "content": page.extract_text() or ""})

            doc_id = str(uuid.uuid4())
            client.documents[doc_id] = {
                "id": doc_id,
                "type": "pdf",
                "path": str(pdf_path.resolve()),
                "doc_name": doc_name,
                "doc_description": payload.get("doc_description", ""),
                "page_count": len(pages),
                "structure": structure,
                "pages": pages,
            }
            client._save_doc(doc_id)
            existing_doc_names.add(doc_name)
            imported += 1
        except Exception as e:
            skipped.append((structure_file, f"导入失败: {e}"))

    return imported, skipped


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


def query_all_docs(
    client: PageIndexClient,
    question: str,
    model: str,
    *,
    stream_output: bool = False,
    max_turns: int = 30,
) -> str:
    tools = build_tools(client)
    agent = Agent(
        name="MultiDocQA",
        instructions=AGENT_SYSTEM_PROMPT,
        tools=tools,
        model=model,
    )

    def _extract_answer_text(result) -> str:
        # 1) 优先使用 SDK 的 final_output
        final_output = result.final_output
        if isinstance(final_output, str) and final_output.strip():
            return final_output.strip()
        if final_output is not None:
            text = str(final_output).strip()
            if text and text.lower() not in {"none", "null"}:
                return text

        # 2) final_output 为空时，从 message 输出项提取文本
        text_from_items = ItemHelpers.text_message_outputs(result.new_items).strip()
        if text_from_items:
            return text_from_items

        # 3) 兜底读取 raw response 文本，避免空白答案
        for raw in reversed(result.raw_responses):
            raw_text = getattr(raw, "output_text", None)
            if isinstance(raw_text, str) and raw_text.strip():
                return raw_text.strip()

        return ""

    async def _run() -> str:
        if not stream_output:
            try:
                result = await Runner.run(agent, question, max_turns=max_turns)
            except NotFoundError as e:
                return (
                    f"模型不可用：{model}。"
                    "DashScope 常用可用名示例：qwen3-max / qwen-max / qwen-plus。"
                    f"原始错误：{e}"
                )

            answer = _extract_answer_text(result)
            if answer:
                return answer
            return "未从模型返回可显示文本（final_output/new_items/raw_responses 均为空）。"

        streamed_chunks: list[str] = []
        try:
            streamed_result = Runner.run_streamed(agent, question, max_turns=max_turns)
            async for event in streamed_result.stream_events():
                if getattr(event, "type", None) != "raw_response_event":
                    continue
                data = getattr(event, "data", None)
                data_type = getattr(data, "type", "")
                delta = getattr(data, "delta", None)
                # 仅流式打印最终答案内容，避免打印 reasoning summary。
                if data_type == "response.output_text.delta" and isinstance(delta, str):
                    print(delta, end="", flush=True)
                    streamed_chunks.append(delta)
        except NotFoundError as e:
            err = (
                f"模型不可用：{model}。"
                "DashScope 常用可用名示例：qwen3-max / qwen-max / qwen-plus。"
                f"原始错误：{e}"
            )
            print(err)
            return err

        if streamed_chunks:
            print()

        answer = _extract_answer_text(streamed_result)
        if answer:
            # 如果流式阶段没有收到可打印 token（少见），这里补打一份完整答案。
            if not streamed_chunks:
                print(answer)
            return answer
        if streamed_chunks:
            return "".join(streamed_chunks).strip()

        fallback = "未从模型返回可显示文本（final_output/new_items/raw_responses 均为空）。"
        print(fallback)
        return fallback

    return asyncio.run(_run())


def main() -> int:
    args = parse_args()
    llm_config = resolve_runtime_llm_config(args)

    # 同步设置环境变量，便于部分 SDK 内部逻辑读取。
    if llm_config.api_key:
        os.environ["OPENAI_API_KEY"] = llm_config.api_key
    if llm_config.base_url:
        os.environ["OPENAI_BASE_URL"] = llm_config.base_url

    if args.list_models:
        print(f"✅ provider: {llm_config.provider}")
        print(f"✅ base_url: {llm_config.base_url}")
        print(f"✅ api_mode: {llm_config.api_mode}")
        return list_available_models(llm_config)

    workspace = args.workspace.expanduser().resolve()
    pdf_dir = args.pdf_dir.expanduser().resolve()
    legacy_dir = args.legacy_dir.expanduser().resolve()

    workspace.mkdir(parents=True, exist_ok=True)

    configure_openai_compatible_backend(llm_config)
    set_tracing_disabled(True)
    client = PageIndexClient(workspace=workspace)
    imported_count, skipped_files = migrate_legacy_structure_jsons(
        client=client,
        legacy_dir=legacy_dir,
        pdf_dir=pdf_dir,
    )
    if imported_count:
        # Reload to ensure we read back the canonical workspace representation.
        client = PageIndexClient(workspace=workspace)
        print(f"✅ 已从 legacy 结构文件迁移 {imported_count} 个文档到 workspace")
    if skipped_files:
        print("⚠️ 以下 legacy 结构文件未迁移：")
        for file_path, reason in skipped_files:
            print(f"- {file_path.name}: {reason}")

    if not client.documents:
        print(f"⚠️ workspace 中没有文档: {workspace}")
        if not legacy_dir.exists():
            print(f"当前 legacy-dir 不存在: {legacy_dir}")
        print("请先执行 build_all.py 构建 *_structure.json，或确认 legacy-dir 与 pdf-dir 配置。")
        return 1

    print(f"✅ workspace: {workspace}")
    print(f"✅ pdf-dir: {pdf_dir}")
    print(f"✅ legacy-dir: {legacy_dir}")
    print(f"✅ 已加载 {len(client.documents)} 个文档")
    print(f"✅ provider: {llm_config.provider}")
    print(f"✅ 使用模型: {llm_config.model}")
    print(f"✅ OPENAI_BASE_URL: {llm_config.base_url}")
    print(f"✅ API mode: {llm_config.api_mode}")
    print(f"✅ Max turns: {args.max_turns}")
    print(f"✅ Stream: {'on' if args.stream else 'off'}")

    if not llm_config.api_key:
        print("❌ 缺少 API Key。可通过 --api-key 或环境变量 DEMO_API_KEY/ALIYUN_API_KEY/OPENAI_API_KEY/LOCAL_API_KEY 配置。")
        return 1

    while True:
        question = input("\n请输入问题（输入 quit 退出）：").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue

        print("\n思考中（跨文档检索）...\n")
        if args.stream:
            print("\n📝 答案（流式）：\n")
            query_all_docs(
                client,
                question,
                llm_config.model,
                stream_output=True,
                max_turns=args.max_turns,
            )
        else:
            answer = query_all_docs(
                client,
                question,
                llm_config.model,
                stream_output=False,
                max_turns=args.max_turns,
            )
            print("\n📝 答案：\n")
            print(answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
