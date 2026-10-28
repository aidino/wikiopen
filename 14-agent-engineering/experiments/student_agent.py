"""Student Coding Agent thuần Python (chỉ dùng thư viện openai chuẩn) gọi Qwen/Qwen3-1.7B qua vLLM.

Kiến trúc (Cách 1 - Vòng lặp ReAct tự chủ với 3 công cụ):
  1. search_knowledge(query): Truy vấn kho tri thức Markdown (knowledge/*.md) do Teacher biên soạn.
  2. test_code(code): Chạy thử code Python với bộ sample_tests để tự kiểm tra lỗi.
  3. submit_solution(code): Nộp code cuối cùng để chấm điểm trên toàn bộ sample + hidden tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
import traceback
from typing import Any

from openai import OpenAI

from tasks import CodingTask, TestCase


EXPERIMENT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = EXPERIMENT_DIR / "knowledge"
LOGS_DIR = EXPERIMENT_DIR / "logs"


@dataclass
class StepLog:
    """Ghi nhận chi tiết 1 bước (turn) trong vòng lặp suy luận của Student Agent."""

    turn: int
    thought: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    observation: str | None = None


@dataclass
class TaskResult:
    """Kết quả giải một bài tập của Student Agent."""

    task_id: str
    title: str
    passed: bool
    passed_count: int
    total_count: int
    turns_used: int
    queried_knowledge: bool
    final_code: str
    failure_summary: str
    steps: list[StepLog] = field(default_factory=list)


def search_markdown_knowledge(query: str, knowledge_dir: Path = KNOWLEDGE_DIR) -> str:
    """Tìm kiếm trong tất cả file .md thuộc thư mục knowledge/ theo từ khóa."""
    if not knowledge_dir.exists():
        return "Kho tri thức hiện đang trống."

    md_files = sorted(knowledge_dir.glob("*.md"))
    if not md_files:
        return "Kho tri thức hiện chưa có file .md nào."

    # Tách từ khóa tìm kiếm (tách cả dấu gạch dưới '_')
    q_lower = query.strip().lower()
    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", q_lower) if len(t) >= 2]
    scored_docs: list[tuple[int, str, str]] = []

    for path in md_files:
        content = path.read_text(encoding="utf-8").strip()
        haystack = f"{path.name}\n{content}".lower()
        score = 10 if (q_lower and q_lower in haystack) else 0
        score += sum(3 if tok in path.name.lower() else (2 if tok in haystack else 0) for tok in tokens)
        if not tokens or score > 0:
            scored_docs.append((score, path.name, content))
    footer = (
        "\n\n[HƯỚNG DẪN BƯỚC TIẾP THEO]: Bạn đã tra cứu xong kho tri thức. "
        "Ở bước tiếp theo, KHÔNG gọi `search_knowledge` nữa; hãy viết code Python và gọi `test_code`."
    )
    if not scored_docs:
        all_docs = [
            f"### [{p.name}]\n{p.read_text(encoding='utf-8').strip()}"
            for p in md_files
        ]
        return (
            "Không có tài liệu chuyên biệt cho từ khóa này. Nội dung tổng quan hiện có:\n\n"
            + "\n\n---\n\n".join(all_docs)
            + footer
        )

    scored_docs.sort(key=lambda item: (-item[0], item[1]))
    top_docs = [f"### [{name}] (score={score})\n{body}" for score, name, body in scored_docs[:2]]
    return "\n\n---\n\n".join(top_docs) + footer


def evaluate_python_code(
    code: str,
    function_name: str,
    test_cases: tuple[TestCase, ...],
) -> tuple[bool, int, int, str]:
    """Thực thi code Python của Student và kiểm tra trên danh sách TestCase.

    Trả về: (all_passed, passed_count, total_count, detailed_report)
    """
    clean_code = _strip_markdown_fences(code)
    namespace: dict[str, Any] = {"__builtins__": __builtins__}

    try:
        exec(compile(clean_code, "<student_solution>", "exec"), namespace)
    except Exception as exc:
        tb = traceback.format_exc(limit=3)
        return False, 0, len(test_cases), f"Lỗi biên dịch/thực thi code: {type(exc).__name__}: {exc}\n{tb}"

    fn = namespace.get(function_name)
    if not callable(fn):
        defined = [k for k in namespace if not k.startswith("__")]
        return (
            False,
            0,
            len(test_cases),
            f"Không tìm thấy hàm `{function_name}`. Các định danh đã khai báo: {defined}",
        )

    passed = 0
    failures: list[str] = []

    for idx, tc in enumerate(test_cases, start=1):
        # Sao chép sâu đối số list/dict để phát hiện hoặc tránh lỗi mutate ngoài ý muốn giữa các test
        safe_args = json_clone_args(tc.args)
        try:
            actual = fn(*safe_args)
        except BaseException as exc:
            if tc.raises is not None and isinstance(exc, tc.raises):
                passed += 1
                continue
            failures.append(
                f"- Test #{idx} ({tc.description}): Gặp ngoại lệ {type(exc).__name__}: {exc} với args={tc.args!r}"
            )
            continue

        if tc.raises is not None:
            failures.append(
                f"- Test #{idx} ({tc.description}): Kỳ vọng raise {tc.raises.__name__} nhưng hàm trả về {actual!r} với args={tc.args!r}"
            )
            continue

        if actual == tc.expected and type(actual) is type(tc.expected):
            passed += 1
        else:
            failures.append(
                f"- Test #{idx} ({tc.description}): Sai kết quả với args={tc.args!r} -> "
                f"Kỳ vọng {tc.expected!r}, nhưng nhận được {actual!r}"
            )

    all_passed = passed == len(test_cases)
    if all_passed:
        return True, passed, len(test_cases), f"PASS: Vượt qua toàn bộ {passed}/{len(test_cases)} test cases."

    report = f"FAIL: Đạt {passed}/{len(test_cases)} test cases.\nChi tiết lỗi:\n" + "\n".join(failures)
    return False, passed, len(test_cases), report


def json_clone_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    """Tạo bản sao độc lập cho các cấu trúc list/dict/tuple lồng nhau."""
    def _clone(obj: Any) -> Any:
        if isinstance(obj, list):
            return [_clone(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _clone(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return tuple(_clone(x) for x in obj)
        return obj

    return tuple(_clone(a) for a in args)


def _strip_markdown_fences(code: str) -> str:
    """Loại bỏ cặp ```python ... ``` nếu model bọc code trong markdown block."""
    text = code.strip()
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Tra cứu kho tri thức Markdown (knowledge/*.md) của Teacher để xem hướng dẫn thuật toán, "
                "công thức Python, và các bẫy (edge-cases) cần tránh trước khi viết code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Từ khóa chủ đề bài toán, ví dụ: 'sum_valid_numbers', 'merge_port_ranges', 'rpn', 'resolve_build_order', 'cron'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "test_code",
            "description": "Chạy thử code Python với bộ sample test cases để kiểm tra lỗi cú pháp hoặc logic trước khi nộp bài.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Toàn bộ mã nguồn Python định nghĩa hàm được yêu cầu (kèm các lệnh import nếu cần).",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_solution",
            "description": "Nộp mã nguồn Python cuối cùng sau khi đã kiểm tra kỹ để chấm điểm chính thức.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Toàn bộ mã nguồn Python hoàn chỉnh của hàm cần nộp.",
                    }
                },
                "required": ["code"],
            },
        },
    },
]


SYSTEM_PROMPT = """You are a Python Student Coding Agent learning from a Teacher's Markdown Knowledge Base.
To solve each Python coding task accurately, follow this strict workflow:
1. FIRST, call `search_knowledge` with keywords from the function name or problem topic to read the Teacher's Markdown guide and edge-case rules.
2. SECOND, write clean Python code implementing the exact required function signature and call `test_code` to verify it against the sample tests.
3. THIRD, if `test_code` reports any failure, fix the bug and re-test. Once `test_code` passes and all Teacher rules are satisfied, call `submit_solution` with your final Python code.

Always call exactly ONE tool per step."""


class StudentCodingAgent:
    """Coding Agent thuần Python điều khiển Qwen/Qwen3-1.7B qua giao thức OpenAI API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_turns: int = 6,
        enable_thinking: bool = False,
    ) -> None:
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        self.model = model or os.environ.get("VLLM_MODEL", "Qwen/Qwen3-1.7B")
        self.max_turns = max_turns
        self.enable_thinking = enable_thinking
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def solve_task(self, task: CodingTask, verbose: bool = True) -> TaskResult:
        """Thực thi vòng lặp ReAct cho một bài tập CodingTask và trả về TaskResult."""
        user_prompt = (
            f"### Bài tập: {task.title}\n"
            f"Chữ ký hàm bắt buộc: `{task.signature}`\n\n"
            f"{task.prompt}\n\n"
            f"Hãy bắt đầu bằng việc gọi `search_knowledge` với từ khóa `{task.function_name}`."
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        steps: list[StepLog] = []
        queried_knowledge = False
        latest_code = ""
        submitted = False

        for turn in range(1, self.max_turns + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=1024,
                extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
            )
            msg = response.choices[0].message
            raw_content = (msg.content or "").strip()
            reasoning = getattr(msg, "reasoning_content", None) or ""
            thought = f"{reasoning}\n{raw_content}".strip() if reasoning else raw_content

            tool_name, tool_args, tool_call_id = self._extract_tool_call(msg, raw_content)

            if tool_name is None:
                # Nếu model trả về trực tiếp một khối code ```python mà quên gọi tool, tự động coi là test_code
                extracted_code = _extract_code_block_if_present(raw_content, task.function_name)
                if extracted_code:
                    tool_name = "test_code"
                    tool_args = {"code": extracted_code}
                    tool_call_id = f"fallback_call_{turn}"
                else:
                    observation = (
                        "Bạn chưa gọi công cụ nào. Hãy gọi `search_knowledge`, `test_code`, hoặc `submit_solution`."
                    )
                    steps.append(StepLog(turn=turn, thought=thought, observation=observation))
                    messages.append({"role": "assistant", "content": raw_content or "Thinking..."})
                    messages.append({"role": "user", "content": observation})
                    continue

            # Thực thi công cụ được chọn
            if tool_name == "search_knowledge":
                queried_knowledge = True
                query = str(tool_args.get("query", task.function_name))
                observation = search_markdown_knowledge(query)
            elif tool_name == "test_code":
                latest_code = str(tool_args.get("code", ""))
                sample_ok, _, _, observation = evaluate_python_code(
                    latest_code, task.function_name, task.sample_tests
                )
                if sample_ok:
                    observation += (
                        "\n[HƯỚNG DẪN BƯỚC TIẾP THEO]: Sample tests đã PASS. "
                        "Hãy gọi `submit_solution` để nộp đoạn code này."
                    )
                else:
                    observation += "\n[HƯỚNG DẪN BƯỚC TIẾP THEO]: Hãy sửa lỗi trong code và gọi lại `test_code`."
            elif tool_name == "submit_solution":
                latest_code = str(tool_args.get("code", "")) or latest_code
                submitted = True
                observation = "Đã nhận bài nộp `submit_solution`."
            else:
                observation = f"Công cụ không hợp lệ: {tool_name}"

            step_log = StepLog(
                turn=turn,
                thought=thought,
                tool_name=tool_name,
                tool_args=tool_args,
                observation=observation,
            )
            steps.append(step_log)

            if verbose:
                print(f"  [Turn {turn}] Tool={tool_name} | Args={_short_str(json.dumps(tool_args, ensure_ascii=False), 90)}")
                print(f"           Obs ={_short_str(observation.replace(chr(10), ' '), 110)}")

            if submitted:
                break

            # Ghi lịch sử tool_call đúng chuẩn OpenAI messages
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": observation,
                }
            )

        # Chấm điểm cuối cùng trên toàn bộ bộ test (sample_tests + hidden_tests)
        all_tests = task.sample_tests + task.hidden_tests
        if not latest_code.strip():
            return TaskResult(
                task_id=task.task_id,
                title=task.title,
                passed=False,
                passed_count=0,
                total_count=len(all_tests),
                turns_used=len(steps),
                queried_knowledge=queried_knowledge,
                final_code="",
                failure_summary="Student không sinh được đoạn code nào.",
                steps=steps,
            )

        all_passed, passed_cnt, total_cnt, full_report = evaluate_python_code(
            latest_code, task.function_name, all_tests
        )
        return TaskResult(
            task_id=task.task_id,
            title=task.title,
            passed=all_passed,
            passed_count=passed_cnt,
            total_count=total_cnt,
            turns_used=len(steps),
            queried_knowledge=queried_knowledge,
            final_code=_strip_markdown_fences(latest_code),
            failure_summary="" if all_passed else full_report,
            steps=steps,
        )

    @staticmethod
    def _extract_tool_call(msg: Any, raw_content: str) -> tuple[str | None, dict[str, Any], str]:
        """Trích xuất lệnh gọi tool từ msg.tool_calls hoặc từ thẻ <tool_call> trong văn bản."""
        if getattr(msg, "tool_calls", None):
            tc = msg.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if isinstance(args, dict) and args:
                return tc.function.name, args, tc.id or "call_0"

        # Dự phòng nếu model in trực tiếp <tool_call>{"name": ..., "arguments": ...}</tool_call>
        match = re.search(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", raw_content, flags=re.DOTALL)
        if match:
            blob = match.group(1).strip()
            try:
                payload = json.loads(blob)
                name = payload.get("name")
                args = payload.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args)
                if isinstance(name, str) and isinstance(args, dict):
                    return name, args, "call_xml"
            except Exception:
                # Khôi phục khi model viết xuống dòng trực tiếp bên trong chuỗi JSON "code"
                name_m = re.search(r'"name"\s*:\s*"(search_knowledge|test_code|submit_solution)"', blob)
                if name_m:
                    t_name = name_m.group(1)
                    if t_name == "search_knowledge":
                        q_m = re.search(r'"query"\s*:\s*"([^"]+)"', blob)
                        return t_name, {"query": q_m.group(1) if q_m else ""}, "call_xml_recovered"
                    code_m = re.search(r'"code"\s*:\s*"(.*)', blob, flags=re.DOTALL)
                    if code_m:
                        raw_code = code_m.group(1)
                        raw_code = re.sub(r'"\s*\}\s*\}\s*(?:</tool_call>)?\s*$', "", raw_code, flags=re.DOTALL)
                        raw_code = raw_code.replace("\\n", "\n").replace('\\"', '"').replace("\\t", "    ")
                        return t_name, {"code": raw_code.strip()}, "call_xml_recovered"
        return None, {}, ""


def _extract_code_block_if_present(text: str, fn_name: str) -> str:
    """Tìm khối code có chứa `def <fn_name>` nếu model quên gọi tool mà in thẳng code."""
    if "<tool_call>" in text:
        return ""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if match and f"def {fn_name}" in match.group(1):
        return match.group(1).strip()
    idx = text.find(f"def {fn_name}")
    if idx != -1:
        return text[idx:].strip()
    return ""

def _short_str(text: str, max_len: int = 100) -> str:
    """Rút gọn chuỗi trên 1 dòng để in log gọn gàng."""
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def save_run_logs(run_tag: str, results: list[TaskResult]) -> Path:
    """Lưu kết quả chạy thực nghiệm ra file JSON và Markdown trong thư mục logs/."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = LOGS_DIR / f"{run_tag}.json"
    md_path = LOGS_DIR / f"{run_tag}.md"

    payload = {
        "run_tag": run_tag,
        "total_tasks": len(results),
        "tasks_passed": sum(1 for r in results if r.passed),
        "total_test_cases_passed": sum(r.passed_count for r in results),
        "total_test_cases": sum(r.total_count for r in results),
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# Báo cáo Thực nghiệm Student Agent (`{run_tag}`)",
        "",
        f"- **Số bài hoàn thành (100% test):** {payload['tasks_passed']}/{payload['total_tasks']}",
        f"- **Tổng số test cases vượt qua:** {payload['total_test_cases_passed']}/{payload['total_test_cases']}",
        "",
        "| Bài tập | Kết quả | Test Cases | Số lượt (Turns) | Đã tra cứu Markdown |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(
            f"| `{r.task_id}` | **{status}** | {r.passed_count}/{r.total_count} | {r.turns_used} | {r.queried_knowledge} |"
        )

    for r in results:
        lines.extend(
            [
                "",
                f"## {r.title} (`{r.task_id}`)",
                f"- **Trạng thái:** {'PASS' if r.passed else 'FAIL'} ({r.passed_count}/{r.total_count} tests)",
                "### Code cuối cùng của Student:",
                "```python",
                r.final_code or "# (empty)",
                "```",
            ]
        )
        if r.failure_summary:
            lines.extend(["### Chi tiết lỗi (dành cho Teacher phân tích):", "```text", r.failure_summary, "```"])
        lines.append("### Nhật ký các bước (Turns):")
        for s in r.steps:
            lines.append(
                f"- **Turn {s.turn}:** Tool=`{s.tool_name}` | Args=`{json.dumps(s.tool_args, ensure_ascii=False)}`"
            )
            if s.observation:
                obs_preview = _short_str(s.observation.replace("\n", " | "), 220)
                lines.append(f"  - *Observation:* `{obs_preview}`")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path
