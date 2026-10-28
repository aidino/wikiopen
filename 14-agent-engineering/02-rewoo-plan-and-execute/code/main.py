"""ReWOO thực tế — Planner, Workers, Solver sử dụng DeepSeek Flash & Tavily Search.

Minh họa kiến trúc tách biệt lập kế hoạch và thực thi từ Xu et al. (arXiv:2305.18323):
  1. Planner (DeepSeek Flash) sinh đồ thị kế hoạch DAG (tool, args) có tham chiếu (#E1, #E2, ...).
  2. Workers chạy từng bước theo thứ tự phụ thuộc tô-pô với Tavily Search thật.
  3. Solver (DeepSeek Flash) tổng hợp câu trả lời cuối cùng từ câu hỏi + kế hoạch + bằng chứng.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sys
from typing import Any

# Thêm thư mục 14-agent-engineering vào sys.path để import gói utils dùng chung
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils import ToolRegistry, chat_completion, tavily_search

# Biểu thức chính quy tìm các mã tham chiếu kết quả bước trước dạng #E1, #E2, ...
REFERENCE_RE = re.compile(r"#E(\d+)")


@dataclass
class PlanStep:
    """Biểu diễn một bước (node) trong đồ thị kế hoạch ReWOO."""

    id: str
    tool: str
    args: dict[str, Any]
    reason: str = ""


@dataclass
class Plan:
    """Biểu diễn toàn bộ kế hoạch gồm danh sách các bước PlanStep."""

    steps: list[PlanStep]


def resolve_references(value: Any, evidence: dict[str, str]) -> Any:
    """Thay thế các biến tham chiếu #E1, #E2 trong tham số bằng kết quả thực tế đã thu thập."""
    # Nếu giá trị không phải kiểu chuỗi thì giữ nguyên không thay thế
    if not isinstance(value, str):
        return value
    # Thay từng mã #E<số> bằng bằng chứng tương ứng trong dict evidence
    return REFERENCE_RE.sub(lambda m: evidence.get(f"E{m.group(1)}", m.group(0)), value)


def topological(plan: Plan) -> list[PlanStep]:
    """Sắp xếp các bước trong kế hoạch theo thứ tự tô-pô để bước phụ thuộc luôn chạy sau."""
    resolved: list[PlanStep] = []
    known: set[str] = set()
    pending = list(plan.steps)

    # Lặp cho tới khi toàn bộ các bước đều được xếp lịch
    while pending:
        progress = False
        rest: list[PlanStep] = []
        for step in pending:
            # Tìm tất cả các mã tham chiếu #E<số> xuất hiện trong tham số của bước hiện tại
            refs = REFERENCE_RE.findall(str(step.args))
            # Nếu mọi bước tiền đề đều đã nằm trong tập known thì bước này đủ điều kiện chạy
            if all(f"E{r}" in known for r in refs):
                resolved.append(step)
                known.add(step.id)
                progress = True
            else:
                rest.append(step)
        # Nếu qua một vòng duyệt mà không thêm được bước nào -> đồ thị bị lặp hoặc thiếu tham chiếu
        if not progress:
            raise RuntimeError("Kế hoạch chứa chu trình hoặc tham chiếu tới bước không tồn tại")
        pending = rest
    return resolved


def run_workers(plan: Plan, tools: ToolRegistry) -> dict[str, str]:
    """Thực thi lần lượt các bước Worker theo thứ tự tô-pô và thu thập bằng chứng (evidence)."""
    evidence: dict[str, str] = {}
    for step in topological(plan):
        # Điền kết quả thực tế của các bước trước vào tham số của bước hiện tại
        bound_args = {k: resolve_references(v, evidence) for k, v in step.args.items()}
        # Gọi công cụ thông qua ToolRegistry và lưu kết quả vào từ điển bằng chứng
        evidence[step.id] = tools.dispatch(step.tool, bound_args)
    return evidence


class DeepSeekPlanner:
    """Bộ lập kế hoạch (Planner) gọi model DeepSeek Flash để sinh toàn bộ DAG kế hoạch ngay từ đầu."""

    def __init__(self, tools: ToolRegistry) -> None:
        """Khởi tạo Planner với danh mục các công cụ hiện có."""
        self.tools = tools

    def _build_messages(self, question: str) -> list[dict[str, str]]:
        """Tạo danh sách messages (system + user) hướng dẫn LLM xuất kế hoạch JSON."""
        system_prompt = (
            "Bạn là ReWOO Planner. Hãy lập kế hoạch giải quyết câu hỏi bằng các công cụ sau:\n"
            f"{self.tools.describe_catalog()}\n\n"
            "Quy tắc:\n"
            "1. Trả về DUY NHẤT một chuỗi JSON hợp lệ theo cấu trúc:\n"
            '   {"steps": [{"id": "E1", "reason": "...", "tool": "search", "args": {"query": "..."}}]}\n'
            "2. Đặt id lần lượt là E1, E2, E3...\n"
            "3. Nếu tham số bước sau cần kết quả của bước trước, hãy dùng mã #E1, #E2... trực tiếp trong chuỗi.\n"
            "4. Với công cụ search, câu truy vấn nên ngắn gọn và rõ ràng."
        )
        user_prompt = question
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def plan_for(self, question: str) -> Plan:
        """Nhận câu hỏi của người dùng, gọi DeepSeek Flash và giải mã JSON thành đối tượng Plan."""
        messages = self._build_messages(question)
        msg = chat_completion(messages=messages, temperature=0.0)
        raw = (msg.content or "").strip()

        # Loại bỏ cặp thẻ markdown ```json ... ``` nếu model sinh kèm
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL)
        data = json.loads(cleaned)
        steps = [
            PlanStep(
                id=s["id"],
                tool=s["tool"],
                args=s.get("args", {}),
                reason=s.get("reason", ""),
            )
            for s in data.get("steps", [])
        ]
        return Plan(steps=steps)


class DeepSeekSolver:
    """Bộ tổng hợp kết quả (Solver) gọi DeepSeek Flash để trả lời câu hỏi dựa trên bằng chứng."""

    def __init__(self, tools: ToolRegistry) -> None:
        """Khởi tạo Solver với danh mục các công cụ hiện có."""
        self.tools = tools

    def _build_messages(
        self, question: str, plan: Plan, evidence: dict[str, str]
    ) -> list[dict[str, str]]:
        """Tạo danh sách messages (system + user) ghép câu hỏi, kế hoạch và bằng chứng cho Solver."""
        system_prompt = (
            "Bạn là ReWOO Solver. Dựa vào bằng chứng thu thập được từ các bước thực thi, "
            "hãy trả lời câu hỏi của người dùng một cách chính xác và ngắn gọn."
        )
        evidence_lines = [
            f"Bước {s.id} ({s.reason}): gọi {s.tool}({s.args}) -> Kết quả: {evidence.get(s.id, '')}"
            for s in plan.steps
        ]
        user_prompt = f"Câu hỏi: {question}\n\nBằng chứng thu thập được:\n" + "\n".join(evidence_lines)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def solve(self, question: str, plan: Plan, evidence: dict[str, str]) -> str:
        """Gọi DeepSeek Flash để tổng hợp câu trả lời cuối cùng từ câu hỏi, kế hoạch và bằng chứng."""
        messages = self._build_messages(question, plan, evidence)
        msg = chat_completion(messages=messages, temperature=0.0)
        return (msg.content or "").strip()
def rounded_million(text: str) -> str:
    """Trích xuất con số theo đơn vị triệu (million) từ văn bản và làm tròn tới hàng triệu gần nhất."""
    # Ưu tiên tìm số đứng ngay trước chữ 'million' hoặc 'triệu'
    m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*(?:million|triệu)", text, re.IGNORECASE)
    if m:
        val = float(m.group(1).replace(",", "."))
        return f"{round(val)} million"
    # Nếu văn bản ghi số nguyên đầy đủ ở hàng triệu (ví dụ: 15,064,617)
    m_raw = re.search(r"\b([0-9]{1,3}(?:,[0-9]{3}){2,})\b", text)
    if m_raw:
        val = float(m_raw.group(1).replace(",", "")) / 1_000_000
        return f"{round(val)} million"
    return f"không rõ số liệu từ: {text[:80]}"

@dataclass
class ReWOORun:
    """Lưu trữ toàn bộ kết quả và thống kê số ký tự của một lần chạy ReWOO."""

    question: str
    plan: Plan
    evidence: dict[str, str] = field(default_factory=dict)
    answer: str = ""
    planner_chars: int = 0
    worker_chars: int = 0
    solver_chars: int = 0


def run_rewoo(
    question: str,
    planner: DeepSeekPlanner,
    tools: ToolRegistry,
    solver: DeepSeekSolver,
) -> ReWOORun:
    """Chạy trọn vẹn luồng ReWOO 3 giai đoạn: Lập kế hoạch (Plan) -> Thực thi (Work) -> Tổng hợp (Solve)."""
    # Giai đoạn 1: Gọi LLM Planner 1 lần để lấy toàn bộ đồ thị kế hoạch
    plan = planner.plan_for(question)
    planner_chars = len(question) + sum(len(s.tool) + len(str(s.args)) for s in plan.steps)

    # Giai đoạn 2: Chạy các Worker thực thi công cụ theo thứ tự phụ thuộc mà không cần gọi lại LLM
    evidence = run_workers(plan, tools)
    worker_chars = sum(len(str(s.args)) + len(v) for s, v in zip(plan.steps, evidence.values()))

    # Giai đoạn 3: Gọi LLM Solver 1 lần cuối để tổng hợp câu trả lời từ bằng chứng
    answer = solver.solve(question, plan, evidence)
    solver_chars = len(question) + worker_chars + len(answer)

    return ReWOORun(
        question=question,
        plan=plan,
        evidence=evidence,
        answer=answer,
        planner_chars=planner_chars,
        worker_chars=worker_chars,
        solver_chars=solver_chars,
    )


def estimate_react_chars(question: str, plan: Plan, evidence: dict[str, str]) -> int:
    """Ước tính lượng ký tự prompt cộng dồn nếu chạy cùng các bước đó theo vòng lặp ReAct truyền thống."""
    # Trong ReAct, mỗi bước gọi LLM phải gửi kèm lại toàn bộ lịch sử (thought + action + observation) trước đó
    prompt_chars = len(question)
    total = 0
    history_chars = 0
    for step in plan.steps:
        obs = evidence.get(step.id, "")
        total += prompt_chars + history_chars + len(step.tool) + len(str(step.args))
        history_chars += len(step.tool) + len(str(step.args)) + len(obs) + 40
    total += prompt_chars + history_chars
    return total


def build_tools() -> ToolRegistry:
    """Khởi tạo bộ công cụ thực tế gồm Tavily Search và hàm làm tròn hàng triệu."""
    tools = ToolRegistry()
    tools.register(
        "search",
        tavily_search,
        description="Tìm kiếm thông tin thực tế trên internet qua Tavily Search API.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    tools.register(
        "round_million",
        rounded_million,
        description="Trích xuất và làm tròn số liệu dân số/số lượng về đơn vị hàng triệu (million).",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    return tools


def main() -> None:
    """Hàm chính chạy kịch bản ReWOO thực tế với DeepSeek Flash và Tavily Search."""
    print("=" * 70)
    print("REWOO — Planner, Workers, Solver (DeepSeek Flash + Tavily Search)")
    print("=" * 70)

    tools = build_tools()
    planner = DeepSeekPlanner(tools)
    solver = DeepSeekSolver(tools)

    question = "What is the metro population of the capital of France in millions, rounded?"
    run = run_rewoo(question, planner, tools, solver)

    print("\nPLAN")
    for step in run.plan.steps:
        print(f"  {step.id}: {step.tool}({step.args})  # {step.reason}")
    print("\nEVIDENCE")
    for k, v in run.evidence.items():
        print(f"  {k} -> {v}")
    print(f"\nFINAL: {run.answer}")

    react_chars = estimate_react_chars(run.question, run.plan, run.evidence)
    rewoo_chars = run.planner_chars + run.worker_chars + run.solver_chars
    print("\nTOKEN INTUITION (chars, approximate)")
    print(f"  react total  : {react_chars}")
    print(f"  rewoo total  : {rewoo_chars}")
    print(f"  ratio        : {react_chars / max(rewoo_chars, 1):.2f}x")


if __name__ == "__main__":
    main()
