"""Các công cụ và hàm gọi LLM dùng chung cho Phase 14 (Agent Engineering)."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient
# Tự động nạp các biến môi trường từ file .env (DEEPSEEK_API_KEY, TAVILY_API_KEY, ...)
load_dotenv()


@dataclass
class ToolCall:
    """Cấu trúc dữ liệu biểu diễn một lệnh gọi công cụ (tên công cụ và tham số)."""

    name: str
    args: dict[str, Any]


class ToolRegistry:
    """Bộ đăng ký và điều phối thực thi các công cụ (tools) cho Agent."""

    def __init__(self) -> None:
        """Khởi tạo danh sách công cụ và schema rỗng."""
        # Lưu ánh xạ từ tên công cụ sang hàm thực thi
        self._tools: dict[str, Callable[..., str]] = {}
        # Lưu danh sách JSON schema phục vụ tính năng function calling của OpenAI/DeepSeek
        self._schemas: list[dict[str, Any]] = []

    def register(
        self,
        name: str,
        fn: Callable[..., str],
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Đăng ký một hàm công cụ mới cùng với mô tả và JSON schema của tham số."""
        self._tools[name] = fn
        self._schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description or f"Run tool {name}",
                    "parameters": parameters or {"type": "object", "properties": {}},
                },
            }
        )

    def names(self) -> list[str]:
        """Trả về danh sách tên các công cụ đã đăng ký (sắp xếp theo bảng chữ cái)."""
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """Trả về danh sách schema của các công cụ theo chuẩn OpenAI tool calling."""
        return self._schemas

    def describe_catalog(self) -> str:
        """Tạo chuỗi mô tả danh mục công cụ để đưa vào prompt cho ReWOO Planner."""
        lines = []
        for s in self._schemas:
            fn = s["function"]
            props = fn["parameters"].get("properties", {})
            args_desc = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
            lines.append(f"- {fn['name']}({args_desc}): {fn['description']}")
        return "\n".join(lines)

    def dispatch(self, name_or_call: str | ToolCall, args: dict[str, Any] | None = None) -> str:
        """Thực thi công cụ theo tên hoặc đối tượng ToolCall và bắt mọi ngoại lệ an toàn."""
        # Hỗ trợ truyền vào cả đối tượng ToolCall (bài 01) lẫn (name, args) rời (bài 02)
        if isinstance(name_or_call, ToolCall):
            name, call_args = name_or_call.name, name_or_call.args
        else:
            name, call_args = name_or_call, args or {}

        fn = self._tools.get(name)
        if fn is None:
            return f"error: unknown tool {name!r}"
        try:
            return str(fn(**call_args))
        except TypeError as e:
            return f"error: bad args for {name}: {e}"
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"


def get_llm_client() -> tuple[OpenAI, str]:
    """Khởi tạo và trả về tuple (OpenAI client kết nối DeepSeek, tên model)."""
    # Lấy cấu hình từ biến môi trường theo mẫu .env.example
    api_key = os.environ["DEEPSEEK_API_KEY"]
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
    return OpenAI(api_key=api_key, base_url=base_url), model

def chat_completion(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
) -> Any:
    """Hàm dùng chung để gửi yêu cầu chat completion tới DeepSeek LLM và trả về message."""
    client, model = get_llm_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    # Chỉ đính kèm danh sách tools nếu có truyền vào
    if tools:
        kwargs["tools"] = tools
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message


def tavily_search(query: str) -> str:
    """Tìm kiếm thông tin thực tế trên web bằng thư viện tavily-python."""
    # Khởi tạo TavilyClient sử dụng biến môi trường TAVILY_API_KEY
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    data = client.search(query=query, include_answer=True, max_results=3)

    # Ưu tiên trả về câu trả lời tổng hợp của Tavily kèm trích đoạn kết quả
    answer = data.get("answer")
    snippets = [r.get("content", "") for r in data.get("results", []) if r.get("content")]
    if answer:
        return f"{answer} | Nguồn: {' // '.join(snippets[:2])}"
    if snippets:
        return " // ".join(snippets[:2])
    return f"Không tìm thấy kết quả cho: {query}"
