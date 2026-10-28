"""Bộ bài tập Python từ Dễ đến Khó dành cho thí nghiệm Teacher - Student Agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class TestCase:
    """Một test case gồm args truyền vào hàm và kết quả kỳ vọng (hoặc ngoại lệ)."""

    args: tuple[Any, ...]
    expected: Any = None
    raises: type[BaseException] | None = None
    description: str = ""


@dataclass(frozen=True)
class CodingTask:
    """Định nghĩa một bài tập lập trình Python cho Student Agent."""

    task_id: str
    level: int
    title: str
    function_name: str
    signature: str
    prompt: str
    sample_tests: tuple[TestCase, ...]
    hidden_tests: tuple[TestCase, ...]


TASKS: tuple[CodingTask, ...] = (
    CodingTask(
        task_id="01_sum_valid_integers",
        level=1,
        title="Level 1 (Easy): Tổng các chuỗi số nguyên hợp lệ",
        function_name="sum_valid_numbers",
        signature="def sum_valid_numbers(items: list[str]) -> int:",
        prompt=(
            "Viết hàm `sum_valid_numbers(items: list[str]) -> int` nhận vào một danh sách các chuỗi.\n"
            "Tính tổng tất cả các chuỗi biểu diễn số nguyên hợp lệ (chấp nhận dấu `+` hoặc `-` đứng liền trước chữ số, "
            "và chấp nhận khoảng trắng ở hai đầu chuỗi như `' -12 '`).\n"
            "Bỏ qua các chuỗi rỗng, chuỗi chứa số thập phân (`'3.14'`), chữ cái (`'12a'`), hoặc ký tự không hợp lệ (`'--5'`, `'+-3'`).\n"
            "Ví dụ:\n"
            "  sum_valid_numbers(['10', ' -3 ', '+5', '3.14', 'abc', '']) == 12"
        ),
        sample_tests=(
            TestCase(args=(["10", " -3 ", "+5", "3.14", "abc", ""],), expected=12, description="cơ bản"),
            TestCase(args=([],), expected=0, description="danh sách rỗng"),
        ),
        hidden_tests=(
            TestCase(args=(["0", "-0", "+0", "--1", "++2", "1_000"],), expected=0, description="chặn dấu kép và dấu gạch dưới"),
            TestCase(args=(["  +42\n", "-100", "99.0", "1e5", "7"],), expected=-51, description="khoảng trắng và ký hiệu khoa học"),
        ),
    ),
    CodingTask(
        task_id="02_merge_adjacent_ranges",
        level=2,
        title="Level 2 (Medium): Gộp các khoảng số nguyên chồng lấp hoặc liền kề",
        function_name="merge_port_ranges",
        signature="def merge_port_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:",
        prompt=(
            "Viết hàm `merge_port_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]`.\n"
            "Mỗi phần tử là một tuple `(start, end)` với `start <= end`.\n"
            "Hãy gộp tất cả các khoảng bị chồng lấp (overlapping) HOẶC liền kề nhau (adjacent, tức là `next_start == current_end + 1`), "
            "và trả về danh sách các tuple `(start, end)` đã sắp xếp tăng dần.\n"
            "Lưu ý: Không làm thay đổi (mutate) danh sách đầu vào `ranges`.\n"
            "Ví dụ:\n"
            "  merge_port_ranges([(10, 12), (13, 15), (20, 25), (22, 30)]) == [(10, 15), (20, 30)]"
        ),
        sample_tests=(
            TestCase(
                args=([(10, 12), (13, 15), (20, 25), (22, 30)],),
                expected=[(10, 15), (20, 30)],
                description="liền kề và chồng lấp",
            ),
            TestCase(args=([],), expected=[], description="danh sách rỗng"),
        ),
        hidden_tests=(
            TestCase(
                args=([(5, 5), (6, 6), (1, 3), (4, 4), (10, 12)],),
                expected=[(1, 6), (10, 12)],
                description="chưa sắp xếp và khoảng đơn điểm liền kề",
            ),
            TestCase(
                args=([(1, 10), (2, 5), (3, 4)],),
                expected=[(1, 10)],
                description="khoảng con nằm trọn bên trong",
            ),
        ),
    ),
    CodingTask(
        task_id="03_evaluate_rpn_safe",
        level=3,
        title="Level 3 (Medium): Tính biểu thức hậu tố (RPN) an toàn",
        function_name="evaluate_rpn_safe",
        signature="def evaluate_rpn_safe(tokens: list[str]) -> int | str:",
        prompt=(
            "Viết hàm `evaluate_rpn_safe(tokens: list[str]) -> int | str` để tính giá trị biểu thức Ký pháp Ba Lan Ngược (RPN).\n"
            "Các toán tử hợp lệ là `'+'`, `'-'`, `'*'`, `'/'`.\n"
            "Quy tắc bắt buộc:\n"
            "1. Phép chia `'/'` phải là chia nguyên làm tròn về 0 (truncate toward zero), ví dụ `-7 / 2` bằng `-3` (KHÔNG phải `-4`).\n"
            "2. Nếu xảy ra chia cho 0, trả về chuỗi `'ERROR:DIV_BY_ZERO'`.\n"
            "3. Nếu biểu thức không hợp lệ (rỗng, thiếu toán hạng khi gặp toán tử, thừa toán hạng sau khi tính xong, hoặc token lạ), "
            "trả về chuỗi `'ERROR:INVALID'`.\n"
            "Ví dụ:\n"
            "  evaluate_rpn_safe(['4', '-7', '2', '/', '+']) == 1"
        ),
        sample_tests=(
            TestCase(args=(["4", "-7", "2", "/", "+"],), expected=1, description="chia âm làm tròn về 0"),
            TestCase(args=(["10", "0", "/"],), expected="ERROR:DIV_BY_ZERO", description="chia cho 0"),
        ),
        hidden_tests=(
            TestCase(args=(["-13", "5", "/"],), expected=-2, description="-13 / 5 làm tròn về 0 là -2"),
            TestCase(args=(["2", "+"],), expected="ERROR:INVALID", description="thiếu toán hạng"),
            TestCase(args=(["2", "3", "4", "+"],), expected="ERROR:INVALID", description="thừa toán hạng trên stack"),
            TestCase(args=([],), expected="ERROR:INVALID", description="danh sách token rỗng"),
            TestCase(args=(["2", "foo", "+"],), expected="ERROR:INVALID", description="token không hợp lệ"),
        ),
    ),
    CodingTask(
        task_id="04_resolve_build_order",
        level=4,
        title="Level 4 (Hard): Sắp xếp thứ tự build gói phụ thuộc (Topological Sort từ điển)",
        function_name="resolve_build_order",
        signature="def resolve_build_order(deps: dict[str, list[str]]) -> list[str] | str:",
        prompt=(
            "Viết hàm `resolve_build_order(deps: dict[str, list[str]]) -> list[str] | str`.\n"
            "Tham số `deps` ánh xạ tên mỗi `package` sang danh sách các gói mà nó phụ thuộc vào "
            "(tức là các gói trong `deps[pkg]` PHẢI được build TRƯỚC `pkg`).\n"
            "Quy tắc bắt buộc:\n"
            "1. Một gói có thể chỉ xuất hiện trong danh sách phụ thuộc mà không nằm ở key của `deps` (coi như không phụ thuộc gói nào khác).\n"
            "2. Tại mỗi bước, nếu có nhiều gói đã sẵn sàng để build, BẮT BUỘC chọn gói có tên nhỏ nhất theo thứ tự từ điển (alphabetical order).\n"
            "3. Nếu phát hiện phụ thuộc vòng (cycle, kể cả tự phụ thuộc chính nó), trả về chuỗi `'ERROR:CYCLE'`.\n"
            "Ví dụ:\n"
            "  resolve_build_order({'app': ['ui', 'db'], 'ui': ['core'], 'db': ['core']}) == ['core', 'db', 'ui', 'app']"
        ),
        sample_tests=(
            TestCase(
                args=({"app": ["ui", "db"], "ui": ["core"], "db": ["core"]},),
                expected=["core", "db", "ui", "app"],
                description="thứ tự phụ thuộc và ưu tiên từ điển db trước ui",
            ),
            TestCase(
                args=({"a": ["b"], "b": ["a"]},),
                expected="ERROR:CYCLE",
                description="phát hiện chu trình 2 đỉnh",
            ),
        ),
        hidden_tests=(
            TestCase(
                args=({"z": ["b"], "a": ["c"], "b": []},),
                expected=["b", "c", "a", "z"],
                description="ưu tiên từ điển toàn cục (heapq) giữa các đỉnh bậc vào 0",
            ),
            TestCase(
                args=({"self_pkg": ["self_pkg"]},),
                expected="ERROR:CYCLE",
                description="tự phụ thuộc chính mình",
            ),
            TestCase(
                args=({"web": ["auth", "auth", "cache"], "auth": ["crypto"]},),
                expected=["cache", "crypto", "auth", "web"],
                description="cạnh trùng lặp và node ngầm định",
            ),
            TestCase(args=({},), expected=[], description="đồ thị rỗng"),
        ),
    ),
    CodingTask(
        task_id="05_parse_cron_field",
        level=5,
        title="Level 5 (Hard): Bộ phân tích biểu thức trường Cron",
        function_name="parse_cron_field",
        signature="def parse_cron_field(expr: str, min_val: int, max_val: int) -> list[int]:",
        prompt=(
            "Viết hàm `parse_cron_field(expr: str, min_val: int, max_val: int) -> list[int]`.\n"
            "Hàm phân tích một trường trong biểu thức cron (như phút hoặc giờ) trong phạm vi `[min_val, max_val]` (bao gồm cả 2 đầu mút) "
            "và trả về danh sách các số nguyên ĐÃ SẮP XẾP TĂNG DẦN và KHÔNG TRÙNG LẶP.\n"
            "Cú pháp hỗ trợ (các phần cách nhau bởi dấu phẩy `,`):\n"
            "- `'*'`: tất cả giá trị từ `min_val` đến `max_val`.\n"
            "- `'*/step'`: từ `min_val` đến `max_val` với bước nhảy `step`.\n"
            "- `'a-b'`: tất cả giá trị từ `a` đến `b` (bao gồm cả `b`).\n"
            "- `'a-b/step'`: từ `a` đến `b` với bước nhảy `step`.\n"
            "- `'n'`: một số nguyên đơn lẻ `n` (không áp dụng `/step` cho số đơn lẻ như `'5/2'` -> phải báo lỗi).\n"
            "Nếu biểu thức rỗng, sai cú pháp, `step <= 0`, `a > b`, hoặc có giá trị nằm ngoài `[min_val, max_val]`, "
            "hãy `raise ValueError`.\n"
            "Ví dụ:\n"
            "  parse_cron_field('*/15,3-7/2,5', 0, 30) == [0, 3, 5, 7, 15, 30]"
        ),
        sample_tests=(
            TestCase(args=("*/15,3-7/2,5", 0, 30), expected=[0, 3, 5, 7, 15, 30], description="kết hợp *, range, step và số lẻ"),
            TestCase(args=("1-5", 1, 10), expected=[1, 2, 3, 4, 5], description="khoảng đơn giản"),
        ),
        hidden_tests=(
            TestCase(args=("*", 1, 4), expected=[1, 2, 3, 4], description="dấu sao"),
            TestCase(args=("10-5", 0, 20), raises=ValueError, description="start > end phải raise ValueError"),
            TestCase(args=("*/0", 0, 10), raises=ValueError, description="step == 0 phải raise ValueError"),
            TestCase(args=("0-60", 0, 59), raises=ValueError, description="vượt quá max_val phải raise ValueError"),
            TestCase(args=("5/2", 0, 10), raises=ValueError, description="số đơn lẻ kèm step là sai cú pháp"),
            TestCase(args=("", 0, 10), raises=ValueError, description="chuỗi rỗng"),
        ),
    ),
)
