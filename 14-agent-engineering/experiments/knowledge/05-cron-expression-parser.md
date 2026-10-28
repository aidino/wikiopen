# Bài học 05: Phân Tích Biểu Thức Trường Cron (`parse_cron_field`)

## Phân tích Lỗi của Student từ Log Baseline
Bài toán yêu cầu phân tích chuỗi `expr` gồm các thành phần phân tách bởi dấu phẩy `,` trong miền `[min_val, max_val]`. Mọi lỗi cú pháp hoặc vượt biên đều phải `raise ValueError`.

## Các Quy tắc Kiểm tra Hợp lệ (Validation Rules)
1. Nếu `not expr` hoặc `not expr.strip()` hoặc `min_val > max_val`: `raise ValueError("Invalid expr")`.
2. Tách `parts = expr.split(",")`. Nếu có phần tử rỗng trong `parts` (ví dụ `"1,,2"`): `raise ValueError`.
3. Với mỗi `part` trong `parts`:
   - **Bước 3a (Tách `/step` nếu có):**
     - Nếu `"/"` trong `part`:
       - Nếu `part.count("/") > 1`: `raise ValueError`.
       - `base, step_str = part.split("/")`
       - Nếu `not step_str.isdigit()`: `raise ValueError` (loại cả số âm và chữ).
       - `step = int(step_str)`
       - Nếu `step <= 0`: `raise ValueError`.
       - **Quan trọng:** Bước nhảy `/step` CHỈ áp dụng cho `base == "*"` hoặc khoảng `a-b` (`"-" in base`). Nếu `base` là một số đơn lẻ (như `"5/2"`), BẮT BUỘC `raise ValueError`!
     - Nếu không có `"/"`:
       - `base = part`
       - `step = 1`
   - **Bước 3b (Xác định khoảng `[start, end]` từ `base`):**
     - Nếu `base == "*"`:
       - `start, end = min_val, max_val`
     - Nếu `"-" in base`:
       - Nếu `base.count("-") > 1`: `raise ValueError`.
       - `a_str, b_str = base.split("-")`
       - Nếu `not (a_str.isdigit() and b_str.isdigit())`: `raise ValueError`.
       - `start, end = int(a_str), int(b_str)`
       - Nếu `start > end` hoặc `start < min_val` hoặc `end > max_val`: `raise ValueError`.
     - Ngược lại (số đơn lẻ `n`):
       - Nếu `not base.isdigit()`: `raise ValueError`.
       - `val = int(base)`
       - Nếu `val < min_val` hoặc `val > max_val`: `raise ValueError`.
       - `start = end = val`
   - **Bước 3c:** Thêm tất cả giá trị `range(start, end + 1, step)` vào một tập hợp `values: set[int]`.
4. Trả về `sorted(values)`.

## Mẫu cài đặt tham khảo
```python
def parse_cron_field(expr: str, min_val: int, max_val: int) -> list[int]:
    if not expr or not expr.strip() or min_val > max_val:
        raise ValueError("Empty or invalid bounds")

    values: set[int] = set()
    for raw_part in expr.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("Empty cron part")

        if "/" in part:
            if part.count("/") > 1:
                raise ValueError("Multiple slashes")
            base, step_str = part.split("/")
            if not step_str.isdigit():
                raise ValueError("Invalid step")
            step = int(step_str)
            if step <= 0:
                raise ValueError("Step must be > 0")
            if base != "*" and "-" not in base:
                raise ValueError("Step not allowed on single value")
        else:
            base = part
            step = 1

        if base == "*":
            start, end = min_val, max_val
        elif "-" in base:
            if base.count("-") > 1:
                raise ValueError("Invalid range")
            a_str, b_str = base.split("-")
            if not (a_str.isdigit() and b_str.isdigit()):
                raise ValueError("Range bounds must be digits")
            start, end = int(a_str), int(b_str)
            if start > end or start < min_val or end > max_val:
                raise ValueError("Range out of bounds")
        else:
            if not base.isdigit():
                raise ValueError("Invalid number")
            val = int(base)
            if val < min_val or val > max_val:
                raise ValueError("Value out of bounds")
            start = end = val

        for x in range(start, end + 1, step):
            values.add(x)

    return sorted(values)
```
