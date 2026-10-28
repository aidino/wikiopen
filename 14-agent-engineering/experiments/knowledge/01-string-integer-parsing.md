# Bài học 01: Kiểm tra và Tính tổng Chuỗi Số Nguyên Hợp Lệ (`sum_valid_numbers`)

## Lỗi thường gặp của Student (từ Log Baseline)
- Không dùng `int(s)` trực tiếp kèm `try/except` đơn thuần vì trong Python `int("1_000")` được chấp nhận là `1000`, trong khi đề bài coi `"1_000"` là ký tự không hợp lệ.
- Không tự viết các nhánh `if item[1] == ' '` rời rạc vì sẽ bỏ sót các chữ số khác `'0'`.

## Phương pháp Chuẩn trong Python
Để kiểm tra một chuỗi `item` có phải là số nguyên hợp lệ hay không:
1. Dùng `s = item.strip()` để loại bỏ khoảng trắng đầu/cuối (bao gồm `' '`, `'\n'`, `'\t'`).
2. Nếu `not s` (chuỗi rỗng): bỏ qua.
3. Nếu `s[0] in ('+', '-')`: phần còn lại là `digits = s[1:]`. Ngược lại: `digits = s`.
4. Kiểm tra `digits.isdigit()`:
   - Phương thức `.isdigit()` đảm bảo chuỗi có ít nhất 1 ký tự và TẤT CẢ đều là chữ số `0-9` (tự động loại bỏ `'--1'`, `'++2'`, `'3.14'`, `'1e5'`, `'1_000'`, `'abc'`).
5. Nếu `digits.isdigit()` là `True`, cộng `int(s)` vào tổng.

## Mẫu cài đặt tham khảo
```python
def sum_valid_numbers(items: list[str]) -> int:
    total = 0
    for item in items:
        s = item.strip()
        if not s:
            continue
        digits = s[1:] if s[0] in ("+", "-") else s
        if digits.isdigit():
            total += int(s)
    return total
```
