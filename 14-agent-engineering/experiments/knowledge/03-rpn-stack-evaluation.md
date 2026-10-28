# Bài học 03: Tính Biểu Thức Hậu Tố RPN An Toàn (`evaluate_rpn_safe`)

## Phân tích Lỗi của Student từ Log Baseline
Trong lần chạy baseline, Student mắc 3 lỗi nghiêm trọng:
1. **Dùng `a // b` cho phép chia nguyên:** Trong Python, `-7 // 2` cho kết quả `-4` (làm tròn xuống $-\infty$), trong khi đề bài yêu cầu làm tròn về 0 (`-7 / 2 -> -3`, `-13 / 5 -> -2`).
   - **Cách sửa:** Dùng `int(a / b)` để chia nguyên làm tròn về 0!
2. **Kiểm tra chia cho 0 sai vị trí:** Student kiểm tra `if stack[0] == 0` ở cuối hàm (khiến kết quả hợp lệ bằng 0 hoặc khác 0 bị nhầm thành `'ERROR:DIV_BY_ZERO'`), thay vì kiểm tra số chia `b == 0` NGAY TRƯỚC phép chia.
   - **Cách sửa:** Khi `token == '/'`, kiểm tra `if b == 0: return 'ERROR:DIV_BY_ZERO'` trước khi tính `int(a / b)`.
3. **Không bắt lỗi token lạ:** Gọi `int(token)` trực tiếp trên `'foo'` gây `ValueError`.

## Quy trình Chuẩn cho `evaluate_rpn_safe`
1. Nếu `not tokens`: trả về `'ERROR:INVALID'`.
2. Duyệt từng `token` trong `tokens`:
   - Nếu `token in ('+', '-', '*', '/')`:
     - Nếu `len(stack) < 2`: trả về `'ERROR:INVALID'`.
     - Lấy `b = stack.pop()`, sau đó `a = stack.pop()` (lưu ý thứ tự: `b` lấy ra trước, `a` lấy ra sau).
     - Nếu `token == '/'`:
       - Nếu `b == 0`: trả về `'ERROR:DIV_BY_ZERO'`.
       - Tính `stack.append(int(a / b))`.
     - Thực hiện tương tự cho `+`, `-`, `*`.
   - Ngược lại (khi `token` là toán hạng):
     - Kiểm tra xem `token` có phải chuỗi số nguyên hợp lệ không (`digits = token[1:] if token.startswith(('+', '-')) else token`). Nếu `not digits.isdigit()`: trả về `'ERROR:INVALID'`.
     - Đẩy `int(token)` vào `stack`.
3. Sau vòng lặp, nếu `len(stack) != 1`: trả về `'ERROR:INVALID'`.
4. Trả về `stack[0]` (kể cả khi `stack[0] == 0` thì đó vẫn là kết quả số nguyên hợp lệ!).

## Mẫu cài đặt tham khảo
```python
def evaluate_rpn_safe(tokens: list[str]) -> int | str:
    if not tokens:
        return "ERROR:INVALID"
    stack: list[int] = []
    for token in tokens:
        if token in ("+", "-", "*", "/"):
            if len(stack) < 2:
                return "ERROR:INVALID"
            b = stack.pop()
            a = stack.pop()
            if token == "+":
                stack.append(a + b)
            elif token == "-":
                stack.append(a - b)
            elif token == "*":
                stack.append(a * b)
            else:
                if b == 0:
                    return "ERROR:DIV_BY_ZERO"
                stack.append(int(a / b))
        else:
            digits = token[1:] if token.startswith(("+", "-")) else token
            if not digits.isdigit():
                return "ERROR:INVALID"
            stack.append(int(token))
    if len(stack) != 1:
        return "ERROR:INVALID"
    return stack[0]
```
