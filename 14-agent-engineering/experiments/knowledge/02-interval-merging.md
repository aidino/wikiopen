# Bài học 02: Gộp Khoảng Chồng Lấp hoặc Liền Kề (`merge_port_ranges`)

## Quy tắc Cốt lõi
1. **Không làm thay đổi đầu vào (Immutability):** Luôn dùng `sorted_ranges = sorted(ranges)` thay vì `ranges.sort()`.
2. **Tránh dùng slice `[1:]` trên dòng `for`:** Các model nhỏ đôi khi nhầm dấu `:` trong `[1:]` với dấu `:` cuối câu lệnh `for`, gây `SyntaxError: expected ':'`. Hãy duyệt trực tiếp `for cur_start, cur_end in sorted_ranges:` và kiểm tra `if not merged:`.
3. **Điều kiện gộp khoảng liền kề hoặc chồng lấp:**
   - Giả sử khoảng cuối cùng trong danh sách kết quả là `(last_start, last_end)` và khoảng đang xét là `(cur_start, cur_end)`.
   - Vì hai khoảng liền kề như `(10, 12)` và `(13, 15)` có `13 == 12 + 1`, điều kiện gộp là:
     `cur_start <= last_end + 1`
   - Khi gộp, cập nhật phần tử cuối: `merged[-1] = (last_start, max(last_end, cur_end))`.

## Mẫu cài đặt tham khảo
```python
def merge_port_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    sorted_ranges = sorted(ranges)
    merged: list[tuple[int, int]] = []
    for cur_start, cur_end in sorted_ranges:
        if not merged:
            merged.append((cur_start, cur_end))
        else:
            last_start, last_end = merged[-1]
            if cur_start <= last_end + 1:
                merged[-1] = (last_start, max(last_end, cur_end))
            else:
                merged.append((cur_start, cur_end))
    return merged
```
