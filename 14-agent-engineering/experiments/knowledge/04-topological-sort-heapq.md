# Bài học 04: Sắp Xếp Topo Thứ Tự Build Theo Từ Điển (`resolve_build_order`)

## Phân tích Lỗi của Student từ Log Baseline
Bài toán yêu cầu:
1. Nếu `pkg` phụ thuộc vào `dep` (`dep in deps[pkg]`), thì `dep` PHẢI đứng trước `pkg` (cạnh có hướng: `dep -> pkg`).
2. Các gói chỉ xuất hiện trong danh sách phụ thuộc (không phải key của `deps`) vẫn phải được đưa vào đồ thị.
3. Có thể có cạnh trùng lặp trong danh sách phụ thuộc (ví dụ `'web': ['auth', 'auth']`) — cần loại bỏ trùng lặp bằng `set` để không tính sai bậc vào (`in_degree`).
4. Khi nhiều gói cùng có `in_degree == 0`, luôn chọn gói có tên nhỏ nhất theo thứ tự từ điển bằng hàng đợi ưu tiên `heapq`.
5. Nếu kết quả thu được có số đỉnh ít hơn tổng số đỉnh (`len(order) < len(all_nodes)`), đồ thị có chu trình $\rightarrow$ trả về `'ERROR:CYCLE'`.

## Thuật toán Kahn kết hợp `heapq`
```python
import heapq

def resolve_build_order(deps: dict[str, list[str]]) -> list[str] | str:
    if not deps:
        return []

    all_nodes: set[str] = set()
    for pkg, reqs in deps.items():
        all_nodes.add(pkg)
        for r in reqs:
            all_nodes.add(r)

    adj: dict[str, set[str]] = {node: set() for node in all_nodes}
    in_degree: dict[str, int] = {node: 0 for node in all_nodes}

    for pkg, reqs in deps.items():
        for r in set(reqs):
            # r phải build trước pkg: cạnh r -> pkg
            adj[r].add(pkg)
            in_degree[pkg] += 1

    heap: list[str] = [node for node, deg in in_degree.items() if deg == 0]
    heapq.heapify(heap)

    order: list[str] = []
    while heap:
        curr = heapq.heappop(heap)
        order.append(curr)
        for nxt in adj[curr]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                heapq.heappush(heap, nxt)

    if len(order) != len(all_nodes):
        return "ERROR:CYCLE"
    return order
```
