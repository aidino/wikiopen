"""Script điều phối chạy thực nghiệm Student Coding Agent trên bộ bài tập từ Dễ đến Khó."""

from __future__ import annotations

import argparse
import sys

from student_agent import StudentCodingAgent, save_run_logs
from tasks import TASKS


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Student Coding Agent experiment.")
    parser.add_argument(
        "--tag",
        default="run",
        help="Nhãn của lần chạy để lưu log vào logs/<tag>.md và logs/<tag>.json",
    )
    parser.add_argument(
        "--task",
        default="",
        help="Chỉ chạy một bài tập cụ thể theo task_id (mặc định chạy toàn bộ 5 bài).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=6,
        help="Số bước gọi tool tối đa cho mỗi bài tập.",
    )
    args = parser.parse_args()

    selected_tasks = [t for t in TASKS if not args.task or args.task in t.task_id]
    if not selected_tasks:
        print(f"Không tìm thấy bài tập nào khớp với {args.task!r}", file=sys.stderr)
        return 1

    agent = StudentCodingAgent(max_turns=args.max_turns)
    results = []

    print(f"=== BẮT ĐẦU CHẠY STUDENT AGENT ({agent.model}) | TAG: {args.tag} ===")
    for task in selected_tasks:
        print(f"\n>>> Đang giải [{task.task_id}] {task.title}")
        res = agent.solve_task(task, verbose=True)
        results.append(res)
        verdict = "PASS" if res.passed else "FAIL"
        print(f"<<< Kết quả [{task.task_id}]: {verdict} ({res.passed_count}/{res.total_count} test cases, {res.turns_used} turns)")
        if not res.passed:
            print(f"    Lý do lỗi:\n{res.failure_summary}")

    log_path = save_run_logs(args.tag, results)
    passed_tasks = sum(1 for r in results if r.passed)
    total_tc_passed = sum(r.passed_count for r in results)
    total_tc = sum(r.total_count for r in results)

    print("\n" + "=" * 68)
    print(f"TỔNG KẾT [{args.tag}]: Hoàn thành {passed_tasks}/{len(results)} bài tập ({total_tc_passed}/{total_tc} test cases).")
    print(f"Log chi tiết đã lưu tại: {log_path}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
