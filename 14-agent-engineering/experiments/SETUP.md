# Hướng dẫn Cài đặt & Khởi chạy Thí nghiệm Teacher – Student Coding Agent

Thí nghiệm này minh họa mô hình **Teacher Agent** (mô hình lớn - Gemini Argon) hướng dẫn **Student Agent** (`Qwen/Qwen3-1.7B` chạy local qua vLLM) giải chuỗi bài tập lập trình Python từ dễ đến khó thông qua **Kho tri thức Markdown (`knowledge/*.md`)**.

---

## 1. Yêu cầu Phần cứng & Môi trường
- **GPU:** NVIDIA RTX 3070 (8 GB VRAM) hoặc tương đương (tối thiểu ~5 GB VRAM trống cho `Qwen/Qwen3-1.7B` BF16).
- **RAM:** 16 GB+ (hệ thống hiện có 30 GB).
- **Python:** Python 3.12 / 3.13 quản lý qua `uv` tại `.venv`.

---

## 2. Cài đặt Thư viện

Từ thư mục gốc của repository (`/home/dino/Documents/wikiopen`), cài đặt `vllm` và `openai` vào môi trường ảo `.venv`:

```bash
uv pip install vllm openai python-dotenv --python .venv/bin/python
```

---

## 3. Khởi chạy Мо hình Student (`Qwen/Qwen3-1.7B`) với vLLM

Vì RTX 3070 (8 GB VRAM) thường dùng sẵn ~800 MB – 1 GB VRAM cho hệ điều hành/màn hình, ta cấu hình:
- `--gpu-memory-utilization 0.70`: Giới hạn vLLM dùng tối đa ~5.6 GB VRAM (tránh lỗi OOM khi mặc định là 0.90).
- `--max-model-len 4096`: Đủ rộng cho multi-turn coding loop + Markdown context mà tiết kiệm KV cache.
```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
.venv/bin/vllm serve Qwen/Qwen3-1.7B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.70 \
  --generation-config vllm \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enforce-eager
```

Kiểm tra server đã sẵn sàng:
```bash
curl http://127.0.0.1:8000/v1/models
```

---

## 4. Cấu trúc Thư mục Thí nghiệm (`14-agent-engineering/experiments/`)

```text
14-agent-engineering/experiments/
├── SETUP.md                 # Tài liệu hướng dẫn cài đặt và chạy vLLM
├── start_vllm.sh            # Script khởi chạy nhanh vLLM server cho Qwen/Qwen3-1.7B
├── student_agent.py         # Coding Agent thuần Python (chỉ dùng openai SDK, không framework)
├── tasks.py                 # Bộ bài tập Python từ Dễ -> Trung bình -> Khó (kèm test cases)
├── run_experiment.py        # Script chạy Student giải bài, ghi log chi tiết cho Teacher
├── knowledge/               # Kho tri thức Markdown do Teacher biên soạn & cập nhật
│   └── 00-overview.md       # Quy tắc cơ bản ban đầu
└── logs/                    # Log chi tiết từng lượt suy nghĩ, gọi tool, kết quả chấm bài
```

---

## 5. Quy trình Thí nghiệm Teacher – Student

1. **Bước 1 (Baseline):** Chạy `student_agent.py` qua `run_experiment.py` khi kho `knowledge/` chỉ có hướng dẫn sơ khai. Quan sát các bài Student làm sai trong `logs/`.
2. **Bước 2 (Teacher phân tích & dạy):** Teacher đọc log lỗi (ví dụ: quên xử lý edge-case, sai thuật toán, dùng sai cấu trúc dữ liệu, lặp vòng) và viết các bài học/mẫu giải pháp vào `knowledge/*.md`.
3. **Bước 3 (Student học & giải lại):** Student Agent sử dụng công cụ `search_knowledge` để truy vấn kho Markdown trong quá trình suy luận, dùng `test_code` để kiểm thử, và `submit_solution` để nộp bài.
4. **Bước 4 (Đánh giá):** So sánh điểm số và số lượt giải trước/sau khi có sự hướng dẫn của Teacher.
