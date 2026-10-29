# Self-Refine and CRITIC: Iterative Output Improvement (Cải thiện đầu ra qua vòng lặp tinh chỉnh)

> `Self-Refine` (Madaan và cộng sự, 2023) sử dụng duy nhất một mô hình `LLM` đóng 3 vai trò — tạo sinh (`generate`), phản hồi phê bình (`feedback`), và tinh chỉnh (`refine`) — xoay vòng liên tục. Mức cải thiện trung bình đạt +20 điểm tuyệt đối trên 7 tác vụ khác nhau. `CRITIC` (Gou và cộng sự, 2023) gia cố thêm bước `feedback` bằng cách điều hướng quá trình kiểm chứng (`verification`) đi qua các công cụ bên ngoài (`external tools`). Đến năm 2026, mẫu thiết kế (`pattern`) này được tích hợp sẵn trong mọi `framework` dưới tên gọi `"evaluator-optimizer"` (của Anthropic) hoặc vòng lặp rào chắn kiểm soát (`guardrail loop` của OpenAI Agents SDK).

- **Loại bài học (Type):** Build (Thực hành xây dựng)
- **Ngôn ngữ (Languages):** Python (`stdlib` — thư viện chuẩn)
- **Kiến thức tiên quyết (Prerequisites):** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
- **Thời gian (Time):** ~60 phút

## Mục tiêu học tập (Learning Objectives)

- Nêu rõ 3 `prompt` cốt lõi của `Self-Refine` (`generate`, `feedback`, `refine`) và giải thích tại sao lịch sử các lần sửa (`history`) lại đóng vai trò sống còn đối với `refine prompt`.
- Giải thích góc nhìn cốt lõi (`critical insight`) của `CRITIC`: các mô hình `LLM` không đáng tin cậy khi tự kiểm chứng (`self-verification`) nếu thiếu đi sự đối chiếu thực tế bên ngoài (`external grounding`).
- Tự cài đặt một vòng lặp `Self-Refine loop` chỉ dùng thư viện chuẩn (`stdlib`) có lưu trữ `history` và tích hợp bộ kiểm chứng bên ngoài tùy chọn (`optional external verifier`).
- Ánh xạ mẫu thiết kế này sang luồng công việc `"evaluator-optimizer"` của Anthropic và cơ chế `output guardrails` của OpenAI Agents SDK.

## Vấn đề (The Problem)

Một `agent` tạo ra câu trả lời "gần đúng" (`almost right`). Có thể một dòng code bị lỗi cú pháp (`syntax error`). Có thể một bản tóm tắt hơi quá dài. Hoặc một bản kế hoạch bỏ sót một trường hợp biên (`edge case`). Điều bạn mong muốn là: `agent` tự phê bình (`critiques`) đầu ra của chính nó, rồi tự tay sửa lại (`fixes it`).

`Self-Refine` chứng minh rằng cơ chế này hoạt động hiệu quả chỉ với một `model` duy nhất, không cần dữ liệu huấn luyện (`no training data`), không cần học tăng cường (`no RL`). Nhưng có một điểm yếu chí mạng: **`LLM` rất kém trong việc tự kiểm chứng các sự thật khách quan (`hard facts`)**. `CRITIC` chỉ ra giải pháp khắc phục — hãy dẫn luồng bước kiểm chứng (`verify step`) đi qua các công cụ bên ngoài (`external tools` như: công cụ tìm kiếm `search`, trình thông dịch code `code interpreter`, máy tính bỏ túi `calculator`, hoặc trình chạy kiểm thử `test runner`).

> **Giải thích dễ hiểu:**
> - **`Self-Refine`** giống như việc bạn viết xong một bài luận rồi tự đọc lại để sửa lỗi diễn đạt, rút gọn câu chữ — bạn hoàn toàn có thể tự làm tốt mà không cần tra cứu gì thêm.
> - **`CRITIC`** giải quyết trường hợp bài luận đó có chứa các con số thống kê hoặc đoạn code: nếu chỉ "nhìn bằng mắt" và tự đoán (`self-verification`), bạn rất dễ bỏ qua lỗi tính nhẩm hoặc lỗi logic ẩn. Thay vì tự đoán, `CRITIC` bắt buộc phải lấy máy tính ra bấm lại hoặc chạy thử đoạn code xem có báo lỗi đỏ không, rồi mới dùng kết quả thực tế đó để sửa bài.

Kết hợp lại, hai bài báo này định hình nên tiêu chuẩn mặc định năm 2026 cho việc cải thiện chất lượng theo vòng lặp (`iterative improvement`): **Tạo sinh (`generate`) $\rightarrow$ Kiểm chứng (`verify` — ưu tiên dùng công cụ ngoài khi có thể) $\rightarrow$ Tinh chỉnh (`refine`) $\rightarrow$ Dừng lại khi bộ kiểm chứng báo đạt (`passes`).**

## Khái niệm cốt lõi (The Concept)

### Self-Refine (Madaan và cộng sự, NeurIPS 2023)

Một `LLM` duy nhất đảm nhận 3 vai trò:

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

- **`generate(task) -> output_0`**: Tạo ra bản nháp đầu tiên.
- **`feedback(task, output_0) -> critique_0`**: Đóng vai người phê bình, chỉ ra các điểm chưa đạt trong `output_0`.
- **`refine(task, output_0, critique_0, history) -> output_1`**: Dựa vào lời phê bình và lịch sử trước đó để viết lại bản mới `output_1` tốt hơn.
- Lặp lại cho tới khi `feedback` báo `"no issues"` (không còn lỗi) hoặc cạn ngân sách vòng lặp (`budget exhausted`).

**Chi tiết kỹ thuật quan trọng:** Bước `refine` nhìn thấy toàn bộ lịch sử (`full history` — bao gồm tất cả các `outputs` và `critiques` từ những vòng trước) để nó không lặp lại những lỗi cũ (ví dụ: tránh việc sửa từ phương án A sang B, rồi vòng sau lại sửa ngược từ B về A). Bài báo đã làm thí nghiệm cắt bỏ (`ablation study`) phần này: nếu bỏ `history` đi, chất lượng đầu ra tụt dốc thảm hại.

**Kết quả nổi bật (`Headline`):** Cải thiện trung bình **+20 điểm tuyệt đối** trên 7 tác vụ (toán học `math`, lập trình `code`, tạo từ viết tắt `acronym`, hội thoại `dialog`) kể cả trên `GPT-4`. Không cần huấn luyện (`no training`), không cần công cụ ngoài (`no external tools`), chỉ dùng một `model` duy nhất.

### CRITIC (Gou và cộng sự, arXiv:2305.11738, v4 tháng 2/2024)

Điểm yếu của `Self-Refine`: bước `feedback` thực chất là `LLM` tự chấm điểm chính mình. Đối với các khẳng định mang tính sự thật/kiến thức (`factual claims`), cách này không đáng tin cậy (một thông tin bịa đặt `hallucination` thường trông rất thuyết phục đối với chính mô hình vừa sinh ra nó). `CRITIC` (*Correcting with Tool-Interactive Critiquing*) thay thế hàm `feedback(task, output)` bằng `verify(task, output, tools)`, trong đó bộ công cụ `tools` bao gồm:

- **Công cụ tìm kiếm (`search engine`)** để đối chiếu các thông tin sự thật (`factual claims`).
- **Trình thông dịch mã (`code interpreter`)** để kiểm tra tính đúng đắn khi chạy code (`code correctness`).
- **Máy tính (`calculator`)** cho các phép tính số học (`arithmetic`).
- **Các bộ kiểm chứng chuyên biệt theo miền (`domain-specific verifiers`)** như `unit tests`, bộ kiểm tra kiểu `type checkers`, công cụ bắt lỗi cú pháp `linters`.

Bộ kiểm chứng (`verifier`) sẽ tạo ra một bản phê bình có cấu trúc (`structured critique`) bám sát vào kết quả trả về từ công cụ (`grounded in tool results`). Sau đó, bộ tinh chỉnh (`refiner`) sẽ dựa vào bản phê bình có căn cứ xác thực này để sửa lại đầu ra.

**Kết quả nổi bật (`Headline`):** `CRITIC` vượt trội hơn hẳn `Self-Refine` trên các tác vụ đòi hỏi tính chính xác thực tế (`factual tasks`) bởi vì lời phê bình có căn cứ xác thực (`grounded`). Còn trên các tác vụ không có bộ kiểm chứng bên ngoài (như viết văn sáng tạo `creative writing`, căn chỉnh định dạng `formatting`), `CRITIC` tự động rút gọn về thành `Self-Refine`.

### Điều kiện dừng vòng lặp (The stop condition)

Có hai dạng điều kiện dừng phổ biến:

1. **Bộ kiểm chứng báo đạt (`Verifier passes`):** Bài kiểm tra bên ngoài trả về trạng thái thành công (`success`). Đây là cách luôn được ưu tiên khi có sẵn công cụ (`unit tests`, `type checker`, `guardrail assertion`).
2. **Không còn phản hồi lỗi nào được đưa ra (`No feedback issued`):** Mô hình tự tuyên bố `"the output is fine"` (đầu ra đã ổn). Cách này rẻ hơn nhưng kém tin cậy; bắt buộc phải đi kèm với giới hạn số vòng lặp tối đa (`max-iteration cap`).

**Tiêu chuẩn mặc định năm 2026:** Kết hợp cả hai điều kiện:
*"Dừng lại nếu `verifier passes` HOẶC (`model says fine` VÀ `iterations >= 2`) HOẶC `iterations >= max_iterations`."*
*(Giải thích: Yêu cầu `iterations >= 2` khi model tự chấm nhằm ép mô hình phải thực hiện ít nhất một vòng suy xét kỹ lưỡng, tránh việc model lười biếng kết luận "ổn rồi" ngay từ lần đầu).*

### Mẫu thiết kế Evaluator-Optimizer (Anthropic, 2024)

Bài viết tháng 12/2024 của Anthropic (*Building Effective Agents*) đã định danh đây là một trong 5 mẫu luồng công việc (`five workflow patterns`) cốt lõi. Kiến trúc gồm 2 vai trò:

- **`Evaluator` (Bộ đánh giá):** Chấm điểm đầu ra và viết bản nhận xét phê bình (`critique`).
- **`Optimizer` (Bộ tối ưu/chỉnh sửa):** Chỉnh sửa lại đầu ra dựa trên bản `critique` đó.

Vòng lặp chạy cho tới khi `Evaluator` chấp thuận (`passes`). Đây chính là `Self-Refine` / `CRITIC` dưới góc nhìn hệ thống của Anthropic. Chi tiết kỹ thuật quan trọng mà Anthropic nhấn mạnh thêm: **`prompt` của `Evaluator` và `Optimizer` phải được thiết kế khác biệt nhau một cách rõ rệt (`substantially different`)** để mô hình không rơi vào tình trạng "vừa đá bóng vừa thổi còi", gật đầu cho qua chuyện (`rubber-stamp`).

### Cơ chế Output Guardrails của OpenAI Agents SDK

OpenAI Agents SDK đóng gói sẵn mẫu thiết kế này dưới tên gọi **`"output guardrails"`** (rào chắn kiểm soát đầu ra). Một `guardrail` là một hàm kiểm tra tính hợp lệ (`validator`) chạy trên kết quả đầu ra cuối cùng của `agent`. Nếu `guardrail` phát hiện vi phạm (ném ra ngoại lệ `OutputGuardrailTripwireTriggered`), đầu ra đó sẽ bị từ chối và `agent` có thể thử lại (`retry`). Các `guardrails` này có thể gọi công cụ bên ngoài (theo phong cách `CRITIC`) hoặc chỉ là các hàm kiểm tra thuần túy (theo phong cách `Self-Refine`).

### Những cạm bẫy thường gặp năm 2026 (2026 pitfalls)

- **Vòng lặp "gật đầu cho qua" (`Rubber-stamp loops`):** Dùng cùng một `model` vừa tạo sinh vừa tự phê bình với cùng một văn phong `prompt` sẽ khiến mô hình nhanh chóng hội tụ về câu trả lời *"Trông ổn đấy"* (`"looks good to me"`). **Cách khắc phục:** Sử dụng các `prompts` có cấu trúc khác hẳn nhau (ví dụ: ép `Evaluator` đóng vai một chuyên gia kiểm toán khó tính bắt buộc phải tìm ra lỗi), hoặc dùng một `model` nhỏ/rẻ độc lập chuyên làm nhiệm vụ `critique`.
- **Tinh chỉnh quá đà (`Over-refinement`):** Mỗi vòng `refine` đều làm tăng độ trễ (`latency`) và tốn thêm `tokens` (đôi khi sửa quá nhiều vòng còn làm câu trả lời mất đi sự tự nhiên ban đầu). Chỉ nên cấp ngân sách từ **1–3 vòng lặp**; nếu sau đó vẫn chưa đạt, hãy chuyển tiếp cho con người xử lý (`escalate to human review`).
- **Lạm dụng `CRITIC` cho các tác vụ tầm thường (`CRITIC on trivial tasks`):** Nếu bài toán thực chất không có bộ kiểm chứng bên ngoài (`external verifier`), `CRITIC` sẽ suy biến về thành `Self-Refine`; đừng trả thêm chi phí độ trễ (`latency`) cho một bộ kiểm chứng rỗng (`stub verifier`).

```figure
self-refine
```

## Thực hành xây dựng (Build It)

File `code/main.py` cài đặt cả `Self-Refine` lẫn `CRITIC` trên một bài toán thu nhỏ (`toy task`): tạo một danh sách gạch đầu dòng ngắn (`bullet list`) cho một chủ đề. Bộ kiểm chứng kiểm tra định dạng (đúng 3 gạch đầu dòng, mỗi dòng dưới 60 ký tự). `CRITIC` bổ sung thêm một "bộ kiểm chứng sự thật" bên ngoài (`external "fact verifier"`) chuyên phạt các lỗi ảo giác/bịa đặt đã biết (`known hallucinations`).

Các thành phần trong code:

- `generate` — bộ tạo sinh theo kịch bản (`scripted producer`).
- `feedback` — bộ tự phê bình theo phong cách `LLM` (`LLM-style self-critique`).
- `verify_external` — bộ kiểm chứng đối chiếu thực tế theo phong cách `CRITIC` (`CRITIC-style grounded verifier`).
- `refine` — viết lại đầu ra dựa trên lịch sử các lần sửa (`history`).
- Điều kiện dừng (`Stop condition`) — dừng khi `verifier passes` hoặc chạm ngưỡng tối đa 4 vòng lặp (`max 4 iterations`).

Chạy thử chương trình:

```
python3 code/main.py
```

Hãy so sánh kết quả giữa lần chạy `Self-Refine` và `CRITIC`. `CRITIC` bắt được một lỗi sai lệch kiến thức thực tế (`factual error`) mà `Self-Refine` đã bỏ lọt, bởi vì `external verifier` có nguồn đối chiếu xác thực (`grounding`) mà bộ tự phê bình (`self-critic`) không hề có.

## Ứng dụng thực tế (Use It)

- Mẫu **`evaluator-optimizer` của Anthropic** chính là kiến trúc này được diễn đạt theo ngôn ngữ thân thiện với hệ sinh thái Claude.
- Cơ chế **`output guardrails` của OpenAI Agents SDK** mang hình thái của `CRITIC` (vì các `guardrails` có thể gọi `tools` để kiểm chứng).
- **LangGraph** cung cấp sẵn nút phản tư (`reflection node`) có luồng hoạt động y hệt `Self-Refine`.
- **Gemini 2.5 Computer Use của Google** bổ sung một bộ đánh giá an toàn trên từng bước (`per-step safety evaluator`) — đây cũng là một biến thể của `CRITIC`: mọi hành động đều được kiểm chứng trước khi thực sự thực thi (`verified before commit`).

## Đóng gói kỹ năng (Ship It)

File `outputs/skill-refine-loop.md` giúp cấu hình một vòng lặp `evaluator-optimizer loop` hoàn chỉnh dựa trên hình thái tác vụ (`task shape`), sự sẵn có của công cụ kiểm chứng (`verifier availability`), và ngân sách số vòng lặp (`iteration budget`). Nó xuất ra bộ `prompts` chuẩn cho `generator`, `evaluator/verifier`, và `optimizer`, kèm theo chính sách dừng (`stop policy`) chặt chẽ.

## Bài tập rèn luyện (Exercises)

1. Chạy bản code mẫu với `max_iterations=1`. Liệu `CRITIC` có còn giúp ích được gì không? *(Gợi ý: chú ý xem nếu chỉ có 1 vòng thì sau khi `verifier` bắt được lỗi, có còn lượt nào để `refine` sửa lại hay ít nhất là chặn đầu ra sai không).*
2. Thay thế `external verifier` bằng một bộ kiểm chứng bị nhiễu (`noisy verifier` — có xác suất 30% báo lỗi giả `false positives`). Vòng lặp sẽ hành xử như thế nào? Đây chính là thực tế phũ phàng năm 2026 của hầu hết các hệ thống `guardrail stacks`.
3. Cài đặt biến thể `"generator-critic on different models"` (dùng 2 model khác nhau): `big model` tạo sinh, `small model` phê bình. Liệu nó có đánh bại cách dùng cùng một model (`same-model`) xét trên tỷ lệ chất lượng/chi phí không?
4. Đọc Mục 3 (Section 3) của bài báo `CRITIC` (arXiv:2305.11738 v4). Kể tên 3 nhóm công cụ kiểm chứng (`verification-tool categories`) và nêu một ví dụ cụ thể cho từng nhóm.
5. Ánh xạ cơ chế `output_guardrails` của OpenAI Agents SDK sang vai trò `verifier` của `CRITIC`. Thiết kế của SDK này có điểm gì chưa tới (`get wrong`), và điểm gì đã làm rất chuẩn (`get right`)?

## Thuật ngữ quan trọng (Key Terms)

| Thuật ngữ (Term) | Cách người ta thường gọi (What people say) | Bản chất thực sự (What it actually means) |
|------|----------------|------------------------|
| **Self-Refine** | "LLM tự sửa lỗi cho chính nó" ("LLM that fixes itself") | Vòng lặp `Generate -> feedback -> refine` ngay trong một `model` duy nhất, có lưu kèm `history` |
| **CRITIC** | "Kiểm chứng dựa trên công cụ" ("Tool-grounded verification") | Thay thế bước `feedback` tự chấm bằng một `external verifier` (như `search`, `code interpreter`, `calculator`, `tests`) |
| **Evaluator-Optimizer** | "Mẫu workflow của Anthropic" ("Anthropic workflow pattern") | Hai vai trò tách biệt — `evaluator` chấm điểm, `optimizer` sửa bài — lặp lại cho đến khi hội tụ đạt yêu cầu |
| **Output guardrail** | "Kiểm tra hậu kỳ" ("Post-hoc check") | Bộ kiểm tra `validator` của OpenAI Agents SDK chạy ngay sau khi `agent` sinh ra kết quả đầu ra |
| **Verify step** | "Pha phê bình/kiểm tra" ("Critique phase") | Quyết định chịu lực quan trọng nhất của kiến trúc: dùng kiểm chứng có căn cứ công cụ (`grounded`) hay tự chấm (`self-rated`) |
| **Refine history** | "Những gì model đã thử qua" ("What the model already tried") | Toàn bộ các `outputs` + `critiques` trước đó được gắn vào đầu `refine prompt`; nếu bỏ đi chất lượng sẽ sụp đổ |
| **Rubber-stamp loop** | "Lỗi tự đồng tình với chính mình" ("Self-agreement failure") | Bước `critique` dùng cùng kiểu `prompt` dễ dãi trả về "trông ổn rồi"; khắc phục bằng cách thiết kế `prompt` khác biệt về cấu trúc |
| **Stop condition** | "Bài kiểm tra hội tụ" ("Convergence test") | Kết hợp: `Verifier passes` HOẶC không còn `feedback` VÀ chạm trần số vòng lặp (`iteration cap`); không bao giờ dùng điều kiện đơn lẻ |

## Đọc thêm (Further Reading)

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — bài báo khoa học nền tảng (`canonical paper`) về `Self-Refine`
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — cơ chế kiểm chứng dựa trên công cụ (`tool-grounded verification`)
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — mẫu thiết kế luồng công việc `evaluator-optimizer`
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — sử dụng `output guardrails` như các bộ kiểm chứng mang hình thái `CRITIC`
