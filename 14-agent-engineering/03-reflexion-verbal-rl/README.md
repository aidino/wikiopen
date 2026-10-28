# Reflexion: Verbal Reinforcement Learning (Học tăng cường bằng ngôn ngữ tự nhiên)

> Học tăng cường dựa trên đạo hàm (`gradient-based RL`) cần hàng nghìn lần thử nghiệm (`trials`) cùng cả một cụm máy chủ `GPU cluster` chỉ để sửa một kiểu lỗi (`failure mode`). `Reflexion` (Shinn và cộng sự, NeurIPS 2023) giải quyết việc đó hoàn toàn bằng ngôn ngữ tự nhiên (`natural language`): sau mỗi lần thử bị thất bại, `agent` tự viết ra một đoạn suy ngẫm rút kinh nghiệm (`reflection`), lưu nó vào bộ nhớ sự kiện (`episodic memory`), và đính kèm bộ nhớ đó làm điều kiện hướng dẫn (`condition`) cho lần thử tiếp theo. Đây chính là mẫu thiết kế (`pattern`) đứng đằng sau cơ chế `sleep-time compute` của Letta, cơ chế ghi nhớ bài học vào `CLAUDE.md` của Claude Code, và lệnh `learn-rule` của pro-workflow.

- **Loại bài học (Type):** Build (Thực hành xây dựng)
- **Ngôn ngữ (Languages):** Python (`stdlib` — thư viện chuẩn)
- **Kiến thức tiên quyết (Prerequisites):** Phase 14 · 01 (Agent Loop), Phase 14 · 02 (ReWOO)
- **Thời gian (Time):** ~60 phút

## Mục tiêu học tập (Learning Objectives)

- Kể tên ba thành phần của `Reflexion` (`Actor`, `Evaluator`, `Self-Reflector`) cùng vai trò của bộ nhớ sự kiện (`episodic memory`).
- Tự cài đặt một vòng lặp `Reflexion loop` chỉ dùng thư viện chuẩn (`stdlib`) với bộ chấm điểm nhị phân (`binary evaluator`), bộ đệm lưu suy ngẫm (`reflection buffer`), và cơ chế thử lại từ đầu (`fresh re-attempts`).
- Lựa chọn hợp lý giữa các nguồn phản hồi (`feedback sources`) gồm: điểm số vô hướng (`scalar`), quy tắc kinh nghiệm (`heuristic`), và tự đánh giá (`self-evaluated`) cho từng tác vụ cụ thể.
- Giải thích tại sao cơ chế củng cố bằng lời (`verbal reinforcement`) lại bắt được những lỗi mà `gradient-based RL` phải mất hàng nghìn lần thử mới sửa được.

## Vấn đề (The Problem)

Một `agent` thực hiện thất bại một tác vụ. Trong học tăng cường (`RL` — Reinforcement Learning) truyền thống, bạn sẽ phải chạy thêm hàng nghìn lần thử nghiệm (`trials`), tính toán đạo hàm (`gradients`), rồi cập nhật trọng số (`update weights`) của mô hình. Cách này vừa đắt đỏ, vừa chậm chạp, và hầu hết các `production agents` trong thực tế không hề có ngân sách huấn luyện (`training budget`) để chạy lại mô hình cho mỗi lần gặp lỗi.

`Reflexion` (Shinn và cộng sự, arXiv:2303.11366) đặt ra một câu hỏi khác: **sẽ ra sao nếu `agent` chỉ cần tự suy nghĩ xem tại sao mình lại thất bại, rồi thử làm lại một lần nữa với suy nghĩ rút kinh nghiệm đó nằm sẵn trong `prompt`?** Không cập nhật trọng số (`no weight updates`). Không tính đạo hàm (`no gradient`). Chỉ có ngôn ngữ tự nhiên được lưu trữ qua lại giữa các lần thử (`trials`).

> **Giải thích dễ hiểu:** Hãy hình dung một lập trình viên giải bài toán code bị lỗi test. Thay vì phải đi học lại đại học (tương đương việc cập nhật lại `weights` bằng `gradient` trong `RL`), người đó chỉ cần viết một tờ giấy nhớ dán lên màn hình: *"Lần trước mình quên xử lý mảng rỗng, lần này nhớ thêm `if not arr`"* rồi code lại từ đầu. Tờ giấy nhớ bằng chữ viết đó chính là **Verbal Reinforcement** (củng cố bằng ngôn ngữ).

Kết quả đạt được: trên môi trường giả lập ALFWorld, nó đánh bại `ReAct` và các phương pháp cơ sở không tinh chỉnh (`non-fine-tuned baselines`) khác. Trên bộ hỏi đáp HotpotQA, nó cải thiện đáng kể so với `ReAct`. Trên bài toán sinh code (`code generation` với HumanEval/MBPP), nó thiết lập kỷ lục tốt nhất (`state of the art` — SOTA) tại thời điểm công bố. Tất cả đều đạt được mà không cần bất kỳ một bước cập nhật đạo hàm (`gradient step`) nào.

## Khái niệm cốt lõi (The Concept)

### Ba thành phần chính (The three components)

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

- **`Actor` (Bộ hành động):** Sinh ra một quỹ đạo thực thi (`trajectory` — tức chuỗi các bước suy nghĩ và hành động theo vòng lặp kiểu `ReAct`).
- **`Evaluator` (Bộ đánh giá/chấm điểm):** Chấm điểm quỹ đạo vừa chạy — dưới dạng nhị phân (`binary` — đúng/sai), theo quy tắc kinh nghiệm (`heuristic`), hoặc tự đánh giá (`self-eval`).
- **`Self-Reflector` (Bộ tự suy ngẫm):** Viết một bản kiểm điểm/rút kinh nghiệm bằng ngôn ngữ tự nhiên (`natural-language reflection`) phân tích nguyên nhân thất bại.

Cộng thêm một cấu trúc dữ liệu quan trọng:

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

- **`Episodic memory` (Bộ nhớ sự kiện / Bộ nhớ trải nghiệm):** Danh sách lưu các bản `reflection` từ những lần thử trước, được gắn vào đầu (`prepended`) `prompt` của lần thử tiếp theo.

**Luồng chạy của một chu kỳ:** Một lần thử (`trial`) sẽ chạy `Actor`. Sau đó `Evaluator` vào chấm điểm kết quả. Nếu điểm thấp (thất bại), `Self-Reflector` sẽ tạo ra một đoạn suy ngẫm (ví dụ: *"Tôi đã chọn sai `tool` bởi vì tôi đọc nhầm câu hỏi là đang hỏi về X trong khi thực chất nó hỏi về Y"*). Đoạn suy ngẫm này được đẩy vào `episodic memory`. Lần thử tiếp theo sẽ bắt đầu lại từ trạng thái sạch (`starts fresh` — xóa bỏ các bước làm sai lộn xộn của lần trước), nhưng nhìn thấy được bài học `reflection` này ngay ở đầu `prompt`.

### Ba loại bộ đánh giá (Three evaluator types)

1. **`Scalar` (Điểm số vô hướng / Nhị phân):** Một tín hiệu đúng/sai (`binary signal`) hoặc điểm số khách quan từ môi trường bên ngoài. Ví dụ: nhiệm vụ trong ALFWorld thành công hay thất bại; bộ `unit tests` của HumanEval chạy `pass` hay `fail`. Đây là loại đơn giản nhất và mang lại tín hiệu chuẩn xác nhất (`highest-signal`).
2. **`Heuristic` (Quy tắc kinh nghiệm định sẵn):** Các dấu hiệu nhận biết lỗi được lập trình trước (`predefined failure signatures`). Ví dụ: *"Nếu `agent` lặp lại cùng một hành động 2 lần liên tiếp, đánh dấu là bị kẹt (`stuck`)."* Hoặc *"Nếu quỹ đạo (`trajectory`) vượt quá 50 bước, đánh dấu là kém hiệu quả (`inefficient`)."*
3. **`Self-evaluated` (Tự đánh giá bằng LLM):** Chính `LLM` tự chấm điểm quỹ đạo thực thi của mình (`LLM-as-judge`). Cần dùng đến khi không có đáp án chuẩn (`ground truth`) hay trình chạy test bên ngoài. Tín hiệu từ cách này yếu hơn và dễ bị ảo tưởng; nó hoạt động hiệu quả nhất khi được kết hợp với cơ chế kiểm chứng dựa trên công cụ (`tool-grounded verification` — sẽ học ở Lesson 05 — `CRITIC`).

Tiêu chuẩn mặc định vào năm 2026 là **kết hợp cả ba (`mix`)**: ưu tiên dùng `scalar` bất cứ khi nào có thể, lùi về dùng `self-eval` khi không có bộ chấm điểm ngoài, và luôn cài đặt các `heuristics` làm hàng rào an toàn (`safety rails`).

### Tại sao cơ chế này có tính tổng quát cao (Why this generalizes)

`Reflexion` không hẳn là một thuật toán hoàn toàn mới mà đúng hơn là một mẫu thiết kế đã được định danh (`named pattern`). Hầu như mọi `agent` có khả năng "tự phục hồi / tự sửa lỗi" (`self-healing agent`) trong môi trường `production` hiện nay đều chạy một biến thể nào đó của nó:

- **`Sleep-time compute` của Letta (Lesson 08):** Một `agent` chạy ngầm riêng biệt suy ngẫm (`reflects`) về các cuộc hội thoại đã qua trong lúc rảnh rỗi và ghi đúc kết vào các khối bộ nhớ (`memory blocks`).
- **Mẫu `CLAUDE.md` / "save memory" của Claude Code:** Các suy ngẫm sửa lỗi được ghi nhận lại thành các bài học (`learnings`), tự động nạp vào đầu các phiên làm việc (`sessions`) trong tương lai.
- **Lệnh `/learn-rule` của pro-workflow:** Các lần người dùng chỉnh sửa lỗi cho agent được lưu thành những quy tắc tường minh (`explicit rules`).
- **Các nút `reflection nodes` của LangGraph:** Một nút (`node`) chuyên chấm điểm đầu ra và điều hướng luồng chạy quay lại bước tinh chỉnh (`route to refine`) nếu chất lượng chưa đạt.

Tất cả đều bắt nguồn từ cùng một góc nhìn sâu sắc (`insight`): **ngôn ngữ tự nhiên là một phương tiện đủ giàu thông tin để truyền tải "bài học rút ra từ thất bại" giữa các lần chạy.**
*(Giải thích: Một con số điểm thưởng như `-1.0` trong `RL` truyền thống chỉ báo rằng "Bạn làm sai rồi", nhưng không nói rõ sai ở đâu. Trong khi đó, một câu ngôn ngữ tự nhiên chứa lượng thông tin ngữ nghĩa cực lớn: chỉ rõ sai ở bước nào, tại sao sai, và lần sau nên đổi sang hướng nào — giúp `LLM` sửa lỗi ngay ở lần thử kế tiếp).*

### Khi nào hiệu quả và khi nào không (When it works and when it does not)

`Reflexion` phát huy hiệu quả khi:

- **Có tín hiệu báo lỗi rõ ràng (`clear failure signal`):** Chạy `test` bị trượt, `tool` báo lỗi cụ thể, hoặc câu trả lời sai định dạng/sai kết quả kiểm tra.
- **Lớp tác vụ có tính lặp lại (`reproducible task class`):** Cùng một dạng câu hỏi hoặc bài toán đó sẽ còn được hỏi lại (hoặc được thử lại trong các `trials` kế tiếp).
- **Bản suy ngẫm có không gian để cải thiện quỹ đạo (`room to improve`):** `Agent` có đủ ngân sách hành động (`action budget`) và phương án thay thế để thử cách tiếp cận khác.

`Reflexion` không giúp ích được gì khi:

- `Agent` đã thành công ngay từ lần thử đầu tiên (`first try`).
- **Lỗi đến từ yếu tố ngoại cảnh (`external failure`):** Mất mạng (`network down`), `tool` bên thứ ba bị sập — việc viết suy ngẫm rằng *"mạng bị sập"* không hề giúp các lần chạy trong tương lai làm tốt hơn.
- **Suy ngẫm biến thành sự "mê tín" (`superstition`):** Khi `agent` gặp một lỗi ngẫu nhiên chập chờn (`one-off flaky run`), nhưng lại suy diễn và lưu vào bộ nhớ một câu chuyện sai lệch (ví dụ: *"Không bao giờ được gọi API vào buổi chiều"*), làm hỏng các lần chạy sau.

**Cạm bẫy phổ biến năm 2026: Sự thối rữa bộ nhớ (`memory rot`).** Các bản `reflection` tích tụ ngày càng nhiều; một số đã lỗi thời (`obsolete`) hoặc sai lệch; các lần chạy lại ngày càng chậm và tốn `token` khi bộ đệm `episodic buffer` phình to.
**Cách khắc phục (`Mitigation`):** Nén/tóm gọn bộ nhớ định kỳ (`periodic compaction` — Lesson 06), gắn thời hạn sống (`TTL` — Time-To-Live) cho từng `reflection`, hoặc dùng một `agent` dọn dẹp chạy ngầm lúc rảnh (`sleep-time cleanup agent` như cách của Letta).

```figure
react-trace
```

## Thực hành xây dựng (Build It)

File `code/main.py` cài đặt mô hình `Reflexion` trên một bài toán đố thu nhỏ (`toy puzzle`): tạo ra một danh sách gồm 3 phần tử có tổng bằng một số mục tiêu (`target`). `Actor` phát ra các danh sách ứng viên (`candidate lists`); `Evaluator` kiểm tra tổng của chúng; và `Self-Reflector` viết một dòng chẩn đoán xem điều gì đã sai. Bản `reflection` này được đưa vào `episodic memory` để phục vụ cho lần thử kế tiếp.

Các thành phần trong code:

- `Actor` — một chính sách theo kịch bản (`scripted policy`) có khả năng cải thiện kết quả khi nhìn thấy các `reflections` trong bộ nhớ.
- `Evaluator.binary()` — chấm đạt/trượt (`pass/fail`) dựa trên việc tổng có khớp với `target` hay không.
- `SelfReflector` — sinh ra một dòng chẩn đoán ngắn gọn chỉ rõ nguyên nhân thất bại (ví dụ: tổng đang lớn hơn hay nhỏ hơn mục tiêu bao nhiêu).
- `EpisodicMemory` — một danh sách có giới hạn kích thước (`bounded list`) kèm theo cơ chế thời hạn sống (`TTL semantics`).

Chạy thử chương trình:

```
python3 code/main.py
```

Bản ghi vết (`trace`) sẽ hiển thị 3 lần thử (`trials`). `Trial 1` thất bại, một `reflection` được lưu lại; `Trial 2` nhìn thấy `reflection` đó nên đã cải thiện được kết quả nhưng vẫn chưa trúng đích; đến `Trial 3` thì thành công. Hãy so sánh với lần chạy cơ sở (`baseline run` — không có `reflection`): nó bị kẹt mãi ở câu trả lời sai của `Trial 1`.

## Ứng dụng thực tế (Use It)

- **LangGraph** cung cấp sẵn cơ chế `reflection` như một mẫu nút đồ thị chuẩn (`node pattern`).
- Lệnh `/memory` của **Claude Code** và lệnh `/learn-rule` của **pro-workflow** đều xuất bộ đệm `episodic buffer` ra thành một file Markdown bên ngoài để lưu trữ lâu dài giữa các phiên.
- Cơ chế `sleep-time compute` của **Letta** chạy bộ `Self-Reflector` vào thời gian nghỉ (`downtime`) để `primary agent` (agent chính giao tiếp với người dùng) luôn giữ được tốc độ phản hồi nhanh (`latency-bound`).
- **OpenAI Agents SDK** không đóng gói sẵn `Reflexion` trực tiếp; bạn có thể tự xây dựng nó bằng cách kết hợp một `custom Guardrail` (từ chối các quỹ đạo có điểm số thấp) với một đối tượng bộ nhớ `Session` tồn tại xuyên suốt qua các lần chạy (`survives across runs`).

## Đóng gói kỹ năng (Ship It)

File `outputs/skill-reflexion-buffer.md` thiết lập và duy trì một bộ đệm `episodic buffer` hoàn chỉnh với khả năng thu thập suy ngẫm (`reflection capture`), giới hạn thời gian sống (`TTL`), và khử trùng lặp (`deduplication`). Khi nhận vào một lớp tác vụ (`task class`) và một lỗi thất bại, nó sẽ phát ra một bản `reflection` thực sự có ích cho lần thử tiếp theo (chứ không phải những lời khuyên chung chung vô thưởng vô phạt kiểu *"hãy cẩn thận hơn"*).

## Bài tập rèn luyện (Exercises)

1. Chuyển từ bộ chấm điểm nhị phân (`binary evaluator`) sang bộ chấm điểm vô hướng (`scalar evaluator`) trả về khoảng cách lệch so với mục tiêu (`distance metric` — còn cách `target` bao xa). Liệu vòng lặp có hội tụ (`converge`) về đáp án đúng nhanh hơn không?
2. Thêm thời hạn sống `TTL` là 10 lần thử (`trials`) cho các `reflections`. Những bản suy ngẫm cũ sau cột mốc đó sẽ gây hại hay giúp ích cho agent?
3. Cài đặt một `heuristic evaluator`: đánh dấu lần thử là bị kẹt (`stuck`) nếu cùng một hành động bị lặp lại. Cơ chế này tương tác như thế nào với `Self-Reflector`?
4. Chạy `Reflexion` với một `adversarial Actor` (Actor đối kháng cố tình phớt lờ các `reflections`). Kỹ thuật `prompt engineering` tối thiểu nào cho phần `reflection` sẽ buộc được `Actor` phải chú ý và làm theo chúng?
5. Đọc Mục 4 (Section 4) của bài báo `Reflexion` về thí nghiệm trên ALFWorld. Hãy tái hiện lại về mặt ý tưởng mức cải thiện 130% tỷ lệ thành công (`success-rate improvement`): đâu là sự khác biệt cốt lõi (`key delta`) so với `vanilla ReAct` (ReAct nguyên bản)?

## Thuật ngữ quan trọng (Key Terms)

| Thuật ngữ (Term) | Cách người ta thường gọi (What people say) | Bản chất thực sự (What it actually means) |
|------|----------------|------------------------|
| **Reflexion** | "Tự sửa lỗi" ("Self-correction") | Kiến trúc của Shinn và cộng sự (2023) — gồm `Actor`, `Evaluator`, `Self-Reflector` cộng với `episodic memory` |
| **Verbal reinforcement** | "Học không cần đạo hàm" ("Learning without gradients") | Bản suy ngẫm bằng ngôn ngữ tự nhiên (`natural-language reflection`) được gắn vào đầu `prompt` của lần thử tiếp theo |
| **Episodic memory** | "Bộ nhớ rút kinh nghiệm theo tác vụ" ("Per-task reflections") | Bộ đệm có giới hạn kích thước (`bounded buffer`) lưu các `reflections` trước đó của cùng một lớp tác vụ (`task class`) |
| **Scalar evaluator** | "Tín hiệu thành công nhị phân/điểm số" ("Binary success signal") | Kết quả `pass/fail` hoặc điểm số định lượng tính từ đáp án chuẩn (`ground truth`) |
| **Heuristic evaluator** | "Bộ phát hiện lỗi theo mẫu" ("Pattern-based detector") | Các dấu hiệu lỗi định sẵn (ví dụ: vòng lặp bị kẹt `stuck-loop`, vượt quá số bước tối đa `too-many-steps`) |
| **Self-evaluator** | "LLM tự chấm điểm vết chạy của mình" ("LLM-as-judge on own trace") | Giải pháp dự phòng có tín hiệu yếu hơn khi không có `ground truth` — nên kết hợp với kiểm chứng bằng công cụ (`tool-grounded verification`) |
| **Memory rot** | "Suy ngẫm bị cũ/lỗi thời" ("Stale reflections") | Bộ đệm `episodic buffer` bị lấp đầy bởi các mục lỗi thời; khắc phục bằng cách nén gọn (`compaction`) hoặc đặt thời hạn sống (`TTL`) |
| **Sleep-time reflection** | "Tự suy ngẫm bất đồng bộ" ("Async self-reflection") | Chạy `Self-Reflector` tách khỏi luồng xử lý chính (`off the hot path`) để `primary agent` luôn phản hồi nhanh |

## Đọc thêm (Further Reading)

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — bài báo khoa học nền tảng (`canonical paper`)
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — cơ chế suy ngẫm bất đồng bộ (`async reflection`) trong môi trường `production`
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — cách quản lý bộ đệm `episodic buffer` như một phần của kỹ thuật ngữ cảnh (`context engineering`)
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — mẫu thiết kế nút suy ngẫm (`reflection node pattern`)
