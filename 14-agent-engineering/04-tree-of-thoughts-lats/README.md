# Tree of Thoughts and LATS: Deliberate Search (Tìm kiếm có chủ đích trên cây suy luận)

> Một quỹ đạo suy luận chuỗi đơn lẻ (`single chain-of-thought trajectory`) không hề có cơ chế quay lui (`backtrack`) khi đi nhầm hướng. `ToT` (`Tree of Thoughts` — Yao và cộng sự, 2023) biến quá trình suy luận thành một cấu trúc cây (`tree`) với bước tự đánh giá (`self-evaluation`) tại mỗi nút (`node`). `LATS` (*Language Agent Tree Search* — Zhou và cộng sự, 2024) hợp nhất `ToT` với `ReAct` và `Reflexion` dưới thuật toán tìm kiếm cây Monte Carlo (`Monte Carlo Tree Search` — `MCTS`). Kết quả: trò chơi toán học Game of 24 tăng vọt tỷ lệ giải đúng từ 4% (`CoT`) lên 74% (`ToT`); còn `LATS` đạt tới 92.7% `pass@1` trên bộ kiểm thử sinh code HumanEval.

- **Loại bài học (Type):** Build (Thực hành xây dựng)
- **Ngôn ngữ (Languages):** Python (`stdlib` — thư viện chuẩn)
- **Kiến thức tiên quyết (Prerequisites):** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
- **Thời gian (Time):** ~75 phút

## Mục tiêu học tập (Learning Objectives)

- Định hình bài toán suy luận (`reasoning`) dưới dạng bài toán tìm kiếm (`search`): các nút (`nodes`) là những "ý nghĩ" (`thoughts`), các cạnh nối (`edges`) là những bước "mở rộng" (`expansions`), và giá trị (`value`) thể hiện "mức độ hứa hẹn" (`how promising`) của hướng đi đó.
- Tự cài đặt thuật toán tìm kiếm cây kiểu `ToT` theo chiều rộng (`BFS` — Breadth-First Search) chỉ dùng thư viện chuẩn (`stdlib`) kèm cơ chế chấm điểm tự đánh giá (`self-evaluation scoring`).
- Mở rộng thành vòng lặp `MCTS` thu nhỏ theo kiến trúc `LATS` với 4 pha: chọn lọc (`select`) / mở rộng (`expand`) / mô phỏng (`simulate`) / lan truyền ngược (`backpropagate`).
- Quyết định khi nào việc tìm kiếm (`search`) xứng đáng với chi phí `token` tăng gấp nhiều lần (`token multiplier` — ví dụ: Game of 24, sinh code phức tạp `code generation`) và khi nào chỉ cần một quỹ đạo đơn lẻ (`single trajectory`) là đủ (ví dụ: hỏi đáp đơn giản `simple Q&A`).

## Vấn đề (The Problem)

Suy luận chuỗi (`Chain-of-thought` — `CoT`) chỉ là một bước đi tuyến tính một chiều (`linear walk`). Nếu ngay ở bước đầu tiên mô hình đã rẽ sai hướng, mọi bước tiếp theo đều sẽ xây dựng dựa trên một tiền đề sai lầm (`bad premise`). Trong trò chơi **Game of 24** (dùng 4 chữ số cho trước kết hợp với các phép tính `+ − × ÷` sao cho ra kết quả bằng 24), `GPT-4` khi chạy `CoT` chỉ đạt độ chính xác vỏn vẹn **4%**. Mô hình chọn nhầm một biểu thức con (`subexpression`) ngay từ sớm và không có cách nào quay đầu sửa lại (`cannot recover`).

Điều mà quá trình suy luận thực sự cần là khả năng đề xuất nhiều phương án ứng viên (`propose multiple candidates`), đánh giá chúng (`evaluate`), chọn ra những hướng đi có triển vọng (`promising ones`), và quay lui (`backtrack`) ngay khi gặp ngõ cụt (`dead ends`). Đó chính là **tìm kiếm (`search`)**. `Tree of Thoughts` và `LATS` là hai công thức tiêu chuẩn (`canonical formulations`) hiện thực hóa ý tưởng này.

> **Giải thích dễ hiểu:** Hãy tưởng tượng bạn đang đi trong mê cung. `Chain-of-Thought (CoT)` giống như việc bạn bịt mắt đi thẳng một mạch và cấm được quay đầu — chỉ cần rẽ nhầm ở ngã ba đầu tiên là bạn kẹt cứng. Ngược lại, `Tree of Thoughts (ToT)` và `LATS` cho phép bạn đứng ở mỗi ngã ba, nhìn thử cả 3 lối rẽ (`candidates`), chấm điểm xem lối nào sáng sủa nhất (`evaluate`), đi thử vào đó, và nếu thấy tường chắn (`dead end`) thì quay lại ngã ba cũ để đi nhánh khác (`backtrack`).

## Khái niệm cốt lõi (The Concept)

### Tree of Thoughts (Yao và cộng sự, NeurIPS 2023)

Mỗi nút (`node`) trên cây là một bước suy luận trung gian mạch lạc (một "ý nghĩ" — `"a thought"`). Từ mỗi nút có thể mở rộng (`expand`) ra $K$ nút con (`child thoughts`). Bản thân `LLM` sẽ tự đánh giá (`self-evaluates`) từng nút bằng một `scoring prompt` (prompt chấm điểm). Thuật toán tìm kiếm sẽ duyệt qua cây này — sử dụng tìm kiếm theo chiều rộng (`BFS` — Breadth-First Search), tìm kiếm theo chiều sâu (`DFS` — Depth-First Search), hoặc tìm kiếm chùm (`beam search`).

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

**Tự đánh giá (`Self-evaluation`)** chính là trụ cột chịu lực quan trọng nhất (`load-bearing piece`) của toàn bộ kiến trúc. Bài báo trình bày 3 biến thể chấm điểm:
1. Phân loại trạng thái thành `sure / likely / impossible` (chắc chắn giải được / có khả năng / bất khả thi),
2. Chấm điểm số từ `1..10` (`numeric score`),
3. Bầu chọn (`vote`) giữa các nhánh ứng viên.

Cả ba cách đều đánh bại `CoT` một cách áp đảo trên bài toán Game of 24 (tăng từ **4% -> 74%** với `GPT-4`).

### LATS (Zhou và cộng sự, ICML 2024)

`LATS` (*Language Agent Tree Search*) hợp nhất cả ba kỹ thuật `ToT`, `ReAct`, và `Reflexion` dưới khung thuật toán tìm kiếm cây Monte Carlo (`MCTS`). Trong đó, `LLM` đóng cùng lúc ba vai trò:

- **`Policy` (Chính sách sinh hành động):** Đề xuất các hành động ứng viên tiếp theo theo phong cách `ReAct` (`ReAct-style`).
- **`Value function` (Hàm giá trị):** Chấm điểm một quỹ đạo chưa hoàn chỉnh (`partial trajectory`) bằng cách tự đánh giá theo phong cách `ToT` (`ToT-style self-eval`).
- **`Self-reflector` (Bộ tự suy ngẫm):** Khi một nhánh bị thất bại, viết một bản rút kinh nghiệm bằng ngôn ngữ tự nhiên theo phong cách `Reflexion` (`Reflexion-style`) và dùng nó làm dữ liệu mồi (`reseed`) định hướng cho các lượt chạy thử (`rollouts`) trong tương lai.

Phản hồi thực tế từ môi trường (`environment feedback` / `observations`) được kết hợp thẳng vào hàm giá trị (`value function`), nhờ đó quá trình tìm kiếm được dẫn dắt bởi kết quả chạy công cụ thực tế (`real tool results`) chứ không chỉ dựa vào "ý kiến chủ quan" của mô hình (`model opinions`). Kết quả tại thời điểm công bố bài báo: đạt **92.7% `pass@1`** trên HumanEval với `GPT-4` (thiết lập kỷ lục `SOTA`), và đạt điểm trung bình **75.9** trên môi trường mua sắm WebShop với `GPT-3.5` (tiệm cận hiệu năng của các mô hình được tinh chỉnh bằng đạo hàm `gradient-based fine-tuning`).

### Thuật toán MCTS ở mức tối giản (MCTS, minimally)

Mỗi vòng lặp (`iteration`) của `MCTS` trải qua bốn pha:

1. **`Select` (Chọn lọc)** — Đi từ nút gốc (`root`) xuống một nút lá (`leaf`) bằng công thức `UCT` (*Upper Confidence Bound for Trees* — Cận trên khoảng tin cậy cho cây).
2. **`Expand` (Mở rộng)** — Sinh ra $K$ nút con (`children`) từ nút lá đó thông qua `policy`.
3. **`Simulate` (Mô phỏng / Rollout)** — Chạy thử (`rollout`) từ một nút con đi tiếp bằng `policy`, sau đó chấm điểm nút lá kết thúc bằng hàm giá trị (`value function`) hoặc phần thưởng từ môi trường (`environment reward`).
4. **`Backpropagate` (Lan truyền ngược)** — Cập nhật số lần ghé thăm (`visit counts`) và ước lượng giá trị (`value estimates`) ngược lên toàn bộ các nút tổ tiên nằm trên đường đi.

Công thức `UCT`:
`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`

- Số hạng đầu tiên `Q(s, a)` là **khai thác (`exploitation`)** — ưu tiên chọn nhánh đã có điểm trung bình cao trong quá khứ.
- Số hạng thứ hai `c * sqrt(ln N(s) / N(s, a))` là **khám phá (`exploration`)** — ưu tiên thử những nhánh ít được ghé thăm (khi mẫu số `N(s, a)` nhỏ thì giá trị biểu thức căn sẽ lớn).
- Hệ số `c` được tinh chỉnh (`tune`) tùy theo đặc thù của từng tác vụ.

### Thực tế về chi phí (The cost reality)

Tìm kiếm cây làm bùng nổ số lượng `token` (`Search explodes tokens`). `ToT` khi giải Game of 24 tiêu tốn lượng `token` **gấp 100–1000 lần** so với `CoT`. `LATS` cũng tương tự. Kỹ thuật này hoàn toàn không miễn phí; chỉ nên dành riêng `search` cho:

- Các tác vụ mà một quỹ đạo đơn lẻ (`single trajectory`) chắc chắn không đủ khả năng giải quyết (ví dụ: Game of 24, viết code thuật toán phức tạp).
- Các tác vụ mà thời gian chờ đợi thực tế (`wall-clock time`) ít quan trọng hơn độ chính xác tuyệt đối (`correctness`).
- Các tác vụ có sẵn một hàm giá trị (`value function`) rẻ và đáng tin cậy (ví dụ: chạy bộ `unit tests` để kiểm tra code, hoặc có công thức kiểm tra kết quả mục tiêu rõ ràng trong toán học).

Nếu tác vụ của bạn chỉ có một đáp án đúng duy nhất nhưng bộ chấm điểm lại nhiễu/thiếu chính xác (`noisy evaluator`), việc chạy tìm kiếm (`search`) thường làm mọi thứ **tệ đi** — nó sẽ lục sục khắp cây và tìm ra một câu trả lời sai nhưng lại vô tình được bộ chấm điểm cho điểm cao (`"good-scoring" wrong answer`).
*(Giải thích: Hiện tượng này tương tự như `reward hacking` — khi bạn thử hàng trăm phương án và dùng một giám khảo chấm điểm thiếu chính xác để lọc, phương án chiến thắng thường là phương án "đánh lừa" được giám khảo tốt nhất chứ không phải phương án thực sự đúng).*

### Vị thế trong thực tế năm 2026 (2026 positioning)

Hầu hết các `production agents` trong thực tế không chạy `LATS` toàn phần. Thay vào đó, chúng chạy `ReAct` kết hợp với cơ chế kiểm chứng dựa trên công cụ (`tool-grounded verification` — mô hình `CRITIC`, sẽ học ở Lesson 05). Tìm kiếm cây (`search`) chủ yếu xuất hiện trong các ngách chuyên biệt (`specialized niches`):

- **`Coding agents`** sử dụng trực tiếp kết quả chạy `tests` làm hàm giá trị (`value function`, theo phong cách HumanEval).
- **`Deep-research agents`** cần khám phá nhiều nhánh truy vấn tìm kiếm (`multiple query paths`) song song.
- **Các luồng công việc nặng về lập kế hoạch (`planning-heavy workflows`)** nằm bên trong các đồ thị con (`subgraphs`) của LangGraph.

**AlphaEvolve (Lesson 11)** là ví dụ cực hạn của hướng đi này vào năm 2025: tìm kiếm tiến hóa (`evolutionary search`) trên không gian code, dùng hàm mục tiêu kiểm chứng được bằng máy (`machine-checkable fitness`), tạo ra những đột phá ở biên giới khoa học (như cải tiến thuật toán nhân ma trận $4 \times 4$ lần đầu tiên sau 56 năm).

```figure
tree-of-thoughts
```

## Thực hành xây dựng (Build It)

File `code/main.py` cài đặt:

- Một thuật toán tìm kiếm `ToT BFS` thu nhỏ trên bài toán cách điệu hóa: "chọn các phép toán số học" (`"pick arithmetic ops"`).
- Một vòng lặp `LATS MCTS` thu nhỏ trên cùng bài toán đó (`Select` / `Expand` / `Simulate` / `Backpropagate`) sử dụng công thức chọn lọc `UCT`.
- Một hàm giá trị (`value function`) kết hợp giữa điểm số ký hiệu toán học (`symbolic score`) cộng với điểm tự đánh giá (`self-eval score`).

Chạy thử chương trình:

```
python3 code/main.py
```

Bản ghi vết (`trace`) sẽ hiển thị quá trình `ToT` mở rộng 3 nhánh ứng viên cho mỗi nút bằng `BFS`, đặt cạnh quá trình `LATS` hội tụ (`converging`) về nhánh chạy thử (`rollout`) tốt nhất thông qua `MCTS`. Số lượng `token` tiêu thụ của cả hai phương pháp đều được in ra để so sánh.

## Ứng dụng thực tế (Use It)

LangGraph cung cấp sẵn kiểu khám phá `ToT-style exploration` dưới dạng các mẫu đồ thị con (`subgraph patterns`); bài blog của đội ngũ LangChain viết về `LATS` (tháng 5/2024) là tài liệu hướng dẫn tham khảo chuẩn (`reference tutorial`). LlamaIndex cũng cung cấp sẵn một agent `TreeOfThoughts`. Đối với hầu hết các `production agents` năm 2026, mẫu thiết kế này thường nằm sau một cổng điều kiện (`gate`) dạng `if task_complexity > threshold: use_search()` (nếu độ phức tạp tác vụ vượt ngưỡng thì mới bật tìm kiếm) — xem thêm mẫu thiết kế `evaluator-optimizer` ở Lesson 05.

## Đóng gói kỹ năng (Ship It)

File `outputs/skill-search-policy.md` đóng vai trò bộ quy tắc giúp lựa chọn giữa `linear ReAct` (ReAct tuyến tính), `ToT`, `LATS`, và `evolutionary search` (tìm kiếm tiến hóa) dựa trên hình thái tác vụ (`task shape`), ngân sách `token` (`budget`), và độ tin cậy của bộ đánh giá (`evaluator fidelity`).

## Bài tập rèn luyện (Exercises)

1. Chạy mô hình `toy LATS` với hệ số `UCT` là `c=0.1` so với `c=2.0`. Điều gì thay đổi trong bản ghi vết (`trace`)? *(Gợi ý: quan sát sự đánh đổi giữa việc đào sâu vào một nhánh điểm cao sớm so với việc chia đều lượt thử cho các nhánh chưa khám phá).*
2. Thay thế hàm giá trị (`value function`) bằng một bộ chấm điểm nhiễu hơn (`noisier scorer` — cộng thêm nhiễu ngẫu nhiên `random jitter`). Liệu `MCTS` có còn tìm ra được nút lá tốt nhất (`best leaf`) không? Tỷ lệ tín hiệu trên nhiễu (`signal-to-noise ratio`) tối thiểu mà thuật toán có thể chịu đựng được là bao nhiêu?
3. Cài đặt biến thể `beam-search ToT` (chỉ giữ lại `top-k` nút điểm cao nhất ở mỗi tầng của cây) và so sánh với `BFS`. Phương pháp nào hoạt động tốt hơn khi ngân sách `token` bị giới hạn chặt (`tight token budget`)?
4. Đọc Mục 5.1 (Section 5.1) của bài báo `LATS`. Tái hiện lại số lượng quỹ đạo (`trajectory count`) trên tập HumanEval: cần bao nhiêu lượt chạy thử (`rollouts`) để đạt được con số `pass@1` như báo cáo?
5. Đọc phần thảo luận của bài báo `LATS` về chủ đề "khi nào LATS ít mang lại hiệu quả hơn" (*"when LATS helps less"*). Viết một đoạn quy tắc quyết định (`decision rule`) dài một đoạn văn ánh xạ từ hình thái tác vụ (`task shape`) sang chiến lược tìm kiếm (`search strategy`) phù hợp.

## Thuật ngữ quan trọng (Key Terms)

| Thuật ngữ (Term) | Cách người ta thường gọi (What people say) | Bản chất thực sự (What it actually means) |
|------|----------------|------------------------|
| **Tree of Thoughts** | "CoT phân nhánh" ("Branching CoT") | Kiến trúc của Yao và cộng sự — cây gồm các nút suy nghĩ (`thought nodes`) đi kèm cơ chế tự đánh giá (`self-evaluation`) |
| **LATS** | "MCTS dành cho LLM" ("MCTS for LLMs") | Kiến trúc của Zhou và cộng sự — hợp nhất `ToT` + `ReAct` + `Reflexion` dưới thuật toán `MCTS` |
| **UCT** | "Cận trên khoảng tin cậy" ("Upper confidence bound") | Công thức chọn nhánh (`select formula`) cân bằng giữa khai thác (`exploitation` — `Q`) và khám phá (`exploration` — `ln N / n`) |
| **Value function** | "Trạng thái này tốt đến đâu" ("How good is this state") | Điểm số do `LLM` chấm qua `prompt` hoặc phần thưởng từ môi trường (`environment reward`); dùng để đưa vào bước `backprop` |
| **Policy** | "Bộ đề xuất hành động" ("Action proposer") | Bộ sinh theo kiểu `ReAct` (`ReAct-style generator`); phát ra các suy nghĩ hoặc hành động ứng viên tiếp theo |
| **Rollout** | "Quỹ đạo mô phỏng" ("Simulated trajectory") | Đi từ một nút xuống tới nút lá bằng `policy`, sau đó chấm điểm bằng `value function` |
| **Backpropagate** | "Cập nhật các nút tổ tiên" ("Update ancestors") | Đẩy phần thưởng (`reward`) của nút lá ngược lên theo đường đi, cập nhật số lượt thăm (`visit counts`) và giá trị `Q` |
| **Search cost** | "Sự bùng nổ token" ("Token explosion") | Tốn gấp 100–1000 lần `CoT` trên bài toán Game of 24; phải tính toán kỹ ngân sách (`budget`) trước khi áp dụng |

## Đọc thêm (Further Reading)

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — bài báo khoa học nền tảng (`canonical paper`) về `Tree of Thoughts`
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) — kết hợp `MCTS` với phản hồi suy ngẫm của `Reflexion` (`Reflexion feedback`)
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — các mẫu đồ thị con (`subgraph patterns`) phục vụ bài toán tìm kiếm
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — tìm kiếm tiến hóa (`evolutionary search`) kết hợp với các bộ đánh giá bằng chương trình (`programmatic evaluators`)
