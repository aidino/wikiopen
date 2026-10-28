# ReWOO and Plan-and-Execute: Decoupled Planning (Lập kế hoạch tách rời thực thi)

> `ReAct` đan xen suy nghĩ (`thought`) và hành động (`action`) trong cùng một luồng xử lý (`stream`). `ReWOO` tách biệt hoàn toàn hai việc này: lập một kế hoạch tổng thể ngay từ đầu (`up front`), sau đó mới đem đi thực thi (`execute`). Kết quả: tốn ít hơn gấp 5 lần số `tokens`, tăng +4% độ chính xác (`accuracy`) trên bộ dữ liệu HotpotQA, và bạn có thể chưng cất (`distill`) bộ lập kế hoạch (`planner`) xuống một `model` nhỏ chỉ 7B tham số. `Plan-and-Execute` đã tổng quát hóa kiến trúc này; còn `Plan-and-Act` mở rộng quy mô (`scale`) của nó cho các tác vụ điều hướng web (`web navigation`).

- **Loại bài học (Type):** Build (Thực hành xây dựng)
- **Ngôn ngữ (Languages):** Python (`stdlib` — thư viện chuẩn)
- **Kiến thức tiên quyết (Prerequisites):** Phase 14 · 01 (Agent Loop)
- **Thời gian (Time):** ~60 phút

## Mục tiêu học tập (Learning Objectives)

- Giải thích tại sao việc tách thành 3 vai trò `Planner` / `Worker` / `Solver` của `ReWOO` lại giúp tiết kiệm `tokens` và tăng độ bền vững (`robustness`) so với vòng lặp đan xen (`interleaved loop`) của `ReAct`.
- Tự cài đặt một đồ thị kế hoạch `plan DAG` (Directed Acyclic Graph — Đồ thị có hướng không chu trình), một bộ thực thi theo thứ tự phụ thuộc (`dependency-ordered executor`), và một bộ giải (`solver`) tổng hợp kết quả từ các `worker` — tất cả chỉ dùng thư viện chuẩn (`stdlib`).
- Quyết định khi nào một tác vụ nên chạy theo kiểu lập-kế-hoạch-rồi-thực-thi (`plan-then-execute`) so với `ReAct` đan xen, dựa trên khung phân loại "5 mẫu workflow" (`five workflow patterns`) năm 2026 của Anthropic.
- Nhận biết khi nào cần đến dữ liệu kế hoạch tổng hợp (`synthetic plan data`) của `Plan-and-Act` cho các tác vụ dài hơi (`long-horizon tasks`) trên web hoặc thiết bị di động (`mobile`).

## Vấn đề (The Problem)

Vòng lặp đan xen `thought-action-observation` của `ReAct` tuy đơn giản và linh hoạt, nhưng mỗi lần gọi công cụ (`tool call`) đều phải mang theo toàn bộ ngữ cảnh trước đó (`full prior context`) — bao gồm tất cả những bước suy nghĩ (`thoughts`) cũ. Lượng `token` tiêu thụ tăng theo hàm bậc hai (`quadratically`) cùng với độ sâu của chuỗi bước chạy.
*(Giải thích: Bước 1 gửi 100 token; bước 2 gửi lại 100 token bước 1 + 100 token mới = 200; bước 10 phải gửi lại toàn bộ lịch sử bước 1..9, nên tổng số token cộng dồn qua N bước tỷ lệ thuận với $1 + 2 + \dots + N \approx O(N^2)$).*
Tệ hơn nữa: khi một `tool` bị lỗi ở giữa vòng lặp, `model` phải tự suy luận lại toàn bộ kế hoạch từ đầu dựa trên thông báo lỗi (`error observation`) đó.

`ReWOO` (*Reasoning Without Observation* — Suy luận không cần chờ kết quả quan sát, Xu và cộng sự, arXiv:2305.18323, tháng 5/2023) nhận ra điểm nghẽn này và đặt cược vào một hướng đi mới: **lên toàn bộ kế hoạch ngay từ đầu, thu thập dữ liệu bằng chứng (`evidence`) song song (`in parallel`), rồi tổng hợp câu trả lời ở bước cuối cùng.** Chỉ cần:
- **1 lần gọi LLM** để lập kế hoạch (`plan`),
- **N lần gọi `tool`** để lấy bằng chứng (có thể chạy song song),
- **1 lần gọi LLM** để giải quyết và tổng hợp kết quả (`solve`).

Sự đánh đổi (`trade-off`) ở đây là giảm tính linh hoạt (vì kế hoạch là tĩnh — `static plan`) để đổi lấy hiệu quả sử dụng `token` (`token efficiency`) vượt trội và khả năng khoanh vùng lỗi (`failure modes`) rõ ràng hơn rất nhiều.

## Khái niệm cốt lõi (The Concept)

### Ba vai trò tách biệt (The three roles)

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

- **`Planner` (Bộ lập kế hoạch):** Nhận câu hỏi của người dùng (`user_question`) và sinh ra một đồ thị `DAG` (`plan_dag`). Mỗi nút (`node`) trong đồ thị chỉ định tên một `tool`, các tham số (`arguments`) của nó, và những nút trước đó mà nó phụ thuộc vào thông qua các biến tham chiếu (`references` dạng placeholder như `#E1`, `#E2`).
- **`Workers` (Các bộ thực thi công cụ):** Chạy các nút trong `DAG` theo thứ tự sắp xếp tô-pô (`topological order` — nút nào không phụ thuộc vào ai thì chạy trước hoặc chạy song song; nút nào cần kết quả của `#E1` thì đợi `#E1` chạy xong rồi điền kết quả thật vào chỗ `#E1`).
- **`Solver` (Bộ tổng hợp kết quả):** Nhận câu hỏi ban đầu, bản kế hoạch `plan_dag`, cùng toàn bộ bằng chứng (`evidence`) thu được để ghép nối thành câu trả lời cuối cùng (`final_answer`).

### Tại sao tiết kiệm gấp 5 lần token (Why 5x fewer tokens)

Trong `ReAct`, độ dài `prompt` tăng tuyến tính theo số bước (`step count`). Ở bước thứ 10, `prompt` chứa `thought 1` + `action 1` + `observation 1` + `thought 2` + `action 2` + `observation 2`... cứ thế nối dài. Chưa kể mỗi bước trung gian đều phải gửi lặp lại phần `system prompt` và câu hỏi gốc một cách dư thừa (`redundantly`).

Ngược lại, `ReWOO` chỉ phải trả chi phí cho:
1. Một `planner prompt` (khá lớn),
2. N `worker prompts` nhỏ gọn (hoặc chỉ là các lời gọi hàm trực tiếp không tốn chuỗi hội thoại dài),
3. Một `solver prompt` cuối cùng.

Trên bộ dữ liệu hỏi đáp đa bước HotpotQA, bài báo đo đạc được `ReWOO` tốn **ít hơn ~5 lần `tokens`** trong khi đạt độ chính xác tuyệt đối cao hơn **+4 điểm phần trăm** so với `ReAct`.

### Tại sao kiến trúc này bền vững hơn (Why it is more robust)

Nếu `worker 3` gặp lỗi trong `ReAct`, vòng lặp phải cố gắng suy luận để thoát khỏi lỗi đó ngay giữa dòng chảy (`mid-stream`), rất dễ dẫn đến việc model bị rối hoặc đi chệch hướng. Trong `ReWOO`, nếu `worker 3` trả về một chuỗi báo lỗi (`error string`), `Solver` ở bước cuối sẽ nhìn thấy lỗi đó đặt cạnh bản kế hoạch gốc (`original plan`) cùng các bằng chứng khác đã lấy thành công, từ đó có thể đưa ra câu trả lời giảm nhẹ hợp lý (`degrade gracefully` — trả lời phần làm được và nêu rõ phần thiếu). Việc khoanh vùng lỗi (`failure localization`) diễn ra độc lập trên từng nút (`per-node`), thay vì làm hỏng cả chuỗi bước (`per-step`).

### Chưng cất bộ lập kế hoạch (Planner distillation)

Kết quả quan trọng thứ hai của bài báo: bởi vì `Planner` không cần nhìn thấy kết quả thực thi trung gian (`observations`), bạn có thể tinh chỉnh (`fine-tune`) một `model` nhỏ cỡ 7B tham số dựa trên các bản kế hoạch đầu ra (`planner outputs`) được sinh bởi một `model` thầy (`teacher model`) khổng lồ cỡ 175B. Khi chạy thực tế (`at inference`), `model` nhỏ đảm nhận việc lập kế hoạch, không cần tốn chi phí gọi `model` lớn nữa. Đây hiện là tiêu chuẩn thiết kế phổ biến: rất nhiều `production agents` năm 2026 kết hợp một `small planner` với một `big executor` (hoặc ngược lại: model suy luận lớn lập kế hoạch một lần, rồi giao cho các model nhỏ/rẻ thực thi từng bước).

### Plan-and-Execute (2023)

Bài viết tháng 8/2023 của đội ngũ LangChain đã tổng quát hóa `ReWOO` thành một mẫu thiết kế (`pattern`) mang tên **Plan-and-Execute**:
- Bộ `Planner` ở đầu vào phát ra một danh sách các bước (`step list`),
- Bộ `Executor` chạy lần lượt từng bước,
- Một bộ lập kế hoạch lại tùy chọn (`optional replanner`) có thể điều chỉnh lại các bước còn lại sau khi quan sát kết quả trung gian.

Hướng tiếp cận này gần với `ReAct` hơn so với `ReWOO` thuần túy (vì `replanner` đưa `observations` quay trở lại quá trình lập kế hoạch), nhưng vẫn giữ được lợi thế tiết kiệm `token` đáng kể.

### Plan-and-Act (Erdogan và cộng sự, arXiv:2503.09572, ICML 2025)

`Plan-and-Act` mở rộng quy mô (`scale`) của mẫu thiết kế này cho các `web and mobile agents` thực hiện tác vụ dài hơi (`long-horizon`). Đóng góp cốt lõi của nghiên cứu này là **dữ liệu kế hoạch tổng hợp (`synthetic plan data`)**: một bộ sinh quỹ đạo có gán nhãn (`labeled trajectory generator`) tạo ra dữ liệu huấn luyện trong đó bản kế hoạch được tách bạch rõ ràng (`explicit`). Dữ liệu này được dùng để `fine-tune` các `planner models` có khả năng duy trì sự mạch lạc vượt qua 30–50 bước trên các tác vụ kiểu WebArena — nơi mà một quỹ đạo `ReAct` đơn lẻ thường bị mất phương hướng (`loses coherence`).

### Khi nào nên chọn mẫu thiết kế nào (When to pick which)

| Mẫu thiết kế (Pattern) | Khi nào nên sử dụng (When) |
|---------|------|
| **ReAct** | Tác vụ ngắn (`short tasks`), môi trường chưa biết trước (`unknown environment`), cần xử lý ngoại lệ tức thì (`reactive exception handling`) |
| **ReWOO** | Tác vụ có cấu trúc rõ ràng với bộ công cụ đã biết (`known tools`), nhạy cảm về chi phí/số lượng `token` (`token-sensitive`), có thể thu thập bằng chứng song song (`parallelizable evidence`) |
| **Plan-and-Execute** | Giống `ReWOO` nhưng cần khả năng lập lại kế hoạch (`replanning`) sau khi thực thi một phần công việc |
| **Plan-and-Act** | Tác vụ dài hơi (`long-horizon` > 30 bước), điều khiển trình duyệt web / ứng dụng mobile / `computer-use` |
| **Tree of Thoughts** | Bài toán khó mà việc bỏ chi phí tìm kiếm nhiều nhánh suy luận (`search`) là xứng đáng (Lesson 04) |

**Lời khuyên từ Anthropic (tháng 12/2024):** Hãy luôn bắt đầu từ kiến trúc đơn giản nhất. Nếu tác vụ chỉ gồm 1 lệnh gọi `tool` cộng với bước tóm tắt, đừng phức tạp hóa bằng `ReWOO`. Ngược lại, nếu tác vụ là một bài nghiên cứu sâu gồm 40 bước, đừng chỉ dùng mỗi một vòng lặp `ReAct` đơn thuần.

```figure
rewoo-plan
```

## Thực hành xây dựng (Build It)

File `code/main.py` cài đặt một mô hình `ReWOO` thu nhỏ (`toy ReWOO`):

- `Planner` — một chính sách theo kịch bản (`scripted policy`) sinh ra đồ thị `plan DAG` từ câu `prompt`.
- `Worker` — điều phối (`dispatch`) lệnh gọi `tool` của từng nút thông qua `tool registry`.
- `Solver` — bước tổng hợp đọc toàn bộ bằng chứng (`evidence`) và tạo ra câu trả lời cuối cùng (`final answer`).
- Giải quyết phụ thuộc (`Dependency resolution`) — các biến tham chiếu như `#E1` được tự động thay thế (`substituted`) bằng kết quả đầu ra thực tế của `worker` chạy trước đó.

Bản demo trả lời câu hỏi *"Dân số của thủ đô nước Pháp là bao nhiêu, làm tròn đến hàng triệu?"* ("What is the population of the capital of France, rounded to millions?") bằng một kế hoạch gồm 2 bước:
1. Tra cứu thủ đô của Pháp (`#E1`),
2. Tra cứu dân số của thành phố tìm được ở `#E1` (`#E2`),
sau đó đưa vào `Solver` để giải.

Chạy thử chương trình:

```
python3 code/main.py
```

Bản ghi vết (`trace`) sẽ hiển thị toàn bộ bản kế hoạch trước tiên, tiếp theo là kết quả của từng `worker`, và cuối cùng là bước tổng hợp của `solver`. Hãy so sánh số lượng `token` (chương trình có in ra số lượng ký tự ước tính) với cách chạy đan xen kiểu `ReAct` — `ReWOO` chiến thắng rõ rệt trên dạng tác vụ có cấu trúc như thế này.

## Ứng dụng thực tế (Use It)

LangGraph cung cấp sẵn `Plan-and-Execute` như một công thức chuẩn (`recipe` — dùng `create_react_agent` cho `ReAct`, và xây dựng đồ thị tùy chỉnh `custom graphs` cho `plan-execute`). Tính năng `Flows` của CrewAI cũng mã hóa trực tiếp mẫu thiết kế này: bạn định nghĩa các tác vụ ngay từ đầu và đồ thị `Flow DAG` sẽ thực thi chúng. Hướng tiếp cận dùng `synthetic data` của `Plan-and-Act` hiện vẫn chủ yếu nằm ở mảng nghiên cứu, nhưng mẫu kiến trúc thực thi (`runtime pattern` — tức `explicit plan DAG`) đã được ứng dụng rộng rãi trong `production` thông qua LangGraph và CrewAI Flows.

## Đóng gói kỹ năng (Ship It)

File `outputs/skill-rewoo-planner.md` giúp sinh ra một đồ thị kế hoạch `ReWOO plan DAG` từ yêu cầu của người dùng dựa trên danh mục công cụ (`tool catalog`) cho trước. Nó tự động kiểm tra tính hợp lệ (`validate`) của bản kế hoạch (đảm bảo đồ thị không có chu trình lặp `acyclic`, mọi tham chiếu `#E...` đều được giải quyết, và mọi `tool` được gọi đều tồn tại) trước khi bàn giao cho bộ thực thi (`executor`).

## Bài tập rèn luyện (Exercises)

1. Song song hóa (`parallelize`) quá trình thực thi `worker` cho các nút kế hoạch độc lập (không phụ thuộc lẫn nhau). Điều này mang lại lợi ích gì về thời gian chạy (`latency`) trên một đồ thị `DAG` gồm 6 nút được chia làm 2 nhóm song song?
2. Thêm một nút lập lại kế hoạch (`replanner node`) chỉ kích hoạt nếu có bất kỳ `worker` nào trả về lỗi. Thay đổi nhỏ nhất đối với `ReWOO` để biến nó thành `Plan-and-Execute` là gì?
3. Thay thế `Planner` bằng một `model` nhỏ (nhóm 7B tham số) và giữ `Solver` trên một `frontier model` (model lớn hàng đầu). So sánh chất lượng đầu-cuối (`end-to-end quality`) — kiến trúc tách đôi này thường thất bại ở điểm nào?
4. Đọc Mục 4 (Section 4) của bài báo `ReWOO` về chưng cất bộ lập kế hoạch (`planner distillation`). Hãy tái hiện lại về mặt ý tưởng kết quả chuyển giao từ `175B -> 7B`: bạn cần dữ liệu huấn luyện (`training data`) gì, và làm thế nào để chấm điểm chất lượng của một bản kế hoạch (`plan quality`)?
5. Chuyển đổi (`port`) bản code mẫu sang hình thái quỹ đạo (`trajectory shape`) của `Plan-and-Act`: trong đó kế hoạch là một chuỗi tuần tự (`sequence`) chứ không phải đồ thị `DAG`. Những sự đánh đổi (`tradeoffs`) nào sẽ thay đổi?

## Thuật ngữ quan trọng (Key Terms)

| Thuật ngữ (Term) | Cách người ta thường gọi (What people say) | Bản chất thực sự (What it actually means) |
|------|----------------|------------------------|
| **ReWOO** | "Suy luận không cần quan sát" ("Reasoning without observations") | Lập kế hoạch trước, sau đó thu thập bằng chứng song song, rồi mới giải quyết — không đưa `observations` vào trong `planning prompt` |
| **Plan-and-Execute** | "Mẫu plan-execute của LangChain" ("LangChain's plan-execute pattern") | Kiến trúc `ReWOO` có gắn thêm một nút `replanner` tùy chọn sau bước thực thi |
| **Plan-and-Act** | "Plan-execute mở rộng quy mô" ("Scaled plan-execute") | Tách biệt rõ `planner`/`executor` kết hợp dữ liệu huấn luyện kế hoạch tổng hợp (`synthetic plan training data`) cho các tác vụ dài hơi (`long-horizon tasks`) |
| **Evidence reference** | "`#E1`, `#E2`, ..." | Biến giữ chỗ (`placeholder`) ở nút kế hoạch, sẽ được thay thế bằng kết quả thực tế của `worker` trước đó ngay tại thời điểm điều phối (`dispatch time`) |
| **Planner distillation** | "Planner nhỏ, Executor lớn" ("Small planner, big executor") | Tinh chỉnh (`fine-tune`) một `model` nhỏ dựa trên các vết lập kế hoạch (`planner traces`) sinh ra từ một `teacher model` lớn |
| **Token efficiency** | "Ít vòng gọi qua lại hơn" ("Fewer round trips") | Tiết kiệm gấp 5 lần số `tokens` trên bộ dữ liệu HotpotQA so với `ReAct` theo kết quả bài báo |
| **DAG executor** | "Bộ điều phối tô-pô" ("Topological dispatcher") | Chạy các nút trong kế hoạch theo đúng thứ tự phụ thuộc (`dependency order`); các nút cùng cấp không phụ thuộc nhau được chạy song song |

## Đọc thêm (Further Reading)

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — bài báo khoa học nền tảng (`canonical paper`)
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — mở rộng quy mô `planner-executor` bằng dữ liệu kế hoạch tổng hợp (`synthetic plans`)
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview) — công thức cài đặt trên `framework` LangGraph
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — nguyên tắc chọn mẫu thiết kế đơn giản nhất mà giải quyết được việc
