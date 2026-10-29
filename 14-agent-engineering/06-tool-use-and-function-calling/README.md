# Tool Use and Function Calling (Sử dụng công cụ và Gọi hàm)

> `Toolformer` (Schick và cộng sự, 2023) đã khởi xướng kỹ thuật tự giám sát gán nhãn công cụ (`self-supervised tool annotation`). Bảng xếp hạng **Berkeley Function Calling Leaderboard V4** (`BFCL V4` — Patil và cộng sự, 2025) thiết lập chuẩn đánh giá cho năm 2026: 40% tác vụ tự chủ (`agentic`), 30% hội thoại đa lượt (`multi-turn`), 10% dữ liệu thực tế (`live`), 10% dữ liệu tổng hợp (`non-live`), và 10% kiểm tra ảo giác (`hallucination`). Bài toán gọi hàm đơn lượt (`single-turn`) coi như đã được giải quyết xong. Nhưng bộ nhớ (`memory`), ra quyết định động (`dynamic decision-making`), và chuỗi gọi công cụ dài hơi (`long-horizon tool chains`) thì vẫn chưa.

- **Loại bài học (Type):** Build (Thực hành xây dựng)
- **Ngôn ngữ (Languages):** Python (`stdlib` — thư viện chuẩn)
- **Kiến thức tiên quyết (Prerequisites):** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
- **Thời gian (Time):** ~60 phút

## Mục tiêu học tập (Learning Objectives)

- Giải thích tín hiệu huấn luyện tự giám sát (`self-supervised training signal`) của `Toolformer`: chỉ giữ lại các nhãn gọi công cụ (`tool annotations`) khi kết quả thực thi của chúng giúp làm giảm hàm mất mát dự đoán từ tiếp theo (`next-token loss`).
- Kể tên 5 hạng mục đánh giá (`evaluation categories`) của `BFCL V4` và ý nghĩa đo lường của từng hạng mục.
- Tự cài đặt một sổ đăng ký công cụ (`tool registry`) chỉ dùng thư viện chuẩn (`stdlib`) với đầy đủ cơ chế kiểm tra lược đồ (`schema validation`), ép kiểu đối số (`argument coercion`), và cô lập môi trường thực thi (`execution sandboxing`).
- Chẩn đoán 3 bài toán mở (`open problems`) của năm 2026: chuỗi gọi công cụ dài hơi (`long-horizon tool chaining`), ra quyết định động (`dynamic decision-making`), và quản lý bộ nhớ (`memory`).

## Vấn đề (The Problem)

Thế hệ `tool use` đời đầu chỉ đặt câu hỏi: *liệu mô hình có dự đoán đúng một lệnh gọi hàm (`function call`) đơn lẻ hay không?*
Thế hệ `tool use` hiện đại đặt câu hỏi khó hơn nhiều: *liệu mô hình có thể xâu chuỗi các công cụ (`chain tools`) xuyên suốt 40 bước, có duy trì bộ nhớ (`memory`), trong môi trường chỉ quan sát được một phần (`partial observability`), có khả năng tự phục hồi khi công cụ báo lỗi (`recovery from tool failures`), mà không bịa ra (`hallucinating`) những công cụ không hề tồn tại hay không?*

`Toolformer` đã đặt nền móng cơ sở (`baseline`): các mô hình có thể tự học *khi nào* cần gọi công cụ thông qua cơ chế tự giám sát (`self-supervision`). Còn `BFCL V4` định nghĩa mục tiêu đánh giá của năm 2026. Khoảng cách giữa hai cột mốc này chính là không gian mà các `production agents` (agent chạy thực tế) đang phải giải quyết hàng ngày.

## Khái niệm cốt lõi (The Concept)

### Toolformer (Schick và cộng sự, NeurIPS 2023)

**Ý tưởng cốt lõi:** Để chính mô hình tự chú thích (`annotate`) vào tập dữ liệu tiền huấn luyện (`pretraining corpus`) của nó bằng các lệnh gọi `API` ứng viên (`candidate API calls`). Với mỗi lệnh gọi ứng viên, hệ thống thực thi chạy thử nó thật. **Chỉ giữ lại chú thích gọi `tool` đó nếu việc chèn kết quả của `tool` vào văn bản giúp làm giảm độ lệch dự đoán các từ tiếp theo (`reduces loss on the next token`).** Sau đó, tinh chỉnh (`fine-tune`) lại mô hình trên tập dữ liệu đã được lọc này.

> **Giải thích dễ hiểu:** Giả sử trong văn bản gốc có câu: *"145 nhân 38 bằng 5510"*. Mô hình thử chèn một lệnh gọi máy tính `[Calculator(145 * 38) -> 5510]` vào trước số `5510`. Khi thấy kết quả `5510` từ máy tính trả về giúp việc đoán chữ `"5510"` tiếp theo trong câu trở nên dễ dàng và chắc chắn hơn hẳn (giảm `next-token loss`), mô hình hiểu rằng: *"À, ở vị trí này gọi `Calculator` là có ích!"* Ngược lại, nếu chèn `Calculator` vào giữa câu *"Hôm nay trời đẹp quá"* mà không giúp đoán chữ *"quá"` tốt hơn, chú thích đó sẽ bị vứt bỏ. Nhờ vậy, mô hình tự học được cách dùng công cụ mà **không cần con người gán nhãn (`no human labels`)**.

Các công cụ được thử nghiệm trong bài báo: máy tính (`calculator`), hệ thống hỏi đáp (`QA system`), công cụ tìm kiếm (`search engines`), trình dịch thuật (`translator`), và lịch (`calendar`). Tín hiệu tự giám sát hoàn toàn dựa trên việc công cụ đó có giúp dự đoán văn bản tiếp theo tốt hơn hay không.

**Kết quả theo quy mô mô hình (`Scale result`):** Khả năng sử dụng công cụ chỉ thực sự trỗi dậy khi mô hình đạt quy mô đủ lớn (`emerges at scale`). Các mô hình quá nhỏ thậm chí còn bị giảm hiệu năng khi thêm `tool annotations`, trong khi các mô hình lớn hưởng lợi rõ rệt. Đây là lý do vì sao các `frontier models` (mô hình hàng đầu) năm 2026 đều có sẵn khả năng `tool use` cực mạnh từ trong lõi, còn hầu hết các mô hình nhỏ cỡ 7B tham số vẫn cần được `fine-tune` chuyên biệt về `tool-use` mới có thể gọi hàm đáng tin cậy.

### Bảng xếp hạng Berkeley Function Calling Leaderboard V4 (Patil và cộng sự, ICML 2025)

`BFCL` là thước đo tiêu chuẩn thực tế (`de facto evaluation`) của năm 2026. Cấu trúc tỷ trọng của phiên bản `V4` bao gồm:

- **`Agentic` (40%)** — Đánh giá toàn bộ quỹ đạo của `agent` (`full agent trajectories`): kết hợp bộ nhớ (`memory`), đa lượt (`multi-turn`), và ra quyết định động (`dynamic decisions`).
- **`Multi-Turn` (30%)** — Các cuộc hội thoại tương tác nhiều lượt có sử dụng chuỗi công cụ (`tool chains`).
- **`Live` (10%)** — Các câu `prompt` thực tế do người dùng gửi lên (phân phối dữ liệu khó và đa dạng hơn nhiều so với dữ liệu nhân tạo).
- **`Non-Live` (10%)** — Các ca kiểm thử tổng hợp (`synthetic test cases`).
- **`Hallucination` (10%)** — Kiểm tra khả năng nhận biết khi nào **không nên** gọi bất kỳ công cụ nào (hoặc phát hiện khi thiếu công cụ phù hợp).

Phiên bản `V3` trước đó đã giới thiệu **đánh giá dựa trên trạng thái (`state-based evaluation`)**: sau khi chạy xong một chuỗi công cụ, hệ thống kiểm tra trạng thái thực sự của `API` / môi trường (ví dụ: *"file đó đã thực sự được tạo ra trên ổ đĩa chưa?"*) thay vì chỉ so khớp cú pháp cây `AST` của câu lệnh gọi hàm (vì có nhiều cách gọi hàm khác nhau cùng tạo ra một kết quả đúng). Phiên bản `V4` bổ sung thêm các hạng mục tìm kiếm web (`web search`), bộ nhớ (`memory`), và độ nhạy với định dạng (`format sensitivity`).

**Phát hiện quan trọng năm 2026:** Bài toán gọi hàm đơn lượt (`single-turn function calling`) gần như đã được giải quyết triệt để. Các điểm gãy đổ (`failures`) hiện tập trung ở 4 mảng:
1. **`Memory` (Bộ nhớ):** Mang ngữ cảnh và thông tin thu thập được xuyên suốt qua nhiều lượt hội thoại.
2. **`Dynamic decision-making` (Ra quyết định động):** Lựa chọn công cụ tiếp theo dựa trên kết quả trả về trước đó (thay vì đi theo kịch bản cứng).
3. **`Long-horizon chains` (Chuỗi dài hơi):** Bị trượt mục tiêu hoặc rối loạn (`drift`) sau hơn 20+ bước gọi công cụ liên tiếp.
4. **`Hallucination detection` (Phát hiện ảo giác):** Biết từ chối gọi hàm khi không có công cụ nào phù hợp hoặc khi thiếu thông tin đầu vào.

### Lược đồ công cụ (Tool schema)

Mọi nhà cung cấp mô hình (`provider`) đều yêu cầu khai báo lược đồ công cụ (`tool schema`). Dù khác nhau ở vài chi tiết đặt tên, tất cả đều chia sẻ chung một cấu trúc:

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

- **`name`**: Tên định danh của hàm/công cụ.
- **`description`**: Mô tả công cụ làm gì, khi nào nên dùng (và khi nào không nên dùng).
- **`input_schema`**: Cấu trúc các tham số đầu vào viết theo chuẩn `JSON Schema` (gồm `properties`, các trường bắt buộc `required`, kiểu dữ liệu `types`, tập giá trị cho phép `enums`).

Anthropic sử dụng trực tiếp trường `input_schema`. OpenAI sử dụng `function.parameters`. Cả hai đều nhận chuẩn `JSON Schema`. **Phần mô tả (`description`) đóng vai trò sống còn (`load-bearing`)** — mô hình đọc dòng mô tả này để quyết định chọn công cụ nào. Viết `tool description` sơ sài, mơ hồ chính là nguyên nhân số #1 dẫn đến lỗi chọn nhầm công cụ (`wrong-tool-picked failures`).

### Kiểm tra tính hợp lệ của đối số (Argument validation)

Nguyên tắc bất di bất dịch: **Không bao giờ tin tưởng tuyệt đối vào lệnh gọi công cụ từ LLM (`Trust no tool call`).** Luôn phải kiểm tra (`validate`):

1. **Ép kiểu an toàn (`Type coercion`):** Mô hình có thể trả về chuỗi `"5"` trong khi `schema` quy định kiểu số nguyên `int`. Nếu việc chuyển đổi là rõ ràng không nhập nhằng (`unambiguous`), hãy tự động ép kiểu (`coerce`); nếu nhập nhằng (như `"five"` hoặc `"5px"`), hãy từ chối ngay.
2. **Kiểm tra tập giá trị (`Enum validation`):** Nếu `schema` quy định `status in {"open", "closed"}` mà mô hình lại phát ra `"in_progress"`, hãy từ chối kèm theo thông báo lỗi mô tả rõ các giá trị hợp lệ.
3. **Các trường bắt buộc (`Required fields`):** Thiếu trường bắt buộc $\rightarrow$ lập tức trả về một `error observation` (kết quả quan sát báo lỗi) cho mô hình tự sửa, tuyệt đối không để chương trình bị sập (`crash`).
4. **Kiểm tra định dạng (`Format validation`):** Ngày tháng (`dates`), `emails`, đường dẫn `URLs` — hãy kiểm tra bằng các bộ phân tích cú pháp cụ thể (`concrete parsers`), đừng chỉ dùng biểu thức chính quy (`regex`) sơ sài.

Mọi lỗi kiểm tra tính hợp lệ (`validation failure`) đều phải trả về một chuỗi quan sát có cấu trúc (`structured observation`) chỉ rõ sai ở trường nào và kỳ vọng kiểu gì, để `model` có thể thử gọi lại (`retry`) với đúng định dạng ở lượt kế tiếp.

### Gọi công cụ song song (Parallel tool calls)

Các nhà cung cấp mô hình hiện đại đều hỗ trợ phát ra nhiều lệnh gọi công cụ song song (`parallel tool calls`) trong cùng một lượt phản hồi của `assistant`. Quy trình diễn ra như sau:

1. `Model` phát ra 3 lệnh gọi công cụ cùng lúc, mỗi lệnh gắn một mã định danh `tool_use_id` riêng biệt.
2. Hệ thống thực thi (`runtime`) chạy cả 3 công cụ đó (chạy song song `in parallel` nếu chúng độc lập với nhau).
3. Kết quả của từng công cụ được gửi trả lại dưới dạng một khối `tool_result` được khớp nối chính xác thông qua mã `tool_use_id`.

**Quy tắc kỹ thuật (`Engineering rule`):** Hãy coi các mã đối chiếu (`correlation IDs` / `tool_use_id`) là thành phần sống còn (`load-bearing`). Chỉ cần ghép nhầm thứ tự hoặc tráo đổi `ID` là bạn sẽ gặp thảm họa định tuyến nhầm kết quả của công cụ này sang câu hỏi của công cụ kia (`wrong-tool-to-wrong-result routing`).

### Cô lập môi trường thực thi (Sandboxing)

Bước thực thi công cụ (`tool execution`) chính là ranh giới bảo mật (`sandbox boundary`). Xem lại Phase 13 · 26 để nắm chi tiết. Tóm tắt ngắn gọn: mỗi công cụ cần phải được giới hạn rõ phạm vi đọc/ghi (`read/write surface`), quyền truy cập mạng (`network access`), thời gian chờ tối đa (`timeout`), và trần bộ nhớ (`memory cap`). Một công cụ vạn năng kiểu `run_shell(cmd)` là một cờ đỏ nguy hiểm (`red flag`); việc tách thành công cụ chuyên biệt, hẹp quyền như `git_status()` sẽ an toàn hơn rất nhiều.

```figure
tool-routing
```

## Thực hành xây dựng (Build It)

File `code/main.py` cài đặt một sổ đăng ký công cụ (`tool registry`) mang hình thái chuẩn `production`:

- Bộ kiểm tra tập con `JSON Schema` (`JSON Schema subset validator` — chỉ dùng thư viện chuẩn `stdlib`).
- Cơ chế đăng ký công cụ (`tool registration`) kèm `description`, `input schema`, `timeout`, và hàm thực thi (`executor`).
- Tự động ép kiểu đối số (`argument coercion`) và kiểm tra tập giá trị (`enum validation`).
- Điều phối gọi công cụ song song (`parallel tool dispatch`) có đối chiếu qua `correlation IDs`.
- Trả về các thông báo lỗi dưới dạng chuỗi quan sát có cấu trúc (`structured error observations`).

Chạy thử chương trình:

```
python3 code/main.py
```

Bản ghi vết (`trace`) sẽ hiển thị một `mini agent` gọi 3 công cụ trong cùng một lượt (`turn`), trong đó có 1 lệnh gọi cố tình truyền sai định dạng tham số (`malformed call`) bị hệ thống từ chối kèm thông báo lỗi chi tiết giúp `model` biết cách tự sửa ở bước tiếp theo.

## Ứng dụng thực tế (Use It)

Mỗi nhà cung cấp đều có định dạng `tool schema` riêng — từ Anthropic, OpenAI, Gemini cho tới AWS Bedrock. Hãy sử dụng một tầng chuyển đổi trung gian (`translation layer` như OpenAI Agents SDK, Vercel AI SDK, hoặc bộ chuyển đổi công cụ của LangChain) nếu hệ thống của bạn cần hỗ trợ đa nhà cung cấp (`multi-provider`). `BFCL` là bộ tiêu chuẩn đánh giá tham chiếu (`reference benchmark`) — hãy chạy kiểm thử `agent` của bạn trên bộ test này trước khi đưa lên `production` nếu `tool use` là tính năng cốt lõi của sản phẩm.

## Đóng gói kỹ năng (Ship It)

File `outputs/skill-tool-registry.md` giúp sinh tự động danh mục công cụ (`tool catalog`), lược đồ `schema`, và mã nguồn `registry` cho một miền bài toán (`task domain`) cho trước. Nó bao gồm cả bước kiểm tra chất lượng mô tả (`description-quality checks` — đảm bảo mô tả của từng công cụ chỉ dẫn rõ cho `model` biết *khi nào* nên dùng và *khi nào không nên dùng* công cụ đó).

## Bài tập rèn luyện (Exercises)

1. Thêm một công cụ `"no-op"` (không làm gì cả) cho phép mô hình chủ động chọn khi từ chối sử dụng bất kỳ công cụ nào khác. Hãy đo lường hiệu quả của nó trên một bài kiểm tra chống ảo giác kiểu `BFCL hallucination test`.
2. Cài đặt cơ chế ép kiểu đối số (`argument coercion`) cho trường hợp số nguyên bị truyền dưới dạng chuỗi (`int-as-string`) và số thực dưới dạng chuỗi (`float-as-string`). Ở ranh giới nào thì việc tự động ép kiểu này bắt đầu che giấu đi các lỗi logic thực sự (`hide real bugs`)? *(Gợi ý: ví dụ mã bưu chính `"01234"` hoặc số điện thoại nếu bị ép sang `int` sẽ mất số `0` ở đầu).*
3. Bổ sung giới hạn thời gian chạy cho từng công cụ (`per-tool timeout`) và một cầu dao ngắt mạch (`circuit breaker` — tạm khóa công cụ trong 60 giây sau 3 lần thực thi thất bại liên tiếp). Điều này làm thay đổi cách mô hình tự phục hồi (`recovers`) như thế nào?
4. Đọc tài liệu mô tả của `BFCL V4`. Chọn một hạng mục (ví dụ: `"multi-turn"`) và chạy thử 10 `prompts` mẫu qua `agent` của bạn. Báo cáo tỷ lệ vượt qua (`pass rate`).
5. Chuyển đổi (`port`) bộ kiểm tra `stdlib validator` sang dùng thư viện `Pydantic` (Python) hoặc `Zod` (TypeScript). `Pydantic`/`Zod` đã bắt được những lỗi ngầm nào mà bộ kiểm tra thu nhỏ (`toy validator`) bỏ sót?

## Thuật ngữ quan trọng (Key Terms)

| Thuật ngữ (Term) | Cách người ta thường gọi (What people say) | Bản chất thực sự (What it actually means) |
|------|----------------|------------------------|
| **Function calling** | "Sử dụng công cụ" ("Tool use") | Lời gọi công cụ thông qua đầu ra có cấu trúc (`structured-output`) được kiểm tra chặt chẽ bằng `schema` |
| **Toolformer** | "Tự giám sát gán nhãn công cụ" ("Self-supervised tool annotation") | Công trình của Schick (2023) — chỉ giữ lại các lệnh gọi công cụ mà kết quả của chúng giúp giảm `next-token loss` |
| **BFCL** | "Bảng xếp hạng gọi hàm Berkeley" ("Berkeley Function Calling Leaderboard") | Chuẩn đánh giá năm 2026: 40% `agentic`, 30% `multi-turn`, 10% `live`, 10% `non-live`, 10% `hallucination` |
| **Tool schema** | "Chữ ký hàm dành cho model" ("Function signature for the model") | Gồm tên hàm (`name`), mô tả (`description`), và lược đồ `JSON Schema` của các đối số đầu vào |
| **tool_use_id** | "Mã đối chiếu" ("Correlation ID") | Gắn kết một lệnh gọi công cụ với kết quả trả về tương ứng của nó; tối quan trọng khi điều phối chạy song song (`parallel dispatch`) |
| **Hallucination detection** | "Biết khi nào không nên gọi hàm" ("Know when not to call") | Hạng mục đánh giá của `BFCL V4`: từ chối gọi công cụ khi không có công cụ nào phù hợp |
| **Argument coercion** | "Sửa lỗi chuỗi thành số" ("String-to-int repair") | Các bước ép kiểu hẹp, an toàn cho những lỗi lệch kiểu dễ đoán trước; từ chối ngay nếu dữ liệu nhập nhằng (`ambiguous`) |
| **Sandboxing** | "Ranh giới thực thi công cụ" ("Tool execution boundary") | Giới hạn phạm vi đọc/ghi, quyền mạng, `timeout`, và trần bộ nhớ riêng cho từng công cụ |

## Đọc thêm (Further Reading)

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — kỹ thuật tự giám sát gán nhãn công cụ (`self-supervised tool annotation`)
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — bộ chuẩn đánh giá (`eval benchmark`) năm 2026
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) — cách thiết kế `tool schema` chuẩn `production` trong Claude Agent SDK
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — kiểu công cụ hàm (`function tool type`) và cơ chế `Guardrails`
