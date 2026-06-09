# Báo Cáo Thực Hành: Xây Dựng Hệ Thống Multi-Agent với A2A Protocol

**Họ tên:** Vũ Quang Vinh  
**Project ID:** Day9_2A2026935_VuQuangVinh  
**Công nghệ:** Python 3.14, LangGraph, LangChain, A2A SDK, OpenAI GPT-4o-mini

---

## Môi Trường Thực Hành

- **OS:** Windows 11
- **Python:** 3.14 (via `uv`)
- **LLM Provider:** OpenAI (`gpt-4o-mini`)
- **Package Manager:** `uv`

### Cấu hình `.env`

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### Cấu hình `common/llm.py`

```python
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
```

> **Lưu ý:** Ban đầu dùng OpenRouter với model `google/gemma-4-31b` (free tier), bị rate limit 429 sau 16 requests. Chuyển sang OpenAI API để tiếp tục.

---

## Phần 1: Direct LLM Calling

### Cách chạy

```bash
uv run python stages/stage_1_direct_llm/main.py
```

### Bài Tập 1.1 — Thay đổi câu hỏi

Sửa biến `QUESTION` trong `stages/stage_1_direct_llm/main.py`:

```python
QUESTION = "Hợp đồng lao động miệng có giá trị pháp lý không?"
```

**Kết quả:** LLM trả lời dựa trên Bộ luật Lao động Việt Nam 2019 — hợp đồng miệng có giá trị với công việc dưới 1 tháng, không có giá trị với hợp đồng từ 1 tháng trở lên.

### Bài Tập 1.2 — Thêm temperature control

Sửa `common/llm.py`, thêm `temperature=0.3`:

```python
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,  # thêm dòng này
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
```

**Tác dụng:** Output ổn định hơn, ít ngẫu nhiên hơn giữa các lần chạy.

### Nhận xét

- LLM Stage 1 hoàn toàn **stateless** — không có memory, không có tools
- Câu trả lời dựa 100% vào training data → có thể outdated
- Không thể cite điều luật cụ thể hay tính toán damages

---

## Phần 2: LLM + RAG & Tools

### Cách chạy

```bash
uv run python stages/stage_2_rag_tools/main.py
```

### Bài Tập 2.1 — Thêm knowledge base entry

Thêm entry về luật lao động vào `LEGAL_KNOWLEDGE` trong `stages/stage_2_rag_tools/main.py`:

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
        "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
        "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
},
```

### Bài Tập 2.2 — Tạo tool mới

Thêm tool `check_statute_of_limitations` vào file, trước dòng `TOOLS = [...]`:

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.
    
    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")
```

Cập nhật TOOLS list:

```python
TOOLS = [search_legal_database, calculate_damages, check_statute_of_limitations]
```

### Nhận xét

So với Stage 1, LLM Stage 2:
- **Tự quyết định** gọi tool `search_legal_database` trước khi trả lời
- **Cite được luật cụ thể**: DTSA, UCC, Economic Espionage Act
- Câu trả lời có cấu trúc rõ ràng hơn, có bảng tóm tắt
- Hạn chế: chỉ 1 vòng tool call, không thể search lại nếu cần

---

## Phần 3: Single Agent với ReAct

### Cách chạy

```bash
uv run python stages/stage_3_single_agent/main.py
```

### Vấn đề gặp phải & cách xử lý

**Vấn đề 1:** `create_react_agent` không nhận `verbose=True`

```
TypeError: create_react_agent() got an unexpected keyword argument 'verbose'
```

**Fix:** Xóa `verbose=True`, dùng `langchain.debug` thay thế:

```python
import langchain
langchain.debug = True
```

> **Lưu ý:** `from langchain.globals import set_debug` không hoạt động với version này. Dùng `import langchain; langchain.debug = True` thay thế.

**Vấn đề 2:** `ModuleNotFoundError: No module named 'langchain'`

**Fix:**
```bash
uv add langchain
```

### Bài Tập 3.1 — Thêm tool tra cứu án lệ

Thêm function `search_case_law` vào file:

```python
@tool
def search_case_law(keywords: str) -> str:
    """Tìm kiếm án lệ theo từ khóa."""
    cases = {
        "breach": "Hadley v. Baxendale (1854) - Consequential damages",
        "negligence": "Donoghue v. Stevenson (1932) - Duty of care",
        "contract": "Carlill v. Carbolic Smoke Ball Co (1893) - Unilateral contract",
    }
    for key, case in cases.items():
        if key in keywords.lower():
            return case
    return "Không tìm thấy án lệ phù hợp"
```

Thêm vào TOOLS list:

```python
TOOLS = [search_legal_database, calculate_penalty, check_compliance_requirements, search_case_law]
```

### Bài Tập 3.2 — Debug agent reasoning

```python
import langchain
langchain.debug = True
```

### Nhận xét

So với Stage 2:
- Agent **tự động** quyết định gọi tool nào, bao nhiêu lần
- Xử lý được câu hỏi phức tạp đa chiều (tax + privacy + compliance cùng lúc)
- Hạn chế: 1 agent xử lý tất cả domain → không chuyên sâu

---

## Phần 4: Multi-Agent In-Process

### Cách chạy

```bash
uv run python stages/stage_4_milti_agent/main.py
```

### Vấn đề gặp phải & cách xử lý

**Vấn đề:** 2 dòng cuối file dùng IPython không chạy được trong terminal:

```python
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

**Fix:** Xóa 2 dòng đó, thay bằng export file ảnh trong hàm `main()`:

```python
graph_viz = create_graph()
png_data = graph_viz.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)
print("\nGraph saved to graph.png")
```

### Bài Tập 4.1 — Thêm privacy_agent

Thêm field vào `LegalState`:

```python
class LegalState(TypedDict):
    # ... các field cũ ...
    privacy_analysis: Annotated[str, _last_wins]  # thêm mới
```

Thêm hàm agent:

```python
async def privacy_agent(state: LegalState) -> dict:
    """Agent chuyên về GDPR và luật bảo vệ dữ liệu cá nhân."""
    print("\n  [Node: privacy_agent] Privacy specialist starting...")
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.

Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Hãy phân tích các vấn đề về privacy và GDPR (nếu có).
"""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    print(f"  [Node: privacy_agent] Done ({len(response.content)} chars)")
    return {"privacy_analysis": response.content}
```

Thêm vào graph trong `create_graph()`:

```python
graph.add_node("privacy_agent", privacy_agent)
graph.add_edge("privacy_agent", "aggregate")
graph.add_conditional_edges(
    "check_routing",
    route_to_specialists,
    ["call_tax_specialist", "call_compliance_specialist", "privacy_agent", "aggregate"],
)
```

Thêm `privacy_analysis: ""` vào `graph.ainvoke(...)`.

### Bài Tập 4.2 — Conditional routing

Sửa hàm `route_to_specialists` (tên thực tế trong code, tương đương `check_routing` trong codelab):

```python
def route_to_specialists(state: LegalState) -> list[Send]:
    question_lower = state["question"].lower()
    sends: list[Send] = []

    if state.get("needs_tax"):
        sends.append(Send("call_tax_specialist", state))
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance_specialist", state))
    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
        sends.append(Send("privacy_agent", state))

    return sends if sends else [Send("aggregate", state)]
```

> **Lưu ý mapping tên node:** Codelab dùng tên `tax_agent`, `compliance_agent`, `aggregate_results` — code thực tế dùng `call_tax_specialist`, `call_compliance_specialist`, `aggregate`.

---

## Phần 5: Distributed A2A System

### Khởi động hệ thống (Windows)

`start_all.sh` không chạy trực tiếp trên Windows. Mở **5 terminal riêng biệt**, chạy lần lượt:

```bash
# Terminal 1
uv run python -m registry

# Terminal 2
uv run python -m law_agent

# Terminal 3
uv run python -m tax_agent

# Terminal 4
uv run python -m compliance_agent

# Terminal 5
uv run python -m customer_agent
```

> **Lưu ý:** Dùng `python -m <module>` thay vì `python <module>/main.py` vì các file entry point tên là `__main__.py`.

### Test hệ thống

```bash
uv run python test_client.py
```

### Bài Tập 5.1 — Đo latency

Thêm đo thời gian vào `test_client.py`:

```python
import time

start = time.time()
# ... gửi request ...
elapsed = time.time() - start
print(f"\nLatency: {elapsed:.2f} s")
```

**Kết quả:** Latency ban đầu = **94.55 giây**

### Bài Tập 5.2 — Test dynamic discovery

Tắt Tax Agent (Ctrl+C ở terminal 3), chạy lại `test_client.py`.

**Quan sát trong log của Law Agent:**

```
ERROR call_tax failed: All connection attempts failed
httpx.ConnectError: All connection attempts failed
```

**Kết luận:** Hệ thống có **fault tolerance** — Tax Agent down nhưng Law Agent bắt lỗi và fallback, hệ thống vẫn trả về response (không có phần tax analysis).

### Bài Tập 5.3 — Modify agent behavior

Mở `tax_agent/graph.py`, tìm system prompt, thêm vào cuối:

```
Keep your response concise, under 100 words.
```

Restart terminal Tax Agent, test lại.

---

## Bài Tập Cộng Điểm: Giảm Latency

### Đo latency ban đầu

| Lần chạy | Latency |
|---|---|
| Baseline | **94.55s** |

### Phân tích nguyên nhân

Flow ban đầu của Law Agent gồm **5 LLM calls**:

```
analyze_law (LLM call #1)
    ↓
check_routing (LLM call #2)
    ↓
call_tax ──────────── call_compliance  (LLM call #3 và #4 — song song ✅)
    ↓                       ↓
         aggregate (LLM call #5)
```

`call_tax` và `call_compliance` đã chạy **song song** nhờ LangGraph Send API. Bottleneck thực sự là `analyze_law` + `check_routing` chạy **tuần tự** — 2 LLM calls riêng biệt.

### Phương án: Gộp `analyze_law` + `check_routing` thành 1 LLM call

Sửa `law_agent/graph.py`, thêm hàm mới:

```python
async def analyze_and_route(state: LawState) -> dict:
    """Gộp analyze_law + check_routing thành 1 LLM call."""
    depth = state.get("delegation_depth", 0)
    if depth >= MAX_DELEGATION_DEPTH:
        return {"law_analysis": "", "needs_tax": False, "needs_compliance": False}

    llm = get_llm()
    messages = [
        SystemMessage(content=(
            "You are a senior corporate litigation attorney. Do two things:\n"
            "1. Analyse the legal aspects of the question (2-3 sentences)\n"
            "2. Decide if specialist agents are needed\n\n"
            "Reply ONLY in valid JSON:\n"
            '{"law_analysis": "<your analysis>", "needs_tax": <true|false>, "needs_compliance": <true|false>}'
        )),
        HumanMessage(content=state["question"]),
    ]
    result = await llm.ainvoke(messages)
    raw = result.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"law_analysis": raw, "needs_tax": True, "needs_compliance": True}

    return {
        "law_analysis": parsed.get("law_analysis", ""),
        "needs_tax": bool(parsed.get("needs_tax", True)),
        "needs_compliance": bool(parsed.get("needs_compliance", True)),
    }
```

Sửa `create_graph()`:

```python
# Thay thế 2 node cũ bằng 1 node mới
graph.add_node("analyze_and_route", analyze_and_route)

graph.set_entry_point("analyze_and_route")
graph.add_conditional_edges(
    "analyze_and_route",
    route_to_subagents,
    ["call_tax", "call_compliance", "aggregate"],
)
```

### Kết quả sau khi tối ưu

| | Trước | Sau | Giảm |
|---|---|---|---|
| **Latency** | 94.55s | **80.26s** | **~14.3s (~15%)** |
| **LLM calls** | 5 | 4 | 1 call |

### Flow sau khi tối ưu

```
analyze_and_route (LLM call #1 — gộp analyze + routing)
    ↓
call_tax ──────────── call_compliance  (LLM call #2 và #3 — song song ✅)
    ↓                       ↓
         aggregate (LLM call #4)
```

---

## Phần 6: Câu Hỏi Ôn Tập

**1. Khi nào nên dùng single agent thay vì multi-agent?**

Single agent phù hợp khi bài toán thuộc 1 domain duy nhất, không cần chuyên môn hóa, hoặc khi cần đơn giản hóa deployment. Multi-agent phù hợp khi cần xử lý nhiều domain song song, cần chuyên môn hóa, hoặc khi mỗi phần có thể scale độc lập.

**2. Ưu điểm của A2A protocol so với gRPC hoặc REST thông thường?**

A2A được thiết kế đặc biệt cho agent-to-agent communication: hỗ trợ streaming responses, có agent card discovery (tự mô tả capability), built-in support cho async tasks, và chuẩn hóa cách agents tìm thấy nhau qua Registry.

**3. Làm thế nào để prevent infinite delegation loops trong A2A?**

Dùng `delegation_depth` counter trong state — mỗi lần delegate thì tăng lên 1, khi đạt `MAX_DELEGATION_DEPTH` thì không delegate thêm nữa. Trong code này `MAX_DELEGATION_DEPTH = 3`.

**4. Tại sao cần Registry service? Có thể hardcode URLs không?**

Registry cho phép dynamic discovery — agents tự đăng ký khi start, tự hủy khi stop. Hardcode URLs được nhưng mất khả năng fault tolerance và auto-scaling: khi 1 agent thay đổi port hoặc scale lên nhiều instances, tất cả agents khác phải update config.

---

## Tổng Kết

| Stage | Pattern | Kết quả |
|---|---|---|
| 1 | Direct LLM | ✅ Chạy thành công |
| 2 | LLM + RAG & Tools | ✅ Chạy thành công |
| 3 | ReAct Agent | ✅ Chạy thành công |
| 4 | Multi-Agent In-Process | ✅ Chạy thành công, vẽ graph.png |
| 5 | Distributed A2A | ✅ Chạy thành công |
| Cộng điểm | Giảm latency | ✅ 94.55s → 80.26s (-15%) |