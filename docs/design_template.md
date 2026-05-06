# Design Template

## Problem

Xây dựng một hệ thống research assistant tự động có khả năng nhận câu hỏi nghiên cứu, tìm kiếm thông tin từ web, phân tích kết quả, và viết câu trả lời tổng hợp chất lượng cao. Hệ thống cần so sánh giữa single-agent (1 agent làm tất cả) và multi-agent (nhiều agent chuyên biệt phối hợp).

## Why multi-agent?

Single-agent gặp các hạn chế sau:
1. **Overload nhiệm vụ**: Một agent vừa search, vừa phân tích, vừa viết dễ bị hallucination và mất focus.
2. **Không có quality check**: Không có bước kiểm tra chéo giữa các bước.
3. **Khó debug**: Khi output sai, không biết sai ở bước nào (search? analysis? writing?).
4. **Thiếu chuyên biệt hóa**: Mỗi task cần prompt/temperature/strategy khác nhau.

Multi-agent giải quyết bằng cách tách rõ responsibility, có shared state để handoff, và có supervisor để điều phối.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Điều phối workflow, quyết định agent tiếp theo, enforce max iterations | ResearchState (toàn bộ) | Route decision (researcher/analyst/writer/done) | Routing loop, vượt max_iterations |
| Researcher | Tìm kiếm web qua Tavily, thu thập sources, viết research notes | Query + state | sources[], research_notes | Search API fail, no results |
| Analyst | Phân tích research notes, đánh giá evidence, tìm gaps | research_notes, sources | analysis_notes | Thiếu input, analysis quá chung |
| Writer | Tổng hợp research + analysis thành final answer có citation | research_notes, analysis_notes, sources | final_answer | Hallucination, thiếu citation |

## Shared state

- `request`: ResearchQuery gốc (query, max_sources, audience) — cần để mọi agent hiểu mục tiêu.
- `iteration`: Đếm vòng lặp — cần cho guardrail max_iterations.
- `route_history`: Lịch sử routing — cần để debug và trace.
- `sources`: Danh sách SourceDocument — cần để writer cite và analyst verify.
- `research_notes`: Ghi chú nghiên cứu — output của Researcher, input cho Analyst.
- `analysis_notes`: Ghi chú phân tích — output của Analyst, input cho Writer.
- `final_answer`: Câu trả lời cuối — output của Writer.
- `agent_results`: Kết quả chi tiết từng agent — cần cho benchmark.
- `trace`: Log sự kiện — cần cho observability.
- `errors`: Danh sách lỗi — cần cho failure analysis.

## Routing policy

```text
START
  |
  v
Supervisor ──────────────────────────────────────┐
  |                                               |
  |── (no research_notes) ──> Researcher ──┐      |
  |── (no analysis_notes) ──> Analyst ─────┤      |
  |── (no final_answer) ───> Writer ───────┤      |
  |── (has final_answer) ──> END           |      |
  |                                        |      |
  <────────────────────────────────────────┘      |
  |                                               |
  |── (iteration >= max_iterations) ──> END ──────┘
```

## Guardrails

- Max iterations: 6 (configurable via .env)
- Timeout: 60s per workflow run
- Retry: 3 retries with exponential backoff on LLM calls (tenacity)
- Fallback: Heuristic routing if LLM-based supervisor fails
- Validation: Pydantic schemas for all inputs/outputs

## Benchmark plan

| Query | Metric | Expected Outcome |
|---|---|---|
| "Research GraphRAG state-of-the-art and write a 500-word summary" | Latency, Cost, Quality, Sources | Multi-agent slower but higher quality with citations |
| "Compare single-agent and multi-agent workflows for customer support" | Latency, Cost, Quality | Multi-agent provides more structured comparison |
| "Summarize production guardrails for LLM agents" | Latency, Cost, Citation coverage | Multi-agent has better citation coverage from web search |
