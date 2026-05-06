# Multi-Agent Research System

Dự án này là một hệ thống nghiên cứu tự động mạnh mẽ được xây dựng dựa trên kiến trúc **Multi-Agent Systems** sử dụng **LangGraph**. Hệ thống bao gồm một **Supervisor** để điều phối và các worker agents chuyên biệt: **Researcher**, **Analyst**, **Writer**, và **Critic**. 

Mục tiêu của dự án là minh chứng khả năng vượt trội của mô hình đa tác vụ (Multi-agent) so với mô hình tác vụ đơn lẻ (Single-agent baseline) trong việc xử lý các truy vấn nghiên cứu phức tạp, yêu cầu tra cứu web và tổng hợp thông tin có trích dẫn.

Hệ thống được vận hành bởi model **Llama-3.3-70b-versatile** qua **Groq API** cho tốc độ phản hồi cực nhanh, sử dụng **Tavily API** cho module tìm kiếm web, và tích hợp sẵn **LangSmith** để tracing toàn bộ luồng chạy.

## Architecture

```text
User Query
   |
   v
Supervisor / Router (LangGraph)
   |
   |------> Researcher Agent  -> Tìm kiếm web (Tavily) & tạo `research_notes`
   |------> Analyst Agent     -> Trích xuất claims & tạo `analysis_notes`
   |------> Writer Agent      -> Viết `final_answer` có trích dẫn
   |------> Critic Agent      -> (Optional) Kiểm tra fact-check & hallucination
   |
   v
Trace + Benchmark Report
```

## Quickstart

### 1. Cài đặt môi trường

```bash
# Tạo và kích hoạt virtual environment
python -m venv .venv
source .venv/bin/activate  # Trên Windows dùng: .venv\Scripts\activate

# Cài đặt package và các dependencies (Bao gồm Groq, LangGraph, Tavily...)
pip install -e ".[llm]"
```

### 2. Cấu hình API keys

Copy file `.env.example` thành `.env` và điền các API key của bạn:

```bash
cp .env.example .env
```

Mở `.env` và điền:
```env
# Core LLM Provider
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Search Provider
TAVILY_API_KEY=your_tavily_api_key_here

# Tracing (Khuyên dùng để xem biểu đồ LangGraph)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=multi-agent-research-lab
```

## Hướng dẫn sử dụng CLI

Dự án cung cấp một CLI entrypoint `malab` (hoặc chạy qua module `python -m multi_agent_research_lab.cli`) với 3 lệnh chính:

### 1. Chạy Baseline (Single-Agent)
Dùng duy nhất 1 lệnh gọi LLM để trả lời câu hỏi, giúp làm cơ sở so sánh (baseline).
```bash
malab baseline --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

### 2. Chạy Multi-Agent Workflow
Kích hoạt toàn bộ luồng LangGraph với Supervisor và các worker.
```bash
malab multi-agent --query "What are the security concerns of LLM agents?"
```

### 3. Chạy Benchmark và xuất Report
Lệnh này sẽ tự động chạy cả `baseline` và `multi-agent` cho cùng một câu hỏi, đo lường tốc độ (latency), ước tính chi phí (cost), đếm token và xuất báo cáo so sánh ra file Markdown.

```bash
malab benchmark --query "Compare single-agent and multi-agent workflows for customer support" --output "reports/my_report.md"
```
*Mặc định nếu không truyền `--output`, báo cáo sẽ được lưu tại `reports/benchmark_report.md`.*

## Cấu trúc thư mục

```text
.
├── src/multi_agent_research_lab/
│   ├── agents/              # Chứa logic của Supervisor, Researcher, Analyst, Writer...
│   ├── core/                # Cấu hình Pydantic schemas, state, config
│   ├── graph/               # Chứa LangGraph StateGraph (workflow.py)
│   ├── services/            # Tích hợp Groq API và Tavily API
│   ├── evaluation/          # Chứa logic Benchmark & Generate Report
│   ├── observability/       # Cấu hình LangSmith tracing
│   └── cli.py               # Lệnh CLI chính
├── configs/                 # YAML configs
├── docs/                    # Tài liệu System Design 
├── reports/                 # Thư mục chứa các file báo cáo Markdown sau khi benchmark
└── tests/                   # Unit tests
```

## References

- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [LangGraph concepts](https://langchain-ai.github.io/langgraph/concepts/)
- [LangSmith tracing](https://docs.smith.langchain.com/)
- [Groq API](https://console.groq.com/docs/quickstart)
