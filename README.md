# Multi-Agent Research System

A production-grade **Multi-Agent Research System** built with [LangGraph](https://langchain-ai.github.io/langgraph/), [Groq API](https://groq.com/) (Llama 3.3 70B), and [Tavily Search](https://tavily.com/). 

This project implements a sophisticated research workflow where multiple specialized AI agents collaborate to search the web, analyze findings, and write highly accurate, citation-backed reports. It also includes an automated benchmarking tool to compare single-agent vs. multi-agent performance.

## 🧠 System Architecture

The workflow uses a `StateGraph` coordinated by a **Supervisor Agent**:

```text
User Query
   |
   v
Supervisor (Router)
   |
   |------> Researcher Agent  (Searches web via Tavily, writes notes)
   |------> Analyst Agent     (Extracts claims, evaluates evidence)
   |------> Writer Agent      (Synthesizes findings, adds citations)
   |------> Critic Agent      (Fact-checks, scores quality)
   |
   v
Final Answer & Benchmark Report
```

## 🚀 Quickstart

### 1. Requirements
- Python 3.11+
- API keys for **Groq**, **Tavily**, and (optionally) **LangSmith** for tracing.

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package and dependencies
pip install -e ".[llm,dev]"
```

### 3. Configuration
Copy the environment template and add your API keys:

```bash
cp .env.example .env
```
Open `.env` and configure:
```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
TAVILY_API_KEY=your_tavily_api_key

# Optional (for tracing):
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=multi-agent-research-lab
```

## 🛠️ Usage

The project provides a CLI `malab` (Multi-Agent Lab) to run the system.

### 1. Run the Multi-Agent Workflow
```bash
python -m multi_agent_research_lab.cli multi-agent --query "What is GraphRAG?"
```

### 2. Run the Single-Agent Baseline
To see how a single LLM handles the prompt without web search or multi-agent collaboration:
```bash
python -m multi_agent_research_lab.cli baseline --query "What is GraphRAG?"
```

### 3. Run the Benchmark Tool
Run both approaches side-by-side and generate a detailed Markdown report comparing latency, cost, and answer quality.
```bash
python -m multi_agent_research_lab.cli benchmark \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary" \
  --output "reports/benchmark_report.md"
```

## 📊 Evaluation & Tracing

- **Reports**: All benchmark evaluations are saved to the `reports/` directory.
- **Observability**: If `LANGSMITH_API_KEY` is provided, every agent's thought process, LLM prompt, latency, and token cost is automatically traced to your LangSmith dashboard.

## 🤝 Project Structure

- `src/multi_agent_research_lab/agents/`: Agent implementations (Supervisor, Researcher, Analyst, Writer, Critic).
- `src/multi_agent_research_lab/graph/`: LangGraph workflow construction.
- `src/multi_agent_research_lab/services/`: LLM and Search integrations.
- `src/multi_agent_research_lab/evaluation/`: Benchmarking logic.
- `configs/`: YAML configs defining agent temperatures and benchmark queries.
