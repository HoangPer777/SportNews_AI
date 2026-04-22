# Sports Weekly Intelligence Agent

A multi-agent AI pipeline that automatically crawls Vietnamese sports news, processes and stores articles, then generates a structured weekly intelligence report using LLMs.

## Architecture

```
START → Planner → Retriever → Ranker → Writer → Reviewer
                                              ↑         |
                                              └─────────┘
                                           (rewrite if rejected, max 2x)
```

The pipeline is built with **LangGraph** and consists of 5 agents:

| Agent | Role |
|---|---|
| **Planner** | Analyzes the article corpus and generates a structured weekly plan with sub-goals |
| **Retriever** | Performs semantic search over articles using FAISS + Gemini embeddings (top 30) |
| **Ranker** | Uses LLM to select the top 8 highest-value articles from retrieved candidates |
| **Writer** | Generates the full weekly report (executive summary, trending keywords, highlighted news) |
| **Reviewer** | Validates report quality and routes back to Writer if rejected (max 2 rewrites) |

## Tech Stack

- **LangGraph** — agent orchestration state machine
- **LangChain + Groq** — LLM calls (Llama 3.1)
- **Google Gemini Embeddings** — semantic vector embeddings
- **FAISS** — vector similarity search
- **FastAPI** — REST API endpoint
- **PostgreSQL + SQLAlchemy** — article persistence
- **BeautifulSoup** — web crawling (VnExpress, Thanh Nien, Tuoi Tre)
- **Docker Compose** — containerized deployment

## Project Structure

```
├── main.py               # FastAPI app
├── graph.py              # LangGraph pipeline definition
├── agents/
│   ├── planner.py        # Planner agent
│   ├── retriever.py      # Retriever agent (FAISS semantic search)
│   ├── ranker.py         # Ranker agent (LLM-based top-8 selection)
│   ├── writer.py         # Writer agent (report generation)
│   └── reviewer.py       # Reviewer agent (quality validation)
├── tools/
│   ├── crawler.py        # Web crawler (VnExpress, Thanh Nien, Tuoi Tre)
│   ├── db.py             # PostgreSQL persistence
│   ├── embeddings.py     # Gemini embeddings + FAISS index
│   └── preprocess.py     # Text cleaning, dedup, filtering
├── models/
│   └── schemas.py        # Pydantic schemas & LangGraph state
├── data/                 # FAISS index (generated at runtime)
├── outputs/              # Generated markdown reports
├── tests/                # Pytest + Hypothesis test suite
├── .env.template         # Environment variable template
├── Dockerfile
└── docker-compose.yml
```

## Setup

### 1. Clone and configure environment

```bash
cp .env.template .env
```

Edit `.env` and fill in your API keys:

```env
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GROQ_LLM_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=models/gemini-embedding-2-preview
```

### 2. Run with Docker Compose

```bash
docker-compose up --build
```

This starts:
- `db` — PostgreSQL 15 on port `5432`
- `agent` — FastAPI app on port `8000`


## API

### `POST /generate-report`

Triggers the full pipeline: crawl → preprocess → embed → plan → retrieve → write → review.

**Response:**

```json
{
  "status": "success",
  "report": {
    "executive_summary": "...",
    "trending_keywords": ["football", "SEA Games", "..."],
    "highlighted_news": [
      {
        "headline": "...",
        "summary": "...",
        "source": "VnExpress",
        "url": "https://..."
      }
    ],
    "generated_at": "2026-04-21T00:00:00"
  }
}
```

The report is also saved as Markdown to `outputs/weekly_report.md`.

## Pipeline Flow

1. **Crawl** — scrapes sports articles from VnExpress, Thanh Nien, Tuoi Tre (last 7 days)
2. **Preprocess** — cleans HTML, deduplicates by URL and title, filters by date and category
3. **Persist** — saves to PostgreSQL (upsert on URL conflict)
4. **Embed** — generates Gemini embeddings and builds a FAISS index
5. **Plan** — LLM generates a structured plan with sub-goals for the week
6. **Retrieve** — semantic search returns the most relevant articles (top 30)
7. **Rank** — LLM selects the top 8 highest-value articles ensuring source diversity
8. **Write** — LLM generates the full structured report (2 separate LLM calls)
9. **Review** — LLM validates quality; rejects and rewrites up to 2 times if needed

## Running Tests

```bash
pytest tests/
```

The test suite uses **Hypothesis** for property-based testing.

## Environment Variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `GOOGLE_API_KEY` | Google API key for Gemini embeddings |
| `GROQ_API_KEY` | Groq API key for LLM inference |
| `GROQ_LLM_MODEL` | Groq model name (default: `llama-3.1-8b-instant`) |
| `EMBEDDING_MODEL` | Gemini embedding model name |
| `FAISS_INDEX_PATH` | Path to save/load FAISS index (default: `data/faiss.index`) |
| `REPORT_OUTPUT_PATH` | Path to save the markdown report (default: `outputs/weekly_report.md`) |
| `TOP_K_RETRIEVAL` | Number of articles to retrieve per query (default: `10`) |
