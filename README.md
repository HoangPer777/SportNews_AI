# SportNews AI

AI service tao ban tin the thao daily/weekly cho he thong SportNews. Service crawl tin tu VnExpress, Thanh Nien, Tuoi Tre, xu ly trung lap, embedding bang Gemini, chon tin quan trong va tao report JSON/Markdown bang LLM qua OpenRouter.

## Chuc nang chinh

- Crawl tin the thao tu 3 nguon bao Viet Nam.
- Tao report `daily` hoac `weekly` qua endpoint `POST /generate-report`.
- Ho tro scheduler tu dong tao report hang tuan.
- Luu bai crawl vao PostgreSQL cua AI service va dung FAISS cho semantic retrieval.
- Dung danh sach OpenRouter free models on dinh hon, co Groq fallback neu OpenRouter loi.
- Tra ve report co `executive_summary`, `trending_keywords`, `highlighted_news`, `metadata`.

## Chay bang Docker

Tao file `.env` tu template va dien key:

```powershell
cp .env.template .env
```

Bien moi truong quan trong:

```env
INTERNAL_API_KEY=dev-ai-secret
GOOGLE_API_KEY=your_google_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
# Optional: override default OpenRouter candidates
OPENROUTER_MODEL_CANDIDATES=nvidia/nemotron-3-super-120b-a12b:free,nvidia/nemotron-3-ultra-550b-a55b:free,openai/gpt-oss-120b:free,google/gemma-4-31b-it:free,openai/gpt-oss-20b:free,nvidia/nemotron-3-nano-30b-a3b:free,openrouter/free
DATABASE_URL=postgresql://sportsuser:sportspass@db:5432/sportsdb
```

Mac dinh `planner`, `ranker`, `writer`, `reviewer` se dung danh sach Deep Research candidates trong `core/model_candidates.py`. Khong nen dung `z-ai/glm-4.5-air:free` vi OpenRouter da bao model nay khong con free.

Khoi dong service:

```powershell
docker compose up -d --build
```

Kiem tra API docs:

```text
http://localhost:8000/docs
```

## Tao report thu cong

Daily report:

```powershell
curl -X POST http://localhost:8000/generate-report `
  -H "Content-Type: application/json" `
  -H "X-Internal-API-Key: dev-ai-secret" `
  -d "{\"period_type\":\"daily\",\"lookback_days\":1}"
```

Weekly report:

```powershell
curl -X POST http://localhost:8000/generate-report `
  -H "Content-Type: application/json" `
  -H "X-Internal-API-Key: dev-ai-secret" `
  -d "{\"period_type\":\"weekly\",\"lookback_days\":7}"
```

Output Markdown duoc ghi vao `outputs/daily_report.md` hoac `outputs/weekly_report.md`.

## Lien ket voi Backend

Backend Go goi AI service qua:

```env
AI_REPORT_SERVICE_URL=http://host.docker.internal:8000
AI_REPORT_API_KEY=dev-ai-secret
```

`AI_REPORT_API_KEY` cua backend phai trung voi `INTERNAL_API_KEY` cua AI service.

## Test

Chay test trong container/local Python env:

```powershell
pytest tests -q
```

Da co test cho API, graph limits, embeddings, planner, retriever, writer va safe LLM fallback.

## Thu muc quan trong

```text
agents/     Planner, Retriever, Ranker, Writer, Reviewer
core/       OpenRouter/Groq LLM routing va safe fallback
tools/      crawler, database, embeddings, preprocess
models/     Pydantic schemas
outputs/    file report Markdown sinh ra
docs/       tai lieu tich hop va huong dan test
tests/      test suite
```

## vào trang admin test tính năng tạo report
http://localhost:5173/admin/reports

username: admin
password: adminadmin
