# Plan fix loi tao bao cao the thao hang ngay

## 1. Ket luan dua tren log

Log hien tai chua cho thay loi crash cuoi cung. Pipeline van dang chay toi buoc embedding:

```text
INFO:graph:Crawled 41 raw articles.
INFO:graph:41 articles after dedup/filter.
INFO:graph:Loaded 60 articles from DB.
INFO:tools.embeddings:Embedding: 19 cached, 41 need API calls (9 batches of 5)
INFO:tools.embeddings:Embedding batch 11-15 of 41 new articles...
```

Van de chinh khong phai backend Neon hay Postgres auth. Van de nam o `SportNews_AI`: buoc tao embedding dang qua cham va de bi rate limit.

## 2. Nguyen nhan goc

### 2.1. Log ghi batch nhung code dang goi API tung article

Trong `SportNews_AI/tools/embeddings.py`, ham `_embed_batch()` nhan list `texts`, nhung ben trong lai lap qua tung `text` va goi:

```python
client.models.embed_content(...)
```

Tuc la voi log:

```text
41 need API calls
```

service thuc te phai goi toi thieu 41 request Gemini embedding. Neu API bi rate limit `429`, moi request co the retry voi backoff 10s, 20s, 40s, 80s.

Ket qua: tren giao dien admin se thay nut tao report chay rat lau, de tuong la bi loi.

### 2.2. Daily report nhung lai load 60 articles tu DB

Log:

```text
41 articles after dedup/filter.
Loaded 60 articles from DB.
```

Pipeline crawl duoc 41 bai moi, nhung sau do load 60 bai tu database theo `lookback_days`. Nghia la report daily khong chi xu ly batch vua crawl, ma con xu ly lai cac bai trong DB nam trong khoang lookback.

Neu DB AI tich luy nhieu bai va embedding cache chua day du, so request API se tang.

### 2.3. Endpoint tao report dang xu ly tac vu nang truc tiep

Flow tao report gom:

1. Crawl 3 nguon bao.
2. Clean/dedup/filter.
3. Save DB.
4. Load lai articles tu DB.
5. Tao embedding.
6. Build FAISS index.
7. Retriever.
8. Ranker.
9. Writer.
10. Reviewer.

Day la job nen chay background. Neu goi truc tiep tu UI/admin API, request co the keo dai, gap timeout hoac bi user tuong la treo.

### 2.4. Rate limit API la rui ro thuc te

Trong cac lan test truoc, log da co:

```text
429 Too Many Requests
Rate limit hit
Retrying...
```

Voi 41+ API calls, viec gap `429` la binh thuong. Khi retry nhieu lan, thoi gian tao report co the tang len vai phut.

## 3. Muc tieu fix

- Tao daily report on dinh, khong de UI bi treo.
- Giam so request toi Gemini/Groq.
- Giu duoc cache embedding trong DB.
- Neu API rate limit, he thong tra ve trang thai ro rang thay vi de user cho vo thoi han.
- Backend van luu report vao Neon.
- `SportNews_AI` van dung Postgres rieng de cache crawl/embedding.

## 4. Huong fix de xuat

## Phase 1 - Fix nhanh de chay on dinh

### 4.1. Gioi han so article xu ly moi lan tao daily report

Them bien cau hinh vao `SportNews_AI/.env`:

```env
MAX_REPORT_ARTICLES=25
```

Trong `graph.py`, sau khi load articles tu DB:

```python
articles = get_articles_by_lookback_days(engine, lookback_days=lookback_days)
articles = articles[:MAX_REPORT_ARTICLES]
```

Ly do:

- Daily report khong can xu ly 60+ bai.
- 20-25 bai la du de rank ra 5-8 tin noi bat.
- Giam embedding calls va LLM context.

Test can co:

- Khi DB co 60 articles, pipeline chi xu ly toi da `MAX_REPORT_ARTICLES`.
- Khi DB co it hon limit, pipeline xu ly toan bo.

### 4.2. Doi tu goi tung embedding sang batch embedding that

Hien tai `_embed_batch()` goi tung text mot. Can doi sang API batch dung nghia:

```python
client.models.embed_content(
    model=model,
    contents=texts,
    config=types.EmbedContentConfig(task_type=task_type),
)
```

Neu SDK/model khong ho tro batch on dinh, can:

- Giam `batch_size` ve 3.
- Them sleep ngan giua cac request.
- Ghi log ro request dang la single-call hay batch-call.

Test can co:

- Mock Gemini client va xac nhan 10 texts voi `batch_size=5` chi goi 2 API calls neu batch duoc ho tro.
- Neu API loi 429, retry dung so lan va co backoff.
- Neu batch fail, fallback khong lam crash toan pipeline.

### 4.3. Them timeout va error message ro cho AI service

Trong `main.py`, endpoint `/generate-report` can bat loi theo tung nhom:

- DB error.
- Crawler error.
- Embedding rate limit.
- LLM rate limit.
- Timeout.

Tra ve JSON ro:

```json
{
  "status": "error",
  "error": "Gemini embedding rate limit. Please retry later.",
  "metadata": {
    "stage": "embedding"
  }
}
```

Test can co:

- Mock pipeline raise rate limit, API tra ve status loi dung format.
- Backend nhan loi tu AI service va luu report status `failed`.

## Phase 2 - Chuyen tao report thanh background job

### 5.1. AI service nen tra job_id ngay

Thay vi UI cho den khi pipeline xong, AI service nen co flow:

```http
POST /generate-report
```

Tra ve:

```json
{
  "job_id": "daily-2026-06-13-xxx",
  "status": "queued"
}
```

Sau do frontend/backend poll:

```http
GET /generate-report/jobs/:job_id
```

Trang thai:

- `queued`
- `running`
- `success`
- `failed`

Ly do:

- Crawl + embedding + LLM la tac vu dai.
- UI khong nen giu HTTP request qua lau.
- Neu API rate limit, user van thay trang thai that.

Test can co:

- Tao job tra ve `queued`.
- Job dang chay tra ve `running`.
- Job thanh cong tra ve report.
- Job loi tra ve `failed` va error.

### 5.2. Backend admin reports can luu trang thai pending/running/failed

Backend hien tai nen xu ly:

- Khi bam tao report: tao report record status `generating`.
- Goi AI service.
- Neu AI thanh cong: update `ready`.
- Neu AI loi: update `failed` + luu error message.

Test can co:

- AI service success -> backend luu `ready`.
- AI service timeout/rate limit -> backend luu `failed`.
- UI hien loi tu backend thay vi loading vo han.

## Phase 3 - Toi uu cache va du lieu

### 6.1. Chi embedding article chua co embedding hop le

Da co logic cache:

```text
19 cached, 41 need API calls
```

Can bo sung log va test de xac nhan:

- Article da co embedding JSON hop le thi khong goi API.
- Article embedding loi JSON thi goi lai API.
- Article bi duplicate URL thi khong insert lai.

### 6.2. Khong build lai FAISS cho qua nhieu article moi daily

Voi daily report, co the chi build FAISS tren tap article trong ngay/lookback da limit. Khong can build tren toan bo DB.

Test can co:

- FAISS index size bang so article sau khi limit.
- Retriever chi tra articles trong period hien tai.

### 6.3. Them cau hinh cho daily/weekly rieng

Trong `.env`:

```env
REPORT_PERIOD=daily
REPORT_LOOKBACK_DAYS=1
MAX_DAILY_REPORT_ARTICLES=25
MAX_WEEKLY_REPORT_ARTICLES=60
EMBEDDING_BATCH_SIZE=5
EMBEDDING_REQUEST_SLEEP_SECONDS=1
```

Test can co:

- Daily dung limit daily.
- Weekly dung limit weekly.
- Neu env invalid thi fallback default an toan.

## 7. Thu tu thuc hien de giam loi

- [x] Buoc 1: Them `MAX_REPORT_ARTICLES` va limit articles trong `graph.py`.
- [x] Buoc 2: Viet unit test cho logic limit article.
- [x] Buoc 3: Sua `_embed_batch()` thanh batch API call that neu SDK ho tro.
- [x] Buoc 4: Viet unit test mock embedding client de xac nhan giam so request.
- [x] Buoc 5: Them log stage ro rang: `crawl`, `db_load`, `embedding`, `faiss`, `ranker`, `writer`, `reviewer`.
- [x] Buoc 6: Them error response co stage khi pipeline fail.
- [x] Buoc 7: Backend xu ly AI service error thanh report status `failed`.
- [x] Buoc 8: Frontend admin reports hien status `generating/failed` va message loi.
- [x] Buoc 9: Chay test `SportNews_AI`.
- [x] Buoc 10: Chay test/build backend.
- [x] Buoc 11: Chay build frontend.
- [ ] Buoc 12: Test end-to-end tu UI admin.

Ghi chu sau khi trien khai:

- Da test runtime truc tiep `POST /generate-report` thanh cong.
- Daily report da gioi han `source_count = 25`.
- Da them generate async o backend cho endpoint admin khi request co `async: true`.
- Da cap nhat frontend `/admin/reports` de goi async va tu refresh danh sach moi 5 giay.
- Chua tao bang job rieng cho AI service. Hien tai background job dang chay bang goroutine trong backend, du de tranh UI bi giu request lau trong giai doan nay.

## 8. Cach verify sau khi fix

### 8.1. Test truc tiep AI service

```powershell
cd SportNews_AI
$body = @{ period_type = "daily"; lookback_days = 1 } | ConvertTo-Json
Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/generate-report" `
  -Headers @{ "X-Internal-API-Key" = "dev-ai-secret" } `
  -ContentType "application/json" `
  -Body $body
```

Ket qua mong doi:

- Khong con chay qua lau voi daily report.
- Log embedding co so API calls giam ro.
- Neu gap rate limit, response bao loi ro thay vi loading vo han.

### 8.2. Test qua giao dien

1. Chay AI service.
2. Chay backend.
3. Chay frontend.
4. Vao `/admin/reports`.
5. Bam tao report ngay.

Ket qua mong doi:

- UI hien dang tao report.
- Neu thanh cong: report status `ready`.
- Neu loi API/rate limit: report status `failed` va hien message loi.

## 9. Ghi chu quan trong

Log ban dua khong phai loi database. Database da qua duoc cac buoc:

```text
Loaded 60 articles from DB.
```

Neu DB loi, pipeline da dung truoc buoc embedding. Loi/thoi gian cho hien tai nam o khu vuc:

- embedding Gemini qua nhieu request,
- rate limit,
- request tao report dang chay dong bo qua UI/backend.

Fix dung trong giai doan nay la giam so article, batch embedding that, va them job/status/error handling ro rang.
