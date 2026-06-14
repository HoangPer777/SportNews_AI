# Plan nang cap SportNews_AI sang OpenRouter GPT-OSS 120B

## 1. Muc tieu

Nang cap cac agent tao report the thao trong `SportNews_AI` de uu tien model:

```text
openai/gpt-oss-120b:free
```

thong qua OpenRouter, nham cai thien chat luong:

- Lap luan trong phan tong quan tot hon.
- Giam lap y giua cac doan.
- Chon tin noi bat hop ly hon.
- Viet tieng Viet tu nhien hon.
- Giu nguyen cac tinh nang cu:
  - Crawl 3 nguon bao.
  - Dedup/filter.
  - Embedding + FAISS.
  - Daily/weekly report.
  - Scheduler daily/weekly.
  - Admin manual generate.
  - Notification/email sau khi tao xong.
  - Groq fallback khi OpenRouter loi/rate limit.

## 2. Hien trang code

Hien tai `SportNews_AI` goi Groq truc tiep trong nhieu agent:

- `agents/planner.py`
- `agents/ranker.py`
- `agents/writer.py`
- `agents/reviewer.py`

Vi du:

```python
llm = ChatGroq(model=model_name, api_key=os.getenv("GROQ_API_KEY"))
response = llm.invoke(prompt)
```

Nhu vay co 3 van de:

- Khong co OpenRouter provider.
- Khong co fallback router dung chung.
- Moi agent tu khoi tao model rieng, kho kiem soat timeout/retry/log.

## 3. Huong thiet ke

Them mot lop LLM provider trung tam:

```text
core/
  model_candidates.py
  safe_llm.py
  llm.py
```

Y tuong:

1. Agent khong goi `ChatGroq` truc tiep nua.
2. Agent goi:

```python
from core.llm import get_safe_llm

llm = get_safe_llm("writer")
response = llm.invoke(prompt)
```

3. `get_safe_llm()` uu tien OpenRouter neu co `OPENROUTER_API_KEY`.
4. Neu OpenRouter fail, tu fallback ve Groq de giu tinh nang cu.
5. Neu khong set OpenRouter key, he thong van chay nhu hien tai bang Groq.

## 4. Cau hinh moi

Them vao `SportNews_AI/.env`:

```env
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_SITE_URL=http://localhost:8000
OPENROUTER_APP_NAME=SportNews AI

OPENROUTER_PRIMARY_MODEL=openai/gpt-oss-120b:free
OPENROUTER_FALLBACK_MODEL=z-ai/glm-4.5-air:free
```

Giu lai cau hinh Groq hien co:

```env
GROQ_API_KEY=...
GROQ_LLM_MODEL=llama-3.1-8b-instant
```

Groq se la fallback khan cap.

## 5. Model candidates theo agent

Tao file:

```text
SportNews_AI/core/model_candidates.py
```

De xuat:

```python
MODEL_CANDIDATES = {
    "planner": [
        "openai/gpt-oss-120b:free",
        "z-ai/glm-4.5-air:free",
    ],
    "ranker": [
        "openai/gpt-oss-120b:free",
        "z-ai/glm-4.5-air:free",
    ],
    "writer": [
        "openai/gpt-oss-120b:free",
        "z-ai/glm-4.5-air:free",
    ],
    "reviewer": [
        "openai/gpt-oss-120b:free",
        "z-ai/glm-4.5-air:free",
    ],
}
```

Ly do:

- `writer` can model lon nhat vi anh huong truc tiep chat luong report.
- `ranker` can hieu ngu canh va chon tin tot hon.
- `reviewer` can bat lap y, generic keyword, loi JSON.
- `planner` co the dung 120B de tao plan chat luong hon, nhung neu can toi uu chi phi/latency co the fallback nhanh.

## 6. SafeLLM Router

Tao file:

```text
SportNews_AI/core/safe_llm.py
```

Chuc nang:

- Nhan `agent_name`, `candidates`, `groq_fallback`.
- Thu tung model OpenRouter theo thu tu.
- Neu model loi/rate limit/timeout, log warning va thu model tiep theo.
- Neu tat ca OpenRouter fail, fallback ve Groq.
- Khong lam pipeline crash neu con fallback kha dung.

Pseudo-code:

```python
class SafeLLM:
    def invoke(self, prompt):
        if not openrouter_enabled:
            return self.groq_fallback.invoke(prompt)

        for model_name in self.candidates:
            try:
                llm = self._build_openrouter_llm(model_name)
                return llm.invoke(prompt)
            except Exception as exc:
                logger.warning("%s failed with %s: %s", self.agent_name, model_name, exc)

        return self.groq_fallback.invoke(prompt)
```

## 7. OpenRouter client

OpenRouter tuong thich OpenAI Chat Completions API. Co 2 cach:

### Cach A - Dung `langchain-openai`

Them vao `requirements.txt`:

```text
langchain-openai
```

Khoi tao:

```python
from langchain_openai import ChatOpenAI

ChatOpenAI(
    model="openai/gpt-oss-120b:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    default_headers={
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "SportNews AI"),
    },
    temperature=0.3,
)
```

### Cach B - Dung OpenAI SDK truc tiep

Them:

```text
openai
```

Tu viet wrapper `.invoke(prompt)` de giong LangChain response.

Khuyen nghi: dung Cach A vi code hien tai da theo LangChain `.invoke()`.

## 8. Factory `get_safe_llm()`

Tao file:

```text
SportNews_AI/core/llm.py
```

Chuc nang:

```python
def get_agent_llm():
    return ChatGroq(...)

def get_safe_llm(agent_name: str):
    if not OPENROUTER_API_KEY:
        return get_agent_llm()

    return SafeLLM(
        agent_name=agent_name,
        candidates=MODEL_CANDIDATES.get(agent_name, [OPENROUTER_PRIMARY_MODEL]),
        groq_fallback=get_agent_llm(),
    )
```

Yeu cau backward compatible:

- Giu `get_agent_llm()` de cac test/code cu co the dung.
- Chi thay agent sang `get_safe_llm()`.
- Neu OpenRouter key thieu, behavior gan nhu y nhu hien tai.

## 9. Sua cac agent

### 9.1 Planner

File:

```text
SportNews_AI/agents/planner.py
```

Thay:

```python
from langchain_groq import ChatGroq
...
llm = ChatGroq(...)
```

Bang:

```python
from core.llm import get_safe_llm
...
llm = get_safe_llm("planner")
```

### 9.2 Ranker

File:

```text
SportNews_AI/agents/ranker.py
```

Dung:

```python
llm = get_safe_llm("ranker")
```

### 9.3 Writer

File:

```text
SportNews_AI/agents/writer.py
```

Dung:

```python
llm = get_safe_llm("writer")
```

Dong thoi nang prompt:

- Yeu cau viet tieng Viet co dau.
- Cam keyword generic nhu `Thể thao`, `Tin tức`, `Bóng đá` neu khong di kem ngu canh.
- Daily report phai noi ro tin trong ngay, weekly report phai tong hop tuan.
- Moi highlight news phai bam sat article source, khong tao tin moi.
- Neu source URL co san, phai giu dung URL.

### 9.4 Reviewer

File:

```text
SportNews_AI/agents/reviewer.py
```

Dung:

```python
llm = get_safe_llm("reviewer")
```

Bo sung deterministic pre-check:

- Reject neu `executive_summary` qua ngan.
- Reject neu co keyword generic.
- Reject neu highlighted news source/url khong nam trong ranked articles.
- Reject neu co qua nhieu doan trung y.

## 10. Nang cap chat luong report

Sau khi dung model 120B, nen nang prompt writer thay vi chi doi model.

### 10.1 Summary prompt moi

Yeu cau:

- 4 doan, moi doan mot vai tro rieng.
- Khong lap cau mo dau.
- Co nhan dinh xu huong, khong chi liet ke.
- Tieng Viet chuan, khong dung tone marketing.
- Khong noi "trong tuan qua" neu `period_type=daily`, thay bang "trong ngày".

### 10.2 Highlight prompt moi

Yeu cau:

- Moi item co headline ngan, ro.
- Summary 2 cau.
- Khong doi source/url.
- Khong them tin khong co trong article list.

### 10.3 Reviewer prompt moi

Yeu cau reviewer tra:

```json
{
  "status": "approved" | "rejected",
  "reason": "...",
  "issues": ["..."]
}
```

Nhung de giu backward compatible, code van chi can doc `status` va `reason`.

## 11. Test can them

### 11.1 Unit test SafeLLM

File moi:

```text
SportNews_AI/tests/test_safe_llm.py
```

Test cases:

- Khi khong co `OPENROUTER_API_KEY`, `get_safe_llm()` tra Groq fallback.
- Khi OpenRouter model dau fail, router thu model thu hai.
- Khi tat ca OpenRouter fail, router fallback Groq.
- Router log agent/model failed.
- Router khong lam thay doi interface `.invoke(prompt)`.

### 11.2 Test agent integration

Cap nhat tests hien co:

- `test_planner.py`
- `test_writer.py`
- `test_reviewer.py`
- `test_ranker.py`

Patch `core.llm.get_safe_llm` thay vi patch `ChatGroq` truc tiep.

### 11.3 Regression test

Dam bao:

- `/generate-report` van tra schema cu.
- Metadata van co `period_type`, `source_count`, `ranked_count`, `stage`.
- Report van save DB backend nhu cu.
- Notification/email flow khong bi anh huong.

## 12. Rollout plan

### Buoc 1 - Them dependency va core LLM

- [x] Them `langchain-openai` vao `SportNews_AI/requirements.txt`.
- [x] Tao `core/model_candidates.py`.
- [x] Tao `core/safe_llm.py`.
- [x] Tao `core/llm.py`.
- [x] Them env OpenRouter vao `SportNews_AI/.env`.

### Buoc 2 - Chuyen agent sang SafeLLM

- [x] Sua `planner.py`.
- [x] Sua `ranker.py`.
- [x] Sua `writer.py`.
- [x] Sua `reviewer.py`.
- [x] Giu Groq fallback.

### Buoc 3 - Nang prompt writer/reviewer

- [x] Sua summary prompt de phan biet daily/weekly ro hon.
- [x] Sua news prompt de giu source/url chinh xac.
- [ ] Sua reviewer prompt va deterministic checks.

### Buoc 4 - Test

- [x] Them `test_safe_llm.py`.
- [x] Cap nhat tests patch dung `get_safe_llm`.
- [x] Chay:

```powershell
cd SportNews_AI
python -m pytest tests -q
```

### Buoc 5 - Docker verify

- [x] Build lai AI service:

```powershell
cd SportNews_AI
docker compose build
docker compose up -d --force-recreate
```

- [x] Test health/docs:

```powershell
curl http://localhost:8000/docs
```

- [ ] Test generate bang OpenRouter sau khi dien `OPENROUTER_API_KEY`:

```powershell
$body = @{ period_type = "daily"; lookback_days = 1 } | ConvertTo-Json
Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/generate-report" `
  -Headers @{ "X-Internal-API-Key" = "dev-ai-secret" } `
  -ContentType "application/json" `
  -Body $body
```

### Buoc 6 - Backend/frontend regression

- [ ] Backend khong can doi API.
- [ ] Tao report tu `/admin/reports`.
- [ ] Xac nhan report duoc luu Neon.
- [ ] Xac nhan notification duoc tao.
- [ ] Xac nhan email duoc gui neu user bat email.
- [ ] Xac nhan trang `/reports/:slug` hien dung.

## 13. Rui ro va cach giam rui ro

### 13.1 OpenRouter free model rate limit

Rui ro:

- `openai/gpt-oss-120b:free` co the bi rate limit.
- Free model co the unstable theo thoi diem.

Giam rui ro:

- SafeLLM fallback sang `z-ai/glm-4.5-air:free`.
- Cuoi cung fallback sang Groq.
- Log ro model nao dang fail.

### 13.2 Output JSON khong on dinh

Rui ro:

- Model lon co the viet giai thich kem JSON.

Giam rui ro:

- Giu parser cat JSON nhu hien tai.
- Prompt "Return ONLY valid JSON".
- Reviewer fail open/fallback neu parse loi.

### 13.3 Latency tang

Rui ro:

- 120B co the cham hon Groq 8B.

Giam rui ro:

- Backend admin generate dang async nen UI khong bi treo.
- Daily article limit da co `MAX_DAILY_REPORT_ARTICLES=25`.
- Scheduler chay nen nen user khong cho truc tiep.

### 13.4 ChatOpenAI dependency

Rui ro:

- Them `langchain-openai` can rebuild Docker image.

Giam rui ro:

- Docker build/test bat buoc trong rollout.
- Neu dependency loi, co the dung OpenAI SDK truc tiep.

## 14. Tieu chi hoan thanh

- [ ] `SportNews_AI` co OpenRouter config.
- [ ] Writer/ranker/reviewer/planner uu tien `openai/gpt-oss-120b:free`.
- [ ] Khong co OpenRouter key thi van chay Groq nhu cu.
- [ ] OpenRouter fail thi fallback Groq.
- [ ] `python -m pytest tests -q` pass.
- [ ] `docker compose build` AI service pass.
- [ ] Tao report daily thanh cong.
- [ ] Tao report weekly thanh cong.
- [ ] Backend/frontend flow cu khong doi API.
- [ ] Notification/email sau khi tao report van hoat dong.

## 15. Khuyen nghi trien khai

Nen trien khai theo thu tu:

1. Them SafeLLM + OpenRouter nhung chua sua prompt.
2. Chay test va generate 1 report daily.
3. Neu on, moi nang prompt writer/reviewer.
4. Chay report weekly de kiem tra latency.
5. Sau 1-2 lan generate thanh cong moi commit/merge.

Ly do: tach loi provider/model khoi loi prompt/parser, de debug nhanh hon.
