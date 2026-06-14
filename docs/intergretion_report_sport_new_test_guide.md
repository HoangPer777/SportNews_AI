# Huong dan test tinh nang Sport Report AI bang giao dien

File nay huong dan cach chay va test flow moi:

- AI service tao daily sport report.
- Backend luu report, tao notification, gui email neu user bat.
- Frontend hien notification tren header.
- Admin co the generate/send/publish report.
- User co the bat/tat nhan report.

## 1. Chay AI service

Trong thu muc `SportNews_AI`, kiem tra file `.env` co cac bien can thiet.

Vi du:

```env
DATABASE_URL=postgresql://sportsuser:secret@db:5432/sportsdb
POSTGRES_USER=sportsuser
POSTGRES_PASSWORD=secret
POSTGRES_DB=sportsdb

GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
GROQ_LLM_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=models/gemini-embedding-001

INTERNAL_API_KEY=dev-ai-secret
REPORT_PERIOD=daily
REPORT_LOOKBACK_DAYS=1
```

Luu y quan trong ve database cua AI service:

- `SportNews_AI` dung Postgres rieng trong Docker volume `sportnews_ai_pgdata`.
- Neu volume da tung duoc tao bang user/db cu, viec sua `.env` sang user/db moi se khong tu tao role/database moi trong Postgres.
- Neu gap loi `password authentication failed` hoac `role does not exist`, hay dung lai dung credential da tao volume, hoac xoa volume de tao database moi.

Phuong an giu volume hien tai:

```env
DATABASE_URL=postgresql://sportsuser:secret@db:5432/sportsdb
POSTGRES_USER=sportsuser
POSTGRES_PASSWORD=secret
POSTGRES_DB=sportsdb
```

Phuong an tao lai database AI tu dau:

```powershell
cd SportNews_AI
docker compose down -v
docker compose up -d --build
```

Chi dung `docker compose down -v` khi chap nhan xoa du lieu Postgres trong AI service.

Chay AI service:

```powershell
cd SportNews_AI
docker compose up --build
```

AI service se chay tai:

```text
http://localhost:8000
```

PostgreSQL rieng cua AI service duoc publish ra may host o cong:

```text
localhost:5433
```

Ly do dung `5433`: tranh dung cong voi database local cua backend o `5432`.

## 2. Database backend

Backend cua ung dung chay tren **Neon PostgreSQL thuc te**, khong dung Postgres local.

Trong `DACN_Backend/.env`, `DB_URL` phai tro den Neon:

```env
DB_URL=postgresql://...neon.tech/neondb?sslmode=require&channel_binding=require
```

Khong can chay container PostgreSQL local cho backend.

Ban chi can chay backend:

```powershell
cd DACN_Backend
docker compose up --build
```

Compose se chay:

- Backend tai `http://localhost:8080`
- Database backend la Neon remote trong `DB_URL`

## 3. Cau hinh backend

Trong `DACN_Backend/.env`, them hoac cap nhat:

```env
DB_URL=postgresql://...neon.tech/neondb?sslmode=require&channel_binding=require
JWT_SECRET=dev-secret
ADMIN_SECRET=admin-secret
APP_BASE_URL=http://localhost:8080

AI_REPORT_SERVICE_URL=http://host.docker.internal:8000
AI_REPORT_API_KEY=dev-ai-secret
REPORT_SCHEDULE_ENABLED=false
REPORT_SCHEDULE_HOUR=8
```

Luu y:

- `AI_REPORT_API_KEY` cua backend phai trung voi `INTERNAL_API_KEY` cua `SportNews_AI`.
- `REPORT_SCHEDULE_ENABLED=false` de test thu cong bang giao dien admin truoc.
- Khi chay backend bang `docker compose`, backend doc `DB_URL` Neon tu `.env`.
- Neu backend `/health` tra `ok`, nghia la backend ket noi duoc Neon.

Chay backend:

```powershell
cd DACN_Backend
docker compose up --build
```

Kiem tra backend:

```powershell
curl http://localhost:8080/health
```

Neu tra ve:

```text
ok
```

la backend da chay duoc.

## 4. Chay frontend

Trong `DACN_Frontend/sportnews-frontend`:

```powershell
cd DACN_Frontend\sportnews-frontend
npm install
npm run dev
```

Mo trinh duyet:

```text
http://localhost:5173
```

## 5. Test flow admin tao report

### 5.1. Dang nhap admin

Dang nhap bang tai khoan admin.

Neu chua co admin, tao admin theo flow hien co cua backend/frontend hoac dung API `/auth/register-admin` voi `ADMIN_SECRET`.

### 5.2. Mo trang quan ly AI Reports

Vao:

```text
http://localhost:5173/admin/reports
```

### 5.3. Tao report

Bam:

```text
Tao report ngay
```

He thong se:

1. Frontend goi backend `POST /admin/reports/generate`.
2. Backend goi AI service `POST /generate-report`.
3. AI service crawl 3 nguon:
   - VnExpress
   - Thanh Nien
   - Tuoi Tre
4. AI service tao daily report.
5. Backend luu report vao bang `ai_reports`.

Qua trinh nay co the lau vi co crawl, embedding va LLM.

Ket qua mong doi:

- Report xuat hien trong bang admin.
- Status la `ready`.

## 6. Test gui notification cho user

Sau khi report co status `ready`, o trang admin reports bam:

```text
Gui thong bao
```

Backend se:

1. Lay danh sach user da bat nhan report.
2. Tao notification cho tung user.
3. Ghi delivery log.
4. Neu user bat email va email verified, gui email report.

Ket qua mong doi:

- User nhan notification tren header.
- Neu cau hinh SMTP dung va user bat email, user nhan email.

## 7. Test user nhan notification tren header

Dang nhap bang tai khoan user thuong.

Tren header, kiem tra icon chuong.

Ket qua mong doi:

- Neu co notification moi, badge hien so notification chua doc.
- Bam icon chuong se mo dropdown notification.
- Bam notification se dieu huong toi trang report detail:

```text
/reports/:slug
```

- Sau khi bam notification, notification duoc mark read va badge giam.

## 8. Test user bat/tat nhan report

Dang nhap user va vao:

```text
http://localhost:5173/me
```

Trong trang thong tin ca nhan se co panel:

```text
Cai dat nhan bao cao the thao
```

Test cac toggle:

- Nhan ban tin hang ngay trong ung dung.
- Gui ban tin hang ngay qua email.
- Nhan tong hop hang tuan trong ung dung.
- Gui tong hop hang tuan qua email.

Flow can test:

1. Tat `Nhan ban tin hang ngay trong ung dung`.
2. Admin gui report.
3. User nay khong nhan notification.
4. Bat lai toggle.
5. Admin gui report moi.
6. User nhan notification.

## 9. Test publish report thanh article dac biet

Vao:

```text
http://localhost:5173/admin/reports
```

Voi report status `ready`, bam:

```text
Publish
```

Backend se:

1. Lay report trong `ai_reports`.
2. Map report thanh `Article`.
3. Tao cac block:
   - Tong quan
   - Tu khoa noi bat
   - Tin noi bat
4. Gan `article_id` vao report.
5. Doi status report thanh `published`.

Ket qua mong doi:

- Report duoc publish thanh article.
- Article co status `approved`.
- Article co the doc nhu bai viet binh thuong.

## 10. Test top bai viet tren trang chu

Mo:

```text
http://localhost:5173
```

Kiem tra section:

```text
Top bai viet
Noi bat trong tuan
```

Backend endpoint duoc dung:

```http
GET /articles/top?period=weekly&limit=5
```

Top bai viet hien tai tinh bang metric san co:

- `view_count`
- `share_count`
- `bookmark_count`
- `comment_count`
- recency theo period

Khong dung pgvector trong phase hien tai.

## 11. Cac API co the test nhanh bang curl

### Backend health

```powershell
curl http://localhost:8080/health
```

### AI service generate report

Neu muon test truc tiep AI service:

```powershell
curl -X POST http://localhost:8000/generate-report `
  -H "Content-Type: application/json" `
  -H "X-Internal-API-Key: dev-ai-secret" `
  -d "{\"period_type\":\"daily\",\"lookback_days\":1}"
```

### Backend latest report

```powershell
curl http://localhost:8080/reports/latest?period_type=daily
```

## 12. Loi thuong gap

### 12.1. Admin bam tao report bi loi 401 tu AI service

Kiem tra:

```env
AI_REPORT_API_KEY=dev-ai-secret
INTERNAL_API_KEY=dev-ai-secret
```

Hai gia tri nay phai giong nhau.

### 12.1.1. AI service tao report bi loi password Postgres

Neu log co loi dang nay:

```text
password authentication failed for user "sportnews"
```

Nguyen nhan thuong gap la Docker volume Postgres cua AI service da duoc khoi tao truoc do bang user/db khac. Postgres chi doc `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` trong lan khoi tao volume dau tien.

Cach sua khuyen dung neu muon giu du lieu cu:

```env
DATABASE_URL=postgresql://sportsuser:secret@db:5432/sportsdb
POSTGRES_USER=sportsuser
POSTGRES_PASSWORD=secret
POSTGRES_DB=sportsdb
```

Sau do restart AI service:

```powershell
cd SportNews_AI
docker compose up -d --force-recreate
```

Kiem tra ket noi DB:

```powershell
docker compose exec -T db psql -U sportsuser -d sportsdb -c "SELECT current_user, current_database();"
```

Cach sua neu muon dung credential moi va chap nhan xoa DB AI hien tai:

```powershell
cd SportNews_AI
docker compose down -v
docker compose up -d --build
```

### 12.2. AI service tao report qua lau

Nguyen nhan co the la:

- Dang crawl nguon bao cham.
- Dang goi Gemini embedding.
- Dang goi Groq LLM.
- API key bi rate limit.

Kiem tra log cua container `agent`.

### 12.3. User khong nhan notification

Kiem tra:

- User da dang nhap dung tai khoan chua.
- User co bat `daily_report_enabled` khong.
- Admin da bam `Gui thong bao` chua.
- Report co status `ready` hoac `published` khong.

### 12.4. Khong gui email

Kiem tra:

- User da bat email report chua.
- User co `email_verified = true` khong.
- Backend da cau hinh SMTP chua.

Neu chua cau hinh SMTP, backend chi log dev va khong gui email that.

### 12.5. Frontend lint fail

`npm run build` da pass cho feature moi.

`npm run lint` toan project hien co the fail vi mot so loi cu ngoai pham vi feature nay. Cac file moi cua tinh nang report da duoc lint rieng va pass.

## 13. Checklist test giao dien

- [ ] Chay duoc AI service tai `http://localhost:8000`.
- [ ] Chay duoc backend tai `http://localhost:8080`.
- [ ] Chay duoc frontend tai `http://localhost:5173`.
- [ ] Admin vao duoc `/admin/reports`.
- [ ] Admin tao duoc daily report.
- [ ] Report co status `ready`.
- [ ] Admin gui duoc notification.
- [ ] User thay badge notification tren header.
- [ ] User click notification mo duoc report detail.
- [ ] User bat/tat nhan report trong `/me`.
- [ ] Admin publish report thanh article.
- [ ] Trang chu hien top bai viet.
