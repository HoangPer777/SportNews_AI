# Plan tích hợp Sport Report AI theo notification, email và bài viết đặc biệt

## 1. Đề xuất cách tích hợp 

Mục tiêu mới của hệ thống không chỉ là tạo một trang report riêng, mà là biến report AI thành **nội dung được phân phối chủ động cho người dùng**.

Luồng nên triển khai theo 3 giai đoạn:

1. **Giai đoạn 1 - Daily sport report notification**
   - `SportNews_AI` crawl 3 nguồn hiện có: VnExpress, Thanh Nien, Tuoi Tre.
   - AI tạo report thể thao mỗi ngày.
   - Backend Go lưu report vào database chính.
   - Backend tạo notification cho user đã bật tính năng nhận report.
   - Frontend hiển thị notification ở thanh header.
   - Nếu user bật email, backend gửi report qua email.
   - User có thể bật/tắt nhận report trong trang thông tin cá nhân hoặc trang cài đặt.

2. **Giai đoạn 2 - Embedding database hệ thống bằng pgvector: SKIP trong scope hiện tại**
   - Backend Go hiện đang dùng PostgreSQL. Đã xác nhận qua:
     - `DACN_Backend/pkg/db/postgres.go` dùng `gorm.io/driver/postgres`.
     - `DACN_Backend/config/config.go` có `DB_URL` mặc định dạng `postgres://...`.
     - `DACN_Backend/go.mod` có dependency `gorm.io/driver/postgres`.
   - Về mặt kỹ thuật có thể dùng `pgvector`, nhưng **không triển khai trong giai đoạn này**.
   - Lý do skip: embedding toàn bộ database, migration vector, job đồng bộ embedding, semantic search và recommendation làm phạm vi dự án phức tạp hơn nhiều so với mục tiêu chính.
   - Chỉ ghi lại như hướng mở rộng tương lai, không đưa vào checklist implement hiện tại.

3. **Giai đoạn 3 - Report thành bài viết đặc biệt hoặc top bài viết hệ thống**
   - Report có thể xuất hiện như:
     - Notification hằng ngày.
     - Email hằng ngày.
     - Bài viết đặc biệt của hệ thống.
     - Block "Top bài viết / AI Daily Sport Brief" trên trang chủ.
   - Admin có thể duyệt hoặc publish report thành article.

Khuyến nghị kiến trúc: **frontend không gọi trực tiếp `SportNews_AI`**. Frontend chỉ gọi `DACN_Backend`. Backend Go là trung tâm quản lý user, notification, email, permission, report storage và publish article. `SportNews_AI` chỉ làm nhiệm vụ crawl + AI generation.

```text
SportNews_AI
  crawl 3 sources + generate daily report
        |
        v
DACN_Backend
  save report + create notifications + send email
        |
        v
DACN_Frontend
  header notification + user preference + special report/article UI
```

## 2. Giai đoạn 1 - Report AI dưới dạng notification và email

### 2.1. Nghiệp vụ chính

Mỗi ngày hệ thống chạy job:

1. Gọi `SportNews_AI` tạo daily report.
2. Lưu report vào bảng `ai_reports`.
3. Lấy danh sách user đã bật `daily_report_enabled`.
4. Tạo notification cho từng user.
5. Nếu user bật `daily_report_email_enabled`, gửi email.
6. Frontend hiển thị số notification chưa đọc trên header.
7. User click notification để xem nhanh report hoặc mở bài viết/report chi tiết.

### 2.2. Bảng dữ liệu cần thêm trong backend

#### Bảng `ai_reports`

Dùng để lưu kết quả report AI.

```go
type AIReport struct {
    ID               int64
    Title            string
    Slug             string
    PeriodType       string // daily, weekly
    PeriodStart      time.Time
    PeriodEnd        time.Time
    ExecutiveSummary string
    TrendingKeywords datatypes.JSON
    HighlightedNews  datatypes.JSON
    Markdown         string
    Status           string // generating, ready, failed, archived, published
    ErrorMessage     string
    ArticleID        *int64 // nếu đã publish thành article
    GeneratedAt      *time.Time
    CreatedAt        time.Time
    UpdatedAt        time.Time
}
```

#### Bảng `user_report_preferences`

Dùng để user bật/tắt nhận report.

```go
type UserReportPreference struct {
    ID                       int64
    UserID                   int64
    DailyReportEnabled       bool
    DailyReportEmailEnabled  bool
    WeeklyReportEnabled      bool
    WeeklyReportEmailEnabled bool
    PreferredHour            int
    CreatedAt                time.Time
    UpdatedAt                time.Time
}
```

Mặc định đề xuất:

- `DailyReportEnabled = true`
- `DailyReportEmailEnabled = false`
- `WeeklyReportEnabled = true`
- `WeeklyReportEmailEnabled = false`

Như vậy user mới sẽ nhận notification trong app, nhưng không bị gửi email nếu chưa bật.

#### Bảng `notifications`

Dùng cho thông báo trên header.

```go
type Notification struct {
    ID        int64
    UserID    int64
    Type      string // ai_report, article_approved, comment, system
    Title     string
    Message   string
    Link      string
    Metadata  datatypes.JSON
    ReadAt    *time.Time
    CreatedAt time.Time
}
```

Với report AI:

- `Type = "ai_report"`
- `Title = "Bản tin thể thao AI hôm nay"`
- `Message = summary ngắn 1-2 câu`
- `Link = "/reports/{slug}"` hoặc `"/articles/{slug}"` nếu publish thành article.
- `Metadata` chứa `report_id`, `period_type`, `trending_keywords`.

#### Bảng `ai_report_deliveries`

Dùng để audit việc gửi report.

```go
type AIReportDelivery struct {
    ID                 int64
    ReportID           int64
    UserID             int64
    NotificationID     *int64
    EmailSent          bool
    EmailError         string
    DeliveredAt        *time.Time
    CreatedAt          time.Time
}
```

Bảng này giúp tránh gửi trùng cho cùng user/report.

### 2.3. Backend module đề xuất

Tạo thêm các module:

```text
DACN_Backend/internal/report/
  domain/report.go
  repository/report_repo.go
  usecase/report_usecase.go
  usecase/ai_client.go
  delivery/http/report_handler.go

DACN_Backend/internal/notification/
  domain/notification.go
  repository/notification_repo.go
  usecase/notification_usecase.go
  delivery/http/notification_handler.go

DACN_Backend/internal/preference/
  domain/report_preference.go
  repository/report_preference_repo.go
  usecase/report_preference_usecase.go
  delivery/http/report_preference_handler.go
```

Nếu muốn gọn cho đồ án, có thể để `UserReportPreference` trong module `report`.

### 2.4. API backend cần thêm

#### User preference

```http
GET /me/report-preferences
PUT /me/report-preferences
```

Payload:

```json
{
  "daily_report_enabled": true,
  "daily_report_email_enabled": false,
  "weekly_report_enabled": true,
  "weekly_report_email_enabled": false,
  "preferred_hour": 8
}
```

#### Notifications cho header

```http
GET /notifications?unread_only=false
GET /notifications/unread-count
POST /notifications/{id}/read
POST /notifications/read-all
```

Frontend header sẽ gọi:

- `GET /notifications/unread-count` để hiện badge.
- `GET /notifications?limit=10` để hiện dropdown.

#### Reports

```http
GET /reports
GET /reports/latest?period_type=daily
GET /reports/{slug}
```

#### Admin

```http
POST /admin/reports/generate
POST /admin/reports/{id}/send
POST /admin/reports/{id}/publish-as-article
GET /admin/reports
```

Payload generate:

```json
{
  "period_type": "daily",
  "lookback_days": 1,
  "send_to_users": true
}
```

### 2.5. Scheduler nên đặt ở đâu?


#### Scheduler nằm trong backend Go

Backend Go chạy job mỗi ngày:

1. Gọi `SportNews_AI`.
2. Nhận report.
3. Lưu DB.
4. Tạo notification.
5. Gửi email.

Đây là hướng khuyến nghị vì backend có đủ dữ liệu user, email, preference, notification và permission.

Với đồ án, có thể dùng goroutine + ticker hoặc thư viện cron Go. Nếu muốn đơn giản hơn, admin trigger thủ công trước, sau đó mới thêm scheduler.

### 2.6. Email report

Backend đã có `SMTPMailer` trong `internal/auth/usecase/mailer.go`, nhưng hiện interface chỉ có:

- `SendPasswordReset`
- `SendEmailVerification`

Cần sửa code backend để thêm method mới:

```go
SendDailyReport(toEmail, fullName, subject, htmlBody string) error
```

Hoặc tạo mailer riêng trong module `report`:

```go
type ReportMailer interface {
    SendReport(toEmail, fullName string, report AIReport) error
}
```

Khuyến nghị: mở rộng mailer hiện tại để tái sử dụng SMTP config.

Email chỉ gửi cho user:

- Có email.
- `EmailVerified = true`.
- `DailyReportEmailEnabled = true`.

## 3. Frontend cần sửa trong giai đoạn 1

### 3.1. Header notification

Hiện `SportsNewsSection.jsx` đang có header riêng trong home. Nên thêm:

- Icon chuông ở header.
- Badge số notification chưa đọc.
- Dropdown 5-10 thông báo mới nhất.
- Click notification thì mark read và điều hướng đến link.

Service mới:

```text
src/services/notificationService.js
```

Functions:

```js
listNotifications(params)
getUnreadNotificationCount()
markNotificationRead(id)
markAllNotificationsRead()
```

### 3.2. Trang cài đặt bật/tắt report

Có thể thêm vào `UserInfoPage.jsx` hoặc tạo page riêng:

```text
/me/report-preferences
```

Controls:

- Toggle "Nhận bản tin thể thao AI hằng ngày trong ứng dụng".
- Toggle "Gửi bản tin thể thao AI hằng ngày qua email".
- Toggle "Nhận tổng hợp thể thao hằng tuần".
- Toggle "Gửi tổng hợp hằng tuần qua email".

Service mới:

```text
src/services/reportPreferenceService.js
```

### 3.3. Trang xem report

Vẫn nên có trang chi tiết report để notification trỏ tới:

```text
/reports/:slug
```

Trang này render:

- Executive summary.
- Trending keywords.
- Highlighted news.
- Source URLs.
- Generated time.

## 4. SportNews_AI cần sửa gì cho giai đoạn 1

Có, cần sửa code ở AI service. Các sửa đổi chính:

### 4.1. Cho phép daily report

Hiện pipeline hard-code 7 ngày. Cần đổi thành tham số:

```python
class GenerateReportRequest(BaseModel):
    period_type: Literal["daily", "weekly"] = "daily"
    lookback_days: int | None = None
```

Logic:

```python
if request.lookback_days is None:
    lookback_days = 1 if request.period_type == "daily" else 7
```

Các file cần sửa:

- `SportNews_AI/main.py`
  - Nhận request body.
  - Truyền `period_type`, `lookback_days` vào `run_pipeline`.
  - Trả thêm metadata.

- `SportNews_AI/graph.py`
  - `run_pipeline()` đổi thành `run_pipeline(period_type="daily", lookback_days=1)`.
  - Gọi crawler/filter/db theo `lookback_days`.

- `SportNews_AI/tools/crawler.py`
  - Bỏ lambda `SEVEN_DAYS_AGO`.
  - `_crawl_source(..., lookback_days: int)`.
  - `crawl_all_sources(lookback_days: int)`.

- `SportNews_AI/tools/preprocess.py`
  - `filter_recent_articles(articles, lookback_days: int)`.

- `SportNews_AI/tools/db.py`
  - `get_articles_last_7_days` đổi thành `get_articles_by_lookback_days(engine, lookback_days)`.

- `SportNews_AI/agents/planner.py`
  - `_get_week_date_range` đổi thành `_get_period_date_range(period_type, lookback_days)`.
  - Prompt không nên luôn nói "weekly intelligence report" nếu đang daily.

- `SportNews_AI/agents/writer.py`
  - Prompt đổi theo daily/weekly.
  - Markdown title đổi:
    - daily: `Báo Cáo Thể Thao Hôm Nay`
    - weekly: `Báo Cáo Thể Thao Tuần`

### 4.2. Response cần thêm metadata

AI service nên trả:

```json
{
  "status": "success",
  "report": {
    "executive_summary": "...",
    "trending_keywords": ["..."],
    "highlighted_news": [],
    "generated_at": "..."
  },
  "metadata": {
    "period_type": "daily",
    "period_start": "2026-06-12",
    "period_end": "2026-06-12",
    "source_count": 20,
    "ranked_count": 8
  }
}
```

### 4.3. Bảo vệ endpoint AI

Thêm header nội bộ:

```http
X-Internal-API-Key: dev-ai-secret
```

Env:

```env
INTERNAL_API_KEY=dev-ai-secret
```

Nếu header sai, trả `401`.

### 4.4. Không nên để AI service gửi email trực tiếp

`SportNews_AI` hiện có `tools/email_sender.py`. Với kiến trúc mới, nên để backend Go gửi email vì:

- Backend biết user nào bật/tắt email.
- Backend có dữ liệu `EmailVerified`.
- Backend lưu delivery log.
- Backend tránh gửi trùng.

AI service chỉ nên trả report JSON.

## 5. Giai đoạn 2 - Embedding toàn bộ database hệ thống bằng pgvector: SKIP

### 5.1. Quyết định scope

Backend Go đang dùng PostgreSQL, nên về mặt kỹ thuật có thể dùng `pgvector`. Tuy nhiên, **giai đoạn 2 sẽ được bỏ qua trong scope hiện tại**.

Lý do:

- Cần đổi hoặc cấu hình PostgreSQL image có extension `pgvector`.
- Cần migration vector và thư viện Go riêng.
- Cần job embedding lại toàn bộ bài viết/report hiện có.
- Cần xử lý cập nhật embedding khi article/report thay đổi.
- Cần thiết kế semantic search/recommendation để dùng vector hiệu quả.
- Phạm vi này quá phức tạp so với mục tiêu trước mắt: gửi report cho user qua notification/email và publish thành bài viết đặc biệt.

Vì vậy, plan hiện tại chỉ triển khai:

1. Giai đoạn 1: notification/email daily report.
2. Giai đoạn 3: publish report thành article đặc biệt và top bài viết hệ thống theo metric sẵn có.

Các nội dung pgvector bên dưới chỉ là ghi chú kỹ thuật cho tương lai, **không nằm trong checklist implement hiện tại**.

Các bằng chứng trong code:

- `DACN_Backend/pkg/db/postgres.go` dùng `postgres.Open(dsn)`.
- `DACN_Backend/config/config.go` default `DB_URL=postgres://sportnews:sportnews@localhost:5432/sportnews?sslmode=disable`.
- `DACN_Backend/go.mod` có `gorm.io/driver/postgres`.

### 5.2. Ghi chú tương lai: cài extension pgvector

Migration SQL:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Nếu dùng Docker Postgres thường, cần image có pgvector, ví dụ:

```text
pgvector/pgvector:pg16
```

### 5.3. Bảng embedding đề xuất

Không nên nhét vector thẳng vào bảng `articles` ngay từ đầu. Nên tạo bảng riêng:

```go
type ContentEmbedding struct {
    ID          int64
    EntityType  string // article, ai_report, category
    EntityID    int64
    ContentHash string
    Model       string
    Dimensions  int
    Embedding   pgvector.Vector
    CreatedAt   time.Time
    UpdatedAt   time.Time
}
```

Unique index:

```text
unique(entity_type, entity_id, model)
```

Các dữ liệu cần embedding:

- `articles.title`
- `articles.summary`
- `article_blocks.content`
- `ai_reports.executive_summary`
- `ai_reports.highlighted_news`
- Có thể thêm category/user interest về sau.

### 5.4. Go library đề xuất

Dùng:

```text
github.com/pgvector/pgvector-go
```

GORM model có thể dùng:

```go
import "github.com/pgvector/pgvector-go"

Embedding pgvector.Vector `gorm:"type:vector(768)"`
```

Số chiều phụ thuộc embedding model:

- Gemini embedding có thể không cố định theo model/version nếu không cấu hình rõ.
- Cần kiểm tra dimension thực tế từ `SportNews_AI`.
- Nếu dùng `models/gemini-embedding-001`, thường nên lưu model và dimensions trong bảng để tránh mismatch về sau.

### 5.5. Ai chịu trách nhiệm embedding DB hệ thống?

Có 2 cách:

#### Cách A - Backend Go gọi embedding API trực tiếp

Backend Go tự gọi Gemini embedding và lưu pgvector.

Ưu điểm:

- Ít service hop.
- Backend kiểm soát DB chính.

Nhược điểm:

- Phải implement Gemini embedding client trong Go.
- Lặp logic với `SportNews_AI`.

#### Cách B - Backend gửi content sang `SportNews_AI` để embedding

Thêm endpoint AI service:

```http
POST /embed-texts
```

Request:

```json
{
  "texts": [
    "title summary content...",
    "another content..."
  ]
}
```

Response:

```json
{
  "model": "models/gemini-embedding-001",
  "dimensions": 768,
  "embeddings": [[0.1, 0.2]]
}
```

Backend nhận vector và lưu vào PostgreSQL bằng pgvector.

Khuyến nghị cho project này: **Cách B** vì `SportNews_AI` đã có code embedding trong `tools/embeddings.py`.

### 5.6. Ghi chú tương lai: chức năng có thể làm sau khi có pgvector

- Semantic search bài viết.
- Gợi ý bài viết liên quan tốt hơn `same category`.
- Top bài viết cá nhân hóa theo sở thích user.
- Report AI dựa trên cả báo ngoài và bài viết nội bộ.
- Gửi notification report phù hợp với từng user thay vì cùng một report cho tất cả.

## 6. Giai đoạn 3 - Report thành bài viết đặc biệt hoặc top bài viết hệ thống

### 6.1. Publish report thành article đặc biệt

Backend map `AIReport` thành `Article`:

- `Title`: `Bản tin thể thao AI ngày DD/MM/YYYY`
- `Summary`: đoạn đầu của `executive_summary`.
- `CategoryID`: category `AI Report` hoặc `Tổng hợp`.
- `AuthorID`: user hệ thống `AI Reporter`.
- `Status`: `pending` nếu muốn admin duyệt, hoặc `approved` nếu tự động publish.
- `Blocks`:
  - heading `Tổng quan`
  - paragraph executive summary
  - heading `Từ khóa nổi bật`
  - paragraph keywords
  - heading `Tin nổi bật`
  - paragraph từng highlighted news + source URL

Sau khi publish:

- `ai_reports.article_id = articles.id`
- Notification link có thể trỏ về `/articles/{slug}`.

### 6.2. Top bài viết hệ thống

Trong scope hiện tại, "Top bài viết hệ thống" **không dùng pgvector**. Backend có thể tính top bài viết bằng các metric đã có trong hệ thống:

Tiêu chí phase đầu:

- `view_count`
- `share_count`
- `bookmark_count`
- `comment_count`
- recency

Tiêu chí phase sau nếu quay lại làm embedding:

- semantic similarity với sở thích user.
- trending keywords trong report AI.
- category user quan tâm.

## 7. Thay đổi cụ thể trong backend Go

### 7.1. `cmd/server/main.go`

Cần AutoMigrate thêm:

- `AIReport`
- `UserReportPreference`
- `Notification`
- `AIReportDelivery`
- Không thêm `ContentEmbedding` trong scope hiện tại vì phase pgvector đã được skip.

Cần khởi tạo:

- report repository/usecase/handler.
- notification repository/usecase/handler.
- preference repository/usecase/handler.

Register routes:

- notification routes.
- report preference routes.
- report routes.
- admin report routes.

### 7.2. `config/config.go`

Thêm:

```go
AIReportServiceURL string
AIReportAPIKey     string
ReportScheduleHour string
```

Env:

```env
AI_REPORT_SERVICE_URL=http://localhost:8000
AI_REPORT_API_KEY=dev-ai-secret
REPORT_SCHEDULE_HOUR=8
```

### 7.3. `auth/usecase/mailer.go`

Cần thêm method gửi report.

Ví dụ:

```go
SendDailyReport(toEmail, fullName, subject, htmlBody string) error
```

### 7.4. `auth/repository/user_repo.go`

Cần thêm method lấy user nhận report:

```go
FindReportSubscribers(ctx context.Context, periodType string, emailOnly bool) ([]authdomain.User, error)
```

Hoặc query nằm trong report repository bằng join `users` + `user_report_preferences`.

## 8. Thay đổi cụ thể trong frontend React

### 8.1. Service mới

```text
src/services/notificationService.js
src/services/reportService.js
src/services/reportPreferenceService.js
```

### 8.2. Component mới

```text
src/features/notifications/components/NotificationBell.jsx
src/features/reports/pages/ReportDetailPage.jsx
src/features/reports/pages/ReportsPage.jsx
src/features/reports/components/ReportPreferencePanel.jsx
```

### 8.3. Header

Thêm `NotificationBell` vào header trong `SportsNewsSection.jsx` hoặc tốt hơn là đưa header vào `ReaderLayout` để mọi trang đều có thông báo.

### 8.4. User settings

Thêm panel bật/tắt report vào `UserInfoPage.jsx`.

## 9. Checklist triển khai

### Phase 1 - Notification/email report

- [ ] Backend: thêm config AI service URL/API key.
- [ ] Backend: thêm bảng `ai_reports`.
- [ ] Backend: thêm bảng `user_report_preferences`.
- [ ] Backend: thêm bảng `notifications`.
- [ ] Backend: thêm bảng `ai_report_deliveries`.
- [ ] Backend: thêm API preference bật/tắt report.
- [ ] Backend: thêm API notification cho header.
- [ ] Backend: thêm AI client gọi `SportNews_AI`.
- [ ] Backend: thêm job generate daily report.
- [ ] Backend: tạo notification cho user đã bật report.
- [ ] Backend: gửi email cho user bật email report.
- [ ] Frontend: thêm notification bell trên header.
- [ ] Frontend: thêm dropdown notification.
- [ ] Frontend: thêm trang/panel bật tắt report.
- [ ] Frontend: thêm trang xem report detail.
- [ ] AI service: sửa report từ weekly hard-code sang daily/weekly configurable.
- [ ] AI service: thêm internal API key.
- [ ] AI service: trả metadata period/source/ranked count.

### Phase 2 - pgvector embedding: SKIP

- [x] Quyết định skip phase này trong scope hiện tại.
- [x] Lý do: quá phức tạp ở giai đoạn này so với mục tiêu notification/email/report article.
- [ ] Chỉ xem lại sau khi phase 1 và phase 3 đã ổn định.

### Phase 3 - Special article/top articles

- [ ] Backend: tạo category `AI Report`.
- [ ] Backend: tạo hoặc seed user hệ thống `AI Reporter`.
- [ ] Backend: API publish report thành article.
- [ ] Frontend: hiển thị article đặc biệt trên home.
- [ ] Backend: tính top bài viết dựa trên view/share/comment/bookmark.
- [ ] Phase sau nếu cần: top bài viết cá nhân hóa bằng pgvector.

## 10. Kết luận

Với yêu cầu mới, hướng đúng nhất là ưu tiên **hệ thống notification report** trước, không chỉ là trang report. `SportNews_AI` tạo nội dung daily/weekly, còn `DACN_Backend` chịu trách nhiệm lưu report, kiểm tra user nào đã bật nhận report, tạo notification trên header và gửi email nếu user bật. Giai đoạn pgvector được xác nhận là khả thi vì backend dùng PostgreSQL, nhưng **skip trong scope hiện tại** vì quá phức tạp. Sau phase notification/email, trọng tâm tiếp theo là publish report thành bài viết đặc biệt và tạo top bài viết hệ thống bằng các metric sẵn có như view/share/comment/bookmark.
