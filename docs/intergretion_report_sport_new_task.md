# Task triển khai Sport Report AI notification, email và bài viết đặc biệt

## Quy ước trạng thái

- `[ ]` Chưa làm.
- `[~]` Đang làm.
- `[x]` Đã làm xong.

Scope hiện tại:

- Làm **Phase 1: Daily sport report notification/email**.
- Skip **Phase 2: pgvector embedding** vì quá phức tạp trong giai đoạn này.
- Làm **Phase 3: publish report thành bài viết đặc biệt/top bài viết hệ thống** sau khi Phase 1 ổn định.

## 0. Chuẩn bị và quyết định kỹ thuật

- [x] Đọc lại `intergretion_report_sport_new.md` và xác nhận scope cuối cùng với team.
- [x] Xác nhận Phase 2 pgvector chỉ là ghi chú tương lai, không tạo table `content_embeddings`, không thêm dependency `pgvector`.
- [x] Chọn strategy scheduler cho Phase 1: backend Go là nơi gọi `SportNews_AI` và gửi notification/email.
- [x] Chọn strategy generate ban đầu: admin trigger thủ công trước, scheduler daily thêm sau.
- [ ] Chọn policy gửi report mặc định:
  - [ ] In-app notification mặc định bật.
  - [ ] Email mặc định tắt.
  - [ ] Chỉ gửi email nếu `email_verified = true`.
- [ ] Chọn policy publish article:
  - [ ] Admin duyệt thủ công.
  - [ ] Hoặc auto publish sau khi report `ready`.
- [ ] Xác định `AI Report` category sẽ seed tự động hay tạo thủ công từ admin.
- [ ] Xác định user hệ thống `AI Reporter` sẽ seed tự động hay dùng admin hiện có.
- [ ] Thống nhất timezone xử lý report: `Asia/Ho_Chi_Minh`.
- [ ] Thống nhất naming slug report: `ban-tin-the-thao-ai-YYYY-MM-DD`.
- [ ] Thống nhất route frontend:
  - [ ] `/reports`
  - [ ] `/reports/:slug`
  - [ ] `/admin/reports`

### Test/verification

- [ ] Kiểm tra không có task nào yêu cầu pgvector trong checklist implement chính.
- [ ] Kiểm tra route và naming không trùng route hiện có.
- [ ] Kiểm tra backend, frontend, AI service có thể chạy độc lập trước khi tích hợp.

## 1. AI service `SportNews_AI`

Mục tiêu: AI service chỉ tạo report JSON theo daily/weekly, không gửi email trực tiếp, không biết user hệ thống.

### 1.1. Request/response contract

- [x] Tạo hoặc cập nhật schema request trong `SportNews_AI/models/schemas.py`.
- [x] Thêm `GenerateReportRequest`:
  - [x] `period_type: "daily" | "weekly"`, default `daily`.
  - [x] `lookback_days: int | None`, default `None`.
- [x] Thêm `ReportMetadata`:
  - [x] `period_type`
  - [x] `period_start`
  - [x] `period_end`
  - [x] `source_count`
  - [x] `ranked_count`
  - [x] `lookback_days`
- [x] Cập nhật `ReportResponse` để có thêm `metadata`.
- [x] Đảm bảo response cũ vẫn không phá test quá nhiều: `status`, `report`, `error` vẫn còn.

### 1.2. Internal API key

- [x] Thêm env `INTERNAL_API_KEY`.
- [x] Thêm dependency kiểm tra header `X-Internal-API-Key`.
- [x] Cho phép bỏ qua auth nếu `INTERNAL_API_KEY` rỗng trong môi trường dev/test, hoặc quyết định luôn bắt buộc.
- [x] Cập nhật `POST /generate-report` để kiểm tra API key.
- [x] Đảm bảo lỗi sai key trả `401`.
- [x] Không log API key ra console.

### 1.3. Chuyển hard-code 7 ngày sang configurable

- [x] Sửa `SportNews_AI/main.py`.
  - [x] Endpoint nhận request body.
  - [x] Tính `lookback_days = 1` nếu `daily`.
  - [x] Tính `lookback_days = 7` nếu `weekly`.
  - [x] Truyền `period_type`, `lookback_days` vào `run_pipeline`.
- [x] Sửa `SportNews_AI/graph.py`.
  - [x] `run_pipeline(period_type="daily", lookback_days=1)`.
  - [x] Truyền `lookback_days` vào crawler/preprocess/db.
  - [x] Lưu `source_count` và `ranked_count` vào state hoặc metadata.
- [x] Sửa `SportNews_AI/tools/crawler.py`.
  - [x] Bỏ lambda `SEVEN_DAYS_AGO`.
  - [x] `_crawl_source(..., lookback_days: int)`.
  - [x] `crawl_vnexpress(lookback_days)`.
  - [x] `crawl_thanhnien(lookback_days)`.
  - [x] `crawl_tuoitre(lookback_days)`.
  - [x] `crawl_all_sources(lookback_days)`.
- [x] Sửa `SportNews_AI/tools/preprocess.py`.
  - [x] `filter_recent_articles(articles, lookback_days)`.
  - [x] Validate `lookback_days >= 1`.
- [x] Sửa `SportNews_AI/tools/db.py`.
  - [x] Đổi `get_articles_last_7_days` thành `get_articles_by_lookback_days`.
  - [x] Giữ wrapper `get_articles_last_7_days` nếu test cũ còn dùng.
- [x] Sửa `SportNews_AI/agents/planner.py`.
  - [x] Đổi `_get_week_date_range` thành `_get_period_date_range`.
  - [x] Prompt nói đúng daily/weekly.
  - [x] Daily date range chỉ là một ngày.
- [x] Sửa `SportNews_AI/agents/writer.py`.
  - [x] Prompt nói đúng daily/weekly.
  - [x] Markdown title daily: `Báo Cáo Thể Thao Hôm Nay`.
  - [x] Markdown title weekly: `Báo Cáo Thể Thao Tuần`.
  - [x] Output path có thể theo period: `outputs/daily_report.md`, `outputs/weekly_report.md`.
- [ ] Sửa `SportNews_AI/agents/retriever.py` nếu cần query daily/weekly khác nhau.
- [ ] Sửa `SportNews_AI/agents/reviewer.py` nếu prompt vẫn nói weekly cứng.

### 1.4. Scheduler trong AI service

- [x] Tắt vai trò gửi email trực tiếp của AI service trong flow mới.
- [x] Giữ scheduler AI service nếu dùng cho dev/demo, nhưng ghi rõ backend mới là scheduler chính.
- [ ] Nếu giữ scheduler:
  - [ ] Scheduler chỉ tạo report local, không gửi email user.
  - [ ] Không gọi `send_report_email` mặc định.
- [ ] Cập nhật README AI service về kiến trúc mới.

### 1.5. AI service tests

- [x] Cập nhật `SportNews_AI/tests/test_schemas.py`.
  - [x] Test `GenerateReportRequest` default là daily.
  - [x] Test weekly lookback default là 7.
  - [x] Test daily lookback default là 1.
  - [x] Test invalid `period_type` bị reject.
- [x] Cập nhật `SportNews_AI/tests/test_preprocess.py`.
  - [x] Test filter với `lookback_days=1`.
  - [x] Test filter với `lookback_days=7`.
  - [x] Test article quá hạn bị loại.
- [x] Cập nhật `SportNews_AI/tests/test_db.py`.
  - [x] Test `get_articles_by_lookback_days`.
  - [x] Test wrapper cũ nếu còn giữ.
- [x] Cập nhật `SportNews_AI/tests/test_planner.py`.
  - [x] Test daily period date range.
  - [x] Test weekly period date range.
- [x] Cập nhật `SportNews_AI/tests/test_writer.py`.
  - [x] Test markdown title daily.
  - [x] Test markdown title weekly.
- [x] Cập nhật `SportNews_AI/tests/test_api.py`.
  - [x] Test thiếu API key trả `401` nếu env bắt buộc.
  - [x] Test sai API key trả `401`.
  - [x] Test đúng API key gọi được endpoint.
  - [x] Mock `run_pipeline` để endpoint trả metadata đúng.
- [x] Chạy `pytest tests/`.
- [x] Ghi lại lỗi LLM/external API nào không test trực tiếp được và đã mock.

## 2. Backend Go - config và database models

Mục tiêu: backend lưu report, preference, notification, delivery log.

### 2.1. Config

- [x] Sửa `DACN_Backend/config/config.go`.
- [x] Thêm field `AIReportServiceURL`.
- [x] Thêm field `AIReportAPIKey`.
- [x] Thêm field `ReportScheduleHour`.
- [x] Thêm field `ReportScheduleEnabled`.
- [ ] Thêm default:
  - [ ] `AI_REPORT_SERVICE_URL=http://localhost:8000`
  - [ ] `REPORT_SCHEDULE_HOUR=8`
  - [ ] `REPORT_SCHEDULE_ENABLED=false`
- [ ] Cập nhật `DACN_Backend/README.md` env mới.
- [ ] Cập nhật `DACN_Backend/docker-compose.yml` env nếu cần.

### 2.2. Domain models

- [x] Tạo `DACN_Backend/internal/report/domain/report.go`.
- [x] Tạo model `AIReport`.
- [ ] Tạo constants status:
  - [ ] `generating`
  - [ ] `ready`
  - [ ] `failed`
  - [ ] `archived`
  - [ ] `published`
- [ ] Tạo constants period:
  - [ ] `daily`
  - [ ] `weekly`
- [x] Tạo model `AIReportDelivery`.
- [x] Tạo model `UserReportPreference`.
- [x] Tạo `DACN_Backend/internal/notification/domain/notification.go`.
- [x] Tạo model `Notification`.
- [x] Dùng JSONB fields không thêm dependency pgvector/datatypes mới.
- [ ] Thêm index cần thiết:
  - [ ] `ai_reports.slug` unique.
  - [ ] `ai_reports.period_type, period_start, period_end` unique nếu muốn tránh trùng.
  - [ ] `user_report_preferences.user_id` unique.
  - [ ] `notifications.user_id, read_at`.
  - [ ] `ai_report_deliveries.report_id, user_id` unique.

### 2.3. AutoMigrate

- [x] Sửa `DACN_Backend/cmd/server/main.go`.
- [x] AutoMigrate thêm:
  - [x] `AIReport`
  - [x] `AIReportDelivery`
  - [x] `UserReportPreference`
  - [x] `Notification`
- [x] Không thêm `ContentEmbedding`.
- [ ] Đảm bảo AutoMigrate không phá bảng hiện có.

### 2.4. Backend model tests

- [ ] Thêm test cho status/period validation nếu có helper.
- [ ] Test slug unique logic bằng repository test hoặc usecase test.
- [ ] Test JSON fields marshal/unmarshal được.
- [ ] Test AutoMigrate chạy được trên PostgreSQL test DB hoặc dùng test container nếu có.
- [ ] Nếu chưa có infra test DB, ghi rõ manual verification bằng local PostgreSQL.

## 3. Backend Go - repositories

### 3.1. Report repository

- [x] Tạo `internal/report/repository/report_repo.go`.
- [x] Implement `CreateReport`.
- [x] Implement `UpdateReport`.
- [x] Implement `FindReportByID`.
- [x] Implement `FindReportBySlug`.
- [x] Implement `FindLatestReadyReport(periodType)`.
- [x] Implement `ListReadyReports(periodType, limit, offset)`.
- [x] Implement `ListAdminReports(limit, offset)`.
- [x] Implement `MarkReportPublished(reportID, articleID)`.
- [x] Implement duplicate check theo period.

### 3.2. Preference repository

- [x] Tạo repository cho `UserReportPreference`.
- [x] Implement `FindByUserID`.
- [x] Implement `GetOrCreateDefaultByUserID`.
- [x] Implement `UpdateByUserID`.
- [x] Implement `FindUsersEnabledForReport(periodType)`.
- [x] Implement `FindUsersEnabledForEmail(periodType)`.
- [x] Query join với `users` để lấy email/full_name/email_verified.

### 3.3. Notification repository

- [x] Tạo `internal/notification/repository/notification_repo.go`.
- [x] Implement `Create`.
- [ ] Implement `BulkCreate`.
- [x] Implement `ListByUser`.
- [x] Implement `UnreadCount`.
- [x] Implement `MarkRead`.
- [x] Implement `MarkAllRead`.
- [x] Đảm bảo user chỉ thao tác notification của chính họ.

### 3.4. Delivery repository

- [x] Implement `CreateDelivery`.
- [x] Implement `DeliveryExists(reportID, userID)`.
- [x] Implement `MarkEmailSent`.
- [x] Implement `MarkEmailFailed`.
- [ ] Implement `ListByReport`.

### 3.5. Repository tests

- [ ] Test `CreateReport` và `FindReportBySlug`.
- [ ] Test duplicate report same period bị xử lý đúng.
- [ ] Test `FindLatestReadyReport` chỉ lấy `ready/published`, không lấy `failed`.
- [ ] Test preference default tạo đúng giá trị mặc định.
- [ ] Test update preference không ảnh hưởng user khác.
- [ ] Test `FindUsersEnabledForReport` chỉ lấy user bật notification.
- [ ] Test `FindUsersEnabledForEmail` chỉ lấy user bật email và verified.
- [ ] Test notification unread count.
- [ ] Test mark read chỉ mark notification của user hiện tại.
- [ ] Test delivery unique report/user chống gửi trùng.

## 4. Backend Go - AI client

### 4.1. Client contract

- [x] Tạo `internal/report/usecase/ai_client.go`.
- [x] Định nghĩa interface:
  - [x] `GenerateReport(ctx, input) (*AIReportResult, error)`.
- [x] Implement HTTP client gọi `POST {AI_REPORT_SERVICE_URL}/generate-report`.
- [x] Gửi header `X-Internal-API-Key`.
- [x] Timeout đủ dài, ví dụ 2-5 phút.
- [x] Decode response `status/report/metadata/error`.
- [x] Nếu AI trả `status=error`, map thành error.
- [x] Nếu HTTP non-2xx, map thành error có status code.
- [x] Validate report tối thiểu:
  - [x] `executive_summary` không rỗng.
  - [x] có `generated_at`.
  - [ ] `highlighted_news` có source/url.
- [x] Không log API key.

### 4.2. AI client tests

- [ ] Dùng `httptest.Server` mock AI service.
- [ ] Test request gửi đúng method/path/header.
- [ ] Test payload daily đúng.
- [ ] Test decode success response.
- [ ] Test AI trả `status=error`.
- [ ] Test HTTP 401.
- [ ] Test invalid JSON.
- [ ] Test timeout/cancel context.

## 5. Backend Go - report usecase

### 5.1. Generate report flow

- [x] Tạo `ReportService`.
- [x] Implement `GenerateReport(ctx, input)`.
- [x] Validate `period_type`.
- [x] Tính `period_start`, `period_end` theo timezone Vietnam.
- [x] Nếu report cùng period đã tồn tại và `force=false`, trả report hiện có.
- [x] Tạo record `ai_reports` status `generating`.
- [x] Gọi AI client.
- [x] Map AI result vào `AIReport`.
- [x] Update status `ready`.
- [x] Nếu AI lỗi, update status `failed`, lưu `error_message`.
- [x] Trả report/error rõ ràng.

### 5.2. Send report flow

- [x] Implement `SendReportToSubscribers(ctx, reportID)`.
- [x] Lấy report, chỉ gửi nếu status `ready` hoặc `published`.
- [x] Lấy user bật report theo period.
- [ ] Với mỗi user:
  - [x] Check delivery chưa tồn tại.
  - [x] Tạo notification.
  - [x] Tạo delivery log.
  - [x] Nếu user bật email và email verified, gửi email.
  - [x] Mark email sent/failed.
- [x] Nếu email lỗi, notification vẫn phải tạo thành công.
- [x] Không để một user lỗi làm fail toàn bộ batch.
- [x] Return summary:
  - [x] total users
  - [x] notifications created
  - [x] emails sent
  - [x] emails failed

### 5.3. Report read APIs

- [ ] Implement `ListReports`.
- [ ] Implement `GetLatestReport`.
- [ ] Implement `GetReportBySlug`.
- [ ] Public API chỉ trả report `ready` hoặc `published`.
- [ ] Admin API thấy cả `generating/failed/archived`.

### 5.4. Report usecase tests

- [ ] Test generate success tạo `generating` rồi `ready`.
- [ ] Test generate AI error tạo `failed`.
- [ ] Test `force=false` không tạo trùng report cùng ngày.
- [ ] Test `force=true` cho phép tạo report mới hoặc regenerate theo policy đã chọn.
- [ ] Test send report tạo notification cho đúng user bật preference.
- [ ] Test user tắt report không nhận notification.
- [ ] Test user bật email nhưng chưa verified không nhận email.
- [ ] Test email failure vẫn tạo notification và delivery có `email_error`.
- [ ] Test delivery chống gửi trùng.
- [ ] Test public list không trả report failed.

## 6. Backend Go - notification usecase và handlers

### 6.1. Notification usecase

- [x] Tạo `NotificationService`.
- [x] Implement `ListForUser(userID, limit, unreadOnly)`.
- [x] Implement `UnreadCount(userID)`.
- [x] Implement `MarkRead(userID, notificationID)`.
- [x] Implement `MarkAllRead(userID)`.
- [x] Validate limit default/max, ví dụ default 10, max 50.

### 6.2. Notification HTTP handlers

- [x] Tạo `internal/notification/delivery/http/notification_handler.go`.
- [x] Register routes:
  - [x] `GET /notifications`
  - [x] `GET /notifications/unread-count`
  - [x] `POST /notifications/{id}/read`
  - [x] `POST /notifications/read-all`
- [x] Tất cả route cần `Authenticate`.
- [x] Response dùng `utils.Response`.
- [x] Error không leak internal details.

### 6.3. Notification tests

- [ ] Test unauthenticated bị reject.
- [ ] Test list chỉ trả notification của user hiện tại.
- [ ] Test unread count đúng.
- [ ] Test mark read notification của user khác bị forbidden/not found.
- [ ] Test read-all chỉ mark user hiện tại.
- [ ] Test limit > max bị clamp hoặc bad request theo policy.

## 7. Backend Go - report preference APIs

### 7.1. Preference usecase

- [x] Tạo `ReportPreferenceService`.
- [x] Implement `GetMyPreference(userID)`.
- [x] Implement `UpdateMyPreference(userID, input)`.
- [x] Validate `preferred_hour` từ 0 đến 23.
- [x] Nếu preference chưa có, tạo default.
- [x] Không cho user update user khác.

### 7.2. Preference HTTP handlers

- [x] Register:
  - [x] `GET /me/report-preferences`
  - [x] `PUT /me/report-preferences`
- [x] Cần `Authenticate`.
- [x] Response trả preference mới nhất.

### 7.3. Preference tests

- [ ] Test GET tạo default nếu chưa có.
- [ ] Test PUT cập nhật daily notification.
- [ ] Test PUT cập nhật email option.
- [ ] Test invalid preferred hour trả 400.
- [ ] Test unauthenticated bị reject.

## 8. Backend Go - mailer report

### 8.1. Mailer implementation

- [x] Mở rộng `internal/auth/usecase/mailer.go` hoặc tạo report mailer riêng.
- [x] Thêm method `SendDailyReport`.
- [ ] Tạo HTML email template:
  - [ ] Greeting user.
  - [ ] Title report.
  - [ ] Executive summary.
  - [ ] Trending keywords.
  - [ ] Highlighted news with source links.
  - [ ] Link mở report trong app.
  - [ ] Ghi chú cách tắt email report.
- [x] Nếu SMTP chưa cấu hình, log nội dung dev và return nil giống mailer hiện tại.
- [x] Subject rõ ràng: `Bản tin thể thao AI ngày DD/MM/YYYY`.
- [x] Không gửi email nếu report rỗng.

### 8.2. Mailer tests

- [ ] Test SMTP disabled không lỗi.
- [ ] Test subject/body chứa title và link report.
- [ ] Test HTML escape nội dung user/report để tránh HTML injection.
- [ ] Nếu có SMTP mock, test gửi đúng recipient.

## 9. Backend Go - admin report APIs

### 9.1. Admin generate/send

- [x] Tạo report handler admin.
- [x] Register:
  - [x] `POST /admin/reports/generate`
  - [x] `POST /admin/reports/{id}/send`
  - [x] `GET /admin/reports`
- [x] Tất cả route dùng `RequireAdmin`.
- [ ] `POST /admin/reports/generate` nhận:
  - [ ] `period_type`
  - [ ] `lookback_days`
  - [ ] `send_to_users`
  - [ ] `force`
- [ ] Nếu `send_to_users=true`, generate xong gọi send.
- [ ] Với request lâu, cân nhắc chạy goroutine và trả status `generating`.

### 9.2. Public report APIs

- [x] Register:
  - [x] `GET /reports`
  - [x] `GET /reports/latest`
  - [x] `GET /reports/{slug}`
- [x] Public không cần login nếu muốn ai cũng xem được.
- [ ] Hoặc yêu cầu login nếu report chỉ dành cho user hệ thống.
- [ ] Nếu report không found hoặc chưa ready, trả 404.

### 9.3. Handler tests

- [ ] Test non-admin generate bị 403.
- [ ] Test admin generate gọi usecase đúng input.
- [ ] Test admin send report gọi usecase đúng.
- [ ] Test public latest trả report ready.
- [ ] Test public detail report failed trả 404.
- [ ] Test JSON request invalid trả 400.

## 10. Backend Go - scheduler daily report

### 10.1. Scheduler implementation

- [ ] Chọn implementation:
  - [ ] goroutine + ticker.
  - [ ] hoặc cron library.
- [x] Chỉ start scheduler nếu `REPORT_SCHEDULE_ENABLED=true`.
- [x] Chạy theo timezone `Asia/Ho_Chi_Minh`.
- [ ] Mỗi ngày giờ `REPORT_SCHEDULE_HOUR`:
  - [x] Generate daily report.
  - [x] Send to subscribers.
- [x] Chống chạy đồng thời nhiều job trong cùng process.
- [x] Chống tạo trùng report cùng ngày.
- [ ] Log start/end/duration.
- [x] Log summary notification/email.
- [x] Không crash server nếu job lỗi.

### 10.2. Scheduler tests

- [ ] Tách logic job thành method testable, ví dụ `RunDailyReportJob(ctx)`.
- [ ] Test job gọi generate và send theo thứ tự.
- [ ] Test generate lỗi thì không send.
- [ ] Test duplicate day không tạo report mới nếu `force=false`.
- [ ] Test scheduler disabled không start job.

## 11. Backend Go - publish report thành article đặc biệt

### 11.1. Seed/setup category và AI user

- [ ] Tạo hoặc đảm bảo tồn tại category `AI Report`.
- [ ] Tạo hoặc đảm bảo tồn tại user hệ thống `AI Reporter`.
- [ ] Quyết định seed ở startup hay tạo thủ công qua admin.
- [ ] Nếu seed ở startup:
  - [ ] Không tạo trùng category.
  - [ ] Không tạo trùng user.
  - [ ] User hệ thống không login được hoặc password random.

### 11.2. Publish usecase

- [x] Implement `PublishReportAsArticle(ctx, reportID, adminID)`.
- [x] Chỉ publish report status `ready`.
- [x] Nếu đã có `article_id`, không publish trùng.
- [x] Map title/summary/blocks đúng.
- [x] Tạo article:
  - [ ] `status=pending` nếu muốn admin duyệt tiếp.
  - [x] hoặc `status=approved` nếu publish trực tiếp.
- [x] Set `published_at` nếu status approved.
- [x] Update `ai_reports.article_id`.
- [x] Update `ai_reports.status=published`.
- [ ] Update notification link nếu cần trỏ sang article.

### 11.3. Publish API

- [x] Register `POST /admin/reports/{id}/publish-as-article`.
- [x] Chỉ admin được gọi.
- [x] Response trả article/report.
- [x] Nếu report đã publish, trả article hiện có.

### 11.4. Publish tests

- [ ] Test publish report ready tạo article.
- [ ] Test publish report failed bị reject.
- [ ] Test publish report đã publish không tạo article trùng.
- [ ] Test blocks có đủ heading/paragraph.
- [ ] Test `article_id` được lưu lại.
- [ ] Test non-admin bị 403.

## 12. Backend Go - top bài viết hệ thống không dùng pgvector

### 12.1. Ranking formula

- [x] Xác định công thức điểm:
  - [x] `score = view_count * 1 + share_count * 3 + bookmark_count * 2 + comment_count * 2 + recency_bonus`.
- [x] Xác định khoảng thời gian:
  - [x] 24h.
  - [x] 7 ngày.
  - [x] 30 ngày.
- [x] Xác định endpoint:
  - [x] `GET /articles/top?period=weekly&limit=5`.
- [x] Repository query cần join/count bookmark/comment nếu chưa có.
- [x] Chỉ tính article `approved`.

### 12.2. Top articles tests

- [ ] Test chỉ lấy approved articles.
- [ ] Test score sorting đúng.
- [ ] Test limit max.
- [ ] Test period filter đúng.
- [ ] Test article mới có recency bonus nếu dùng.

## 13. Frontend - services

### 13.1. Notification service

- [x] Tạo `src/services/notificationService.js`.
- [x] Implement `listNotifications(params)`.
- [x] Implement `getUnreadNotificationCount()`.
- [x] Implement `markNotificationRead(id)`.
- [x] Implement `markAllNotificationsRead()`.
- [x] Handle API response shape `{ success, data }`.

### 13.2. Report service

- [x] Tạo `src/services/reportService.js`.
- [x] Implement `listReports(params)`.
- [x] Implement `getLatestReport(periodType)`.
- [x] Implement `getReportBySlug(slug)`.
- [x] Implement admin `generateReport(payload)`.
- [x] Implement admin `sendReport(id)`.
- [x] Implement admin `publishReportAsArticle(id)`.

### 13.3. Preference service

- [x] Tạo `src/services/reportPreferenceService.js`.
- [x] Implement `getReportPreferences()`.
- [x] Implement `updateReportPreferences(payload)`.

### 13.4. Frontend service tests

- [ ] Nếu project có test setup, mock axios và test service gọi đúng URL/method.
- [x] Nếu chưa có test setup, thêm tối thiểu manual API checklist trong docs.
- [x] Test auth interceptor vẫn gửi token.

## 14. Frontend - NotificationBell header

### 14.1. Component

- [x] Tạo `src/features/notifications/components/NotificationBell.jsx`.
- [x] Hiển thị icon chuông.
- [x] Hiển thị badge unread count.
- [x] Click mở dropdown.
- [x] Dropdown list notification mới nhất.
- [x] Mỗi item hiển thị title/message/time.
- [x] Click item:
  - [x] gọi mark read.
  - [x] navigate tới `notification.link`.
  - [x] giảm unread count.
- [x] Có button mark all read.
- [x] Loading state.
- [x] Empty state.
- [x] Error state nhẹ, không phá header.
- [x] Poll unread count định kỳ hoặc refresh khi mở dropdown.
- [x] Chỉ hiển thị khi user authenticated.

### 14.2. Gắn vào layout

- [x] Gắn `NotificationBell` vào header hiện tại trong `SportsNewsSection.jsx`.
- [ ] Cân nhắc refactor header sang `ReaderLayout` để mọi trang có notification.
- [ ] Đảm bảo admin/author layout cũng có notification nếu cần.

### 14.3. UI tests/checks

- [ ] Test unauthenticated không gọi notification API.
- [ ] Test authenticated gọi unread count.
- [ ] Test dropdown render list đúng.
- [ ] Test click item mark read và navigate.
- [ ] Test mark all read reset badge.
- [ ] Kiểm tra responsive mobile.
- [x] Kiểm tra badge không che text/header.

## 15. Frontend - report preference UI

### 15.1. Component

- [x] Tạo `src/features/reports/components/ReportPreferencePanel.jsx`.
- [x] Thêm vào `UserInfoPage.jsx`.
- [x] Load preference khi user mở trang.
- [x] Toggle daily in-app report.
- [x] Toggle daily email report.
- [x] Toggle weekly in-app report.
- [x] Toggle weekly email report.
- [x] Input/select preferred hour nếu triển khai.
- [x] Save button hoặc auto-save theo policy.
- [x] Loading state.
- [x] Success message.
- [x] Error message.
- [ ] Disable email toggle nếu email chưa verified, hoặc hiển thị cảnh báo.

### 15.2. UI tests/checks

- [ ] Test load preference thành công.
- [ ] Test toggle gửi payload đúng.
- [ ] Test error API hiển thị message.
- [ ] Test email chưa verified không bật email hoặc có cảnh báo.
- [ ] Kiểm tra mobile layout.

## 16. Frontend - report pages

### 16.1. Routes

- [x] Sửa `src/routes/AppRoutes.jsx`.
- [x] Add `reports` route.
- [x] Add `reports/:slug` route.
- [x] Add admin `admin/reports` route.

### 16.2. Report list/detail

- [x] Tạo `src/features/reports/pages/ReportsPage.jsx`.
- [x] Tạo `src/features/reports/pages/ReportDetailPage.jsx`.
- [x] Report detail render:
  - [x] title.
  - [x] generated_at.
  - [x] period.
  - [x] executive_summary.
  - [x] trending_keywords.
  - [x] highlighted_news.
  - [x] source links.
- [x] Link source mở tab mới an toàn `rel="noreferrer"`.
- [x] Empty/not found state.
- [x] Loading state.
- [x] Error state.

### 16.3. Admin reports page

- [x] Tạo `src/features/reports/pages/AdminReportsPage.jsx`.
- [x] Add nav item trong `AdminLayout.jsx`.
- [x] List reports admin.
- [x] Button generate daily report.
- [x] Button send to users.
- [x] Button publish as article.
- [x] Status badge.
- [ ] Confirmation khi send/publish.
- [x] Disable action khi status không hợp lệ.

### 16.4. Report page tests/checks

- [ ] Test route `/reports/:slug` render data.
- [x] Test source links đúng.
- [x] Test admin buttons gọi API đúng.
- [x] Test action disabled theo status.
- [ ] Kiểm tra mobile và desktop.

## 17. Frontend - article đặc biệt/top articles

### 17.1. Home integration

- [ ] Hiển thị report/article đặc biệt trên home.
- [ ] Nếu report đã publish article, link tới article.
- [ ] Nếu chưa publish, link tới report detail.
- [x] Thêm section top bài viết hệ thống nếu backend endpoint có.
- [x] Không làm UI phụ thuộc pgvector.

### 17.2. UI tests/checks

- [ ] Test home có report mới nhất.
- [ ] Test fallback khi không có report.
- [ ] Test top articles sorting từ API.
- [ ] Kiểm tra layout không vỡ trên mobile.

## 18. Security và permissions

- [x] AI service endpoint yêu cầu internal API key.
- [x] Backend admin generate/send/publish yêu cầu admin.
- [x] Notification APIs yêu cầu authenticated user.
- [x] User chỉ xem/mark notification của chính mình.
- [x] Preference APIs chỉ thao tác user hiện tại.
- [x] Public report APIs không trả report failed/generating.
- [x] Không log JWT, API key, SMTP password.
- [x] Email report chỉ gửi user verified nếu bật email.
- [x] Thêm CORS nếu route mới cần.

### Security tests

- [ ] Test admin endpoint với user thường bị 403.
- [ ] Test notification user A không mark read notification user B.
- [ ] Test AI client không gọi được khi API key sai.
- [ ] Test report failed không public.

## 19. Integration/e2e manual flow

### 19.1. Local setup

- [ ] Start PostgreSQL backend.
- [ ] Start `DACN_Backend` ở `:8080`.
- [ ] Start `SportNews_AI` ở `:8000`.
- [ ] Start frontend Vite ở `:5173`.
- [ ] Tạo admin user.
- [ ] Tạo user thường có email verified.
- [ ] Bật daily report notification cho user.
- [ ] Bật daily report email nếu muốn test SMTP/dev log.

### 19.2. Flow admin generate

- [ ] Admin login.
- [ ] Mở `/admin/reports`.
- [ ] Click generate daily report.
- [ ] Backend gọi AI service thành công.
- [ ] Report status thành `ready`.
- [ ] Admin click send to users.
- [ ] User nhận notification.
- [ ] Nếu bật email, email được gửi hoặc log dev xuất hiện.

### 19.3. Flow user notification

- [ ] User login.
- [ ] Header hiện badge unread.
- [ ] Mở dropdown thấy report notification.
- [ ] Click notification.
- [ ] Notification được mark read.
- [ ] Điều hướng tới `/reports/:slug`.
- [ ] Badge giảm.

### 19.4. Flow publish article

- [ ] Admin publish report thành article.
- [ ] Article được tạo với block đúng.
- [ ] Article xuất hiện ở trang đọc nếu status approved.
- [ ] Nếu status pending, admin approval flow xử lý tiếp.
- [ ] Report lưu `article_id`.

## 20. Automated test command checklist

### AI service

- [x] Chạy `pytest tests/` trong `SportNews_AI`.
- [x] Tất cả test schema/preprocess/db/api/writer/planner pass.
- [x] Ghi lại test nào mock LLM/crawler.

### Backend

- [x] Chạy `go test ./...` trong `DACN_Backend`.
- [ ] Nếu cần PostgreSQL test DB, ghi rõ env test.
- [x] Đảm bảo handler/usecase/repository tests pass.
- [ ] Chạy backend và kiểm tra `/health` (chưa chạy runtime vì cần DB/backend service thực tế).

### Frontend

- [ ] Chạy `npm run lint` hoặc script lint hiện có (toàn repo còn lỗi lint cũ ngoài phạm vi; các file mới đã lint riêng pass).
- [ ] Nếu có test runner, chạy test frontend.
- [x] Chạy `npm run build`.
- [ ] Mở app kiểm tra notification/header/report pages.

## 21. Acceptance criteria

Phase 1 được xem là xong khi:

- [ ] Admin tạo được daily report từ backend.
- [ ] Report được lưu trong DB backend.
- [ ] User bật nhận report nhận được notification.
- [ ] User tắt nhận report không nhận notification.
- [ ] User bật email và verified nhận email hoặc dev log email.
- [ ] User click notification mở được report detail.
- [ ] Có test cho AI request/metadata/API key.
- [ ] Có test cho backend generate/send/preference/notification.
- [ ] Frontend build pass và UI notification hoạt động.

Phase 3 được xem là xong khi:

- [ ] Admin publish được report thành article đặc biệt.
- [ ] Article có title/summary/blocks đúng.
- [ ] Không publish trùng cùng report.
- [ ] Report lưu `article_id`.
- [ ] Article đặc biệt hoặc report mới nhất hiển thị được trên home.
- [ ] Top bài viết hệ thống chạy bằng metric sẵn có, không dùng pgvector.
- [ ] Có test cho publish report và top articles.

Ngoài scope hiện tại:

- [x] Skip pgvector.
- [x] Không làm semantic search.
- [x] Không làm personalized recommendation bằng embedding.
- [x] Không embedding toàn bộ database.
