# Plan nâng cấp email report thể thao AI

## 1. Vấn đề hiện tại

Email report hiện tại chưa đạt yêu cầu vì backend chỉ gửi một đoạn tóm tắt ngắn:

- Chỉ có tiêu đề, câu đầu tiên của `executive_summary`, một link và dòng hướng dẫn tắt email.
- Link đang dùng đường dẫn tương đối `/reports/{slug}` nên khi vào email client có thể bị hiểu sai thành `http://reports/...`.
- Người dùng không thể đọc report ngay trong email.
- Template email chưa có bố cục đầy đủ, chưa đủ đẹp và chưa tối ưu cho Gmail.
- Subject và một số text tiếng Việt trong backend cần được kiểm tra lại encoding để tránh lỗi tiếng Việt không dấu hoặc mojibake.

Yêu cầu mới: email phải chứa nguyên nội dung report do AI tạo ra, người dùng đọc được trực tiếp trong Gmail mà không cần mở web app. Link mở web chỉ là tùy chọn phụ.

## 2. Mục tiêu sau khi sửa

- Gửi full nội dung report vào email cho tất cả user có bật email report.
- Email có HTML template đẹp, dễ đọc trên Gmail desktop và mobile.
- Vẫn có nút "Đọc trên SportNews", nhưng đây không phải nội dung chính.
- Link phải là URL tuyệt đối theo môi trường:
  - Local: `http://localhost:5173/reports/{slug}`
  - Production: `https://domain-that-cua-ban/reports/{slug}`
- Không gửi link lỗi dạng `http://reports/...`.
- Nội dung AI phải được escape HTML để tránh lỗi layout hoặc injection.
- Có test để đảm bảo email luôn chứa full report, keyword, tin nổi bật và link đúng.

## 3. Phạm vi triển khai

### 3.1. Backend config

Thêm config riêng cho frontend:

```env
FRONTEND_BASE_URL=http://localhost:5173
```

Lý do không dùng `APP_BASE_URL`: biến hiện tại đang đại diện cho backend API, ví dụ `http://localhost:8080`. Link trong email cần trỏ về frontend, nên phải tách riêng.

Các thay đổi dự kiến:

- Thêm field `FrontendBaseURL` trong `DACN_Backend/config/config.go`.
- Default local là `http://localhost:5173`.
- Cập nhật `DACN_Backend/README.md` và file env mẫu nếu có.

### 3.2. Report service

Hiện tại `SendReportToSubscribers()` gọi:

```go
buildEmailBody(report)
```

Sẽ đổi thành:

```go
buildEmailBody(report, frontendBaseURL)
```

Hoặc tốt hơn là đưa phần build email vào một helper/testable riêng:

```go
BuildReportEmailHTML(report, frontendBaseURL string) string
BuildReportEmailText(report, frontendBaseURL string) string
```

Nội dung cần lấy từ `AIReport`:

- `Title`
- `PeriodType`
- `GeneratedAt`
- `ExecutiveSummary`
- `TrendingKeywords`
- `HighlightedNews`
- `Markdown`, nếu cần fallback khi JSON thiếu dữ liệu
- `Slug`

### 3.3. Template HTML email mới

Thiết kế email nên dùng inline CSS vì Gmail không hỗ trợ đầy đủ CSS hiện đại.

Bố cục đề xuất:

1. Header thương hiệu
   - SportNews AI Report
   - Badge: `Bản tin ngày` hoặc `Bản tin tuần`

2. Hero/title block
   - Tiêu đề report
   - Thời gian tạo report
   - Một đoạn mô tả ngắn

3. Tổng quan
   - Toàn bộ `ExecutiveSummary`
   - Chia đoạn theo newline hoặc câu dài để dễ đọc

4. Từ khóa nổi bật
   - Render `TrendingKeywords` thành pill/tag

5. Tin nổi bật
   - Render từng item trong `HighlightedNews`
   - Mỗi item có:
     - Headline
     - Summary
     - Source
     - Link nguồn gốc nếu có URL hợp lệ

6. CTA phụ
   - Nút "Đọc trên SportNews"
   - URL tuyệt đối: `${FRONTEND_BASE_URL}/reports/${slug}`
   - Có thêm plain link bên dưới để user copy nếu nút không hoạt động

7. Footer
   - Dòng thông báo: user có thể tắt email report trong phần thông tin cá nhân.
   - Không để footer thay thế nội dung chính.

### 3.4. Plain text fallback

Mailer hiện tại gửi `Content-Type: text/html`. Nên nâng cấp thành `multipart/alternative`:

- Part 1: `text/plain; charset=UTF-8`
- Part 2: `text/html; charset=UTF-8`

Lý do:

- Một số email client hoặc chế độ bảo mật có thể ưu tiên text.
- Dễ debug nội dung mail.
- Tăng độ tin cậy khi gửi qua SMTP.

Plain text fallback vẫn phải chứa đủ:

- Title
- Executive summary
- Keywords
- Highlighted news
- Link SportNews tuyệt đối

### 3.5. Subject email

Subject hiện tại đang nối string trực tiếp:

```go
"Subject: " + subject
```

Sẽ đổi sang MIME encoded subject:

```go
mime.QEncoding.Encode("utf-8", subject)
```

Lý do: subject tiếng Việt cần được encode chuẩn để Gmail không bị lỗi dấu.

### 3.6. An toàn dữ liệu

Tất cả dữ liệu từ AI hoặc DB khi render HTML phải dùng:

```go
html.EscapeString(...)
```

Các URL cần kiểm tra:

- Nếu URL nguồn tin hợp lệ `http://` hoặc `https://` thì render link.
- Nếu URL rỗng hoặc không hợp lệ thì chỉ render source text, không tạo link.

### 3.7. Không thay đổi nghiệp vụ gửi email hiện tại

Giữ nguyên rule đã sửa trước đó:

- Tất cả user có email và chưa tắt email report sẽ nhận email.
- Không bắt buộc `EmailVerified`, vì user đăng nhập Google có thể chưa đi qua flow verify email nội bộ.
- Sau khi report daily/weekly/manual tạo xong và `send_to_users=true`, backend gửi notification và email ngay.

## 4. Test cần có

### 4.1. Unit test email HTML

Tạo test cho email builder:

- `[x]` Email HTML chứa full `ExecutiveSummary`, không chỉ câu đầu tiên.
- `[x]` Email HTML chứa toàn bộ keyword từ `TrendingKeywords`.
- `[x]` Email HTML chứa từng item trong `HighlightedNews`.
- `[x]` Email HTML chứa URL tuyệt đối `http://localhost:5173/reports/{slug}`.
- `[x]` Email HTML không chứa link tương đối `href="/reports/..."`
- `[x]` Email HTML escape được nội dung nguy hiểm như `<script>`.
- `[x]` Email HTML vẫn render được khi `TrendingKeywords` hoặc `HighlightedNews` rỗng.

### 4.2. Unit test plain text

- `[x]` Plain text chứa title.
- `[x]` Plain text chứa full summary.
- `[x]` Plain text chứa keywords.
- `[x]` Plain text chứa highlighted news.
- `[x]` Plain text chứa frontend absolute URL.

### 4.3. Unit test mailer

- `[x]` `SendDailyReport` tạo `multipart/alternative`.
- `[x]` Subject tiếng Việt được MIME encode.
- `[x]` Header có `MIME-Version: 1.0`.
- `[x]` HTML part có `Content-Type: text/html; charset=UTF-8`.
- `[x]` Text part có `Content-Type: text/plain; charset=UTF-8`.

### 4.4. Integration/manual test

Sau khi triển khai:

1. Chạy backend với:

```env
FRONTEND_BASE_URL=http://localhost:5173
```

2. Tạo report bằng admin UI hoặc API.

3. Kiểm tra DB:

```sql
select report_id, user_id, email_sent, email_error
from ai_report_deliveries
order by created_at desc;
```

4. Kiểm tra Gmail nhận được email:

- `[ ]` Email có full report.
- `[ ]` Có section Tổng quan.
- `[ ]` Có section Từ khóa nổi bật.
- `[ ]` Có section Tin nổi bật.
- `[ ]` Nút đọc trên SportNews trỏ tới `http://localhost:5173/reports/{slug}` khi đang chạy local.
- `[ ]` Người dùng vẫn đọc được đầy đủ nội dung nếu không bấm nút.

## 5. Thứ tự triển khai đề xuất

- `[x]` Thêm `FRONTEND_BASE_URL` vào config backend.
- `[x]` Truyền `FrontendBaseURL` vào `ReportService`.
- `[x]` Tách email builder ra function riêng để test dễ.
- `[x]` Viết HTML template full report.
- `[x]` Viết plain text template full report.
- `[x]` Nâng cấp SMTP mailer sang `multipart/alternative`.
- `[x]` Encode subject tiếng Việt bằng MIME encoding.
- `[x]` Cập nhật README/env hướng dẫn `FRONTEND_BASE_URL`.
- `[x]` Viết unit test cho email builder.
- `[x]` Viết unit test cho mailer.
- `[x]` Chạy `go test ./...`.
- `[x]` Build/restart backend.
- `[ ]` Gửi thử report mới hoặc resend latest report.
- `[ ]` Kiểm tra Gmail thực tế.

## 6. Kết quả kỳ vọng

Email mới sẽ có dạng:

- Tiêu đề: `Bản tin thể thao AI ngày 13/06/2026`
- Nội dung chính nằm ngay trong email.
- Có đầy đủ phần tổng quan, keyword và tin nổi bật.
- Có nút mở trên SportNews nhưng không bắt buộc.
- Link local đúng là `http://localhost:5173/reports/...`.
- Khi deploy thật, chỉ cần đổi:

```env
FRONTEND_BASE_URL=https://sportnews-domain-cua-ban.com
```

Không cần sửa code.

## 7. Rủi ro cần lưu ý

- Link `localhost:5173` chỉ mở được trên máy đang chạy frontend local. Đây là đúng trong môi trường dev, nhưng không dùng được cho user thật ngoài máy local.
- Với production, bắt buộc cấu hình domain frontend thật trong `FRONTEND_BASE_URL`.
- Email HTML phải giữ inline CSS đơn giản để tương thích Gmail.
- Nội dung report dài có thể làm email dài; cần thiết kế section rõ ràng để vẫn dễ đọc.
