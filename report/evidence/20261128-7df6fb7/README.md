# Minh chứng giao diện tại commit 7df6fb7

Nhánh `main` được pull fast-forward từ `5373e1b` đến `7df6fb7` ngày
28/11/2026. Hai commit mới đổi giao diện, thêm mẫu thử, đồng thuận và cẩm
nang. Sáu thuật toán, CSV, model, phép chia và metric không thay đổi.

Ảnh JPEG chụp trực tiếp từ ứng dụng tại `127.0.0.1:8501`, không dựng lại:

- `web-overview.jpg`: đầu trang và năm tab.
- `web-form.jpg`: ba nhóm chỉ tiêu, các nút nạp mẫu và chín ô nhập trung vị.
- `web-result.jpg`: NB nhãn 0, điểm lớp 1 là 24,79%; 6/6 mô hình chọn 0.
- `web-evaluation.jpg`: bảng đánh giá và biểu đồ cột.
- `web-roc.jpg`: tab con ROC của kết quả thực nghiệm.
- `web-csv.jpg`: ba dòng hợp lệ, không có lỗi, NB chọn nhãn 1 cho 1/3 mẫu.
- `web-csv-detail.jpg`: chi tiết theo dòng và mô hình của CSV trên.
- `web-raw-data.jpg`: dữ liệu nguyên trạng cùng nút tải CSV.
- `web-guide.jpg`: tab cẩm nang đúng như mã nguồn vừa pull.

CSV mẫu sử dụng các ID gốc 0, 27 và 100; không dùng nhãn làm đầu vào.
CSV gốc tải qua web đã được so sánh nguyên byte. Hai preset nạp nhanh đã
được thao tác và kiểm tra giá trị từng ô; không coi tên preset là nhãn thật.

## Phạm vi diễn giải ảnh

Ảnh ghi lại trạng thái giao diện, không xác nhận tính đúng của mọi nội
dung y tế trên đó. Trong bản Git này, tên tab/tooltip có gắn WHO và tiêu đề
kết quả có chữ “đạt chuẩn”. Báo cáo diễn giải nhãn mô hình theo Potability,
phân biệt đồng thuận với accuracy và nêu giới hạn của phần tham khảo.

Đối chiếu tài liệu WHO 2026: pH, hardness, sulfate, TDS trong bảng 8.8
không có guideline value như cách gán ngưỡng sức khỏe chung trên giao diện.
Factsheet WHO ghi monochloramine 3 mg/L; cấu hình UI dùng Chloramines 4 ppm
và chưa xác minh hai đại lượng có cùng cách đo. Preset “nước sạch” có
Solids=8500, vượt chính max=1000 trong cấu hình UI. Các ngưỡng này không
tham gia hàm dự đoán Naive Bayes và không được dùng tính lại metric.

`manifest.json` lưu checksum ảnh, mã giao diện và URL tài liệu đã đối chiếu.
Thực nghiệm vẫn tham chiếu hồ sơ cố định `20261124-six-v1`.
