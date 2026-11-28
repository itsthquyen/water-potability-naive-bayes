# Ảnh thao tác bổ sung ngày 28/11/2026

Ảnh được chụp trực tiếp trên máy, không dựng lại giao diện. `manifest.json`
lưu SHA-256 của từng ảnh. Chúng bổ sung cho hồ sơ thực nghiệm cố định
`results/comparison/20261124-six-v1`; không ghi đè kết quả học máy.

## Weka 3.8.7 Explorer

- `weka-raw-rows.png`: Viewer mở CSV gốc, đủ 10 cột, các ô thiếu còn trống.
- `weka-raw-ph.png`, `weka-raw-sulfate.png`: toàn bộ 3.276 mẫu, lần lượt
  491 ô pH thiếu và 781 ô Sulfate thiếu.
- `weka-class-options.png`: NumericToNominal chỉ chọn `last`.
- `weka-class-nominal.png`: nhãn đã đổi kiểu, các thuộc tính số và ô thiếu
  vẫn giữ nguyên.
- `weka-class-counts.png`: Potability Nominal, lớp 0 có 1.998 mẫu, lớp 1
  có 1.278 mẫu.
- `weka-discretize-options.png`: mở `development_raw.arff` gồm 2.620 mẫu,
  chọn Discretize với `bins=3`, `useEqualFrequency=True`, `ignoreClass=False`.
  Đây là ảnh cấu hình; không phải ảnh đầu ra bộ lọc. Kết quả sau điền thiếu
  và rời rạc hóa dùng ảnh của hồ sơ thực nghiệm đã lưu.

CSV gốc chỉ được mở và đổi kiểu nhãn trong bộ nhớ để quan sát. Không lưu
đè CSV. Các tham số học tiền xử lý được lấy từ tập phát triển; ảnh toàn bộ
CSV không được coi là bằng chứng fit bộ lọc trên train.

## Ứng dụng Streamlit cục bộ

- `web-form.jpg`: chín ô nhập, giá trị minh họa là trung vị train được làm
  tròn bốn chữ số trên biểu mẫu.
- `web-result.jpg`: dự đoán thực tế từ chính bộ số trên biểu mẫu, NB lớp 0,
  điểm lớp 1 hiển thị 24,79%.
- `web-evaluation.jpg`: chỉ số test đã lưu và biểu đồ so sánh sáu cấu hình.
- `web-raw-data.jpg`: bảng dữ liệu gốc, thống kê và nút tải toàn bộ CSV.
- `web-csv.jpg`: tải CSV mẫu từ ứng dụng, nạp lại và bấm dự đoán; ba mẫu
  trích từ ID gốc 0, 27, 100 tạo 18 kết quả, không có lỗi đầu vào. ID dòng
  hiển thị 0, 1, 2 là vị trí trong tệp tải lên; `sample_id` trong CSV kết
  quả giữ `raw_0`, `raw_27`, `raw_100`.

Đã tải CSV gốc qua giao diện và đối chiếu nguyên byte với tệp trong dự án.
Ảnh chụp trình duyệt ở định dạng JPEG; ảnh Weka ở định dạng PNG.
