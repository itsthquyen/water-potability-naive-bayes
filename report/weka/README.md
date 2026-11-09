# Tái tạo hình Weka cho báo cáo

Weka dùng các panel **Explorer → Preprocess** và **Visualize** để mở tệp
ARFF và xuất ảnh. Model, cách điền thiếu, phép chia, các chỉ số và các
điểm ROC đến từ mã Python tự cài. Không dùng classifier, bộ lọc hay hàm
metric Weka để thay phần tự cài.

Chạy từ gốc repo sau khi có `.venv`, JDK và Weka 3.8.7:

```powershell
rtk proxy .venv/Scripts/python.exe -X utf8 report/weka/render_weka.py
```

Lệnh dùng run hiện hành `20261109-NB-final` và Weka ở `C:/Program Files/Weka-3-8-7`.
Có thể chỉ định `--weka-home`, `--java` và `--javac` theo đường dẫn trên máy.
Tệp `evidence/evidence_manifest.json` liên kết SHA-256 của ARFF và ảnh với lần chạy.
Có thể mở các tệp dữ liệu bằng **Explorer → Preprocess → Open file**,
đặt Class là `Potability` và chọn thuộc tính muốn xem:

| Ảnh | Tệp ARFF | Nội dung |
| --- | --- | --- |
| 01 | `raw.arff` | 3.276 bản ghi và phân bố lớp gốc |
| 02 | `development_raw.arff` | Sulfate trước điền thiếu |
| 03 | `development_processed.arff` | Sulfate đã điền thiếu và mã hóa ba khoảng nominal |
| 04 | `test_processed.arff` | 656 bản ghi kiểm tra, dùng ngưỡng và trung vị từ development |
| 05 | `roc.arff` | Trục X = FPR, Y = TPR; nối các điểm theo thứ tự |
| 06 | `errors.arff` | Trục X = xác suất lớp 1, Y = ph gốc, màu = kết cục |
| 07 | `development_processed.arff` | Phân bố lớp của 2.620 mẫu phát triển |
| 08 | `development_processed.arff` | Ba khoảng pH sau rời rạc hóa |

Hai tệp `*_processed.arff` chỉ minh họa trạng thái pipeline cuối.
Không dùng chúng làm đầu vào cho CV; mỗi fold cần fit riêng. `roc.arff`
và `errors.arff` chỉ dùng để vẽ. Hàng thiếu ph không có tọa độ Y trên
biểu đồ lỗi, nhưng vẫn được tính trong ma trận nhầm lẫn.

Để dựng lại PDF công khai, chạy `rtk proxy <python-có-ReportLab> -X utf8 report/build_report.py`
trên Windows có Times New Roman và Consolas. Lệnh ghi
`output/pdf/BaiTapLon-cong-khai.pdf`. Để dựng bản nộp có thông tin thành viên,
thêm `--students report/students.local.json` với JSON là danh sách các cặp
`[họ tên, MSSV]`. Tệp JSON và bản nộp `output/pdf/BaiTapLon.pdf` lưu cục bộ.
Ảnh biểu trưng trang bìa được trích từ PDF mẫu do người dùng cung cấp.
