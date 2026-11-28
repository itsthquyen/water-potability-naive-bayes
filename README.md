# Phân loại chất lượng nước bằng Naive Bayes và năm thuật toán so sánh

Bài tập lớn Khai phá dữ liệu, Nhóm 11, lớp 65CNTT. Naive Bayes đếm tần
suất là thuật toán chính; ID3, C4.5, KNN, Logistic Regression và SVM tuyến
tính dùng để đối chiếu. Sáu thuật toán Python tự cài phần huấn luyện và
dự đoán; tiền xử lý, chia tập, thống kê và metric cũng tự viết.

## Cấu trúc đang sử dụng

- `src/algorithms/`: sáu file thuật toán đặt cạnh nhau; common.py dùng chung kiểm tra và entropy.
- `src/preprocessing.py`: kiểm ma trận, trung vị, z-score, audit, chia train/test và fold.
- `src/discretization.py`: học mã khoảng cho NB/ID3; `src/data_validation.py`: kiểm CSV và nhãn.
- `src/six_experiments.py`, `src/metrics.py`: thực nghiệm và đánh giá.
- `src/eda.py`, `src/plots.py`: thống kê và hình từ số liệu tự tính.
- `app/comparison_app.py`, `src/ui_theme.py`: web hiện hành.
- `weka/`: mã đối chiếu bằng Weka, tách khỏi thuật toán Python.
- `tests/`: đáp án tính tay, tình huống biên và kiểm thử luồng dữ liệu.

C4.5 có ngưỡng số, thuộc tính danh nghĩa, ô thiếu có trọng số, MDL,
collapse, cắt tỉa bi quan và nâng cây con. Xem [công thức và kiểm chứng](DOCS/C45.md).
Không dùng thư viện học máy trong mã Python; Weka dùng mô hình tham chiếu riêng.

## Chạy từ gốc dự án trên máy hiện tại

```powershell
rtk proxy .venv/Scripts/python.exe -m pip install -r requirements.txt
rtk proxy .venv/Scripts/python.exe -X utf8 -m src.six_experiments --run-id NEW-SIX-RUN
rtk proxy .venv/Scripts/python.exe -X utf8 -m src.plots --run-id NEW-SIX-RUN
rtk proxy .venv/Scripts/python.exe -X utf8 weka/run_six_weka.py --run-id NEW-SIX-RUN
rtk proxy .venv/Scripts/python.exe -m pytest -q
rtk proxy .venv/Scripts/python.exe -m streamlit run app/comparison_app.py --server.address 127.0.0.1 --server.port 8501
```

Gói source cho máy khác có hướng dẫn Python trực tiếp, không bắt buộc cài RTK.
Nếu chưa có dữ liệu, chạy `python -m src.download_data` để tải CSV thô từ
Kaggle phiên bản 3. Dữ liệu gốc được giữ nguyên; thông tin nguồn ở [data](data/README.md).
Không ghi đè lần chạy đã có. Web mặc định đọc `20261128-c45-full-v1`.

## Kết quả mới

Giữ seed 42, 2.620 mẫu development, 656 mẫu test và năm fold. Test đã
từng được sử dụng, chưa phải kiểm chứng ngoài độc lập. C4.5/J48 nhận
cùng dữ liệu số còn ô thiếu; NB/ID3 dùng mã khoảng và ba mô hình số còn
lại dùng trung vị, z-score học từ train.

| Thuật toán Python | Accuracy | BA | AUC |
| --- | ---: | ---: | ---: |
| Naive Bayes | 60,37% | 51,61% | 0,5218 |
| ID3 | 55,03% | 52,86% | 0,5385 |
| C4.5 | 61,89% | 53,70% | 0,5732 |
| KNN | 64,79% | 60,72% | 0,6518 |
| Logistic Regression | 60,98% | 50,00% | 0,5182 |
| SVM tuyến tính | 60,98% | 50,00% | 0,5112 |

C4.5 khớp nhãn J48 trên toàn bộ test và năm fold; cây sau cắt tỉa có
21 nút, 11 lá. Sai số xác suất lớn nhất trên test dưới 1,56e-14. Accuracy
tăng so với cây gain ratio danh nghĩa cũ, nhưng recall lớp 1 chỉ 16,41%.
NB nhận đúng 30/256 mẫu lớp 1. Báo cáo phân tích các hạn chế này.

Bộ kiểm thử có 328 ca đạt; tám cảnh báo deprecation từ Streamlit/pandas.
Gói source được kiểm thêm trong bản sao độc lập, gồm tải CSV, unit test và web.

## Bàn giao

```powershell
rtk proxy .venv/Scripts/python.exe tools/package_source.py
```

Lệnh tạo `output/source/BaiTapLon_Nhom11_Source` và ZIP cùng tên, chỉ
gồm source cần cho đề tài, kiểm thử và hướng dẫn. Không chứa báo cáo,
dữ liệu mẫu, model sẵn, cache hay bản cũ. Danh sách và checksum gói nằm
ngoài thư mục source. Xem [hướng dẫn kèm gói](DOCS/README_NOP_SOURCE.md).

Báo cáo nộp có thông tin thành viên ở `output/docx/BaiTapLon-C45.docx` và
`output/pdf/BaiTapLon-C45.pdf`, lưu cục bộ. Bản công khai ở
[Word](output/docx/BaiTapLon-cong-khai.docx) và [PDF](output/pdf/BaiTapLon-cong-khai.pdf).
Trình dựng `report/build_six_report.py` lấy số liệu từ manifest lần chạy mới,
theo mẫu DOCX người dùng cung cấp. Ảnh thao tác mới ở
[hồ sơ C4.5](report/evidence/20261128-c45-full/README.md).
Hai bản báo cáo gồm 59 trang, 22 bảng và 25 hình; xem
[đối chiếu mẫu và kiểm chứng](DOCS/DOI_CHIEU_BAO_CAO_MAU.md).

Các lần chạy và archive cũ giữ để truy vết trong repository, không có
trong gói source gửi giảng viên. [Kế hoạch](DOCS/KE_HOACH_NAIVE_BAYES_CHAT_LUONG_NUOC.md)
ghi cấu hình và cách giữ tách kết quả Python với Weka.
