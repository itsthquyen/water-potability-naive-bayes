# Phân loại nước uống được hay không bằng Naive Bayes

Bài tập lớn Khai phá dữ liệu của **Quyền, Lộc, Séc**. Bản hiện tại là
**Naive Bayes đếm tần suất theo giáo trình**, tự viết bằng Python. Đầu vào
gồm 9 chỉ tiêu đo; đầu ra là nhãn `Potability`: `0` = không uống được,
`1` = uống được theo bộ dữ liệu. Đây là demo học thuật.

## Phương pháp

1. Kiểm tra dữ liệu, làm sạch và chia phân tầng development/test theo seed 42.
2. Với từng cột trên phần train, tính trung vị từ ô quan sát và học tối đa ba
   khoảng theo thứ tự giá trị. Điền ô thiếu bằng trung vị đã học, rồi mã hóa
   khoảng. Trong 5 fold, mỗi fold chỉ học thống kê từ train của fold đó.
3. Tự đếm `n_c` và `n(c,j,v)`. Tính `P(c) = n_c/N` và
   `P(A_j=v | c) = n(c,j,v)/n_c`. Nhân prior và các xác suất có điều kiện,
   chọn lớp có tích lớn hơn; nếu hai điểm dương bằng nhau chọn 0.
4. Giữ tần suất 0 theo công thức trực tiếp, không thêm giả mẫu. Nếu cả hai
   tích bằng 0, dòng đó được báo lỗi, không được tự gán nhãn. Mã lưu bảng
   đếm và ngưỡng vào JSON kèm SHA-256; demo dùng lại trạng thái đã học.

Không dùng Gaussian, hàm phân phối hoặc thư viện ML trong bản chạy hiện tại.
Weka đọc ARFF và dựng hình từ dữ liệu/kết quả Python; Weka không học mô hình.

## Chạy trên Windows

Chạy từ thư mục gốc dự án. Nếu chưa có môi trường:

```powershell
rtk proxy uv venv --python 3.12 .venv
rtk proxy uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

```powershell
rtk proxy .venv/Scripts/python.exe -X utf8 -m pytest -q
rtk proxy .venv/Scripts/python.exe -X utf8 -m streamlit run app/app.py
```

Demo mặc định nạp `artifacts/20261109-NB-final/pipeline.json` và
checksum bên cạnh. Có thể nhập một mẫu hoặc tải CSV; cột `sample_id` và
`Potability` không tham gia suy luận. CSV tải xuống giữ đủ kết quả, kể cả
lỗi từng dòng. Model schema v1 của bản thử nghiệm cũ không được app nạp.

Để chạy lại, chọn ID mới; lệnh train không đánh giá test và không ghi đè:

```powershell
rtk proxy .venv/Scripts/python.exe -X utf8 -m src.train --run-id MY-NB-RUN
rtk proxy .venv/Scripts/python.exe -X utf8 -m src.evaluate --run-id MY-NB-RUN --verification-confirmed
rtk proxy .venv/Scripts/python.exe -X utf8 -m src.predict --model artifacts/MY-NB-RUN/pipeline.json --input samples.csv --output predictions.csv
```

## Mã nguồn và kết quả

- [Thuật toán đếm](src/naive_bayes.py), [chia khoảng](src/discretization.py),
  [pipeline và lưu/nạp](src/pipeline.py).
- [Nguồn dữ liệu](data/README.md), [đặc tả](DOCS/KE_HOACH_NAIVE_BAYES_CHAT_LUONG_NUOC.md),
  [ghi chú rà soát](DOCS/RA_SOAT_LY_THUYET_VA_WEKA.md).
- [Quy trình Weka](report/weka/README.md), [báo cáo công khai](output/pdf/BaiTapLon-cong-khai.pdf).
- [Run `20261109-NB-final`](results/runs/20261109-NB-final.json) dùng 2.620 mẫu
  phát triển và 656 mẫu test. Bộ kiểm thử bản chạy hiện tại có 293 ca đạt.
  Accuracy **60,37%**, Balanced Accuracy **51,61%**, macro-F1 **0,4627**,
  ROC-AUC **0,5218**. Baseline lớp đa số đạt accuracy **60,98%**.
  Mô hình nhận ra **30/256** mẫu nhãn 1; kết quả không thay thế kiểm nghiệm nước.
- Các run `20261024-E4-*` và mã ở `archive/gaussian/` là hồ sơ lịch sử.
  Notebook và app hiện tại không sử dụng chúng.
