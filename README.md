# Phân loại chất lượng nước bằng Naive Bayes

Project học phần Khai phá dữ liệu — phân loại nhãn `Potability` bằng Gaussian Naive Bayes.

**Trạng thái:** Đã triển khai luồng tự cài, CLI, demo Streamlit; 457 tests đạt. Đã audit dữ liệu Kaggle, chạy CV và đánh giá test giữ riêng một lần ở run `20261024-E4-reviewed-v3`; bằng chứng tại [checklist triển khai](DOCS/NGHIEM_THU_TRIEN_KHAI.md). Chưa là bộ nộp cuối đã ký duyệt hoặc xác minh trên ba máy.

> **BẮT BUỘC: CODE TƯỜNG MINH, TỰ CÀI ĐẶT PHẦN HỌC MÁY.** Mọi agent và thành viên phải đọc [AGENTS.md](AGENTS.md) trước khi viết code. Tự viết Gaussian Naive Bayes, tiền xử lý, chia tập/fold, cross-validation và các chỉ số đánh giá; không dùng scikit-learn hoặc hàm có sẵn làm thay. Thư viện chỉ hỗ trợ công cụ nền theo phạm vi cho phép trong quy tắc. Demo cũng phải dùng bản tự cài đã kiểm chứng.

Kế hoạch chi tiết: [KE_HOACH_NAIVE_BAYES_CHAT_LUONG_NUOC.md](DOCS/KE_HOACH_NAIVE_BAYES_CHAT_LUONG_NUOC.md).

## Cấu trúc

```text
.
├── README.md
├── AGENTS.md                        # Quy tắc bắt buộc cho mọi agent
├── requirements.txt
├── DOCS/
│   └── KE_HOACH_NAIVE_BAYES_CHAT_LUONG_NUOC.md
├── data/
│   ├── README.md
│   ├── raw/                         # Dữ liệu gốc
│   └── splits/                      # Chỉ số các tập dữ liệu
├── notebooks/
│   ├── 01_data_audit_eda.ipynb       # Kiểm tra và khám phá dữ liệu
│   └── 02_experiments.ipynb          # Thực nghiệm
├── src/
│   ├── __init__.py
│   ├── data_validation.py           # Kiểm tra dữ liệu
│   ├── preprocessing.py             # Tiền xử lý
│   ├── gaussian_nb_numpy.py         # Gaussian NB tự cài
│   ├── config.py                    # Cấu hình chung
│   ├── pipeline.py                  # Chuỗi tự cài, JSON + checksum
│   ├── metrics.py                   # Chỉ số tự tính
│   ├── experiments.py               # CV và chọn cấu hình
│   ├── eda.py                       # Thống kê khám phá tự tính
│   ├── visualization.py             # Chỉ vẽ số liệu đã tính
│   ├── download_data.py             # Tải đúng nguồn Kaggle
│   ├── train.py                     # Audit/CV/fit, không mở test
│   ├── evaluate.py                  # Đánh giá
│   └── predict.py                   # Dự đoán
├── app/
│   └── app.py                       # Demo Streamlit, không train
├── tests/
│   ├── test_gaussian_nb.py
│   └── test_prediction_flow.py
├── artifacts/                       # Pipeline/mô hình sinh ra sau này
├── results/
│   ├── metrics/
│   └── figures/
├── report/
└── slides/
```

Notebook gọi logic trong `src`, không có bản thuật toán riêng. Các file `.gitkeep` giữ thư mục chưa có sản phẩm. `report/` và `slides/` chưa phải báo cáo/slide hoàn chỉnh.

## Chuẩn bị môi trường khi bắt đầu lập trình

Dùng Python 3.12; môi trường xác minh hiện tại là 3.12.3. Chạy tại thư mục gốc project trên Windows, mọi lệnh shell qua RTK theo AGENTS.md:

```powershell
rtk proxy py -3.12 -m venv .venv
rtk proxy .\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements.txt
rtk proxy .\.venv\Scripts\python.exe -X utf8 -m pytest -q
```

Nếu launcher `py` trỏ tới runtime cũ không tồn tại, dùng đường dẫn Python đang cài trên máy để tạo `.venv`; không ghi cứng đường dẫn riêng vào mã chia sẻ. `-X utf8` tránh lỗi giải mã requirements/docstring trên Windows. Phiên bản thực tế được ghi trong `data/manifest.json` và manifest lần chạy; chỉ có Python/math tự cài phần học máy, JSON thay cho joblib/pickle.

## Dữ liệu và đầu ra

Xem [data/README.md](data/README.md) cho nguồn CC0, checksum và giới hạn đơn vị/nhãn. Theo yêu cầu chia sẻ toàn bộ ngày 24/10/2026, repository chứa CSV gốc CC0, file phép chia, model JSON/sidecar và các kết quả đã tạo. `.gitattributes` giữ nguyên bytes các file có checksum. Không đưa môi trường ảo, cache, secrets hoặc file khóa tiến trình lên Git. Dữ liệu tổng hợp trong tests không phải số liệu thực nghiệm.

Sau khi clone và cài dependencies, có thể nạp ngay model `artifacts/20261024-E4-reviewed-v3/pipeline.json` trong app; không cần train/evaluate lại để xem kết quả đã lưu. Run `20261024-E4-local-v1` là lịch sử trước review, chưa mở test; thư mục `20261024-E4-reviewed-v2` là lần chạy bị timeout, không coi là run hoàn tất. Chỉ dùng v3 cho kết quả thực nghiệm hiện tại.

## Quy Trình Chạy

Không huấn luyện lại khi bấm dự đoán. Chạy kiểm thử trước khi thực nghiệm; chọn `run_id` mới, không ghi đè lần cũ:

```powershell
rtk proxy .\.venv\Scripts\python.exe -X utf8 -m src.download_data
rtk proxy .\.venv\Scripts\python.exe -X utf8 -m src.train --prepare-only
rtk proxy .\.venv\Scripts\python.exe -X utf8 -m src.train --run-id MY-RUN
rtk proxy .\.venv\Scripts\python.exe -X utf8 -m src.evaluate --run-id MY-RUN --verification-confirmed
rtk proxy .\.venv\Scripts\python.exe -X utf8 -m src.predict --model artifacts/MY-RUN/pipeline.json --input samples.csv --output predictions.csv
rtk proxy .\.venv\Scripts\python.exe -X utf8 -m streamlit run app/app.py
```

Lệnh tải chỉ chạy khi chưa có CSV. `--prepare-only` không fit; chặn checksum khác manifest. Train thực hiện E0–E3 và 28 cấu hình E4 trên cùng 5 fold, chọn theo Balanced Accuracy rồi fit trên dev. Evaluate yêu cầu xác nhận kiểm chứng, nạp đúng model và chỉ đánh giá một lần/run; không sửa cấu hình theo test. Mã nguồn chưa commit được ghi `dirty=true`, kết quả là lần chạy phát triển, chưa đủ điều kiện gói nộp chốt theo kế hoạch. Không tự commit/push trong quá trình thực nghiệm.

Trong app, nhập `artifacts/<run_id>/pipeline.json` vào ô đường dẫn model; cần file `.sha256` đi kèm. Model, metrics và hình phải khớp manifest. Thiếu model báo lỗi, không tạo dự đoán giả. Hai notebook chạy bằng kernel `.venv`, từ gốc project hoặc thư mục notebooks.

CLI dự đoán không ghi đè đầu vào, model, checksum hoặc file đầu ra đã có. Nguồn `source.json` và file phép chia vật lý phải khớp checksum; không tự sửa file bị thay đổi. Lần train lỗi giữ ID đã chiếm, cần ID mới. Requirements chốt phiên bản trực tiếp đã kiểm tra, không phải lock tất cả dependency gián tiếp.

Nếu evaluate lỗi khi vẽ/xuất sau checkpoint, chạy lại cùng lệnh chỉ phục hồi output từ checkpoint, không dự đoán test lại. Nếu inference đã bắt đầu mà checkpoint chưa hợp lệ, chương trình chặn và báo interrupted; không tự xóa trạng thái để mở test lần hai. Khóa OS ngăn hai tiến trình evaluate cùng run.

Kết quả run v3 trên 656 mẫu: accuracy 60.67%, Balanced Accuracy 53.20%, macro-F1 0.5027, AUC 0.5496. Baseline accuracy 60.98%, BA 50%; mô hình không vượt baseline về accuracy. Có 51 FP và 207 FN; recall lớp 1 chỉ 19.14%. Kết quả yếu được giữ nguyên, không tối ưu lại theo test. Model: `artifacts/20261024-E4-reviewed-v3/pipeline.json`; kết quả/checksum: `results/runs/20261024-E4-reviewed-v3.json`.

## Cấu Hình Và Đầu Vào

- `src/config.py`: seed mặc định 42, test 20%, 5 fold, thứ tự 9 đặc trưng và lớp `[0,1]`.
- Điền thiếu `mean`/`median`; scaler bật/tắt bằng bool, phương sai mẫu số `n`, cột hằng scale 1. Thống kê chỉ học từ train.
- `var_smoothing` mặc định `1e-9`, lưới `1e-12` tới `1e-6`; `variance_floor=1e-12`, cả hai dương hữu hạn. Epsilon tính trên ma trận sau tiền xử lý. Thay cấu hình ảnh hưởng kết quả; phải chốt trước CV và ghi manifest.
- CSV UTF-8 dấu phẩy, header đúng hoa/thường; đủ 9 cột, có thể thêm `sample_id` và `Potability` khi dự đoán. Cột đảo thứ tự được sắp theo tên. Nhãn và ID không vào mô hình.
- Chỉ ô rỗng hoặc token `NaN` (sau strip giá trị) là thiếu; `NULL`, `N/A`, `nan`, vô hạn và chuỗi sai là lỗi. API dùng `None` cho thiếu; không nhận numeric NaN ở tầng kiểm tra nghiệp vụ.
- Từ chối thiếu toàn bộ; cảnh báo thiếu từ 3/9 và ngoài min/max train. Dòng lỗi giữ ID/thứ tự, nhãn/xác suất để trống. Không coi cảnh báo là tiêu chuẩn nước uống.

**Demo học thuật; không thay thế kiểm nghiệm chất lượng nước.** Nguồn có giới hạn quy trình đo/gán nhãn; xác suất mô hình chưa được hiệu chỉnh thành xác suất an toàn thực tế.
