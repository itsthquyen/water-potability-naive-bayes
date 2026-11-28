# Bài tập lớn Khai phá dữ liệu Nhóm 11

Đề tài: Phân loại chất lượng nước uống được hay không uống được bằng
thuật toán Naive Bayes và so sánh với các thuật toán khác.

Gói này chỉ chứa mã nguồn cần cho bài tập, kiểm thử và hướng dẫn chạy.
Không chứa báo cáo, dữ liệu mẫu, bản chạy cũ, mô hình huấn luyện sẵn,
môi trường ảo hoặc file của IDE. Các thư mục dữ liệu/kết quả được tạo
khi thực thi. Báo cáo và CSV gốc được bàn giao riêng.

## Tổ chức mã nguồn

```text
src/
├── algorithms/           Sáu thuật toán học máy, mỗi thuật toán một file
├── data_processing/      Đọc/kiểm tra CSV, điền thiếu, chia tập và chia khoảng
├── six_experiments.py    Chạy huấn luyện và thực nghiệm sáu thuật toán
├── metrics.py            Tính các chỉ số đánh giá tường minh
├── eda.py, plots.py      Phân tích dữ liệu và vẽ kết quả
├── ui_theme.py           Tên nhãn và các thành phần hiển thị
└── config.py             Cột dữ liệu, seed và đường dẫn dùng chung
app/                     Web Streamlit và giao diện CSS
weka/                    Mã đối chiếu tham chiếu với Weka
tests/                   Kiểm thử thuật toán, dữ liệu và ứng dụng
```

| Tệp | Nội dung |
| --- | --- |
| `src/algorithms/naive_bayes.py` | Đếm tần suất, prior, tích xác suất và MAP; thuật toán chính |
| `src/algorithms/id3.py` | Entropy, Information Gain và cây danh nghĩa |
| `src/algorithms/c45.py` | Thuộc tính số/danh nghĩa, MDL, thiếu có trọng số, cắt tỉa và nâng cây con |
| `src/algorithms/knn.py` | Khoảng cách Euclid và bỏ phiếu k láng giềng |
| `src/algorithms/logistic.py` | Sigmoid, gradient và cập nhật hệ số Logistic |
| `src/algorithms/svm.py` | Hinge loss và SGD cho SVM tuyến tính |
| `src/algorithms/common.py` | Kiểm tra dữ liệu có nhãn và entropy dùng chung |
| `src/data_processing/preprocessing.py` | Kiểm tra ma trận, trung vị, z-score, audit và chia phân tầng/fold |
| `src/data_processing/discretization.py` | Học các khoảng từ train, dùng cho NB và ID3 |
| `src/data_processing/data_validation.py` | Đọc CSV, tách nhãn, kiểm tra kiểu, thiếu và trùng |
| `src/six_experiments.py` | Chạy 6 thuật toán, 5 fold, test và lưu kết quả |
| `src/metrics.py` | Tự đếm ma trận, Accuracy, Precision, Recall, F1, BA và ROC-AUC |
| `src/eda.py`, `src/plots.py` | Tự tính thống kê và vẽ các kết quả đã tính |
| `src/artifacts.py` | JSON, checksum và kiểm tra ID lần chạy |
| `src/download_data.py` | Tải CSV thô từ Kaggle phiên bản 3 |
| `src/config.py` | Thứ tự cột, seed, tỷ lệ test và đường dẫn theo gốc dự án |
| `app/comparison_app.py`, `src/ui_theme.py`, `app/assets/comparison.css` | Web nhập số đo/CSV, dự đoán và xem thực nghiệm |
| `weka/` | Mã Java/Python chạy bộ phân lớp Weka để đối chiếu riêng |
| `tests/` | Ví dụ tính tay, tình huống biên và kiểm thử luồng dự đoán |

Phần học máy và thống kê cốt lõi dùng vòng lặp, cấu trúc dữ liệu và phép
toán Python; không dùng thư viện học máy. pandas hỗ trợ trình bày bảng,
matplotlib vẽ hình, Streamlit tạo giao diện, pytest chạy kiểm thử.
Weka là công cụ tham chiếu, không được gọi bên trong thuật toán Python.

## Cài đặt và chạy

Đã kiểm tra trên Windows với Python 3.12. Chạy lệnh từ thư mục chứa README.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m src.download_data
.venv\Scripts\python.exe -m src.six_experiments
.venv\Scripts\python.exe -m src.plots
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m streamlit run app/comparison_app.py
```

Nguồn tải: Water Quality, Aditya Kadiwal, Kaggle, phiên bản 3. Lệnh tải
cần HTTPS và giữ nguyên ô thiếu. CSV hợp lệ có SHA-256
`904004bde729bfe3d2e195f46343bceead09e32a0eb95bb8184e7e20e029b2bf`.
Nếu không truy cập được Kaggle, có thể chép CSV gốc được bàn giao riêng
vào `data/raw/water_potability.csv`; không dùng bản đã điền hoặc chuẩn hóa.

Lần chạy mặc định là `20261128-c45-full-v1`, tạo model, dự đoán và bảng
đánh giá trong `results/comparison/`. Chạy lại với `--run-id TEN_MOI` để
giữ các kết quả trước; chương trình không ghi đè lần chạy đã có.
Web mặc định đọc lần chạy trong hằng `RUN_ID` ở `src/six_experiments.py`.
Khi sử dụng ID khác cho web, đổi hằng này và chạy lại web.

## Sử dụng web

Thanh điều hướng bên trái mở năm trang chức năng, có nút thu gọn/mở lại.
Giao diện tự điều chỉnh trên điện thoại; bảng rộng cuộn ngang trong khung.
Các ô nhập được giữ khi chuyển trang rồi quay lại.

Nhãn trên web và CSV kết quả dùng **KHÔNG UỐNG ĐƯỢC**, **UỐNG ĐƯỢC**;
dòng lỗi dùng **KHÔNG XÁC ĐỊNH**. CSV dự đoán có `classification` và
`potable_score`; CSV đối chiếu có nhãn gốc và sáu dự đoán bằng chữ.
Khi tệp đầu vào có Potability, có thể chọn từng thuật toán để xem bảng
đúng/sai sau dự đoán. Potability không được dùng làm đặc trưng đầu vào.
CSV nguồn vẫn được tải nguyên byte, giữ cách mã hóa của tác giả dữ liệu.

## Đối chiếu Weka

Cần cài Weka 3.8.7 và JDK có `javac` trong PATH; JRE riêng không đủ để
biên dịch. Gói giáo dục ID3 chính thức được tải ở lần chạy đầu.

```powershell
.venv\Scripts\python.exe weka/run_six_weka.py --weka-home "C:/Program Files/Weka-3-8-7"
```

C4.5 nhận ARFF số còn dấu `?`, J48 dùng `-C 0.25 -M 2`, bật MDL,
collapse và subtree raising. NB/ID3 dùng mã khoảng; KNN/Logistic/SVM
dùng số chuẩn hóa theo tham số train của Python. Weka chạy trên cùng ID;
không thay số liệu Python bằng số liệu Weka. DirectFrequencyNB là mã
Java tự viết theo cùng công thức tần suất, vì NaiveBayes mặc định của
Weka không cùng cách ước lượng với NB trong đề tài.

## Quy trình và giới hạn

Giữ seed 42, 2.620 mẫu development và 656 mẫu test. Mỗi fold học lại
tiền xử lý từ train. C4.5 giữ giá trị thiếu và phân phối trọng số; các
mô hình còn lại học trung vị để điền. Không đưa Potability hoặc ID vào
đặc trưng. Không thay cấu hình theo điểm test.

Test là holdout đã từng sử dụng, chưa phải dữ liệu ngoài độc lập.
Đầu ra là nhãn Potability theo dữ liệu nghiên cứu, không chứng nhận mẫu
đạt tiêu chuẩn nước uống. Thông tin kết quả và phân tích nằm trong báo
cáo riêng, không nhúng báo cáo vào gói mã nguồn.
