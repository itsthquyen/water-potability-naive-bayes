# Phân loại nước uống được hay không bằng thuật toán Naive Bayes

**Học phần:** Khai phá dữ liệu, Trường Đại học Thủy Lợi.

**Sinh viên thực hiện:** Quyền, Lộc, Séc.

**Đặc tả hiện hành:** Naive Bayes đếm tần suất, 09/11/2026.

## 0. Phạm vi và quy tắc

- Tự viết thuật toán, tiền xử lý, chia tập/fold và metric bằng Python.
  Không dùng Gaussian, xác suất theo mật độ, mô hình học máy có sẵn hoặc
  phương pháp cộng giả mẫu. Weka chỉ dùng để kiểm tra ARFF và trình bày.
- Đầu vào là chín số đo theo `FEATURE_COLUMNS`. Nhãn `Potability` có
  hai giá trị 0/1 theo bộ dữ liệu. Không đưa nhãn hoặc ID vào đặc trưng.
- Giữ nguyên CSV nguồn và chia tập đã xác minh. Không chọn số khoảng,
  tham số hoặc seed dựa trên tập test. Không coi đầu ra là kết luận y tế.

## 1. Tiền xử lý

Kiểm tra kiểu, nhãn, dữ liệu thiếu và bản ghi trùng trước khi chia dữ liệu.
Với mỗi cột, chỉ lấy các giá trị quan sát trên phần train, sắp xếp tăng dần
và tính trung vị. Cho n giá trị, thử hai vị trí `ceil(n/3)` và
`ceil(2*n/3)`. Mỗi vị trí là số phần tử ở bên trái điểm cắt. Dùng trung
điểm của hai quan sát kề nếu khác nhau; bỏ điểm cắt nếu chúng bằng nhau.
Nếu trung điểm bị làm tròn lên bằng đầu mút trên, dùng đầu mút dưới.
Cột hằng chỉ có một khoảng; cột toàn thiếu ở train bị từ chối.

Khi transform, điền thiếu bằng trung vị train. Mã hóa 0, 1, 2 theo thứ tự
khoảng, với số đo đúng điểm cắt thuộc khoảng dưới. Các mã là giá trị rời
rạc, không được dùng như số đo hay ngưỡng an toàn nước. Giá trị ngoài
min/max train vẫn vào khoảng ngoài cùng và phát cảnh báo.

## 2. Naive Bayes

Với mỗi lớp c, đếm số mẫu `n_c`; với mỗi thuộc tính j và mỗi giá trị v,
đếm số mẫu `n(c,j,v)` trong lớp c mang giá trị đó. Dùng đúng công thức
tần suất ở Giáo trình Khai phá dữ liệu (2022), mục 4.2.2:

- Prior: `P(c) = n_c / N`.
- Xác suất có điều kiện: `P(A_j=v | c) = n(c,j,v) / n_c`.
- Điểm lớp: `P(c) * product_j P(A_j=x_j | c)`.
- Chuẩn hóa: chia điểm mỗi lớp cho tổng hai điểm nếu tổng dương.

Tử/mẫu được nhân và quy đồng bằng số nguyên trước khi so sánh, tránh
chọn nhãn từ xác suất đã làm tròn. Hai điểm dương bằng nhau chọn 0.
Nếu cả hai bằng 0, báo dòng không phân lớp được. Không có Gaussian hoặc
Laplace trong thuật toán hiện hành. JSON v2 ghi rõ loại thuật toán,
thứ tự đặc trưng, lớp, ngưỡng, trung vị, số mẫu lớp và bảng đếm.

## 3. Thực nghiệm và bàn giao

- Giữ phép chia phân tầng xấp xỉ 80/20 seed 42: development 2.620, test 656.
- Chạy 5 fold với một cấu hình cố định: tối đa ba khoảng, điền median,
  tần suất trực tiếp. So với baseline lớp đa số học từ train của từng fold.
- Fit lại trên toàn development. Đánh giá trên test sau khi khóa quy
  trình. Phép chia này được dùng để đối chiếu với báo cáo trước, không phải
  bộ dữ liệu ngoài độc lập.
- Tự tính ma trận nhầm lẫn, Accuracy, Balanced Accuracy, precision, recall,
  macro-F1 và ROC-AUC. Nêu giới hạn của nhãn, giả định độc lập có điều
  kiện và thông tin bị mất khi chia khoảng.
- Run hiện hành `20261109-NB-final`, báo cáo
  [báo cáo công khai](../output/pdf/BaiTapLon-cong-khai.pdf) và
  [hướng dẫn Weka](../report/weka/README.md).
