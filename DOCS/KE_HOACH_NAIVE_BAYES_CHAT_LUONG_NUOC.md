# Kế hoạch phân loại nước với sáu thuật toán học máy

Cập nhật 28/11/2026: C4.5 đầy đủ các bước dựng cây và cắt tỉa cho nhãn 0/1. Naive Bayes là phương pháp chính.
Năm phương pháp còn lại phục vụ so sánh. Sản phẩm gồm mã Python tự cài,
thực nghiệm có kết quả và phân tích, thực hành Weka, báo cáo chi tiết theo
mẫu DOCX mới và web dự đoán cục bộ.

## 0. Phạm vi và quy tắc

- Tự viết thuật toán, tiền xử lý, chia fold và metric bằng Python.
- Giữ Naive Bayes đếm tần suất trực tiếp trên thuộc tính đã chia khoảng.
- Dùng Weka thực hành tiền xử lý và chạy bộ phân lớp tham chiếu theo cùng
  ID train/test. Công bố riêng cấu hình và số liệu Python với Weka.
- Dự đoán Potability 0/1 từ chín cột; loại nhãn và ID khỏi vector đầu vào.
- Giữ nguyên byte CSV gốc; chỉ fit trung vị, mean/std và ngưỡng từ train.
- Chỉ trình bày kết quả đã chạy và kiểm chứng. Mọi giới hạn của dữ liệu,
  thuật toán hoặc phép đánh giá phải nêu trong phần phân tích.
- Báo cáo Nhóm 11 giữ thông tin thành viên đã xác nhận. Roster/MSSV
  lưu cục bộ. Commit/push theo ủy quyền của người dùng, tách theo chức năng.

## 1. Chọn thuật toán từ bài giảng

| Vai trò | Thuật toán | Tài liệu đã kiểm tra |
| --- | --- | --- |
| Chính | Naive Bayes | Bài 5, trang PDF 40–45; giáo trình 2022, mục 4.2 |
| Đối chiếu | ID3 | Bài 5, trang 25–33; giáo trình, mục 4.3 |
| Đối chiếu | C4.5 | Bài 5, trang 34 |
| Đối chiếu | KNN | Bài 5, trang 46–49; giáo trình, mục 4.4 |
| Đối chiếu | Logistic Regression | Giáo trình, mục 4.5.4, trang in 93–94 |
| Bổ sung | SVM tuyến tính | Bài giảng học máy Cornell và tài liệu Weka SMO |

C4.5 tự cài xử lý thuộc tính số/danh nghĩa, ô thiếu bằng trọng số, MDL,
gain ratio theo gain trung bình, collapse, cắt tỉa bi quan và nâng cây con.
Chi tiết công thức và kiểm chứng nằm trong [C45.md](C45.md). Các nội dung K-Means, RFM và Apriori của báo cáo mẫu phục vụ
đề tài bán hàng, không dùng để thay một trong sáu bộ phân lớp của đề tài.

## 2. Dữ liệu thô và nguồn

Tải lại Water Quality của Aditya Kadiwal từ Kaggle phiên bản 3 để xác
minh CSV. Bản tải ngày 24/11/2026 khớp nguyên byte bản đang có, SHA-256
`904004bde729bfe3d2e195f46343bceead09e32a0eb95bb8184e7e20e029b2bf`.
Tệp có 3.276 mẫu, chín chỉ tiêu số và nhãn Potability. Thiếu pH 491 ô,
Sulfate 781 ô và Trihalomethanes 162 ô. Sử dụng CSV này làm đầu vào thô,
không sử dụng tệp đã điền khuyết hoặc chuẩn hóa để bắt đầu thực nghiệm.

Kiểm tên cột, kiểu số, số vô hạn, nhãn 0/1 và bản ghi trùng trước chia
tập. Giữ bản đầu nếu trùng cùng nhãn, dừng khi cùng đầu vào khác nhãn.
Báo cáo số hàng trước/sau bước xử lý. Dữ liệu gốc không bị ghi đè.
Phân tích thống kê, histogram, ngoại lệ và tương quan trên development.

Nguồn công bố xác định được CSV và phiên bản, chưa xác nhận địa điểm,
thời gian lấy mẫu hay quy trình tạo nhãn. Nêu giới hạn này. Không suy
diễn dữ liệu là phép đo thực địa đã được kiểm chứng.

## 3. Thiết kế thực nghiệm

Giữ split phân tầng seed 42 đã lưu. Development có 2.620 mẫu, gồm
1.598 lớp 0 và 1.022 lớp 1. Test có 656 mẫu, gồm 400 lớp 0 và 256 lớp 1.
Hai tập không giao nhau và phủ toàn bộ mẫu hợp lệ. Test đã được dùng
trong phiên bản cũ, vì vậy đây là holdout lịch sử và không gọi là tập
kiểm chứng ngoài độc lập. Không chỉnh tham số theo điểm test.

| Thuật toán | Tiền xử lý train-only | Cấu hình chốt trước khi chạy |
| --- | --- | --- |
| Naive Bayes | Trung vị và tối đa ba khoảng | Tần suất trực tiếp, MAP, không giả mẫu |
| ID3 | Cùng mã khoảng với NB | Gain, cây chưa cắt tỉa, thuộc tính mỗi đường dùng một lần |
| C4.5 | Số thô còn ô thiếu; không điền trung vị/chia khoảng | Gain ratio, MDL, min leaf 2, CF 0,25, pruning/collapse/subtree raising |
| KNN | Trung vị và z-score | k=5, Euclid, bỏ phiếu đều |
| Logistic | Trung vị và z-score | 400 epoch, tốc độ 0,1, L2=0,001 |
| SVM tuyến tính | Trung vị và z-score | Hinge loss, SGD 100 epoch, tốc độ 0,01, L2=0,01 |

Năm fold giống nhau cho mọi phương pháp. Trong mỗi lượt tạo mới toàn bộ
pipeline, chỉ fit trên bốn fold train và biến đổi validation bằng thống
kê của lượt đó. Báo cáo mean/std ddof=0 cho Accuracy, Balanced Accuracy,
macro-F1, ROC-AUC. Học lại trên development rồi dự đoán toàn bộ test.
Ghi thời gian fit/predict riêng. SVM dùng biên quyết định cho ROC-AUC.
Không gọi biên SVM là xác suất uống được. Baseline lớp đa số là mốc phụ,
không tính vào sáu thuật toán.

## 4. Các bước tiền xử lý phải trình bày

1. Xác minh nguồn và checksum. Trích bảng mẫu thô còn ô thiếu.
2. Audit dữ liệu và bản ghi trùng, thống kê số lượng trước/sau.
3. Chia train/test rồi năm fold trong development, lưu ID.
4. Với NB, ID3, KNN, Logistic và SVM, học trung vị từ ô quan sát train
   rồi điền thiếu bằng trung vị đã học. C4.5 không áp dụng bước điền này.
5. Với NB/ID3, học vị trí làm tròn lên n/3 và 2n/3 trên ô quan sát.
   Lấy trung điểm hai giá trị kề khác nhau. Bỏ cắt giữa hai số bằng nhau.
   Điểm cắt thuộc khoảng dưới. Đổi sang mã danh nghĩa 0/1/2.
6. C4.5 giữ số đo và ô thiếu; tự học ngưỡng và tỷ trọng nhánh từ train.
7. Với KNN/Logistic/SVM, tính mean và std ddof=0 trên train đã điền.
   Cột hằng dùng scale 1; giữ lại cả chín chỉ tiêu.
8. Lưu tham số theo từng mô hình và kiểm tra sau khi nạp lại.

Không loại quan sát chỉ vì hiếm hoặc làm kết quả xấu. Không fit bộ lọc
trên test. Thống kê mô tả toàn bộ CSV phải phân biệt với thống kê học
tham số trên development/train.

## 5. Weka

Thực hiện trên Weka 3.8.7 có trên máy. Mở ARFF thô trong Explorer,
chọn Potability nominal, kiểm số mẫu và số ô thiếu. Thực hành
ReplaceMissingValues và Discretize trên development để có hình trước
và sau xử lý. Weka mặc định điền mean, khác median trong Python. Ghi rõ
sự khác biệt và không dùng hình minh họa làm kết quả CV.

Bảng đối chiếu Weka dùng cùng các ID. NB/ID3 và ba mô hình số dùng ARFF
đã biến đổi bằng tham số train của Python để kiểm soát đầu vào; C4.5
giữ số đo và ô thiếu trong ARFF. NB dùng bộ phân lớp Weka tự viết
theo công thức tần suất trực tiếp. ID3 dùng gói giáo dục chính thức,
C4.5 dùng J48 -C 0.25 -M 2 trên ARFF số còn ô thiếu; bật MDL, collapse và subtree raising; KNN dùng IBk k=5, Logistic dùng Logistic,
SVM dùng SMO kernel tuyến tính. Lưu tùy chọn, phiên bản, log và dự đoán.
Giải thích khác biệt tối ưu/ràng buộc giữa Python và Weka. Tự tính lại
ma trận/chỉ số từ dự đoán Weka để kiểm chứng, không ép hai bảng khớp.

## 6. Báo cáo theo mẫu mới

Tham chiếu `C:/Users/Asus/Downloads/BTL_KPDL_Nhom8_66TTNT2.docx`.
Giữ khổ A4, cách trình bày bìa, Times New Roman 13 ở thân bài, mục lục,
đánh số phần và tiểu mục theo mẫu. Thay toàn bộ nội dung bán hàng bằng
đề tài nước. Dùng Nhóm 11 và thông tin thành viên đã xác nhận.

- Lời mở đầu.
- I. Mô tả bài toán. Lý do, tổng quan, quy trình và phân tích dữ liệu thô.
- II. Quy trình khai phá dữ liệu. Tiền xử lý trên Python, trên Weka,
  biến đổi, phân tích dữ liệu và cơ sở sáu thuật toán.
- Thực nghiệm. Thiết kế, cấu hình, kiểm định chéo, test, sáu ma trận,
  so sánh, ROC và phân tích sai số. Naive Bayes trình bày chi tiết nhất,
  gồm ví dụ tính tay và bảng đếm của mẫu nước thật.
- Ứng dụng. Web nhập chín chỉ tiêu/CSV, kết quả NB và năm phương pháp
  đối chiếu, ảnh chạy thật và giải thích đầu ra.
- Kết luận, giới hạn và tài liệu tham khảo có nguồn xác định.

Sản phẩm là Word chỉnh sửa được và PDF nộp giảng viên. Hình có chú thích,
được dẫn trong phần nhận xét. Không sao chép danh tính và số liệu của
Nhóm 8. Kết quả trong báo cáo phải đọc từ manifest của lần chạy mới.

## 7. Web và nghiệm thu

Streamlit chạy cục bộ, NB là kết quả chính. Năm mô hình còn lại trả nhãn
và điểm so sánh. Mỗi dòng CSV giữ ID và lỗi, không bỏ lỗi để nâng chỉ số.
Nút dự đoán chỉ dùng trạng thái đã học có hash khớp, không huấn luyện.
Có tab trình bày số liệu thực nghiệm và giới hạn sử dụng.

Hoàn thành khi kiểm thử tính tay đạt, sáu mô hình chạy đủ các fold,
manifest khớp dữ liệu/model/kết quả, Weka có log và hình thật, Word/PDF
đã kiểm tra bố cục, web dự đoán được nhập tay và CSV. Ghi trung thực mọi
phần chưa chạy được thay vì tạo số liệu minh họa.

## 8. Tổ chức source và gói gửi giảng viên

Mỗi thuật toán nằm trong một file ở src/algorithms. Tiền xử lý tập trung
ở src/preprocessing.py và src/discretization.py, kiểm CSV ở
src/data_validation.py. Thực nghiệm và đánh giá được tách khỏi mô hình.
Weka tham chiếu ở weka/, giao diện ở app/.

tools/package_source.py tạo output/source/BaiTapLon_Nhom11_Source theo
whitelist và ZIP tương ứng. Thư mục chỉ có mã nguồn, kiểm thử, cấu hình
pytest, requirements và README chạy; không có báo cáo, dữ liệu, mô hình,
ảnh, môi trường, cache hoặc hồ sơ cũ. Dữ liệu và kết quả được sinh khi
người nhận chạy lệnh. Kiểm tra gói trong một bản sao độc lập, không chạy
pytest trực tiếp trong thư mục sạch được bàn giao để tránh sinh cache.
