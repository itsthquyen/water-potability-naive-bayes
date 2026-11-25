# Rà soát Naive Bayes đếm tần suất

Ghi chú này mô tả lần chạy `20261109-NB-final`. Phạm vi hiện hành ngày
24/11/2026 là sáu mô hình tại `results/comparison/20261124-six-v1`, có
thực hành bộ lọc và bộ phân lớp Weka tham chiếu. Xem kế hoạch hiện hành.

Bản chính bám theo nhánh công thức đếm cho thuộc tính rời rạc của mục
4.2.2, Giáo trình Khai phá dữ liệu ĐH Kinh tế - Kỹ thuật Công nghiệp
(2022), và nguyên tắc MAP của slide phân lớp (trang PDF 40–44).

| Nội dung | Kết luận đã đối chiếu |
| --- | --- |
| Bài toán | Nhãn nhị phân Potability, 0/1 theo dữ liệu, không đưa nhãn vào đầu vào. |
| Đặc trưng số | Tối đa ba khoảng học theo thứ tự quan sát của từng cột trên train. Rời rạc hóa là bước tiền xử lý. |
| Điền thiếu | Trung vị học trên các ô quan sát của train; không fit trên validation/test. |
| Xác suất | Prior n_c/N và tần suất điều kiện n(c,j,v)/n_c. Không Gaussian, không smoothing. |
| Quyết định | So tích các xác suất sau khi quy đồng; nếu cả hai bằng 0, không gán nhãn. |
| Đánh giá | Cấu hình cố định trước CV/test; 5 fold trên development, test 656 dòng theo split đã chốt. |
| Kết quả | TN=366, FP=34, FN=226, TP=30. FPR 34/400 = 8,50%; tỷ lệ sai trong dự đoán dương 34/64 = 53,125%; recall lớp 1 là 30/256 = 11,72%. |
| Kết luận | Accuracy 60,37% thấp hơn baseline 60,98%; Balanced Accuracy 51,61%, ROC-AUC 0,5218. Không coi là chứng nhận nước an toàn. |

Manifest và các tệp kết quả của run `20261109-NB-final` liên kết theo
SHA-256. Weka 3.8.7 đọc và hiển thị dữ liệu gốc, khoảng nominal và kết quả
tự tính. Tám ảnh trong báo cáo thuộc run hiện hành.

Bản Gaussian và các kết quả cũ được lưu để truy vết, không phải phương
pháp hoặc bằng chứng của bản hiện hành.
