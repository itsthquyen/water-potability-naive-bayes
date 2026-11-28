# Đối chiếu báo cáo ngày 28/11/2026

Mẫu sử dụng: `BTL_KPDL_Nhom8_66TTNT2.docx` do người dùng cung cấp.
Giữ khung bìa, biểu trưng, phông Times New Roman, cấu trúc mô tả bài toán,
quy trình khai phá, thực nghiệm, kết luận và tài liệu tham khảo. Nội dung
RFM, bán hàng, K-Means, Apriori và marketing của mẫu được thay bằng bài
toán phân lớp nước và sáu thuật toán theo yêu cầu hiện hành.

| Nội dung cần kiểm tra | Vị trí trong bản hoàn thiện |
| --- | --- |
| Đề tài, giảng viên, nhóm và thành viên | Trang bìa; TS. Trần Mạnh Tuấn theo cách ghi mới nhất của người dùng |
| Lời cảm ơn | Cuối báo cáo, sau kết luận và ngay trước tài liệu tham khảo |
| Mục lục, danh mục bảng và hình | Trường tự động, đã cập nhật số trang bằng Word |
| Mô tả đầu vào, đầu ra, mục tiêu và giới hạn dữ liệu | Phần I |
| Dữ liệu thô, nguồn và ô thiếu | Phần I.4; ảnh Viewer Weka; phụ lục A đủ mười cột |
| Toàn bộ dữ liệu gốc | CSV nguyên byte trong gói nộp và chức năng tải trên web |
| Tiền xử lý và chống rò rỉ dữ liệu | Phần II.1; tham số học từ train/development; không fit trên test |
| Thao tác Weka chi tiết | Mở CSV, kiểm tra pH/Sulfate, NumericToNominal cho nhãn, điền thiếu và Discretize |
| Phân tích dữ liệu | Phần II.2; thống kê, histogram, tương quan và phân bố lớp |
| Sáu thuật toán | Phần II.3; Naive Bayes chính, năm thuật toán đối chiếu; công thức, giả mã và ví dụ |
| Thực nghiệm và phân tích kết quả | Phần III; năm fold, test, baseline, ma trận, ROC, lỗi và đối chiếu Weka |
| Ứng dụng có đầu ra | Phần IV; ảnh nhập một mẫu, dự đoán sáu mô hình, CSV, đánh giá và dữ liệu gốc |
| Tài liệu tham khảo | Dạng `[1]. …`; giáo trình và bài giảng ghi chung, không liệt kê bài hoặc trang |
| Chú thích | Nằm dưới bảng hoặc dưới ảnh, cùng trang với đối tượng |

## Kiểm chứng bản bàn giao

- PDF riêng và công khai cùng 54 trang, 21 bảng, 26 hình. Phần thân hai
  bản giống nhau về nội dung và ảnh render; bản riêng có thông tin thành viên.
- Đã xem toàn bộ trang render. Thu gọn bảng phụ lục để tiêu đề, dữ liệu,
  chú thích và ghi nguồn nằm cùng trang ngang A4; tên Potability không bị ngắt.
- 94 mục trong mục lục/danh mục trỏ đúng nội dung và số trang.
- Giữ nguyên 19 bảng có số liệu của bản trước; bảng nguồn lý thuyết được
  viết gọn và thêm một bảng dữ liệu thô. Sáu mã nguồn thuật toán và 56 tệp
  của lần thực nghiệm cố định vẫn khớp checksum.
- CSV 3.276 dòng, 10 cột giữ nguyên ô thiếu và độ chính xác. Đã tải từ web
  và so sánh nguyên byte với CSV nguồn; CSV trong ZIP cũng khớp.
- Toàn bộ 306 kiểm thử đạt. Kiểm tra trực tiếp trên trình duyệt gồm dự đoán
  một mẫu, CSV ba mẫu, bảng so sánh và tải dữ liệu gốc. Có sáu cảnh báo
  deprecation từ phần biểu đồ Streamlit/pandas, không có kiểm thử thất bại.

Những giới hạn thực nghiệm vẫn được nêu trong báo cáo: tập test đã từng
được xem, nguồn chưa mô tả đầy đủ cách lấy mẫu/gán nhãn, C4.5 tự cài có
phạm vi danh nghĩa chưa cắt tỉa, và cấu hình SVM Python/Weka khác nhau.
Không bổ sung số liệu hoặc ảnh giả để làm đẹp kết quả.

Lời mở đầu, lý do chọn đề tài và tổng quan được viết theo mạch trình bày
của mẫu: bối cảnh, mục tiêu của nhóm, dữ liệu và các nội dung thực hiện.
Lời cảm ơn nằm sau kết luận, ngay trước tài liệu tham khảo.
