# Đối chiếu báo cáo ngày 28/11/2026

Mẫu sử dụng: `BTL_KPDL_Nhom8_66TTNT2.docx` do người dùng cung cấp.
Giữ khung bìa, biểu trưng, Times New Roman, cấu trúc mô tả bài toán,
quy trình khai phá, thực nghiệm, ứng dụng, kết luận và tài liệu tham khảo.
Nội dung bán hàng của mẫu được thay bằng bài toán phân lớp nước và sáu
thuật toán theo yêu cầu hiện hành.

| Nội dung cần kiểm tra | Vị trí trong bản hoàn thiện |
| --- | --- |
| Đề tài, giảng viên, nhóm và thành viên | Trang bìa; TS. Trần Mạnh Tuấn theo cách ghi mới nhất của người dùng |
| Lời cảm ơn | Cuối báo cáo, sau kết luận và ngay trước tài liệu tham khảo |
| Mục lục, danh mục bảng và hình | Trường tự động, đã cập nhật số trang bằng Word |
| Mô tả đầu vào, đầu ra, mục tiêu và giới hạn dữ liệu | Phần I |
| Dữ liệu thô, nguồn và ô thiếu | Phần I.4; ảnh Viewer Weka; phụ lục A đủ mười cột |
| Toàn bộ dữ liệu gốc | CSV nguyên byte trong data/raw, lệnh tải trong gói source và nút tải trên web |
| Tiền xử lý và chống rò rỉ dữ liệu | Phần II.1; tham số học từ train/development; không fit trên test |
| Thao tác Weka chi tiết | Mở CSV, kiểm tra pH/Sulfate, NumericToNominal cho nhãn, điền thiếu và Discretize |
| Phân tích dữ liệu | Phần II.2; thống kê, histogram, tương quan và phân bố lớp |
| Sáu thuật toán | Phần II.3; Naive Bayes chính, năm thuật toán đối chiếu; công thức, giả mã và ví dụ |
| C4.5 đầy đủ cho bài toán hai lớp | II.3.4.1–3.4.4: ngưỡng số, danh nghĩa, MDL, thiếu phân số, collapse, cắt tỉa bi quan và nâng cây con |
| Thực nghiệm và phân tích kết quả | Phần III; năm fold, test, baseline, ma trận, ROC, lỗi và đối chiếu Weka |
| Kiểm chứng C4.5/J48 | III.8.1; đối chiếu từng mẫu test và từng fold, ảnh cấu hình và ảnh kết quả Weka thực tế |
| Ứng dụng có đầu ra | Phần IV; năm tab, lần chạy 20261128-c45-full-v1, nhập một mẫu, đồng thuận, CSV, đánh giá, ROC, dữ liệu gốc và thông tin chỉ tiêu |
| Tài liệu tham khảo | Dạng `[1]. …`; giáo trình và bài giảng ghi chung, không liệt kê bài hoặc trang |
| Chú thích | Nằm dưới bảng hoặc dưới ảnh, cùng trang với đối tượng |

## Kiểm chứng bản bàn giao

- Word/PDF riêng và công khai cùng 59 trang, 22 bảng, 25 hình. Phần thân
  giống nhau cả nội dung và ảnh render. Bản công khai không chứa tên/MSSV
  thành viên trong XML, PDF hoặc metadata; bản nộp giữ roster đã xác nhận.
- Mục lục và danh mục có 98 mục đã đối chiếu nội dung và số trang. Bố cục
  được xem qua toàn bộ trang; phần không thay đổi được so sánh pixel với
  bản đã kiểm tra trước, phần thay đổi được xem ở độ phân giải đầy đủ.
  Phụ lục đủ mười cột, chú thích và nguồn nằm trên cùng trang ngang A4.
- Lần thực nghiệm mới có 75 file được khóa checksum và 15 mã nguồn tham
  gia huấn luyện được lưu hash riêng. Dự đoán của năm thuật toán ngoài
  C4.5 giữ nguyên byte so với lần chạy trước.
- C4.5 Python và J48 khớp nhãn trên 656 mẫu test và 2.620 lượt validation
  trong năm fold. Sai số xác suất test tối đa 1,5543e-14. Cây sau cắt tỉa
  có 21 nút và 11 lá. Thêm 24 bộ kiểm tra tổng hợp với 2.009 lượt dự đoán,
  không có khác biệt nhãn hoặc số nút với J48.
- C4.5 mới đạt accuracy 61,89%, BA 53,70%, AUC 0,5732; recall lớp 1 chỉ
  16,41%. So với cây danh nghĩa cũ, accuracy tăng nhưng recall và macro-F1
  giảm. Báo cáo trình bày cả hai chiều thay đổi này.
- CSV 3.276 dòng, 10 cột giữ nguyên ô thiếu và độ chính xác; SHA-256 là
  `904004bde729bfe3d2e195f46343bceead09e32a0eb95bb8184e7e20e029b2bf`.
- Có 328 kiểm thử đạt; tám cảnh báo deprecation từ Streamlit/pandas,
  không có kiểm thử thất bại. Gói source được tải CSV và chạy 197 unit
  test cùng ba kiểm thử ứng dụng trong bản sao độc lập.
- Năm ảnh bổ sung và hash nằm trong `report/evidence/20261128-c45-full`.
  Cấu hình và kết quả J48 được chụp từ Weka Explorer thao tác thực tế;
  hai ảnh web mới chụp từ Chrome, không giữ ảnh C4.5 cũ trong báo cáo.

Những giới hạn thực nghiệm vẫn được nêu: test là holdout lịch sử đã từng
được xem, nguồn chưa mô tả đầy đủ cách lấy mẫu/gán nhãn, ID3 Weka có 53
mẫu chưa gán nhãn và cấu hình/tối ưu SVM Python/Weka khác nhau.

Lời mở đầu, lý do chọn đề tài và tổng quan được viết theo mạch trình bày
của mẫu: bối cảnh, mục tiêu của nhóm, dữ liệu và các nội dung thực hiện.
Lời cảm ơn nằm sau kết luận, ngay trước tài liệu tham khảo.

Phần IV mô tả giao diện mới, tách dữ liệu thử và tỷ lệ đồng thuận khỏi
đánh giá mô hình. Giao diện bỏ các ngưỡng an toàn chưa kiểm chứng, đổi
preset thành bộ số thử và luôn hiển thị điểm lớp 1 đúng chiều. Đầu ra là
nhãn Potability nghiên cứu, không phải chứng nhận nước uống.

## Gói source riêng

`tools/package_source.py` tạo thư mục và ZIP `BaiTapLon_Nhom11_Source`
theo whitelist. Có 36 file mã nguồn, kiểm thử, requirements, pytest.ini
và README. Sáu thuật toán đặt gần nhau trong `src/algorithms`; tiền xử lý
ở `src/preprocessing.py`, chia khoảng ở `src/discretization.py`.
Gói không chứa báo cáo, dữ liệu, model sẵn, ảnh, mẫu cũ, môi trường ảo,
cache, Git hoặc cấu hình IDE. Đã kiểm tra ZIP và từng byte theo manifest.
Báo cáo được bàn giao riêng để thư mục source giữ đúng phạm vi yêu cầu.
