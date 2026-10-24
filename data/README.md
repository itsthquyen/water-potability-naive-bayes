# Dữ liệu

Đã tải CSV gốc ngày 24/10/2026 từ [Water Quality, Aditya Kadiwal](https://www.kaggle.com/datasets/adityakadiwal/water-potability), version 3. Kaggle API công khai ghi giấy phép **CC0: Public Domain**. Không thay bằng bản mirror hoặc dữ liệu tổng hợp.

- `raw/`: dành cho dữ liệu gốc; giữ nguyên nội dung khi bắt đầu xử lý.
- `splits/`: dành cho chỉ số mẫu thuộc các tập dữ liệu, tạo sau khi triển khai.

## Tải Và Xác Minh

Chạy tại gốc project: `rtk proxy .\.venv\Scripts\python.exe -X utf8 -m src.download_data`. Lệnh tải đúng version 3, giữ nguyên bytes `raw/water_potability.csv`, không ghi đè file đã có. Nếu tải thủ công, dùng cùng nguồn/version và đối chiếu SHA256 trước thực nghiệm.

- `source.json`: URL nguồn/API/download, ngày UTC, version, license và checksum archive/CSV thực tế.
- `manifest.json`: audit/làm sạch, phiên bản quy tắc, seed 42, môi trường, checksum phép chia.
- SHA256 CSV: `904004bde729bfe3d2e195f46343bceead09e32a0eb95bb8184e7e20e029b2bf`.
- Kiểm tra: `rtk proxy .\.venv\Scripts\python.exe -X utf8 -m src.train --prepare-only`.
- Tệp thật có 3276 dòng; nhãn 0: 1998, nhãn 1: 1278; không phát hiện nhóm đầu vào trùng hoặc mâu thuẫn. Dev 2620 dòng, test 656 dòng. ID bắt đầu từ 0 trước lọc; không đưa vào đặc trưng.
- Thiếu: `ph` 491 ô, `Sulfate` 781 ô, `Trihalomethanes` 162 ô; các cột khác không thiếu.
- Làm sạch: giữ bản đầu của trùng cùng nhãn, từ chối xung đột nhãn trước mọi phép chia. Không lọc outlier theo kết quả mô hình.
- Danh sách ID trong `splits/` là UTF-8 không BOM, LF, có LF cuối; checksum khác manifest thì dừng. Không đổi seed để lấy điểm tốt hơn.

## Từ Điển Dữ Liệu

Tất cả 9 cột là số thực, có thể thiếu theo hợp đồng đầu vào. Đơn vị dưới đây được đọc từ mô tả nguồn Kaggle, **không phải xác nhận độc lập quy trình đo**. Giữ giá trị nguyên trạng; không tự quy đổi hoặc suy ra ngưỡng an toàn.

- `ph`: chỉ số pH, không đơn vị. Nguồn không mô tả quy trình đo từng dòng.
- `Hardness`: độ cứng do muối calcium/magnesium; trang nguồn chưa nêu rõ đơn vị cột, ghi **chưa xác minh**.
- `Solids`: tổng chất rắn hòa tan TDS; phần mô tả nhắc mg/L, chưa có metadata đo riêng cho CSV.
- `Chloramines`: chloramine; mô tả nhắc mg/L hoặc ppm nhưng không chứng minh phép đo cụ thể của cột, không tự coi hai đơn vị tương đương cho mọi bối cảnh.
- `Sulfate`: sulfate; mô tả tham chiếu nồng độ mg/L, chưa xác minh phương pháp đo.
- `Conductivity`: độ dẫn điện; mô tả dùng µS/cm, chưa có điều kiện nhiệt độ đo.
- `Organic_carbon`: carbon hữu cơ TOC; mô tả dùng mg/L, chưa có quy trình đo.
- `Trihalomethanes`: THM; nguồn ghi ppm trong ví dụ giới hạn. Đơn vị cột **chưa xác minh**; không tự sửa thành µg/L hoặc dùng giới hạn đó trong app.
- `Turbidity`: độ đục; mô tả dùng NTU, chưa có phương pháp đo từng bản ghi.
- `Potability`: số nguyên 0/1; ý nghĩa theo nguồn là không uống được/uống được. Không đưa vào đặc trưng; chưa xác minh cách gán nhãn hoặc chuẩn kiểm nghiệm.

## Giới Hạn Nguồn

Metadata được tham khảo tại `https://www.kaggle.com/api/v1/datasets/view/adityakadiwal/water-potability`, ngày 24/10/2026. Lịch sử version 2 ghi **“Introduced null values to a few columns”**, version 3 ghi **“Removed garbage column”**. Vì vậy không diễn giải missing trong bản 3 như bằng chứng thiếu đo tự nhiên hoặc quy trình thực địa. Mô tả pH có khoảng tham chiếu không phản ánh toàn bộ CSV; không sao chép các ngưỡng WHO/EPA chưa kiểm chứng thành tiêu chuẩn quyết định.

Chưa xác minh địa điểm/thời gian lấy mẫu, nhóm trạm hoặc phương pháp tạo nhãn. Không tuyên bố đây là dữ liệu Việt Nam, Wondo Genet hoặc do nhóm tự đo chỉ từ các đoạn mô tả trên trang nguồn. Demo học thuật, không thay thế kiểm nghiệm chất lượng nước.

Theo yêu cầu chia sẻ toàn bộ ngày 24/10/2026, CSV gốc CC0 và các file phép chia được đưa vào Git cùng nguồn/checksum. Các file này giữ nguyên bytes qua `.gitattributes`; không tự sửa CSV hoặc ID khi clone/chạy lại.
