# KẾ HOẠCH BÀI TẬP KẾT THÚC HỌC PHẦN KHAI PHÁ DỮ LIỆU

## Phân loại khả năng uống được của nước bằng thuật toán Naive Bayes

- **Đơn vị học tập:** Trường Đại học Thủy Lợi.
- **Thành viên:** Quyền, Lộc, Séc.
- **Phiên bản:** 1.3 — bổ sung đặc tả số học, kiểm thử độc lập, chống trùng trong CV và điều kiện nghiệm thu sau rà soát; giữ yêu cầu tự cài tường minh của bản 1.2.
- **Ngày lập:** 24/10/2026.
- **Thời gian đề xuất:** 6 tuần kể từ ngày thống nhất kế hoạch; điều chỉnh theo hạn nộp thực tế.
- **Công nghệ chính:** Python, NumPy ở mức mảng/phép toán cơ bản, pandas cho đọc/ghi bảng, Jupyter Notebook và Streamlit. Không dùng scikit-learn trong mã dự án.
- **Trạng thái:** Đã triển khai bản tự cài, 457 tests đạt, tải/audit dữ liệu, chạy CV và đánh giá test giữ riêng một lần (run `20261024-E4-reviewed-v3`). Chưa nghiệm thu gói nộp cuối; bằng chứng và giới hạn theo [checklist triển khai](NGHIEM_THU_TRIEN_KHAI.md).

> Kế hoạch được thiết kế cho nhóm 3 sinh viên, ưu tiên hiểu thuật toán, thực nghiệm đúng và bảo vệ được kết quả. Nhóm chưa cung cấp đề cương, mẫu báo cáo hoặc tiêu chí chấm của giảng viên; các yêu cầu về số trang, số slide và thời lượng bên dưới là đề xuất, không phải quy định chính thức của trường.

## 0. BẮT BUỘC: CODE TƯỜNG MINH, KHÔNG DÙNG HÀM CÓ SẴN LÀM THAY

> **ĐÂY LÀ YÊU CẦU TRỰC TIẾP CỦA NGƯỜI DÙNG, ÁP DỤNG CHO MỌI AGENT VÀ THÀNH VIÊN.**
> **Phải tự viết và giải thích được từng bước của thuật toán và quy trình học máy. Không thay bằng mô hình/hàm thư viện, kể cả để làm nhanh, đối chiếu trong tests hoặc cứu demo khi bản tự cài chưa đạt.**

Quy tắc được lưu tại [AGENTS.md](../AGENTS.md) ở gốc repository và `.cursor/rules/explicit-implementation.mdc` để các agent đọc trước khi làm. Khi giao việc cho agent/subagent/worker, phải đưa yêu cầu này vào brief. Các nội dung bên dưới là kế hoạch triển khai, không cho phép bỏ qua mục này.

**Các phần phải tự cài:**

- Gaussian Naive Bayes: nhóm mẫu theo lớp, đếm và tính prior, trung bình, phương sai MLE, ổn định phương sai, log-likelihood Gaussian, chuẩn hóa xác suất và chọn nhãn.
- Kiểm tra/làm sạch dữ liệu, nhận diện trùng, tính trung bình/trung vị, điền thiếu và chuẩn hóa theo công thức.
- Chia phân tầng 80/20, tạo 5 fold, baseline, vòng lặp cross-validation, duyệt các cấu hình và chọn mô hình.
- Đếm TP/FP/TN/FN, confusion matrix, accuracy, precision, recall, F1, Balanced Accuracy, macro-F1 và ROC-AUC.

**Giới hạn dùng thư viện:** không import/gọi `sklearn` hoặc thư viện ML tương đương. Không dùng `GaussianNB`, `DummyClassifier`, `Pipeline`, `SimpleImputer`, `StandardScaler`, `train_test_split`, `StratifiedKFold`, `cross_val_score`, `GridSearchCV`, hàm metric có sẵn hoặc wrapper gọi lại chúng. Không dùng `np.mean/var/std/median`, các hàm tương đương của pandas/`statistics`, hàm Gaussian PDF, softmax hoặc logsumexp để làm thay phép tính cốt lõi.

Được dùng Python và công cụ nền: vòng lặp, điều kiện, list/dict, `len/sum/min/max/sorted`, phép toán số học, `math.log/exp/sqrt/isfinite`; NumPy để lưu/truy cập mảng và phép toán phần tử cơ bản; pandas để đọc/ghi CSV, chọn cột, trình bày bảng; bộ sinh số ngẫu nhiên/shuffle có seed; thư viện đọc/ghi, checksum, lưu/nạp trạng thái, vẽ biểu đồ, Streamlit và pytest. Biểu đồ dùng số liệu đã được mã tự viết tính ra. Cho phép tách hàm tự viết để tái sử dụng, không yêu cầu chép lại logic ở mọi nơi.

**Cách viết và nghiệm thu:** tên biến rõ nghĩa, hàm có đầu vào/đầu ra cụ thể, chú thích công thức và các bước chính, tránh biểu thức nén khó giải thích. Kiểm chứng bằng đáp án tính tay, dữ liệu tổng hợp có tham số biết trước và các tính chất/bất biến; không lấy đầu ra của chính hàm đang kiểm tra làm đáp án kỳ vọng. Code chạy được nhưng gọi hàm có sẵn thay phần phải tự cài vẫn chưa đạt. Khi có lỗi, sửa bản tự cài và báo đúng trạng thái; demo chỉ dùng bản tự cài đã được kiểm chứng.

Trong tài liệu này, **pipeline** chỉ là chuỗi xử lý do nhóm tự viết và trạng thái đã học được lưu lại; **fit/predict/predict_proba** là tên phương thức của lớp tự cài, không phải lời gọi estimator có sẵn. Các tài liệu scikit-learn ở phần tham khảo chỉ dùng để đọc lý thuyết/quy ước, không cho phép sử dụng mã thư viện trong luồng thực thi.

## 1. Định hướng và mục tiêu

### 1.1. Tên đề tài đề xuất

**“Ứng dụng thuật toán Gaussian Naive Bayes trong phân loại khả năng uống được của nước dựa trên các chỉ tiêu chất lượng nước.”**

Tên này chỉ rõ thuật toán, đầu vào và đầu ra. Trong báo cáo, giải thích đây là bài toán dự đoán nhãn `Potability` của bộ dữ liệu, không phải hệ thống cấp chứng nhận nước uống.

### 1.2. Bài toán

- **Đầu vào:** Các thông số lý hóa của một mẫu nước.
- **Đầu ra:** Một trong hai nhãn: `0 — Không uống được` hoặc `1 — Uống được`, theo cách gán nhãn của bộ dữ liệu.
- **Loại bài toán:** Phân lớp nhị phân có giám sát.
- **Thuật toán trọng tâm:** Gaussian Naive Bayes.
- **Quy trình:** Hiểu bài toán → hiểu dữ liệu → tiền xử lý → xây dựng mô hình → đánh giá → trình diễn và báo cáo.

Ứng dụng đến chất lượng nước tạo bối cảnh phù hợp để trình bày tại Đại học Thủy Lợi. Phần học thuật vẫn cần tập trung vào khai phá dữ liệu: chất lượng dữ liệu, giả định của mô hình, thiết kế thực nghiệm và phân tích sai số.

### 1.3. Những câu hỏi nhóm cần trả lời

1. Bộ dữ liệu có thiếu giá trị, mất cân bằng lớp hoặc điểm bất thường như thế nào?
2. Vì sao Gaussian Naive Bayes là lựa chọn phù hợp để thử nghiệm với các biến liên tục?
3. Điền giá trị thiếu bằng trung bình hoặc trung vị ảnh hưởng thế nào đến kết quả?
4. Gaussian Naive Bayes có tốt hơn cách luôn dự đoán lớp phổ biến nhất không?
5. Mô hình nhầm những mẫu nào, đặc biệt là mẫu không uống được bị dự đoán uống được?
6. Bản tự cài đặt có khớp ví dụ tính tay và thỏa các tính chất kiểm chứng về xác suất, nhãn và tính ổn định số học không?

### 1.4. Mục tiêu hoàn thành

- Có quy trình chạy lại được từ dữ liệu gốc đến kết quả đánh giá.
- Cả ba thành viên giải thích được Bayes, giả định độc lập có điều kiện và phân phối Gaussian.
- Có bản tự cài đặt tường minh và bộ đáp án tính tay/kiểm tra tính chất độc lập để kiểm chứng.
- Có kết quả thực nghiệm trung thực, có mô hình cơ sở để đối chiếu.
- Có báo cáo, slide, demo và bằng chứng đóng góp của từng thành viên.

**Không đặt trước yêu cầu “phải đạt 90% hoặc 95%”.** Chỉ tiêu chất lượng của bài là tính đúng đắn, khả năng giải thích và tái lập; mức hiệu năng sẽ được kết luận sau khi thực nghiệm.

## 2. Lựa chọn ngôn ngữ và công cụ

### 2.1. Chốt đề xuất: dùng Python

Python giúp nhóm dùng một ngôn ngữ xuyên suốt từ đọc dữ liệu, tự viết thuật toán đến xây dựng demo. NumPy hỗ trợ lưu mảng và phép toán cơ bản, pandas hỗ trợ đọc/ghi bảng, còn Streamlit hỗ trợ giao diện. Mọi phần học máy và thống kê cốt lõi tuân thủ mục 0. [Nguồn: NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [Streamlit](https://docs.streamlit.io/).

Đối với đề tài này, nhóm nên dành thời gian cho phương pháp và thực nghiệm. Chưa cần xây dựng hệ thống frontend/backend riêng, tài khoản người dùng, cơ sở dữ liệu hoặc ứng dụng di động.

### 2.2. Bộ công cụ dự kiến

- **Python 3:** Ngôn ngữ chung; chốt cùng một phiên bản tương thích trên ba máy trong tuần 1.
- **Python và `math`:** Tự viết thuật toán, thống kê, tiền xử lý, chia tập/fold, thực nghiệm và đánh giá bằng công thức tường minh.
- **NumPy:** Chỉ lưu/truy cập mảng và phép toán phần tử cơ bản; không gọi hàm thống kê có sẵn thay phần tự cài.
- **pandas:** Đọc/ghi CSV, chọn cột và trình bày bảng; không dùng aggregation hoặc tiện ích làm thay logic xử lý cốt lõi.
- **Matplotlib, seaborn:** Biểu đồ phân bố, dữ liệu thiếu, tương quan và kết quả.
- **Jupyter Notebook:** Giải thích từng bước và trình bày thực nghiệm.
- **Streamlit:** Demo chạy cục bộ trên máy tính.
- **Git/GitHub:** Quản lý phiên bản và ghi nhận đóng góp tại [itsthquyen/water-potability-naive-bayes](https://github.com/itsthquyen/water-potability-naive-bayes). Repository cục bộ đã liên kết remote `origin`; việc liên kết không đồng nghĩa mã nguồn đã được push.
- **pytest:** Kiểm tra đáp án tính tay, tính chất thuật toán và luồng dự đoán; không import thư viện ML làm oracle.

Tạo môi trường ảo, ghi lại phiên bản thư viện và hướng dẫn cài đặt. Máy tính cá nhân chạy CPU là cấu hình dự kiến cho phạm vi dữ liệu này; không lập kế hoạch thuê GPU.

## 3. Phạm vi cần làm và phần mở rộng

### 3.1. Phần cốt lõi đề xuất đưa vào bản được duyệt

1. Thu thập, mô tả và kiểm tra dữ liệu.
2. Phân tích khám phá dữ liệu có giải thích bằng lời.
3. Xây dựng quy trình tiền xử lý tránh rò rỉ dữ liệu.
4. Tự cài Gaussian Naive Bayes bằng Python; NumPy chỉ hỗ trợ mảng và phép toán cơ bản.
5. Kiểm chứng bằng đáp án tính tay và tính chất đã biết; không dùng mô hình thư viện làm đối chứng thực thi.
6. Thực nghiệm có mô hình cơ sở và cross-validation.
7. Phân tích kết quả, các loại sai và giới hạn.
8. Demo nhỏ, báo cáo, slide và hướng dẫn chạy.

### 3.2. Chỉ mở rộng khi phần cốt lõi đã hoàn thành

- So sánh thêm một mô hình đơn giản như Logistic Regression nếu giảng viên cho phép.
- Khảo sát một số ngưỡng phân loại bằng dự đoán ngoài fold trên tập phát triển.
- Bổ sung khoảng tin cậy bootstrap cho một vài thước đo trên tập kiểm tra, ghi rõ đây chỉ là bất định trong phạm vi mẫu dữ liệu hiện có.

Không đưa nhiều thuật toán, deep learning, IoT hoặc triển khai đám mây vào mục tiêu bắt buộc. Không mặc định dùng SMOTE; nếu bổ sung sau này, cần lý do thực nghiệm và chỉ áp dụng trong phần huấn luyện của từng fold.

## 4. Kế hoạch dữ liệu

### 4.1. Nguồn được đề xuất

Sử dụng **Water Quality / Water Potability** do Aditya Kadiwal đăng trên Kaggle: [trang bộ dữ liệu](https://www.kaggle.com/datasets/adityakadiwal/water-potability).

Một nghiên cứu sử dụng bộ dữ liệu này mô tả 3.276 mẫu, 9 thuộc tính đầu vào, 1.998 mẫu nhãn 0 và 1.278 mẫu nhãn 1; các cột có giá trị thiếu gồm pH, Sulfate và Trihalomethanes. Đây là thông tin tham khảo, phải kiểm tra lại trên tệp thực tế nhóm tải về. [Nguồn: nghiên cứu trên JSDEWES](https://www.sdewes.org/jsdewes/pid13.0634).

Không mô tả dữ liệu là mẫu đo tại Việt Nam, tại trường hoặc do nhóm tự thu thập. Kiểm tra mô tả nguồn gốc, điều kiện sử dụng và mức độ rõ ràng của cách tạo nhãn trước khi chốt dữ liệu.

**Cách đưa dữ liệu vào project khi bắt đầu triển khai:**

1. Truy cập trang Kaggle ở trên, đăng nhập nếu được yêu cầu và tải bộ dữ liệu.
2. Giải nén, kiểm tra tệp `water_potability.csv` và đặt tại `data/raw/water_potability.csv`, tính từ thư mục gốc repository.
3. Ghi nguồn, ngày tải, phiên bản nếu có và điều kiện sử dụng vào `data/README.md`; kiểm tra số dòng, tên cột và nhãn trên tệp thực tế.
4. Giữ nguyên CSV gốc. Mã xử lý sẽ đọc từ đây và lưu chỉ số chia tập vào `data/splits/`.

Hiện repository chưa có dataset. `.gitignore` đang bỏ qua dữ liệu trong `data/raw/` và `data/splits/`, ngoại trừ `.gitkeep`; mỗi thành viên tải dữ liệu theo cùng hướng dẫn. Chưa tự động tải hoặc đưa dataset lên GitHub.

**Bảo đảm cùng dữ liệu và phép chia giữa ba máy:**

- Lộc phụ trách tạo `data/manifest.json` trong giai đoạn triển khai và đưa file thông tin này vào Git, bên ngoài các thư mục đang bị bỏ qua. Manifest ghi nguồn, ngày tải, SHA-256 của CSV gốc, số dòng trước/sau làm sạch, phiên bản quy tắc làm sạch, phiên bản môi trường và cấu hình chia tập.
- Gán `row_id` bằng vị trí bản ghi trong CSV gốc, bắt đầu từ 0, trước khi lọc hoặc sắp xếp; ID chỉ dùng truy vết, không đưa vào đặc trưng mô hình. Cùng một CSV được xác minh bằng checksum sẽ có cùng ánh xạ ID.
- Lộc tạo danh sách ID chuẩn cho tập phát triển, test và train/validation của từng fold trong `data/splits/`. Manifest lưu checksum của từng danh sách và chỉ rõ phiên bản mã nguồn tạo ra chúng.
- Các máy khác chạy cùng quy trình làm sạch/chia tập trên CSV có checksum khớp, rồi đối chiếu checksum danh sách với manifest. Thống nhất cách tính checksum danh sách: ID theo đúng thứ tự sử dụng, mỗi ID một dòng, UTF-8 không BOM, xuống dòng LF và có LF cuối file.
- Nếu checksum không khớp, dừng việc so sánh thực nghiệm và kiểm tra dữ liệu, thứ tự dòng, quy tắc làm sạch, cấu hình và môi trường; không tự đổi seed để tạo kết quả mới. Dữ liệu và danh sách chia vẫn lưu cục bộ, không cần thêm dịch vụ quản lý dữ liệu.

Manifest chỉ được điền từ dữ liệu và phép chia thực tế: hoàn thành phần nguồn/checksum CSV ở M1, phần làm sạch và phép chia ở M2; không tạo checksum hoặc số liệu minh họa như kết quả thật.

### 4.2. Các trường dữ liệu dự kiến

- `ph`: Độ pH.
- `Hardness`: Độ cứng.
- `Solids`: Tổng chất rắn hòa tan theo mô tả dữ liệu.
- `Chloramines`: Hàm lượng chloramine.
- `Sulfate`: Hàm lượng sulfate.
- `Conductivity`: Độ dẫn điện.
- `Organic_carbon`: Hàm lượng carbon hữu cơ.
- `Trihalomethanes`: Hàm lượng trihalomethane.
- `Turbidity`: Độ đục.
- `Potability`: Nhãn đích, không được đưa vào tập thuộc tính đầu vào.

Lộc lập từ điển dữ liệu cho từng cột: ý nghĩa, đơn vị theo nguồn, kiểu dữ liệu, tỷ lệ thiếu và các điểm cần xác minh. Nếu đơn vị hoặc cách đo chưa rõ, ghi rõ giới hạn; không tự suy ra ngưỡng an toàn từ khoảng giá trị trong CSV.

### 4.3. Việc cần kiểm tra trước khi huấn luyện

- Tệp có đúng tên cột, số cột và mã nhãn không?
- Có dữ liệu không phải số, vô hạn, mất nhãn hoặc nhãn ngoài 0/1 không?
- Có bản ghi lặp hoàn toàn hoặc cùng đầu vào nhưng khác nhãn không?
- Mỗi cột thiếu bao nhiêu giá trị? Có mẫu thiếu nhiều chỉ tiêu không?
- Tỷ lệ hai lớp thực tế là bao nhiêu?
- Nguồn có thông tin địa điểm, thời gian hoặc nhóm mẫu không? Nếu có, cân nhắc chia theo nhóm hoặc thời gian thay cho chia ngẫu nhiên.

Giữ nguyên tệp gốc; mọi sửa đổi phải thực hiện bằng mã và có thống kê số mẫu trước/sau. Bản ghi trùng không được lọt sang cả tập huấn luyện và kiểm tra. Quy tắc xử lý các bản ghi mâu thuẫn cần được chốt trước khi đánh giá mô hình.

Trước mọi phép chia, tại M2 phải chốt một trong hai cách: loại bản ghi trùng theo quy tắc cố định, hoặc giữ các bản ghi có cùng đầu vào thành một nhóm không tách rời. Ràng buộc này áp dụng cả phép chia phát triển/test và train/validation trong từng fold; ID khác nhau không đủ chứng minh mẫu khác nhau. Nhận diện nhóm trên 9 giá trị đầu vào gốc, với ô thiếu có biểu diễn thống nhất, không dựa trên giá trị đã điền hoặc dự đoán. Ghi riêng quyết định cho nhóm cùng đầu vào nhưng khác nhãn và ánh xạ các ID bị loại/gom. Nếu giữ nhóm, ưu tiên không tách nhóm hơn yêu cầu cân bằng tuyệt đối của fold và ghi sai lệch phân tầng thực tế trong manifest.

### 4.4. Phân tích khám phá dữ liệu

Sau kiểm tra cấu trúc và chia dữ liệu, dùng tập phát triển để phân tích phục vụ lựa chọn phương pháp:

1. Biểu đồ số lượng và tỷ lệ mỗi lớp.
2. Biểu đồ tỷ lệ thiếu theo thuộc tính.
3. Histogram theo từng lớp; không bắt buộc ước lượng mật độ KDE.
4. Boxplot để quan sát phân bố và giá trị cực đoan.
5. Ma trận tương quan giữa các thuộc tính; có thể xem thêm theo từng lớp.

Mỗi hình cần có câu hỏi, nhận xét và hệ quả đối với mô hình. Tương quan tuyến tính thấp không chứng minh các biến độc lập có điều kiện. Cũng không xóa một mẫu chỉ vì nằm ngoài ngưỡng boxplot: đó có thể là mẫu nước có đặc điểm đáng quan tâm.

Lộc tự tính số đếm bin, phân vị boxplot và tương quan Pearson; thư viện chỉ dựng hình từ số liệu đã tính, không gọi tiện ích tự thống kê thay phần này. Chốt số bin và biên bin trên tập phát triển; phân vị dùng nội suy tuyến tính tại vị trí `(n - 1) * q` của dãy đã sắp xếp. Boxplot dùng hàng rào `Q1 - 1,5 * IQR`, `Q3 + 1,5 * IQR` và râu tại quan sát xa nhất còn trong hàng rào. Tương quan dùng các cặp giá trị cùng không thiếu, ghi số cặp; dưới hai cặp hoặc mẫu số bằng 0 thì ghi `N/A`. Cột rỗng không có phân vị. Quyền kiểm tra chéo bằng ví dụ nhỏ trước khi dùng hình trong báo cáo.

## 5. Thiết kế phương pháp và thuật toán

### 5.1. Vì sao chọn Gaussian Naive Bayes?

Các đầu vào là số đo liên tục. Gaussian Naive Bayes mô hình hóa phân bố từng thuộc tính trong từng lớp bằng phân phối chuẩn và dùng giả định độc lập có điều kiện. Đây là giả định cần kiểm tra và thảo luận, không phải tính chất đã được chứng minh của bộ dữ liệu. [Nguồn: tài liệu Naive Bayes của scikit-learn](https://scikit-learn.org/stable/modules/naive_bayes.html).

### 5.2. Nội dung lý thuyết phải trình bày

- Định lý Bayes và ý nghĩa xác suất tiên nghiệm, likelihood, xác suất hậu nghiệm.
- Giả định độc lập của các thuộc tính khi đã biết lớp.
- Phân phối Gaussian, trung bình và phương sai theo từng lớp.
- Quy tắc chọn lớp có điểm hậu nghiệm lớn nhất.
- Lý do dùng log để tính tổng thay cho nhân nhiều giá trị nhỏ.
- Vai trò của phần cộng ổn định phương sai khi phương sai quá nhỏ.

Công thức phân loại dùng trong phần giải thích:

$$
\hat{y}=\arg\max_{c\in\{0,1\}}\left[\log P(c)+\sum_{j=1}^{9}\log p(x_j\mid c)\right].
$$

Nhóm chuẩn bị một ví dụ số nhỏ với 2 thuộc tính và 2 lớp để tính từng bước. Ghi rõ đó là ví dụ minh họa thuật toán, không phải kết quả thực nghiệm trên toàn bộ dữ liệu.

### 5.3. Bản tự cài đặt

Quyền phụ trách chính, gồm các chức năng:

- `fit`: Ước lượng xác suất mỗi lớp, trung bình và phương sai của từng thuộc tính theo lớp.
- `predict_log_proba` hoặc hàm tính điểm log: Tính điểm và chuẩn hóa ổn định khi cần xác suất.
- `predict`: Trả về nhãn dự đoán.
- `predict_proba`: Trả về xác suất mô hình ước lượng cho hai lớp.

Đếm số mẫu mỗi lớp để tính prior; tự cộng giá trị và chia số mẫu để tính trung bình; tự cộng bình phương độ lệch và chia số mẫu của lớp để tính phương sai MLE (`ddof=0`). Tính log-likelihood theo công thức Gaussian, cộng với log prior; chuẩn hóa bằng cách trừ điểm log lớn nhất trước khi lấy exp, rồi chia tổng exp do mã tự tính. Tự so sánh điểm để chọn nhãn, thống nhất thứ tự lớp `[0, 1]` và quy tắc chọn lớp 0 nếu hai điểm bằng nhau.

Thống nhất công thức: với `X_train` là ma trận sau điền thiếu và sau chuẩn hóa nếu cấu hình có bật, tự tính phương sai MLE toàn cục `v_j` của từng cột trên đúng phần fit. Đặt `epsilon = max(var_smoothing * max_j(v_j), variance_floor)` và `variance_effective[c, j] = variance_mle[c, j] + epsilon`. `var_smoothing` phải hữu hạn và dương; `variance_floor` mặc định `10^-12`, là sàn số học dương hữu hạn trong đơn vị bình phương của đầu vào mô hình, không phải ngưỡng chất lượng nước. Sàn này bảo vệ trường hợp dữ liệu hằng hoàn toàn hoặc tích smoothing bị quá nhỏ. Giữ cùng sàn cho mọi cấu hình E1–E4; nếu thay thì chốt trước CV, không chọn theo test. Lưu phương sai MLE chưa cộng, epsilon, sàn và phương sai hiệu dụng; không cộng epsilon lần nữa khi nạp. Không gọi hàm thống kê, Gaussian PDF hoặc chuẩn hóa xác suất có sẵn.

Kiểm tra tính hữu hạn sau tính thống kê, transform và tính điểm log; đầu vào hữu hạn không bảo đảm phép bình phương hoặc chia không tràn số. Khi fit có tham số không hữu hạn/phương sai hiệu dụng không dương, dừng và báo lỗi, không lưu model. Khi predict gặp lỗi số học hoặc điểm log không hữu hạn, từ chối đúng dòng đó và để trống nhãn/xác suất; không để lỗi một dòng làm mất kết quả các dòng hợp lệ. Không tự gán nhãn khi cả hai điểm là `-inf` hoặc xác suất là `NaN`.

Kiểm chứng trên bộ mẫu nhỏ có prior, trung bình, phương sai, điểm log và xác suất được tính tay độc lập; ghi rõ sai số số học cho phép trước khi chạy kiểm thử. Bổ sung các tính chất: tổng xác suất bằng 1 trong sai số cho phép, xác suất không âm, đổi thứ tự dòng không làm thay đổi tham số ngoài sai số và lưu/nạp giữ nguyên dự đoán.

Chỉ dùng bản tự cài đã kiểm chứng cho huấn luyện chính và demo. Nếu chưa đạt, tiếp tục sửa, ghi rõ hạn chế và áp dụng tiến độ xử lý tại M3; **không chuyển sang bản thư viện hoặc mô hình huấn luyện sẵn**.

### 5.4. Pipeline tiền xử lý

Luồng mặc định: **Dữ liệu đầu vào → điền giá trị thiếu → Gaussian Naive Bayes → nhãn dự đoán.**

- Thử điền thiếu bằng trung bình và trung vị.
- Tự viết chuẩn hóa z-score trên ma trận train đã điền thiếu: tính phương sai với mẫu số `n` (`ddof=0`), lấy căn để có độ lệch chuẩn, rồi áp dụng `(x - mean) / scale`. Nếu độ lệch chuẩn bằng 0, lưu `scale=1`, vẫn trừ trung bình; cột hằng trong train trở thành 0. Khảo sát ảnh hưởng đến tính toán và ổn định phương sai; không coi chuẩn hóa là điều kiện bắt buộc của Gaussian NB.
- Giữ các giá trị cực đoan có thể hợp lệ trong cấu hình chính.
- Không chọn đặc trưng chỉ dựa trên tương quan tuyến tính với nhãn.
- Không điền thiếu theo nhãn thật `Potability`: khi dự đoán mẫu mới, nhãn này chưa biết.

Tự tính trung bình hoặc trung vị trên các giá trị không thiếu của phần huấn luyện; với trung vị, sắp xếp rồi lấy phần tử giữa hoặc trung bình hai phần tử giữa. Tự lưu các giá trị dùng để điền thiếu/chuẩn hóa; khi dự đoán chỉ áp dụng giá trị đã lưu. Nếu một cột thiếu hoàn toàn trong phần fit, báo lỗi dữ liệu rõ ràng thay vì âm thầm bỏ cột hoặc dùng thông tin từ tập khác.

Mọi bộ điền thiếu và chuẩn hóa phải được học trên phần huấn luyện tương ứng. Tự viết chuỗi gọi các bước và khởi tạo trạng thái mới bên trong từng fold; không gọi `sklearn.pipeline.Pipeline`. Không để thống kê của fold trước hoặc validation/test đi vào lần fit sau. [Nguồn lý thuyết về rò rỉ dữ liệu](https://scikit-learn.org/stable/common_pitfalls.html).

## 6. Thiết kế thực nghiệm

### 6.1. Chia dữ liệu và giữ tập kiểm tra độc lập

Đề xuất sau khi xử lý các vấn đề cấu trúc và bản ghi trùng:

- **80% tập phát triển:** Dùng cho phân tích, huấn luyện và lựa chọn cấu hình.
- **20% tập kiểm tra cuối:** Giữ riêng đến khi chốt mô hình.
- Chia phân tầng theo nhãn; đề xuất cố định `random_state=42` cho phép chia 80/20 và bộ fold, ghi vào manifest trước khi thực nghiệm.
- Lưu danh sách ID mẫu của từng tập và từng fold, kiểm tra khớp manifest theo mục 4.1 để cả nhóm sử dụng cùng một phép chia.

Tự viết phép chia phân tầng: nhóm ID theo nhãn, xáo trộn trong từng lớp bằng bộ sinh số có seed, phân bổ theo tỷ lệ 80/20 với quy tắc làm tròn cố định được ghi trong manifest. Trên 80% tập phát triển, tự tạo **5 fold phân tầng** bằng phân phối ID của mỗi lớp vào 5 nhóm sao cho chênh lệch số mẫu cùng lớp giữa các nhóm không quá 1. Mỗi lượt lấy một nhóm làm validation, bốn nhóm còn lại làm train. Kiểm tra đủ hai lớp trong các phần cần đánh giá, không giao nhau giữa train/validation và mỗi mẫu được validation đúng một lượt. Không gọi hàm chia tập/fold có sẵn. [Nguồn tham khảo khái niệm phân tầng](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html).

Nếu metadata cho thấy nhiều mẫu cùng địa điểm hoặc cùng đợt lấy mẫu, nhóm phải rà soát lại chiến lược chia. Chia ngẫu nhiên chỉ là phương án mặc định khi không có cấu trúc nhóm/thời gian cần bảo vệ.

### 6.2. Các thí nghiệm cần làm

1. **E0 — Mô hình cơ sở tự viết:** Đếm nhãn trên phần huấn luyện, chọn lớp phổ biến nhất và tự tạo dự đoán hằng; nếu hòa, chọn lớp 0. Điểm lớp 1 cho mọi mẫu là prior lớp 1 học trên phần train tương ứng, không dùng tỷ lệ lớp của validation/test. Khi đánh giá có đủ hai lớp, điểm hằng cho ROC-AUC bằng 0,5. Mục đích: xác định mức tham chiếu tối thiểu, không dùng `DummyClassifier`.
2. **E1 — Gaussian NB + điền trung bình:** Cấu hình đơn giản đầu tiên.
3. **E2 — Gaussian NB + điền trung vị:** Đánh giá tác động của lựa chọn cách điền thiếu.
4. **E3 — Gaussian NB + điền trung vị + chuẩn hóa:** Khảo sát tác động của chuẩn hóa khi các đại lượng có thang đo khác nhau.
5. **E4 — Gaussian NB được lựa chọn bằng cross-validation:** Tìm cấu hình trong không gian nhỏ, được chốt trước.
6. **E5 — Kiểm chứng bản tự cài:** So sánh từng bước với đáp án tính tay và kiểm tra tính chất trên dữ liệu nhỏ/tổng hợp. Đây là kiểm chứng tính đúng đắn, không phải một mô hình học máy mới hay phép thử trên test cuối.

E1–E3 dùng chung `var_smoothing=10^-9`, `variance_floor=10^-12`, cùng prior theo tần suất train, cùng bộ fold và quy tắc hòa điểm chọn lớp 0. E1/E2 không chuẩn hóa; E3 có chuẩn hóa. Chỉ thay yếu tố đã nêu để so sánh; chỉ E4 duyệt lưới. Các mặc định và danh sách cấu hình có một nơi định nghĩa dùng chung, được kiểm tra giá trị và ghi vào manifest thực tế.

Không gian tìm kiếm E4 đề xuất:

- Chiến lược điền thiếu: `mean`, `median`.
- Chuẩn hóa: có hoặc không.
- `var_smoothing`: 7 giá trị `10^-12`, `10^-11`, `10^-10`, `10^-9`, `10^-8`, `10^-7`, `10^-6`.

Tổng cộng 28 cấu hình, 140 lượt fit với 5 fold, chưa tính fit cuối. Tự viết vòng lặp duyệt cấu hình và vòng lặp fold, khởi tạo mô hình/tiền xử lý mới cho mỗi lượt; tự tổng hợp chỉ số, không gọi grid search hoặc cross-validation có sẵn. Đây là giới hạn thực nghiệm chủ động để tránh thử tùy ý quá nhiều cấu hình. `var_smoothing` là tên tham số của bản tự cài, dùng để tính lượng ổn định cộng vào phương sai theo công thức đã ghi ở mục 5.3. [Nguồn tham khảo khái niệm ổn định phương sai](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html).

### 6.3. Quy tắc lựa chọn mô hình

- Chỉ tiêu chính: **Balanced Accuracy trung bình trên 5 fold**.
- Chỉ tiêu theo dõi bổ sung: macro-F1, precision/recall từng lớp và độ lệch chuẩn giữa các fold.
- Điểm CV là trung bình cộng không trọng số của 5 điểm fold. Độ lệch chuẩn mô tả 5 fold dùng mẫu số 5 (`ddof=0`), không phải khoảng tin cậy; tự tính và lưu cả điểm từng fold. Nếu một lượt fit/metric bắt buộc lỗi, ghi cấu hình chưa đánh giá thành công, không bỏ fold lỗi để lấy trung bình; sửa nguyên nhân trước khi chọn cấu hình cuối.
- Xét các cấu hình có điểm Balanced Accuracy trung bình cách điểm tốt nhất không quá 0,005, dùng điểm chưa làm tròn. Trong nhóm này, ưu tiên không chuẩn hóa; nếu vẫn còn nhiều lựa chọn, ưu tiên `mean` trước `median`, rồi `var_smoothing` có số mũ gần `-9` nhất; nếu khoảng cách bằng nhau, chọn giá trị nhỏ hơn. Hai quy tắc cuối là quy ước phân xử cố định, không khẳng định một cách điền thiếu hay mức smoothing tốt hơn về lý thuyết. Nhóm duyệt thứ tự này trước khi chạy, không thay theo điểm test.
- Chốt cấu hình rồi fit lại trên toàn bộ 80% tập phát triển.
- Đánh giá một đợt cuối trên 20% tập kiểm tra cho mô hình đã chốt và mô hình cơ sở.

Điểm cross-validation của cấu hình được chọn có thể lạc quan do quá trình tìm kiếm. Kết quả trên tập kiểm tra độc lập là cơ sở báo cáo khả năng khái quát trong phạm vi bộ dữ liệu này.

Không đổi seed hoặc điều chỉnh mô hình theo kết quả test để lấy số đẹp hơn. Nếu phát hiện lỗi lập trình sau đánh giá cuối, phải sửa và ghi lại việc đánh giá lại; không che giấu lịch sử.

## 7. Cách đánh giá và phân tích kết quả

### 7.1. Những kết quả bắt buộc xuất ra

- Accuracy.
- Balanced Accuracy.
- Precision, recall và F1-score của từng lớp.
- Macro-F1.
- Confusion matrix, ghi rõ hàng là nhãn thật và cột là nhãn dự đoán.
- ROC-AUC cho điểm dự đoán lớp 1; chỉ dùng như thước đo bổ sung.
- Điểm trung bình và độ lệch chuẩn cross-validation, tách biệt với điểm test.

Balanced Accuracy là trung bình recall của các lớp, giúp quan sát việc mô hình có bỏ qua lớp ít mẫu hơn hay không. [Nguồn: balanced_accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html).

Với tỷ lệ lớp được tài liệu mô tả, cách luôn dự đoán nhãn 0 có thể đạt khoảng 61% accuracy trên dữ liệu có cùng tỷ lệ. Đây chỉ là phép tính minh họa từ tỷ lệ lớp, chưa phải kết quả baseline của nhóm.

**Quy ước thống nhất khi tính và hiển thị chỉ số:**

- Cố định thứ tự lớp `[0, 1]` trong chỉ số theo lớp, confusion matrix và xác suất trả về. Ghi số mẫu thật của từng lớp (`support`) cùng kết quả.
- Tự kiểm tra mẫu số khi tính precision/recall/F1; nếu bằng 0, dùng giá trị thay thế 0 trong tính toán và lưu thêm dấu hiệu không xác định để bảng trình bày ghi `N/A` cùng chú thích. Đây là quy ước tự cài tương đương ý nghĩa `zero_division=0`, không phải gọi hàm metric thư viện. Không dùng `np.nan` rồi âm thầm loại một lớp khỏi trung bình.
- Macro-F1 là trung bình F1 của đúng hai lớp `[0, 1]`, theo quy ước trên. Precision không xác định không đồng nghĩa F1 cũng không xác định: nếu có mẫu thật lớp 1 nhưng baseline không dự đoán lớp 1, precision lớp 1 là `N/A`, còn recall và F1 lớp 1 đều bằng 0.
- ROC-AUC dùng điểm/xác suất cho lớp 1, không dùng nhãn dự đoán cứng của Gaussian NB. Nếu tập đánh giá chỉ có một lớp thật, ghi ROC-AUC là `N/A`, không thay bằng 0; đồng thời kiểm tra lại phép chia trước khi công nhận kết quả thực nghiệm.

Tự duyệt từng cặp nhãn thật/dự đoán để đếm TP/FP/TN/FN, tính các tỷ số theo công thức và lấy trung bình các lớp. ROC-AUC được tự tính bằng sắp xếp điểm lớp 1, xử lý các điểm bằng nhau theo cùng nhóm, dựng các điểm ROC rồi cộng diện tích hình thang; có ví dụ tính tay cho trường hợp hòa điểm và xếp hạng hoàn hảo. Không gọi hàm metric/ROC/AUC có sẵn.

Quy ước áp dụng giống nhau cho baseline và Gaussian NB tự cài, trong CV và đánh giá cuối. [Nguồn tham khảo quy ước chỉ số](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html).

### 7.2. Phân tích loại sai quan trọng

Giữ quy ước **lớp dương là 1 — uống được**:

- **False positive (FP):** Mẫu thực tế nhãn 0 nhưng mô hình dự đoán nhãn 1.
- **False negative (FN):** Mẫu thực tế nhãn 1 nhưng mô hình dự đoán nhãn 0.

Phân tích riêng FP vì đây là lỗi dự đoán “uống được” đối với mẫu được dữ liệu gán nhãn “không uống được”. Báo cáo số lượng và hai tỷ lệ có ý nghĩa khác nhau:

- `FP / (FP + TN)`: Trong các mẫu thực tế không uống được, bao nhiêu mẫu bị dự đoán uống được?
- `FP / (TP + FP)`: Trong các mẫu được dự đoán uống được, bao nhiêu mẫu thực tế mang nhãn không uống được?

Nếu mẫu số bằng 0, ghi rõ không xác định thay vì tự diễn giải là không có rủi ro.

### 7.3. Phân tích định tính

- Xem khoảng 5–10 trường hợp dự đoán sai, ưu tiên có cả FP và FN nếu tồn tại.
- So sánh mức độ thiếu dữ liệu, giá trị thuộc tính và điểm mô hình giữa mẫu đúng/sai.
- Thảo luận khả năng các phân bố hai lớp chồng lấn, giả định Gaussian chưa phù hợp hoặc thông tin đầu vào chưa đủ.
- Không kết luận một chỉ tiêu “gây ra” việc không uống được chỉ từ tương quan hoặc sai số mô hình.
- Nếu mô hình không vượt baseline, báo cáo đúng và phân tích nguyên nhân; không thay dữ liệu hoặc lọc mẫu theo kết quả dự đoán.

Việc phân tích lỗi trên tập test chỉ phục vụ thảo luận kết quả cuối. Nếu muốn dùng phát hiện đó để cải tiến mô hình, phải coi đó là vòng nghiên cứu mới và cần một bộ đánh giá độc lập mới.

## 8. Demo đề xuất

### 8.1. Hình thức

Ứng dụng Streamlit chạy cục bộ, dùng đúng pipeline đã lưu từ thực nghiệm. Giao diện tiếng Việt, rõ ràng, ưu tiên thao tác ổn định trên máy thuyết trình.

### 8.2. Chức năng cốt lõi

1. **Giới thiệu:** Bài toán, nhóm thực hiện, nguồn dữ liệu và phạm vi áp dụng.
2. **Khám phá dữ liệu:** Một số biểu đồ đã chuẩn bị và giải thích ngắn.
3. **Dự đoán một mẫu:** Nhập 9 chỉ tiêu hoặc chọn mẫu minh họa có sẵn.
4. **Dự đoán CSV:** Kiểm tra cột, xử lý theo pipeline và tải kết quả.
5. **Kết quả thực nghiệm:** Hiển thị chỉ số thật, confusion matrix và cấu hình đã chốt.

### 8.3. Quy tắc hành vi của demo

- Hiển thị “Mô hình dự đoán: uống được/không uống được theo nhãn dữ liệu”.
- Kiểm tra đầu vào không phải số, vô hạn, thiếu cột hoặc sai cấu trúc.
- Với ô thiếu được cho phép, thông báo thuộc tính nào đã được điền bằng giá trị học từ tập huấn luyện.
- Không dự đoán một mẫu thiếu toàn bộ 9 chỉ tiêu; cảnh báo khi mẫu thiếu quá nhiều hoặc nằm ngoài phạm vi dữ liệu huấn luyện.
- Không tự sắp xếp sai cột hoặc âm thầm lấy `Potability` làm đặc trưng khi tải CSV.
- Nếu hiển thị `predict_proba`, ghi “xác suất do mô hình ước lượng”, không gọi là tỷ lệ an toàn thực tế. Naive Bayes có thể cho xác suất chưa được hiệu chỉnh tốt. [Nguồn: tài liệu Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html).
- Hiển thị ngắn gọn: “Demo học thuật; không thay thế kiểm nghiệm chất lượng nước”.

Mẫu minh họa được ghi rõ nguồn và chọn sau khi chốt mô hình. Chuẩn bị cả trường hợp dự đoán sai để giải thích giới hạn; không chỉ trình diễn những mẫu thuận lợi.

### 8.4. Quy ước đầu vào, lỗi và file kết quả

Quyền và Séc chốt đặc tả đầu vào từ M1; Lộc xác nhận các quy tắc liên quan đến dữ liệu sau audit ở M2, trước khi hoàn thiện demo ở M3.

- **Cấu trúc CSV:** nhận CSV UTF-8, dấu phẩy phân tách, có header chứa đủ 9 cột ở mục 4.2. Thiếu cột, tên cột trùng hoặc file không đọc được thì từ chối toàn bộ file và báo rõ lý do. Cột đủ nhưng đảo thứ tự được sắp lại theo tên cột chuẩn, không dự đoán theo vị trí cột người dùng gửi.
- **Đọc nguyên trạng:** kiểm tra header trước khi parser có thể tự đổi tên cột trùng. Tên cột phải khớp chính xác, phân biệt hoa/thường, không tự cắt khoảng trắng ở tên cột. Đọc ô đặc trưng dưới dạng chuỗi và tắt suy diễn missing mặc định; cắt khoảng trắng hai đầu giá trị rồi chỉ coi chuỗi rỗng hoặc đúng `NaN` là thiếu. `NULL`, `N/A`, `nan`, chuỗi khác không chuyển được thành số và mọi biểu diễn vô hạn đều là lỗi. Không chuẩn hóa `sample_id` như số để tránh mất số 0 đầu; quy tắc ô thiếu tương đương áp dụng cho nhập tay và API trực tiếp.
- **Cột thừa:** chỉ cho phép thêm `sample_id` để đối chiếu và `Potability` nếu người dùng lấy mẫu từ dataset. `Potability` phải được tách bỏ với thông báo rõ, không đưa vào đặc trưng. Cột thừa khác làm file bị từ chối để người dùng kiểm tra tên cột.
- **Thiếu ô:** ô rỗng hoặc `NaN` trong cột hợp lệ được coi là giá trị thiếu và xử lý bằng imputer đã fit. Chuỗi không phải số hoặc giá trị vô hạn là lỗi, không tự chuyển thành ô thiếu. Dòng thiếu cả 9 chỉ tiêu bị từ chối.
- **Thiếu nhiều chỉ tiêu:** đề xuất cảnh báo khi thiếu từ 3/9 chỉ tiêu trở lên nhưng vẫn cho dự đoán nếu chưa thiếu toàn bộ. Đây chỉ là quy ước giao diện cần xác nhận ở M2, không phải ngưỡng an toàn hay kết luận rút ra từ dữ liệu. Nếu điều chỉnh, ghi quyết định trước M3 và trước đánh giá test cuối.
- **Ngoài phạm vi huấn luyện:** cảnh báo khi một giá trị hữu hạn nằm ngoài min/max quan sát của thuộc tính trên phần dữ liệu fit mô hình, trước điền thiếu/chuẩn hóa. Lưu các mốc này cùng pipeline; không lấy min/max từ test hoặc CSV người dùng. Cảnh báo không tự cắt ngưỡng, không tự xóa mẫu và không diễn giải thành tiêu chuẩn nước uống.
- **Lỗi số học:** ngoài cảnh báo min/max, nếu transform hoặc mô hình không tính được kết quả hữu hạn theo mục 5.3 thì từ chối dòng với thông báo lỗi số học. Không âm thầm clip đầu vào, trả xác suất đều hoặc gán nhãn thay thế.
- **CSV có cả dòng đúng và sai:** nếu cấu trúc file hợp lệ, dự đoán các dòng hợp lệ và trả lỗi từng dòng còn lại; không làm mất dòng hoặc đổi thứ tự. Dòng không được dự đoán có nhãn và xác suất để trống. Nếu mọi dòng đều lỗi, chỉ xuất báo cáo lỗi và không gọi mô hình.
- **Đầu ra:** mỗi dòng có `input_row_id` theo vị trí bản ghi đầu vào, `sample_id` nếu được cung cấp, nhãn dự đoán, xác suất lớp 0/1 nếu hiển thị, trạng thái, thông báo lỗi/cảnh báo và `run_id` của pipeline. `input_row_id` là khóa đối chiếu kể cả khi `sample_id` bị trùng; không đưa hai ID vào mô hình.

Các quy tắc kiểm tra dùng chung cho nhập tay, CSV và gọi `src/predict.py` trực tiếp. Không coi cảnh báo thiếu dữ liệu hoặc ngoài phạm vi là bằng chứng mẫu nước an toàn hay không an toàn.

## 9. Phân công cho Quyền, Lộc và Séc

Phân công dưới đây là đề xuất theo mảng công việc, chưa giả định năng lực sẵn có của từng bạn. Có thể hoán đổi người phụ trách nhưng phải giữ đầu ra và trách nhiệm kiểm tra chéo.

### 9.1. Quyền — Thuật toán và tích hợp

**Trách nhiệm chính:**

- Viết cơ sở lý thuyết và ví dụ tính tay.
- Tự cài Gaussian Naive Bayes tường minh bằng Python; NumPy chỉ hỗ trợ mảng/phép toán cơ bản.
- Phối hợp viết chuỗi xử lý và kiểm chứng từng bước bằng đáp án tính tay/tính chất đã biết.
- Chốt giao diện hàm huấn luyện, dự đoán và định dạng mô hình lưu.
- Tích hợp mã nguồn, kiểm tra khả năng chạy lại toàn bộ dự án.

**Sản phẩm bàn giao:** Module thuật toán; kiểm thử đối chiếu; ví dụ tính tay; chương phương pháp; hướng dẫn luồng chương trình.

**Phần thuyết trình:** Bài toán, lựa chọn thuật toán và cơ chế hoạt động.

**Người kiểm tra chéo:** Lộc kiểm tra số học và dữ liệu đầu vào; Séc kiểm tra kết quả khi gọi từ demo.

### 9.2. Lộc — Dữ liệu và thực nghiệm

**Trách nhiệm chính:**

- Tải dữ liệu, ghi nguồn và lập từ điển dữ liệu.
- Kiểm tra dữ liệu thiếu, trùng, nhãn và giá trị bất thường.
- Thực hiện EDA và giải thích biểu đồ.
- Chuẩn bị phép chia cố định và pipeline tiền xử lý.
- Chạy các thí nghiệm đã thống nhất, lưu cấu hình, chỉ số và biểu đồ.

**Sản phẩm bàn giao:** Notebook dữ liệu; báo cáo kiểm tra dữ liệu; chỉ số chia tập; mã chạy thí nghiệm; nhật ký kết quả; chương dữ liệu và thiết kế thực nghiệm.

**Phần thuyết trình:** Nguồn dữ liệu, tiền xử lý và cách tổ chức đánh giá.

**Người kiểm tra chéo:** Quyền rà soát rò rỉ dữ liệu; Séc kiểm tra số liệu trước khi đưa vào báo cáo.

### 9.3. Séc — Đánh giá, demo và trình bày

**Trách nhiệm chính:**

- Xây phần tổng hợp chỉ số, confusion matrix và phân tích lỗi.
- Phát triển demo Streamlit dùng pipeline chung.
- Kiểm tra luồng nhập mẫu, tải CSV và tải kết quả.
- Viết chương kết quả, thảo luận và giới hạn.
- Tổng hợp slide, thống nhất hình thức báo cáo và chuẩn bị kịch bản demo.

**Sản phẩm bàn giao:** Mã đánh giá; ứng dụng demo; phân tích FP/FN; chương kết quả; slide; kịch bản trình diễn.

**Phần thuyết trình:** Kết quả, demo, hạn chế và hướng phát triển.

**Người kiểm tra chéo:** Lộc đối chiếu số liệu; Quyền kiểm tra tính nhất quán giữa mô hình và ứng dụng.

### 9.4. Nguyên tắc làm việc chung

- Mỗi người có phần lập trình, phần viết và phần thuyết trình; không giao một người chỉ làm slide.
- Mỗi người viết bản thảo phần mình phụ trách. Séc tổng hợp hình thức, không chịu trách nhiệm viết thay cả nhóm.
- Mỗi tuần họp khoảng 30–45 phút để xem sản phẩm thực tế và chốt việc tiếp theo.
- Dùng một danh sách công việc chung: người phụ trách, hạn nội bộ, đầu ra, trạng thái và người kiểm tra.
- Mọi thành viên phải hiểu toàn bộ pipeline ở mức đủ trả lời câu hỏi cơ bản.
- Không sửa tệp báo cáo, số liệu hoặc mã nguồn chung theo các bản rời không có lịch sử.
- Mỗi thành viên làm trên nhánh công việc riêng; người kiểm tra chéo xem thay đổi trước khi Quyền tích hợp vào `main`. Không đưa secrets, môi trường ảo hoặc dữ liệu/mô hình cục bộ bị bỏ qua vào Git; gói mô hình bàn giao theo mục 11.4.

**Ước lượng:** 6–8 giờ/người/tuần trong 6 tuần, khoảng 36–48 giờ/người. Sau tuần 1 điều chỉnh theo tốc độ thực tế và kinh nghiệm Python của nhóm.

## 10. Lộ trình 6 tuần và điều kiện qua từng mốc

### Tuần 1 — Chốt yêu cầu, môi trường và dữ liệu

- **Quyền:** Tìm hiểu Gaussian NB, thống nhất cấu trúc dự án, chuẩn bị ví dụ tính tay và chốt đầu vào/đầu ra dùng chung theo mục 11.3.
- **Lộc:** Kiểm tra nguồn, tải dữ liệu và lập báo cáo chất lượng dữ liệu ban đầu.
- **Séc:** Làm rõ sản phẩm nộp, phác thảo demo và thống nhất khung báo cáo.
- **Cả nhóm:** Xác nhận với giảng viên công cụ được dùng, yêu cầu tự cài thuật toán, mẫu báo cáo và thời lượng bảo vệ.

**Mốc M1:** Cả ba máy chạy được môi trường; có dữ liệu gốc cùng checksum, nguồn trích dẫn, danh sách vấn đề dữ liệu, đặc tả đầu vào/đầu ra kèm ví dụ minh họa và kế hoạch được nhóm thống nhất. Hoàn thành đặc tả chậm nhất đầu tuần 2, trước khi viết các phần phụ thuộc song song.

### Tuần 2 — EDA, chia tập và mô hình đầu tiên

- **Quyền:** Hoàn thành bản Gaussian NB tự cài đầu tiên và kiểm tra trên ví dụ nhỏ.
- **Lộc:** Chốt quy tắc làm sạch, lưu phép chia, thực hiện EDA trên tập phát triển.
- **Séc:** Chuẩn bị hàm đánh giá và khung ứng dụng nhận dữ liệu đúng cấu trúc.
- **Cả nhóm:** Chạy baseline tự viết và cấu hình Gaussian NB tự cài đơn giản sau khi kiểm tra ví dụ tính tay; không dùng mô hình thư viện.

**Mốc M2:** Có luồng huấn luyện–dự đoán–đánh giá hoạt động trên tập phát triển; test cuối vẫn được giữ riêng; biểu đồ có nhận xét. Manifest phép chia/fold khớp giữa các máy; quy tắc làm sạch, xử lý metric và đầu vào demo đã được ghi nhận thống nhất.

### Tuần 3 — Kiểm chứng thuật toán và hoàn thiện thực nghiệm

- **Quyền:** Kiểm chứng bản tự cài bằng đáp án tính tay và tính chất đã biết; xử lý ổn định số học và hoàn thiện lý thuyết.
- **Lộc:** Chạy E1–E4 bằng cùng bộ fold, lưu đầy đủ cấu hình và kết quả.
- **Séc:** Tổng hợp chỉ số, kiểm tra cách tính FP/FN và phát triển demo dự đoán một mẫu.

**Mốc M3:** Có kết quả cross-validation; bản tự cài được kiểm chứng; chưa sử dụng test để chọn cấu hình.

**Nếu kiểm chứng bản tự cài chưa đạt M3:** đánh dấu phần thuật toán chưa hoàn thành, không công nhận M3 đã đạt đầy đủ. Quyền sửa bản tự cài, Lộc kiểm tra từng phép tính trên ví dụ nhỏ/tập phát triển; hạn xử lý đề xuất là cuối tuần 4. Lộc và Séc tiếp tục các phần độc lập đã đúng như audit dữ liệu, tài liệu, giao diện nhập liệu và kiểm tra lỗi; chưa hiển thị dự đoán khi thuật toán chưa được kiểm chứng. Nếu vẫn chưa đạt, nhóm ghi rõ lỗi và trao đổi với giảng viên về tiến độ; **không dùng pipeline thư viện hoặc mô hình có sẵn thay thế, không tạo kết quả giả**. Không mở test cuối trước khi mô hình đạt kiểm chứng và cấu hình được chốt.

### Tuần 4 — Chốt mô hình và đánh giá cuối

- **Quyền:** Rà soát pipeline, xác nhận cấu hình cuối và lưu mô hình có metadata.
- **Lộc:** Fit trên toàn bộ tập phát triển, thực hiện đợt đánh giá test đã thống nhất.
- **Séc:** Phân tích lỗi, hoàn thiện dự đoán CSV và hiển thị kết quả thật trong demo.
- **Cả nhóm:** Viết nhận xét dựa trên số liệu, chốt thông điệp chính của bài.

**Mốc M4:** Chỉ được công nhận sau khi hoàn tất kiểm chứng M3 và chốt cấu hình trước khi mở test. Có bộ kết quả cuối, phân tích hạn chế, demo dùng đúng mô hình và phiên bản dữ liệu đã ghi nhận; model, chỉ số và biểu đồ liên kết bằng cùng `run_id`, có gói bàn giao và checksum theo mục 11.4. Nếu kiểm chứng chưa đạt, lùi đánh giá test và bàn giao mô hình, điều chỉnh tiến độ với giảng viên; chỉ tiếp tục các phần độc lập, không công nhận M4 bằng cách báo riêng lỗi còn tồn đọng.

### Tuần 5 — Báo cáo, slide và chạy lại trên máy khác

- **Quyền:** Hoàn thiện chương phương pháp và kiểm tra tính nhất quán của công thức với mã.
- **Lộc:** Hoàn thiện chương dữ liệu, thực nghiệm và phụ lục cấu hình.
- **Séc:** Hoàn thiện chương kết quả, kết luận, tổng hợp báo cáo và slide.
- **Cả nhóm:** Một thành viên cài lại môi trường từ hướng dẫn và chạy dự án trên máy khác.

**Mốc M5:** Có bản báo cáo đầy đủ, slide đầu tiên, demo ổn định và bằng chứng tái lập kết quả.

### Tuần 6 — Phản biện thử và đóng gói

- Tổ chức ít nhất hai buổi bảo vệ thử, đổi người hỏi và trả lời.
- Kiểm tra đồng nhất số liệu giữa báo cáo, slide, notebook và demo.
- Chuẩn bị video hoặc ảnh dự phòng để minh họa nếu thiết bị trình chiếu gặp sự cố.
- Sửa lỗi trình bày, kiểm tra tài liệu tham khảo và đóng gói tệp nộp.
- Dành 2–3 ngày cuối làm khoảng dự phòng; không bổ sung tính năng lớn.

**Mốc M6:** Bộ nộp hoàn chỉnh; cả ba thành viên chạy được demo và trả lời được câu hỏi cốt lõi.

Nếu còn nhiều thời gian sau 6 tuần, ưu tiên đọc sâu, cải thiện giải thích và luyện bảo vệ. Chỉ mở rộng thực nghiệm khi có câu hỏi nghiên cứu cụ thể.

## 11. Cấu trúc mã nguồn và sản phẩm nộp

Cấu trúc bộ khung đã tạo; các file Python chỉ có mô tả/TODO, notebook chưa có code hoặc kết quả:

```text
water-potability-naive-bayes/
  README.md
  AGENTS.md
  .cursor/
    rules/
      explicit-implementation.mdc
  requirements.txt
  DOCS/
    KE_HOACH_NAIVE_BAYES_CHAT_LUONG_NUOC.md
  data/
    README.md
    raw/
    splits/
  notebooks/
    01_data_audit_eda.ipynb
    02_experiments.ipynb
  src/
    __init__.py
    data_validation.py
    preprocessing.py
    gaussian_nb_numpy.py
    train.py
    evaluate.py
    predict.py
  app/
    app.py
  tests/
    test_gaussian_nb.py
    test_prediction_flow.py
  artifacts/
  results/
    metrics/
    figures/
  report/
  slides/
```

`README.md` cần hướng dẫn cài môi trường, lấy dữ liệu, huấn luyện, đánh giá và chạy demo. Nếu điều kiện sử dụng dữ liệu không cho phép đóng gói lại, cung cấp hướng dẫn tải thay vì phân phối bản sao.

### 11.1. Luồng từ dữ liệu đến mô hình và demo

Đây là luồng dự kiến để triển khai sau giai đoạn tạo bộ khung; chưa có bước xử lý, huấn luyện hoặc dự đoán nào được thực thi.

**Dữ liệu đầu vào và đáp án:** `X` gồm 9 chỉ số chất lượng nước đã liệt kê ở mục 4.2; `y` là cột `Potability` (`0` — không uống được, `1` — uống được theo nhãn dataset). Khi huấn luyện, mô hình nhận cả `X` và `y`. Khi dự đoán mẫu mới, chỉ nhận `X`; không đưa `Potability` vào đặc trưng.

**Luồng huấn luyện và đánh giá:**

1. **Đọc và kiểm tra CSV — `src/data_validation.py`:** đọc `data/raw/water_potability.csv`; kiểm tra cột, kiểu dữ liệu, giá trị thiếu, nhãn không hợp lệ và bản ghi trùng. Xử lý vấn đề cấu trúc theo quy tắc đã chốt trước khi chia tập.
2. **Chia dữ liệu — `src/preprocessing.py`:** chia phân tầng 80% tập phát triển và 20% tập kiểm tra cuối, dùng seed cố định; lưu chỉ số mẫu vào `data/splits/`. Không để bản ghi trùng lọt sang cả hai tập.
3. **Khám phá và chuẩn bị tiền xử lý — `notebooks/01_data_audit_eda.ipynb`, `src/preprocessing.py`:** phân tích tập phát triển, xây dựng pipeline điền thiếu bằng trung bình/trung vị và khảo sát chuẩn hóa. Các tham số tiền xử lý chỉ được học trên phần huấn luyện tương ứng trong mỗi fold, không học từ tập validation hoặc test.
4. **Huấn luyện, chọn cấu hình và lưu pipeline — `src/train.py`, `src/gaussian_nb_numpy.py`, `notebooks/02_experiments.ipynb`:** tự viết vòng lặp 5-fold phân tầng trên tập phát triển; so sánh baseline tự viết và các cấu hình Gaussian NB đã qua kiểm chứng tính tay/tính chất. Tự tính tỷ lệ từng lớp, trung bình và phương sai của từng thuộc tính theo lớp. Chọn cấu hình theo Balanced Accuracy tự tính như mục 6.3, rồi fit lại toàn bộ chuỗi xử lý tự cài trên 80% tập phát triển. Lưu cả trạng thái tiền xử lý và mô hình trong `artifacts/<run_id>/`, kèm thứ tự đặc trưng, thứ tự lớp, cấu hình và phiên bản môi trường.
5. **Đánh giá cuối — `src/evaluate.py`:** nạp đúng pipeline đã lưu để dự đoán trên 20% tập test giữ riêng; so sánh với baseline, xuất chỉ số và biểu đồ dưới cùng `run_id` theo mục 11.4. Không fit lại hoặc chỉnh mô hình theo điểm test.
6. **Chốt và bàn giao:** hoàn thiện manifest, checksum và gói mô hình theo mục 11.4. Demo dùng đúng pipeline đã được đánh giá, không tạo lại một pipeline khác khi bàn giao.

**Luồng dự đoán trên demo:**

1. Người dùng nhập 9 chỉ số hoặc tải CSV qua `app/app.py`.
2. `src/predict.py` kiểm tra đầu vào và thứ tự cột, tải pipeline đã lưu trong `artifacts/`.
3. Pipeline áp dụng các giá trị tiền xử lý đã học rồi dự đoán nhãn; không fit lại bộ tiền xử lý hoặc mô hình trên mẫu mới.
4. App hiển thị nhãn theo dataset và xác suất mô hình ước lượng nếu có, tuân theo giới hạn ở mục 8.3.

**Không huấn luyện lại mỗi lần bấm dự đoán.** Nếu chưa có pipeline đã huấn luyện, app phải thông báo cần hoàn thành bước huấn luyện trước; không tạo kết quả giả. Demo phục vụ học thuật, không xác nhận nước thực tế an toàn để uống.

### 11.2. Nhật ký và sản phẩm nộp

Nhật ký thí nghiệm tối thiểu gồm: mã thí nghiệm, phiên bản dữ liệu, seed, cách chia, cấu hình tiền xử lý, tham số mô hình, chỉ số từng fold, điểm tổng hợp và ngày chạy. Lưu toàn bộ pipeline để demo không tự tạo lại bộ tiền xử lý khác với lúc huấn luyện.

**Bộ sản phẩm nộp đề xuất:**

1. Báo cáo hoàn chỉnh dưới dạng DOCX và PDF.
2. Slide thuyết trình.
3. Mã nguồn, notebook và hướng dẫn chạy.
4. Mô hình/pipeline đã huấn luyện và cấu hình đi kèm.
5. Kết quả thực nghiệm, biểu đồ và dự đoán test để kiểm tra lại.
6. Nhật ký phân công và đóng góp.
7. Video demo ngắn nếu giảng viên yêu cầu hoặc nhóm cần bản dự phòng.

### 11.3. Đặc tả chung trước khi lập trình song song

Quyền soạn, Lộc và Séc kiểm tra, chốt ở M1 hoặc chậm nhất đầu tuần 2. Các ví dụ đầu vào/đầu ra dùng để thống nhất cấu trúc phải ghi rõ là minh họa, không phải dự đoán thật.

- **Kiểm tra dữ liệu:** đầu vào là bảng dữ liệu và chế độ huấn luyện/dự đoán; đầu ra gồm dữ liệu hợp lệ, ID dòng, danh sách lỗi/cảnh báo và thống kê số dòng nhận/loại. Huấn luyện bắt buộc có nhãn hợp lệ; dự đoán không yêu cầu nhãn.
- **Đặc trưng:** cố định thứ tự `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`. Đơn vị và ô thiếu hợp lệ theo từ điển dữ liệu đã kiểm tra, không tự suy đoán đơn vị mới.
- **Thuật toán:** lõi GNB nhận ma trận số đã tiền xử lý dạng `(n_samples, n_features)` với cả hai kích thước dương để dùng được ví dụ tính tay hai thuộc tính; pipeline nghiệp vụ bắt buộc đúng 9 đặc trưng theo thứ tự chuẩn. `fit(X, y)` yêu cầu `len(y) == n_samples`, nhãn chỉ thuộc `{0, 1}` và có đủ hai lớp; lưu `n_features` để kiểm tra lúc dự đoán. `predict(X)` trả nhãn `(n_samples,)`; `predict_proba(X)` trả xác suất `(n_samples, 2)` theo thứ tự `[0, 1]`. Dữ liệu rỗng, hàng không cùng số cột, số cột khác lúc fit, giá trị không hữu hạn, nhãn sai/thiếu lớp hoặc gọi trước `fit` phải có lỗi rõ ràng. API lõi báo lỗi; tầng dự đoán nghiệp vụ chuyển lỗi số học thành kết quả từng dòng theo mục 8.4.
- **Huấn luyện và đánh giá:** thống nhất cách truyền cấu hình, manifest phép chia và nơi lưu kết quả. `train.py` chịu trách nhiệm fit/lưu pipeline; `evaluate.py` chỉ nạp hoặc nhận pipeline đã fit để đánh giá, không tự fit trên dữ liệu đánh giá.
- **Dự đoán:** `predict.py` nhận bảng đầu vào và pipeline đã lưu, trả kết quả cùng lỗi/cảnh báo theo từng dòng như mục 8.4. App chỉ gọi luồng chung và hiển thị, không tự xây imputer hoặc huấn luyện mô hình.
- **Nguồn logic chung:** thuật toán, kiểm tra, tiền xử lý và đánh giá nằm trong `src`; notebook và app gọi các hàm này. Notebook bổ sung giải thích/biểu đồ, không sao chép một phiên bản logic huấn luyện khác.
- **Bàn giao:** ghi tên hàm thực tế, tham số, kiểu trả về, lỗi dự kiến, định dạng pipeline và một ví dụ gọi trong tài liệu dự án khi triển khai. Mọi thay đổi giao diện sau M1 phải được báo cho cả ba thành viên và cập nhật ví dụ trước khi tích hợp.

### 11.4. Liên kết lần chạy và bàn giao mô hình

Mã thí nghiệm E0–E5 mô tả loại thí nghiệm; `run_id` định danh một lần chạy cụ thể, ví dụ theo dạng thời gian và mã thí nghiệm. Không ghi đè kết quả cũ bằng lần chạy mới cùng ID.

- Dự kiến lưu pipeline và metadata tại `artifacts/<run_id>/`, chỉ số tại `results/metrics/<run_id>/`, hình tại `results/figures/<run_id>/`, manifest lần chạy tại `results/runs/<run_id>.json`. Đây là các đường dẫn sẽ tạo khi có lần chạy thật; bộ khung hiện chưa chứa mô hình hoặc kết quả.
- Manifest lần chạy được lưu Git, ghi `run_id`, mã E0–E5, thời gian, commit mã nguồn, checksum dữ liệu và manifest phép chia, cấu hình, phiên bản Python/thư viện, đường dẫn và checksum của model/metrics/figures. Lần chạy chốt để báo cáo dùng mã nguồn đã commit; thay đổi chưa commit phải được giải quyết trước khi chốt.
- Với lần chạy có fit cuối, `train.py` lưu pipeline một lần; `evaluate.py` đánh giá đúng pipeline đó và bổ sung kết quả dưới cùng `run_id`, không huấn luyện lại. Model dùng cho demo phải khớp checksum model đã được đánh giá.
- Quyền đóng gói thư mục pipeline cùng manifest, cấu hình, phiên bản môi trường và hướng dẫn nạp trong gói nộp/gói bàn giao riêng. Gói này được chuyển trực tiếp cho máy trình chiếu; chưa yêu cầu đăng mô hình lên GitHub. Séc kiểm tra checksum, nạp đúng môi trường và thử lại các mẫu đối chiếu trước buổi demo.
- Chốt `run_id` cuối dùng cho báo cáo, slide và app tại M4. App hiển thị mã phiên bản mô hình trong phần thông tin; báo cáo/slide ghi ID để truy vết. Chạy lại do sửa lỗi tạo ID mới và ghi lý do, không trộn model của lần cũ với chỉ số của lần mới.

**Hợp đồng lưu/nạp tối thiểu:** Quyền chốt tại M1 định dạng JSON UTF-8, `schema_version=1`, không chứa `NaN`/vô hạn. Trạng thái bắt buộc gồm `run_id`, thứ tự bước/cột/lớp, `n_features`, cấu hình và checksum dữ liệu/phép chia; giá trị imputer; cờ bật scaler cùng mean/scale; min/max trước tiền xử lý; prior, trung bình lớp, phương sai MLE, epsilon, `variance_floor` và phương sai hiệu dụng. Các mảng ghi rõ kích thước và ý nghĩa. Bản triển khai lưu mean/scale sau điền thiếu kể cả khi `standardize=false` để kiểm tra liên kết thống kê, nhưng không áp dụng phép chuẩn hóa khi tắt; không có trường scaler riêng `null`. Không cần hệ thống định dạng mở rộng hoặc pickle cho phạm vi này.

Loader kiểm tra checksum theo manifest, phiên bản schema được hỗ trợ, trường bắt buộc, thứ tự cột/lớp, kích thước mảng, số hữu hạn, prior dương có tổng xấp xỉ 1, scale/phương sai hiệu dụng dương và quan hệ `variance_effective = variance_mle + epsilon` trong sai số đã chốt. Từ chối trạng thái không hợp lệ bằng lỗi rõ ràng; chỉ khôi phục trạng thái đã fit, không fit lại hoặc cộng epsilon lần hai. Kiểm tra round-trip cho cả cấu hình có/không chuẩn hóa và từ chối schema lạ, trạng thái thiếu, sai kích thước hoặc sai checksum.

Không cần dịch vụ quản lý thí nghiệm riêng cho phạm vi đồ án; các file manifest, thư mục và quy ước trên là đủ để kiểm tra bàn giao.

## 12. Đề cương báo cáo và thuyết trình

### 12.1. Báo cáo đề xuất 25–35 trang nội dung

Điều chỉnh theo mẫu giảng viên; không kéo dài bằng ảnh chụp mã nguồn hoặc lý thuyết không dùng đến.

- **Chương 1 — Giới thiệu:** Bối cảnh, bài toán, mục tiêu, phạm vi và câu hỏi nghiên cứu.
- **Chương 2 — Cơ sở lý thuyết:** Bayes, Gaussian NB, giả định, công thức, ví dụ minh họa và các chỉ số đánh giá.
- **Chương 3 — Dữ liệu và tiền xử lý:** Nguồn, từ điển dữ liệu, kiểm tra chất lượng, EDA, quy tắc làm sạch và chia tập.
- **Chương 4 — Xây dựng mô hình và thực nghiệm:** Giải thích mã tự cài theo từng công thức, kiểm chứng tính tay/tính chất, chuỗi xử lý, các cấu hình và quy tắc lựa chọn.
- **Chương 5 — Kết quả và thảo luận:** So sánh baseline, kết quả CV/test, confusion matrix, phân tích lỗi và giới hạn.
- **Chương 6 — Demo và kết luận:** Cách sử dụng, kết quả đạt được, hạn chế và hướng phát triển.
- **Phụ lục:** Phân công, môi trường, cấu hình, hướng dẫn chạy và ví dụ tính tay.

Tài liệu tham khảo phải có nguồn dữ liệu, tài liệu thuật toán và tài liệu thư viện. Đọc và kiểm tra phương pháp của bài nghiên cứu trước khi áp dụng; không mặc định một quy trình đúng chỉ vì đã xuất hiện trong một bài báo.

### 12.2. Slide đề xuất 15–18 trang, thuyết trình 12–15 phút

- Quyền: khoảng 4 phút cho bài toán và thuật toán.
- Lộc: khoảng 4 phút cho dữ liệu và thiết kế thực nghiệm.
- Séc: khoảng 4–5 phút cho kết quả và demo.
- Phần chuyển tiếp và kết luận chung: khoảng 1–2 phút.

Điều chỉnh khi biết thời lượng chính thức. Slide cần dành đủ chỗ cho kết quả và phân tích; hạn chế chép nguyên công thức dài hoặc đoạn mã lớn.

### 12.3. Câu hỏi cần luyện trả lời

1. Vì sao dùng Gaussian NB thay vì Multinomial NB?
2. “Độc lập có điều kiện” nghĩa là gì, và dữ liệu có chắc thỏa mãn không?
3. Vì sao điền thiếu sau khi chia tập và bên trong từng fold?
4. Vì sao không điền thiếu dựa vào nhãn uống được/không uống được?
5. Cross-validation khác tập test cuối như thế nào?
6. Vì sao accuracy khá cao vẫn có thể là mô hình kém?
7. Trong bài này FP/FN tương ứng với tình huống nào?
8. Không dùng hàm thuật toán có sẵn thì kiểm chứng prior, trung bình, phương sai, xác suất và metric bằng cách nào?
9. Vì sao dùng log và phải ổn định phương sai?
10. Nếu có mẫu đo ở Việt Nam, có thể dùng mô hình này ngay không?
11. Tại sao xác suất dự đoán cao chưa đồng nghĩa nước thực tế an toàn?
12. Nếu Naive Bayes không vượt baseline thì nhóm kết luận thế nào?

## 13. Kiểm chứng và tiêu chí nghiệm thu

### 13.1. Kiểm tra thuật toán

- Ví dụ nhỏ có đáp án tính tay cho kết quả phù hợp.
- Trung bình, phương sai, prior, điểm log và xác suất khớp đáp án tính tay độc lập trong sai số cho phép đã chốt.
- Trường hợp thuộc tính có phương sai bằng 0 không làm xuất hiện NaN hoặc crash.
- Có đáp án tính tay cho epsilon và phương sai hiệu dụng ở dữ liệu thường, một cột hằng và toàn bộ cột hằng; kiểm tra smoothing/sàn không hợp lệ và không cộng epsilon hai lần sau nạp.
- Lõi thuật toán được kiểm tra trực tiếp trên ví dụ hai thuộc tính; từ chối `X/y` lệch số mẫu, hàng lệch số cột, nhãn sai, thiếu lớp, số cột predict khác fit và gọi trước fit.
- Số hữu hạn cực lớn hoặc phương sai rất nhỏ không tạo nhãn/xác suất giả: lỗi fit dừng huấn luyện; lỗi predict được quy về đúng dòng, các dòng hợp lệ khác vẫn giữ kết quả.
- Tổng xác suất hai lớp xấp xỉ 1 trong sai số số học cho phép.
- Đổi thứ tự dòng dữ liệu không làm thay đổi ánh xạ lớp hoặc kết quả ngoài sai số dự kiến.
- Không có import/lời gọi thư viện làm thay phần học máy hoặc thống kê cốt lõi trong source, notebook, tests và demo. Người kiểm tra đọc được trình tự tính toán và giải thích được công thức từ code.
- Chia tập/fold tự viết thỏa các điều kiện không giao nhau, giữ phân tầng và mỗi mẫu được validation một lần; metric tự viết khớp ví dụ tính tay, kể cả mẫu số 0 và ROC có điểm bằng nhau.

### 13.2. Kiểm tra quy trình

- Không có cùng đầu vào lọt sang cả phát triển/test hoặc train/validation trong mỗi fold; kiểm tra bằng nội dung/nhóm mẫu, không chỉ ID, theo quy tắc M2.
- Không có nhãn `Potability` trong đầu vào của mô hình.
- Imputer và scaler không fit trên validation/test.
- Lưu rồi nạp lại pipeline cho cùng kết quả trên cùng đầu vào.
- Notebook có thể chạy từ đầu đến cuối, không phụ thuộc biến được tạo thủ công ở phiên cũ.
- Cùng dữ liệu, cấu hình, seed và môi trường cho kết quả tái lập trong sai số dự kiến.
- CSV gốc và danh sách ID train/test/fold khớp checksum manifest trên máy khác; phát hiện sai khác phải dừng thực nghiệm.
- Baseline chỉ đoán một lớp vẫn xuất chỉ số theo đúng quy ước `N/A`, giá trị thay thế và macro-F1 ở mục 7.1.
- Cùng bảng điểm CV chưa làm tròn cho cùng cấu hình được chọn theo thứ tự ưu tiên ở mục 6.3.
- Bộ đáp án độc lập cho mean/median chẵn/lẻ có ô thiếu, cột thiếu toàn bộ, cột hằng và z-score dùng tham số train; xác nhận mẫu số phương sai scaler là `n`. Không lấy đầu ra của hàm đang kiểm tra làm đáp án.
- Bảng 5 điểm fold có đáp án tính tay cho trung bình/độ lệch chuẩn (`ddof=0`), thử cả biên chênh lệch 0,005 và các mức phân xử cấu hình; fold lỗi không bị bỏ qua để tính điểm đẹp hơn. E1–E3 giữ cùng smoothing; baseline điểm hằng cho AUC 0,5 khi đủ hai lớp.
- Thống kê EDA có ví dụ nhỏ kiểm tra số đếm bin, phân vị, tương quan với ô thiếu và mẫu số 0; biểu đồ chỉ nhận số liệu đã tự tính.

### 13.3. Kiểm tra demo và tài liệu

- Nhập mẫu hợp lệ, mẫu thiếu một phần, CSV sai cột và dữ liệu không phải số đều được xử lý rõ ràng.
- Kết quả demo khớp với gọi pipeline trực tiếp.
- CSV đảo thứ tự cột được xử lý đúng; thiếu/trùng/thừa cột bị báo theo mục 8.4. CSV có dòng lỗi giữ nguyên ID/thứ tự dòng, không gán nhãn giả cho dòng bị từ chối.
- Fixtures CSV phân biệt header trùng nguyên trạng, sai hoa/thường/khoảng trắng tên cột, ô rỗng, `NaN`, `NULL`, `N/A`, `nan`, `inf` và số hữu hạn quá lớn; `sample_id` giữ được số 0 đầu. Dòng lỗi parser/số học không âm thầm trở thành mẫu đã điền thiếu.
- Lưu/nạp qua JSON đạt các kiểm tra schema/trạng thái tại mục 11.4 cho cả cấu hình có và không chuẩn hóa; checksum đúng không thay thế kiểm tra nội dung model.
- Model, chỉ số và biểu đồ của bản bàn giao khớp `run_id` và checksum; máy trình chiếu dùng đúng gói đã kiểm tra.
- Mọi hình có tiêu đề, nhãn, chú giải cần thiết và nội dung giải thích.
- Báo cáo, slide và ứng dụng sử dụng cùng phiên bản kết quả.
- Không có số liệu minh họa bị trình bày như kết quả thật.
- Cả ba thành viên đều có thể giải thích luồng dữ liệu và ý nghĩa các chỉ số chính.

Không coi “vượt baseline” là điều kiện để được báo cáo kết quả. Nếu mô hình kém, yêu cầu nghiệm thu là chứng minh quy trình đúng và giải thích trung thực các giới hạn.

## 14. Rủi ro và phương án xử lý

- **Nguồn dữ liệu hoặc cách tạo nhãn chưa rõ:** Ghi rõ trong giới hạn nghiên cứu; trao đổi với giảng viên nếu cần nguồn khác trước khi triển khai sâu.
- **Kết quả cao bất thường:** Kiểm tra leakage, nhãn lẫn vào đặc trưng, chia trùng mẫu và điền thiếu theo nhãn trước khi kết luận.
- **Kết quả thấp:** Đối chiếu baseline, kiểm tra mã và giả định; không cố thay seed để lấy kết quả đẹp.
- **Bản tự cài không khớp đáp án:** Kiểm tra từng phép tính, phương sai MLE, prior, lượng ổn định và thứ tự lớp trên ví dụ nhỏ; sửa bản tự cài, không thay bằng thuật toán thư viện.
- **Khác phiên bản thư viện giữa ba máy:** Chốt môi trường, lưu phiên bản và kiểm tra lại trên máy khác từ tuần 5.
- **Phân công mất cân bằng:** Kiểm tra giờ thực tế mỗi tuần; chuyển các phần có thể tách như biểu đồ, kiểm thử và tài liệu.
- **Demo lỗi khi bảo vệ:** Chạy cục bộ, thử trên máy thuyết trình, chuẩn bị dữ liệu mẫu và video dự phòng.
- **Phạm vi phình to:** Hoàn thành M1–M4 trước khi xét chức năng bổ sung.

## 15. Các quyết định cần nhóm duyệt

- [ ] Thống nhất tên đề tài và phạm vi dự đoán nhãn `Potability`.
- [ ] Chọn Python và bộ công cụ được đề xuất.
- [ ] Chọn Water Potability làm nguồn dữ liệu, với điều kiện kiểm tra nguồn và tệp trong tuần 1.
- [x] Yêu cầu người dùng đã chốt: tự cài tường minh toàn bộ phần học máy theo mục 0; không dùng mô hình/hàm thư viện làm thay, kể cả trong tests/demo.
- [ ] Dùng phép chia 80/20 và 5-fold CV trên tập phát triển, có rà soát cấu trúc nhóm/thời gian nếu metadata cho phép.
- [ ] Chọn Balanced Accuracy làm chỉ tiêu lựa chọn chính; phân tích riêng lỗi FP.
- [ ] Thống nhất phân công Quyền — thuật toán; Lộc — dữ liệu/thực nghiệm; Séc — đánh giá/demo.
- [ ] Thống nhất lộ trình 6 tuần và khoảng 6–8 giờ/người/tuần.
- [ ] Làm demo Streamlit chạy cục bộ với chức năng dự đoán mẫu và CSV.
- [ ] Xác nhận yêu cầu chính thức của giảng viên trước khi chốt mẫu báo cáo và slide.
- [ ] Duyệt manifest dữ liệu/phép chia, đặc tả module, quy tắc metric và đầu vào demo trước các mốc tương ứng M1–M3.
- [ ] Duyệt thứ tự chọn cấu hình gần điểm nhau tại mục 6.3 trước khi chạy thực nghiệm.
- [ ] Thống nhất quy ước `run_id`, gói bàn giao và cách xử lý khi bản tự cài chưa đạt M3.

**Việc đầu tiên sau khi duyệt:** Ghi lại yêu cầu của giảng viên, kiểm tra dữ liệu thực tế và chốt đầu ra tuần 1. Những con số về hiệu năng chỉ được bổ sung sau khi chạy thực nghiệm đúng quy trình.

## 16. Tài liệu tham khảo khởi đầu

Các tài liệu dưới đây dùng để đọc lý thuyết, công thức và quy ước. Việc trích dẫn tài liệu thư viện không cho phép import/gọi các hàm bị cấm ở mục 0; dự án không có bản tham chiếu scikit-learn thực thi.

1. [Kaggle — Water Quality / Water Potability, Aditya Kadiwal](https://www.kaggle.com/datasets/adityakadiwal/water-potability).
2. [JSDEWES — Machine Learning-Based Water Quality Prediction](https://www.sdewes.org/jsdewes/pid13.0634), dùng để đối chiếu mô tả bộ dữ liệu; không thay thế kiểm tra tệp thực tế.
3. [scikit-learn — Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html).
4. [scikit-learn — GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html).
5. [scikit-learn — Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html).
6. [scikit-learn — StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html).
7. [scikit-learn — Balanced Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html).
8. [NumPy](https://numpy.org/).
9. [pandas](https://pandas.pydata.org/).
10. [Streamlit Documentation](https://docs.streamlit.io/).
11. [scikit-learn — Precision, recall, F-score và support](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html), tham chiếu quy ước `zero_division` và trung bình chỉ số.

Ngày tra cứu nguồn: 24/10/2026. Khi triển khai, ghi lại phiên bản tài liệu và thư viện thực tế được dùng.
