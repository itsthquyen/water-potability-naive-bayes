# QUY TẮC BẮT BUỘC CHO MỌI AGENT

> **ƯU TIÊN CAO: CODE TƯỜNG MINH — TỰ CÀI ĐẶT PHẦN HỌC MÁY.**
> Đây là yêu cầu trực tiếp của người dùng. Không thay bằng hàm/thuật toán có sẵn để tiết kiệm thời gian, kể cả trong notebook, tests hay demo.

## Phạm vi

- Áp dụng toàn repository, mọi agent/subagent/worker và mọi thay đổi sau này.
- Đọc mục 0 của [kế hoạch](DOCS/KE_HOACH_NAIVE_BAYES_CHAT_LUONG_NUOC.md) trước khi làm; khi giao việc, đưa yêu cầu này vào brief.
- Hiện chỉ chuẩn bị bộ khung/tài liệu. Chỉ triển khai chức năng chính khi người dùng yêu cầu.
- Dự án là đồ án Python phân loại nhãn `Potability` bằng Gaussian Naive Bayes, demo Streamlit chạy cục bộ. Giữ phạm vi này; không tự thêm backend web, database, đăng nhập hoặc framework frontend.
- Ưu tiên yêu cầu hiện hành của người dùng so với nội dung cũ trong tài liệu dự án. Các quy tắc tham khảo từ dự án khác chỉ được áp dụng sau khi điều chỉnh cho phạm vi này, không tự đưa công nghệ hoặc cấu hình của dự án khác vào đây.

## RTK — bắt buộc cho lệnh shell

- Mọi lệnh chạy qua Bash/PowerShell phải đi qua `rtk`. Ví dụ: `rtk git status --short`, `rtk rg --files`, `rtk rg -n "pattern" src`. Ưu tiên `rg` để tìm file/nội dung.
- Lệnh chưa có wrapper riêng dùng `rtk proxy <command>`, không giả định mọi tên lệnh đều được RTK nhận trực tiếp. Với PowerShell có thể dùng `rtk proxy powershell -NoProfile -Command '...'`; đọc/ghi văn bản bằng UTF-8 và quote an toàn.
- Khi có nhiều lệnh shell độc lập, mỗi lệnh đều qua RTK. Có thể bọc cả script bằng `rtk proxy`; không bọc lặp RTK cho từng câu lệnh nội bộ của script đó.
- Công cụ đọc/sửa file tích hợp không phải shell thì không cần RTK. Nếu RTK hoặc runtime chưa có, báo đúng lỗi công cụ; không tự cài lại toàn bộ môi trường hoặc tuyên bố đã chạy thành công.
- Khi đã có môi trường và kiểm thử thực thi, dùng Python trong `.venv`, ví dụ `rtk proxy .\.venv\Scripts\python.exe -m pytest -q`. Không coi file test chỉ chứa TODO là kiểm thử đã đạt.

## Quy trình làm việc và ranh giới thay đổi

- Trước khi sửa, đọc file liên quan, phần kế hoạch tương ứng và kiểm tra trạng thái Git. Giữ nguyên thay đổi đang có của người dùng; không reset, xóa hoặc ghi đè công việc ngoài phạm vi.
- Tìm logic/hàm tương tự trước khi thêm mới; thay đổi vừa đủ giải quyết yêu cầu. Không gộp đổi kiến trúc, nâng dependencies hay sửa hàng loạt file không liên quan vào một việc nhỏ.
- Khi người dùng chỉ yêu cầu tài liệu/bộ khung, chỉ làm đúng phần đó. Khi đã được yêu cầu triển khai, tiếp tục các bước cần thiết trong phạm vi, không hỏi lại cho từng thao tác thông thường.
- Cập nhật tài liệu cùng thay đổi hành vi: cấu trúc/cách chạy ở `README.md`, thiết kế/quy ước ở `DOCS`, nguồn dữ liệu ở `data/README.md`. Không đánh dấu hoàn thành khi mới có TODO hoặc chưa xác minh kết quả.
- Không tự commit, push, triển khai hoặc thay đổi repository khác nếu hành động đó chưa thuộc phạm vi người dùng đã cho phép. Đọc HR-V5 để tham khảo không có nghĩa được sửa HR-V5.

## Phải tự viết rõ từng bước

- Gaussian Naive Bayes: đếm lớp, prior, trung bình, phương sai MLE, ổn định phương sai, log-likelihood, chuẩn hóa xác suất và chọn nhãn.
- Tiền xử lý: kiểm tra dữ liệu, nhận diện trùng, tính trung bình/trung vị, điền thiếu và chuẩn hóa.
- Thực nghiệm: chia phân tầng train/test, tạo fold, baseline, vòng lặp cross-validation, duyệt cấu hình và chọn mô hình.
- Đánh giá: đếm TP/FP/TN/FN, confusion matrix, accuracy, precision, recall, F1, Balanced Accuracy, macro-F1 và ROC-AUC.
- Mỗi hàm có đầu vào/đầu ra rõ ràng, tên biến dễ hiểu, chú thích công thức và các bước; được tách thành hàm tự viết để dùng lại. Không viết wrapper gọi thư viện để che phần phải tự cài.

## Không được dùng để thay phần tự cài

- Không import/gọi `sklearn` hoặc thư viện ML tương đương trong mã dự án, notebook, tests hay demo.
- Không dùng `GaussianNB`, `DummyClassifier`, `Pipeline`, `SimpleImputer`, `StandardScaler`, `train_test_split`, `StratifiedKFold`, `cross_val_score`, `GridSearchCV` hoặc các hàm metric có sẵn.
- Không dùng `np.mean/var/std/median`, `statistics.mean/variance/median`, pandas aggregation hoặc hàm tương đương để làm thay phép tính thống kê cốt lõi. Không dùng hàm phân phối Gaussian, softmax hoặc logsumexp có sẵn.
- Không lấy mô hình huấn luyện sẵn; không chuyển sang mô hình thư viện nếu bản tự cài lỗi. Không dùng thư viện ML làm oracle trong tests khi chưa có chỉ đạo mới của người dùng.

## Được dùng làm công cụ nền

- Python: vòng lặp, điều kiện, list/dict, `len`, `sum`, `min`, `max`, `sorted`, phép toán số học; `math.log/exp/sqrt/isfinite`.
- NumPy chỉ để lưu/truy cập mảng và phép toán phần tử cơ bản. pandas chỉ hỗ trợ đọc/ghi CSV, chọn cột và trình bày bảng; logic xử lý cốt lõi vẫn tự viết.
- Bộ sinh số ngẫu nhiên/shuffle có seed cho thao tác xáo trộn; cách phân lớp/chia tập/phân fold phải tự viết.
- Thư viện đọc/ghi file, JSON, checksum, lưu/nạp trạng thái, biểu đồ và Streamlit; pytest/assert cho kiểm thử. Biểu đồ nhận số liệu do code tự tính.

## Đặt tên, chú thích và tổ chức hàm

- Dùng `snake_case` cho hàm/biến, `PascalCase` cho lớp, `UPPER_SNAKE_CASE` cho hằng số. Tên nói rõ mục đích, ví dụ `calculate_class_priors`, `calculate_gaussian_log_likelihood`, `validate_feature_columns`; tránh `process`, `get_data`, `tmp`, `list1` nếu không thể hiểu vai trò từ ngữ cảnh.
- Giữ các tên giao diện đã thống nhất như `fit`, `predict`, `predict_proba`. Ký hiệu chuẩn `X`, `y`, `TP/FP/TN/FN` được dùng khi có giải thích ý nghĩa và kích thước; không rút gọn tùy tiện các tên khác.
- Docstring/chú thích tiếng Việt, UTF-8, giải thích đầu vào/đầu ra, shape, thứ tự cột/lớp, đơn vị nếu đã xác minh, công thức, trường hợp rỗng/thiếu và lỗi dự kiến. Với tham số/trường trạng thái mới, nêu vì sao cần và giá trị hợp lệ; tránh chú thích chỉ lặp lại tên biến.
- Mỗi hàm có trách nhiệm rõ ràng, tách các bước tính toán đủ nhỏ để đọc và kiểm chứng; không nén cả thuật toán thành một biểu thức khó giải thích. Không tạo tầng abstraction hoặc lớp chỉ để bọc một lời gọi không cần thiết.
- Một công thức/quy tắc có một cài đặt dùng chung trong `src`. Notebook và `app` gọi lại, không sao chép tiền xử lý, huấn luyện hoặc metric thành phiên bản riêng. Tests vẫn phải có đáp án kỳ vọng độc lập.
- Dự đoán nhập tay và CSV dùng cùng hàm kiểm tra/tính toán cho mẫu hợp lệ; xử lý hàng loạt giữ ID, thứ tự dòng và lỗi từng dòng theo kế hoạch, không viết một thuật toán dự đoán khác.

## Cấu hình, dữ liệu và hiệu năng

- Thứ tự đặc trưng, nhãn, seed, ngưỡng, smoothing và cấu hình thực nghiệm có một nơi định nghĩa dùng chung; không lặp các giá trị khác nhau trong notebook, `src` và app. Lưu cấu hình thực tế cùng manifest của lần chạy.
- Khi thêm cấu hình, cập nhật đủ phần đọc/kiểm tra giá trị, giá trị mặc định hoặc mẫu cấu hình, và tài liệu sử dụng. Giải thích mục đích, giá trị hợp lệ, hành vi khi bỏ trống và ảnh hưởng khi thay đổi; không tự thêm hệ thống cấu hình phức tạp.
- Đường dẫn trong mã/cấu hình chia sẻ tính từ gốc project hoặc vị trí module, không ghi cứng ổ đĩa/tên người dùng. Cấu hình riêng và secrets nằm ngoài Git; mẫu cấu hình chỉ chứa giá trị minh họa không nhạy cảm.
- Giữ nguyên CSV gốc; dùng checksum/manifest để xác nhận dữ liệu, phép chia và phiên bản kết quả. Không fit tiền xử lý trên validation/test, không đưa nhãn hoặc ID vào đặc trưng, không đổi seed theo điểm test.
- Tránh đọc lại CSV, nạp model hoặc ghi file trong vòng lặp từng mẫu khi có thể đọc/nạp một lần cho một lượt xử lý. Tính và lưu tham số lớp lúc fit, không tính lại trên mỗi dự đoán; không train lại khi bấm dự đoán.
- Mỗi fold phải có trạng thái tiền xử lý/mô hình mới. Không tái sử dụng tham số đã fit giữa các fold để tối ưu tốc độ; không đổi sang hàm học máy có sẵn hoặc làm code khó hiểu vì tối ưu.
- Model, metrics và figures phải khớp `run_id`/checksum. Dữ liệu minh họa và dữ liệu tổng hợp cho test phải ghi rõ nguồn/mục đích, không trình bày như kết quả thực nghiệm thật.

## Giao diện và thông báo

- Giao diện, nhãn và thông báo dùng tiếng Việt rõ ràng; tận dụng thành phần Streamlit có sẵn, thống nhất cách nhập và hiển thị giữa các màn.
- Hướng dẫn dài có thể đặt trong help/expander. Lỗi chặn dự đoán, cảnh báo đầu vào và giới hạn “demo học thuật, không thay thế kiểm nghiệm chất lượng nước” phải nhìn thấy rõ, không giấu trong phần trợ giúp.
- App chỉ hiển thị kết quả từ luồng tự cài đã kiểm chứng; khi thiếu model hoặc đầu vào lỗi, báo đúng trạng thái. Không hiển thị dự đoán giả để làm giao diện trông hoàn chỉnh.

## Phối hợp agent

- Chỉ dùng thêm agent khi người dùng hoặc hướng dẫn áp dụng yêu cầu/cho phép; việc tham khảo quy tắc này không tự mở một đợt làm việc song song.
- Mỗi brief nêu phạm vi file, đầu vào/đầu ra, yêu cầu tự cài, tiêu chí nghiệm thu và trạng thái chưa triển khai nếu còn áp dụng. Các agent không ghi đè cùng file ngoài phần đã phân công.
- Codex giữ phần thuật toán, dữ liệu, logic xử lý, thiết kế, review và tích hợp. Theo chỉ dẫn người dùng, Antigravity chỉ có thể nhận phần giao diện độc lập đủ lớn qua helper/worktree riêng; không giao thuật toán hay tiền xử lý, việc nhỏ làm trực tiếp.
- Agent chính phải đọc thay đổi và tự xác minh phần được giao trước khi tích hợp; báo cáo worker là đã xong chưa đủ để coi là đạt.

## Kiểm chứng và bàn giao

- Dùng ví dụ nhỏ có đáp án tính tay, bộ dữ liệu tổng hợp có tham số biết trước và kiểm tra tính chất/bất biến; không lấy kết quả của chính hàm đang kiểm tra làm đáp án kỳ vọng.
- Khi chưa đúng, ghi rõ chưa đạt và sửa bản tự cài; không tạo kết quả giả hoặc bỏ kiểm chứng.
- Review phải kiểm tra các import, lời gọi hàm và tính tường minh; chức năng chạy được nhưng vi phạm quy tắc này vẫn chưa đạt nghiệm thu.
- Kiểm thử tập trung vào hành vi có ý nghĩa: đáp án tính tay, dữ liệu lỗi/thiếu, phương sai 0, mẫu số 0, phân tầng/fold, lưu–nạp và dự đoán một mẫu/CSV. Không sửa kỳ vọng chỉ để hợp thức hóa lỗi của implementation.
- Chạy kiểm tra phù hợp với phần thay đổi khi đã có môi trường; phân biệt lỗi code, dữ liệu và công cụ. Với sửa tài liệu/bộ khung, kiểm tra liên kết, nội dung và tính nhất quán; không tạo test hình thức hoặc train mô hình chỉ để xác minh tài liệu.
- Trước khi báo xong, kiểm tra phạm vi file đã đổi, không trùng logic, không lọt dữ liệu/secrets, tài liệu đã đồng bộ và các kết quả có bằng chứng. Báo ngắn gọn đã sửa gì, đã kiểm tra gì, phần nào chưa chạy và lý do; không báo “test pass” khi chưa có test thực thi.

## Nguồn quy tắc

- [Kế hoạch dự án](DOCS/KE_HOACH_NAIVE_BAYES_CHAT_LUONG_NUOC.md), [README](README.md) và [rule tự cài](.cursor/rules/explicit-implementation.mdc) phải nhất quán với file này.
- Đã chọn lọc nguyên tắc RTK, tên rõ nghĩa, một nguồn logic, chú thích ngữ cảnh, quản lý cấu hình và tự kiểm từ `D:\HR-V5\AGENTS.md` và `D:\HR-V5\CLAUDE.md` ngày 24/10/2026. Nội dung áp dụng đã ghi đầy đủ ở đây; không cần truy cập HR-V5 để làm việc với project này.
