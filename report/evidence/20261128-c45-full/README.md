# Minh chứng C4.5 đầy đủ ngày 28/11/2026

- `weka-c45-raw.png`: mở development_raw.arff của lần chạy mới, vẫn có ô thiếu.
- `weka-c45-settings.png`: J48 -C 0.25 -M 2, bật collapse, subtree raising, MDL, không Laplace.
- `weka-c45-result.png`: chạy Start thật với Supplied test set; 406/656 mẫu đúng,
  accuracy 61,8902%, TN=364, FP=36, FN=214, TP=42; cây 21 nút.
- `web-overview.png`: giao diện mới có C4.5, nhãn Potability và năm tab.
- `web-form.png`: đủ chín ô nhập và các bộ số thử chưa có nhãn kiểm nghiệm.

Ảnh Weka chụp cửa sổ trực tiếp, ảnh web chụp viewport bằng công cụ sẵn có
của Chrome, không tái dựng màn hình. Không đưa thanh tab hoặc lịch sử
trình duyệt vào ảnh bàn giao. `manifest.json` lưu checksum ảnh và mã nguồn.
Các panel Evaluation còn lại do Weka dựng từ lần chạy mới được lưu cùng
log và dự đoán trong results/comparison/20261128-c45-full-v1/weka.
