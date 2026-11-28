"""Web cục bộ dự đoán Potability bằng sáu thuật toán đã học."""

import csv
from hashlib import sha256
from html import escape
import io
import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import FEATURE_COLUMNS, RAW_DATA_PATH
from src.data_validation import validate_records, read_csv_records
from src.pipeline import sha256_file
from src.six_experiments import ALGORITHMS, NAMES, RUN_ID, load_models

FOLDER = ROOT / "results" / "comparison" / RUN_ID
FEATURE_LABELS = {
    "ph": "pH",
    "Hardness": "Độ cứng (Hardness)",
    "Solids": "Chất rắn hòa tan (Solids)",
    "Chloramines": "Chloramines",
    "Sulfate": "Sulfate",
    "Conductivity": "Độ dẫn điện (Conductivity)",
    "Organic_carbon": "Carbon hữu cơ (Organic_carbon)",
    "Trihalomethanes": "Trihalomethanes",
    "Turbidity": "Độ đục (Turbidity)",
}


@st.cache_resource
def resources(manifest_hash):
    manifest = json.loads((FOLDER / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(FOLDER / "summary.json") != manifest["files"]["summary.json"]:
        raise ValueError("Checksum số liệu thực nghiệm không khớp.")
    return load_models(FOLDER), json.loads((FOLDER / "summary.json").read_text(encoding="utf-8"))


@st.cache_data
def dataset(expected_hash, modified_ns):
    """Đọc CSV nguyên trạng; mtime tham gia khóa cache khi tệp thay đổi."""
    payload = RAW_DATA_PATH.read_bytes()
    if sha256(payload).hexdigest() != expected_hash:
        raise ValueError("CSV gốc không khớp dữ liệu của lần thực nghiệm.")
    records = read_csv_records(payload.decode("utf-8-sig"), from_text=True, training=True)
    missing = {name: 0 for name in FEATURE_COLUMNS}
    classes = {0: 0, 1: 0}
    for record in records:
        classes[int(record["Potability"])] += 1
        for name in FEATURE_COLUMNS:
            if not record[name].strip():
                missing[name] += 1
    return payload, records, {"missing": missing, "classes": classes}


def percent(value):
    return f"{value * 100:.2f}%".replace(".", ",")


def fill_example(values):
    """Nạp mẫu trung vị; kết quả cũ chỉ có hiệu lực với đúng bộ ô nhập trước đó."""
    for name, value in zip(FEATURE_COLUMNS, values):
        st.session_state["six_" + name] = f"{value:.4f}"
    st.session_state.pop("six_manual", None)


def clear_inputs():
    for name in FEATURE_COLUMNS:
        st.session_state["six_" + name] = ""
    st.session_state.pop("six_manual", None)


def example_csv(records):
    """Trích ba mẫu gốc để minh họa định dạng dự đoán, bỏ cột nhãn."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=("sample_id", *FEATURE_COLUMNS))
    writer.writeheader()
    for index in (0, 27, 100):
        writer.writerow({"sample_id": f"raw_{index}",
                         **{name: records[index][name] for name in FEATURE_COLUMNS}})
    return output.getvalue().encode("utf-8-sig")


def predict(records, models):
    """Một kết quả cho mỗi dòng và mỗi mô hình, giữ lỗi dữ liệu theo dòng."""
    validation = validate_records(records, training=False)
    valid = dict(zip(validation["row_ids"], validation["X"]))
    errors = {entry["input_row_id"]: entry["error"] for entry in validation["errors"]}
    output = []
    for row_id, record in enumerate(records):
        for algorithm in ALGORITHMS:
            result = {"row_id": row_id, "sample_id": record.get("sample_id", ""),
                      "algorithm": NAMES[algorithm], "label": None, "score_1": None,
                      "score_type": "biên quyết định" if algorithm == "svm" else "xác suất ước lượng",
                      "error": errors.get(row_id, "")}
            if row_id in valid:
                try:
                    labels, scores = models[algorithm].predict_with_scores([valid[row_id]])
                    result.update(label=labels[0], score_1=scores[0])
                except (ValueError, ArithmeticError) as error:
                    result["error"] = str(error)
            output.append(result)
    return output


def table(headers, records, primary_column=None):
    html = ["<div class='comparison-table'><table><thead><tr>",
            *(f"<th scope='col'>{escape(item)}</th>" for item in headers), "</tr></thead><tbody>"]
    for values in records:
        primary = primary_column is not None and values[primary_column] == NAMES["naive_bayes"]
        html.append(("<tr class='primary-model'>" if primary else "<tr>")
                    + "".join(f"<td>{escape(str(value))}</td>" for value in values) + "</tr>")
    st.html("".join(html) + "</tbody></table></div>")


def show_results(results, key):
    display = []
    for result in results[:120]:
        label = result["label"]
        label_text = "Chưa có kết quả" if label is None else f"{label} · " + ("Uống được" if label else "Không uống được")
        score = ("" if result["score_1"] is None else
                 f"{result['score_1']:.4f}" if result["score_type"] == "biên quyết định"
                 else percent(result["score_1"]))
        display.append((result["row_id"], result["algorithm"], label_text, score, result["score_type"], result["error"]))
    table(("ID dòng", "Thuật toán", "Phân loại", "Điểm đầu ra", "Loại điểm", "Ghi chú / lỗi"), display, primary_column=1)
    if len(results) > 120:
        st.caption(f"Đang xem 120/{len(results)} kết quả. CSV tải xuống chứa toàn bộ kết quả.")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=("row_id", "sample_id", "algorithm", "label", "score_1", "score_type", "error"))
    writer.writeheader()
    writer.writerows(results)
    st.download_button("Tải toàn bộ kết quả CSV", stream.getvalue().encode("utf-8-sig"), "du_doan_nuoc.csv", "text/csv", key=key)
    st.caption("Các điểm xác suất chưa được hiệu chỉnh. Biên quyết định của SVM có thể âm hoặc dương và không phải xác suất.")


def main():
    st.set_page_config(page_title="Chất lượng nước · Nhóm 11", page_icon="💧", layout="wide")
    st.html("<style>" + (Path(__file__).parent / "assets/comparison.css").read_text(encoding="utf-8") + "</style>")
    try:
        models, summary = resources(sha256_file(FOLDER / "manifest.json"))
        manifest = json.loads((FOLDER / "manifest.json").read_text(encoding="utf-8"))
        raw_bytes, raw_records, raw_info = dataset(manifest["raw_sha256"], RAW_DATA_PATH.stat().st_mtime_ns)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Chưa nạp được mô hình đã kiểm chứng. {error}")
        return
    raw_count = f"{len(raw_records):,}".replace(",", ".")
    st.html(f"""<header class="water-header">
      <div><p class="eyebrow">KHAI PHÁ DỮ LIỆU · NHÓM 11 · 65CNTT</p>
      <h1>Phân loại chất lượng nước</h1>
      <p class="subtitle">Naive Bayes và năm thuật toán so sánh</p></div>
      <div class="dataset-badges"><span><strong>{raw_count}</strong> mẫu dữ liệu</span>
      <span><strong>{len(FEATURE_COLUMNS)}</strong> chỉ tiêu</span>
      <span><strong>{len(ALGORITHMS)}</strong> thuật toán</span></div>
    </header>""")
    st.warning("Ứng dụng học thuật theo nhãn Water Quality. Kết quả không thay thế kiểm nghiệm chất lượng nước.")
    manual, upload, experiments, raw_tab = st.tabs(("Dự đoán một mẫu", "Dự đoán CSV", "Kết quả thực nghiệm", "Dữ liệu gốc"))
    with manual:
        medians = models["naive_bayes"].processor.to_state()["fill_values"]
        record = {}
        with st.container(border=True):
            st.subheader("Thông số mẫu nước")
            st.caption("Nhập số thập phân bằng dấu chấm; để trống chỉ tiêu chưa có. Mẫu minh họa ban đầu dùng trung vị tập huấn luyện.")
            fields = st.columns(3)
            for i, name in enumerate(FEATURE_COLUMNS):
                with fields[i % 3]:
                    record[name] = st.text_input(FEATURE_LABELS[name], value=f"{medians[i]:.4f}",
                                                 key="six_" + name, help=f"Tên cột trong CSV: {name}")
            actions = st.columns((1.4, 1, 1))
            submit = actions[0].button("Dự đoán và so sánh", type="primary", width="stretch")
            actions[1].button("Nạp mẫu minh họa", on_click=fill_example, args=(medians,), width="stretch")
            actions[2].button("Xóa ô nhập", on_click=clear_inputs, width="stretch")
        identity = tuple(record.values())
        if st.session_state.get("six_manual_identity") != identity:
            st.session_state.pop("six_manual", None)
            st.session_state["six_manual_identity"] = identity
        if submit:
            st.session_state["six_manual"] = predict([record], models)
        if "six_manual" in st.session_state:
            st.subheader("Kết quả phân loại")
            result = st.session_state["six_manual"]
            primary = result[0]
            if primary["label"] is not None:
                verdict, score = st.columns((2, 1))
                with verdict:
                    (st.success if primary["label"] else st.warning)(
                        "Naive Bayes dự đoán " + ("uống được (1)" if primary["label"] else "không uống được (0)"))
                score.metric("Điểm ước lượng lớp uống được", percent(primary["score_1"]))
            else:
                st.error(primary["error"])
            show_results(result, "manual_download")
    with upload:
        st.subheader("Dự đoán nhiều mẫu từ CSV")
        st.write("Tải tệp UTF-8 có đủ chín cột chỉ tiêu. Mỗi dòng được dự đoán bằng cả sáu mô hình.")
        with st.expander("Định dạng tệp và dữ liệu mẫu"):
            st.code(",".join(FEATURE_COLUMNS), language="text")
            st.caption("Có thể kèm sample_id và Potability. Potability không được sử dụng làm đầu vào dự đoán.")
            st.download_button("Tải CSV mẫu", example_csv(raw_records), "mau_du_doan.csv", "text/csv")
        file = st.file_uploader("Chọn tệp CSV cần dự đoán", type="csv")
        identity = sha256(file.getvalue()).hexdigest() if file is not None else None
        if st.session_state.get("six_csv_identity") != identity:
            st.session_state.pop("six_csv", None)
            st.session_state["six_csv_identity"] = identity
        if st.button("Dự đoán CSV", disabled=file is None, type="primary"):
            st.session_state.pop("six_csv", None)
            try:
                records = read_csv_records(file.getvalue().decode("utf-8-sig"), from_text=True)
                st.session_state["six_csv"] = predict(records, models)
            except (ValueError, UnicodeError) as error:
                st.error(str(error))
        if "six_csv" in st.session_state:
            csv_results = st.session_state["six_csv"]
            row_ids = {r["row_id"] for r in csv_results}
            error_ids = {r["row_id"] for r in csv_results if r["error"]}
            counts = st.columns(3)
            counts[0].metric("Tổng số dòng", len(row_ids))
            counts[1].metric("Dòng không có lỗi", len(row_ids) - len(error_ids))
            counts[2].metric("Dòng có ít nhất một lỗi", len(error_ids))
            show_results(csv_results, "csv_download")
    with experiments:
        st.subheader("So sánh sáu thuật toán")
        st.caption("Cùng 2.620 mẫu phát triển, 656 mẫu kiểm tra và năm lượt kiểm định chéo. Naive Bayes được tô nổi bật trong bảng.")
        rows = []
        chart = []
        for algorithm in ALGORITHMS:
            metric = summary["models"][algorithm]["test"]
            rows.append((NAMES[algorithm], percent(metric["accuracy"]), percent(metric["balanced_accuracy"]),
                         f"{metric['macro_f1']:.4f}", f"{metric['roc_auc']:.4f}"))
            chart.append({"Thuật toán": NAMES[algorithm], "Accuracy (%)": metric["accuracy"] * 100,
                          "Balanced Accuracy (%)": metric["balanced_accuracy"] * 100})
        table(("Thuật toán", "Accuracy", "Balanced Accuracy", "Macro-F1", "ROC-AUC"), rows, primary_column=0)
        st.bar_chart(pd.DataFrame(chart).set_index("Thuật toán"), height=320,
                     color=["#1469b6", "#10a8a2"], stack=False, width="stretch")
        st.caption("Tập kiểm tra đã được sử dụng trong phiên bản trước của đề tài. Kết quả mô tả các cấu hình đã chạy; chưa phải đánh giá trên dữ liệu ngoài độc lập.")
    with raw_tab:
        st.subheader("Dữ liệu trước tiền xử lý")
        st.write("CSV Water Quality nguyên trạng, bao gồm ô thiếu và nhãn Potability. Bảng này chưa điền thiếu, chuẩn hóa hoặc chia khoảng.")
        stats = st.columns(4)
        stats[0].metric("Số mẫu", f"{len(raw_records):,}".replace(",", "."))
        stats[1].metric("Thuộc tính đầu vào", len(FEATURE_COLUMNS))
        stats[2].metric("Không uống được · 0", raw_info["classes"][0])
        stats[3].metric("Uống được · 1", raw_info["classes"][1])
        st.dataframe(pd.read_csv(io.BytesIO(raw_bytes)), width="stretch", height=390,
                     column_config={**{name: st.column_config.NumberColumn(format="%.4f")
                                        for name in FEATURE_COLUMNS},
                                    "Potability": st.column_config.NumberColumn(format="%d")})
        st.caption("Bảng hiển thị bốn chữ số thập phân để dễ đọc; tệp tải xuống giữ nguyên độ chính xác của CSV.")
        st.download_button("Tải toàn bộ CSV gốc", raw_bytes, "water_potability.csv", "text/csv", type="primary")
        st.caption("Ô thiếu: " + " · ".join(f"{name}: {count}" for name, count in raw_info["missing"].items() if count))
        st.caption("Nguồn: Water Quality, Aditya Kadiwal, Kaggle, phiên bản 3. Tệp tải xuống giữ nguyên các byte của dữ liệu gốc.")


if __name__ == "__main__":
    main()
