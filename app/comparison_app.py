"""Web cục bộ dự đoán Potability bằng sáu thuật toán đã học."""

import csv
from hashlib import sha256
from html import escape
import io
import json
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import FEATURE_COLUMNS
from src.data_validation import validate_records, read_csv_records
from src.pipeline import sha256_file
from src.six_experiments import ALGORITHMS, NAMES, RUN_ID, load_models

FOLDER = ROOT / "results" / "comparison" / RUN_ID


@st.cache_resource
def resources(manifest_hash):
    manifest = json.loads((FOLDER / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(FOLDER / "summary.json") != manifest["files"]["summary.json"]:
        raise ValueError("Checksum số liệu thực nghiệm không khớp.")
    return load_models(FOLDER), json.loads((FOLDER / "summary.json").read_text(encoding="utf-8"))


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


def table(headers, records):
    html = ["<div style='overflow-x:auto'><table><tr>", *(f"<th>{escape(item)}</th>" for item in headers), "</tr>"]
    for values in records:
        html.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in values) + "</tr>")
    st.html("".join(html) + "</table></div>")


def show_results(results, key):
    display = []
    for result in results[:120]:
        label = result["label"]
        label_text = "" if label is None else f"{label} — " + ("uống được" if label else "không uống được")
        score = "" if result["score_1"] is None else f"{result['score_1']:.6f}"
        display.append((result["row_id"], result["algorithm"], label_text, score, result["score_type"], result["error"]))
    table(("Dòng", "Thuật toán", "Nhãn", "Điểm lớp 1", "Loại điểm", "Lỗi"), display)
    if len(results) > 120:
        st.caption(f"Đang xem 120/{len(results)} kết quả. CSV tải xuống chứa toàn bộ kết quả.")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=("row_id", "sample_id", "algorithm", "label", "score_1", "score_type", "error"))
    writer.writeheader()
    writer.writerows(results)
    st.download_button("Tải toàn bộ kết quả CSV", stream.getvalue().encode("utf-8-sig"), "du_doan_nuoc.csv", "text/csv", key=key)


def main():
    st.set_page_config(page_title="Nhóm 11 — Chất lượng nước", layout="wide")
    st.title("Phân loại chất lượng nước bằng Naive Bayes")
    st.caption("Nhóm 11 · 65CNTT · So sánh với ID3, C4.5, KNN, Logistic Regression và SVM tuyến tính")
    st.warning("Demo học thuật theo nhãn dữ liệu. Kết quả không thay thế kiểm nghiệm chất lượng nước.")
    try:
        models, summary = resources(sha256_file(FOLDER / "manifest.json"))
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Chưa nạp được mô hình đã kiểm chứng. {error}")
        return
    manual, upload, experiments = st.tabs(("Dự đoán một mẫu", "Dự đoán CSV", "Kết quả thực nghiệm"))
    with manual:
        st.write("Nhập chín chỉ tiêu, để trống ô thiếu. Giá trị mặc định là trung vị tập huấn luyện để thử giao diện.")
        medians = models["naive_bayes"].processor.to_state()["fill_values"]
        fields = st.columns(3)
        record = {}
        for i, name in enumerate(FEATURE_COLUMNS):
            with fields[i % 3]:
                record[name] = st.text_input(name, value=f"{medians[i]:.4f}", key="six_" + name)
        identity = tuple(record.values())
        if st.session_state.get("six_manual_identity") != identity:
            st.session_state.pop("six_manual", None)
            st.session_state["six_manual_identity"] = identity
        if st.button("Dự đoán và so sánh", type="primary"):
            st.session_state["six_manual"] = predict([record], models)
        if "six_manual" in st.session_state:
            result = st.session_state["six_manual"]
            primary = result[0]
            if primary["label"] is not None:
                st.success("Naive Bayes dự đoán " + ("uống được (1)" if primary["label"] else "không uống được (0)"))
            else:
                st.error(primary["error"])
            show_results(result, "manual_download")
    with upload:
        st.caption("CSV UTF-8, dấu phẩy, chín tên cột bên dưới. Có thể kèm sample_id và Potability.")
        st.code(",".join(FEATURE_COLUMNS), language="text")
        file = st.file_uploader("Tải dữ liệu cần dự đoán", type="csv")
        identity = sha256(file.getvalue()).hexdigest() if file is not None else None
        if st.session_state.get("six_csv_identity") != identity:
            st.session_state.pop("six_csv", None)
            st.session_state["six_csv_identity"] = identity
        if st.button("Dự đoán CSV", disabled=file is None):
            st.session_state.pop("six_csv", None)
            try:
                records = read_csv_records(file.getvalue().decode("utf-8-sig"), from_text=True)
                st.session_state["six_csv"] = predict(records, models)
            except (ValueError, UnicodeError) as error:
                st.error(str(error))
        if "six_csv" in st.session_state:
            show_results(st.session_state["six_csv"], "csv_download")
    with experiments:
        st.write("Cùng 2.620 mẫu phát triển, 656 mẫu test và năm fold. Naive Bayes là mô hình chính của đề tài.")
        rows = []
        for algorithm in ALGORITHMS:
            metric = summary["models"][algorithm]["test"]
            rows.append((NAMES[algorithm], f"{metric['accuracy']:.2%}", f"{metric['balanced_accuracy']:.2%}",
                         f"{metric['macro_f1']:.4f}", f"{metric['roc_auc']:.4f}"))
        table(("Thuật toán", "Accuracy", "Balanced Accuracy", "Macro-F1", "ROC-AUC"), rows)
        st.caption("SVM trong bảng dự đoán dùng biên quyết định. Các xác suất mô hình còn lại là ước lượng, chưa được hiệu chỉnh.")


if __name__ == "__main__":
    main()
