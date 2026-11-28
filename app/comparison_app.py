"""Web cục bộ dự đoán Potability bằng sáu thuật toán đã học với giao diện hiện đại."""

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
from src.ui_theme import (
    CUSTOM_CSS,
    PRESET_SAMPLES,
    WATER_STANDARDS,
    render_consensus_board,
    render_hero_header,
    render_metric_card,
)
from src.config import FEATURE_COLUMNS, RAW_DATA_PATH
from src.data_validation import read_csv_records, validate_records
from src.pipeline import sha256_file
from src.six_experiments import ALGORITHMS, NAMES, RUN_ID, load_models

FOLDER = ROOT / "results" / "comparison" / RUN_ID

FEATURE_LABELS = {
    "ph": "pH (Độ pH)",
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
    """Xóa trắng 9 ô nhập và hủy kết quả dự đoán cũ."""
    for name in FEATURE_COLUMNS:
        st.session_state["six_" + name] = ""
    st.session_state.pop("six_manual", None)


def example_csv(records):
    """Trích ba mẫu gốc để minh họa định dạng dự đoán, bỏ cột nhãn."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=("sample_id", *FEATURE_COLUMNS))
    writer.writeheader()
    for index in (0, 27, 100):
        writer.writerow({
            "sample_id": f"raw_{index}",
            **{name: records[index][name] for name in FEATURE_COLUMNS},
        })
    return output.getvalue().encode("utf-8-sig")


def predict(records, models):
    """Một kết quả cho mỗi dòng và mỗi mô hình, giữ lỗi dữ liệu theo dòng."""
    validation = validate_records(records, training=False)
    valid = dict(zip(validation["row_ids"], validation["X"]))
    errors = {entry["input_row_id"]: entry["error"] for entry in validation["errors"]}
    output = []
    for row_id, record in enumerate(records):
        for algorithm in ALGORITHMS:
            result = {
                "row_id": row_id,
                "sample_id": record.get("sample_id", ""),
                "algorithm": NAMES[algorithm],
                "label": None,
                "score_1": None,
                "score_type": "biên quyết định" if algorithm == "svm" else "xác suất ước lượng",
                "error": errors.get(row_id, ""),
            }
            if row_id in valid:
                try:
                    labels, scores = models[algorithm].predict_with_scores([valid[row_id]])
                    result.update(label=labels[0], score_1=scores[0])
                except (ValueError, ArithmeticError) as error:
                    result["error"] = str(error)
            output.append(result)
    return output


def table(headers, records, primary_column=None):
    """Hiển thị bảng dữ liệu với phong cách giao diện hiện đại."""
    html = ["<div class='custom-table-wrapper'><table class='custom-table'><thead><tr>"]
    html.extend(f"<th scope='col'>{escape(str(item))}</th>" for item in headers)
    html.append("</tr></thead><tbody>")
    for values in records:
        primary = primary_column is not None and len(values) > primary_column and NAMES["naive_bayes"] in str(values[primary_column])
        row_class = " class='primary-model'" if primary else ""
        html.append(f"<tr{row_class}>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
    html.append("</tbody></table></div>")
    st.html("".join(html))


def show_results(results, key):
    """Hiển thị kết quả dự đoán với các huy hiệu và định dạng số trực quan."""
    display = []
    for result in results[:120]:
        label = result["label"]
        algorithm_name = result["algorithm"]
        if algorithm_name == "Naive Bayes":
            algo_cell = f"<span class='badge-model-primary'>{escape(algorithm_name)} [Chính]</span>"
        else:
            algo_cell = f"<strong>{escape(algorithm_name)}</strong>"

        if label is None:
            label_cell = "<span class='badge-error'>Không xác định</span>"
        elif label == 1:
            label_cell = "<span class='badge-potable'>1 · Uống được</span>"
        else:
            label_cell = "<span class='badge-non-potable'>0 · Không uống được</span>"

        score_val = result["score_1"]
        if score_val is None:
            score_cell = "-"
        elif result["score_type"] == "biên quyết định":
            score_cell = f"<code>{score_val:+.4f}</code>"
        else:
            score_cell = f"<code>{score_val:.4f}</code> ({percent(score_val)})"

        error_text = f"<span style='color: #dc2626;'>{escape(result['error'])}</span>" if result["error"] else "-"
        display.append((
            str(result["row_id"]),
            algo_cell,
            label_cell,
            score_cell,
            escape(result["score_type"]),
            error_text,
        ))

    table(("ID dòng", "Thuật toán", "Phân loại", "Điểm đầu ra", "Loại điểm", "Ghi chú / lỗi"), display, primary_column=1)
    if len(results) > 120:
        st.caption(f"Đang hiển thị 120/{len(results)} kết quả. Tệp CSV tải xuống chứa toàn bộ kết quả.")

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=("row_id", "sample_id", "algorithm", "label", "score_1", "score_type", "error"),
    )
    writer.writeheader()
    writer.writerows(results)
    st.download_button(
        "Tải toàn bộ kết quả CSV",
        stream.getvalue().encode("utf-8-sig"),
        "du_doan_nuoc.csv",
        "text/csv",
        key=key,
    )
    st.caption("Các điểm xác suất chưa được hiệu chỉnh. Biên quyết định của SVM có thể âm hoặc dương và không phải xác suất.")


def check_water_safety(record):
    """So sánh từng chỉ số với ngưỡng an toàn khuyến nghị của WHO."""
    safety_rows = []
    for col in FEATURE_COLUMNS:
        raw_val = record.get(col, "")
        info = WATER_STANDARDS.get(col, {})
        name = info.get("name", col)
        unit = info.get("unit", "")
        who_range = info.get("who_range", "-")
        min_safe = info.get("min_safe", 0)
        max_safe = info.get("max_safe", float("inf"))

        try:
            val = float(raw_val)
            is_safe = min_safe <= val <= max_safe
            status_badge = (
                "<span class='badge-potable'>Trong khoảng khuyến nghị</span>"
                if is_safe
                else "<span class='badge-non-potable'>Vượt ngưỡng khuyến cáo</span>"
            )
            val_display = f"<strong>{val:.2f}</strong> {unit}"
        except (ValueError, TypeError):
            val_display = f"<em>{escape(str(raw_val))}</em>"
            status_badge = "<span class='badge-error'>Thiếu / Chưa nhập</span>"

        safety_rows.append((name, val_display, who_range, status_badge))
    return safety_rows


def main():
    st.set_page_config(
        page_title="Chất lượng nước · Nhóm 11",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    css_path = ROOT / "app" / "assets" / "comparison.css"
    if css_path.is_file():
        st.html("<style>" + css_path.read_text(encoding="utf-8") + "</style>")
    st.html(CUSTOM_CSS)

    try:
        models, summary = resources(sha256_file(FOLDER / "manifest.json"))
        manifest = json.loads((FOLDER / "manifest.json").read_text(encoding="utf-8"))
        raw_bytes, raw_records, raw_info = dataset(manifest["raw_sha256"], RAW_DATA_PATH.stat().st_mtime_ns)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Chưa nạp được mô hình đã kiểm chứng. {error}")
        return

    raw_count = f"{len(raw_records):,}".replace(",", ".")
    st.html(
        render_hero_header(
            title="Hệ thống Phân loại Chất lượng Nước & Đối chiếu Sáu Thuật toán Học máy",
            subtitle="Bài tập lớn Khai phá dữ liệu — Nhóm 11 · 65CNTT · Trường Đại học Thủy Lợi",
            badges=[
                f"{raw_count} mẫu dữ liệu",
                f"{len(FEATURE_COLUMNS)} chỉ tiêu đo",
                "Naive Bayes [Chính]",
                "ID3",
                "C4.5",
                "KNN",
                "Logistic Regression",
                "Linear SVM",
                "Tự cài đặt 100% Python",
                "Xác thực SHA-256",
            ],
        )
    )

    st.warning("Ứng dụng học thuật theo nhãn Water Quality. Kết quả không thay thế kiểm nghiệm chất lượng nước.")

    manual, upload, experiments, raw_tab, guide_tab = st.tabs(
        ("Dự đoán một mẫu", "Dự đoán CSV", "Kết quả thực nghiệm", "Dữ liệu gốc", "Cẩm nang 9 chỉ tiêu nước (WHO)")
    )

    with manual:
        medians = models["naive_bayes"].processor.to_state()["fill_values"]

        st.subheader("Thông số mẫu nước")
        st.caption("Nhập số thập phân bằng dấu chấm; để trống chỉ tiêu chưa có. Mẫu minh họa ban đầu dùng trung vị tập huấn luyện.")

        st.markdown("##### Mẫu thử nghiệm nhanh:")
        btn_cols = st.columns([1.2, 1.2, 1.2, 1.2, 2])
        if btn_cols[0].button("Nạp mẫu nước sạch", help="Nạp chỉ số trong ngưỡng an toàn khuyến nghị của WHO"):
            for k, v in PRESET_SAMPLES["clean"]["values"].items():
                st.session_state[f"six_{k}"] = f"{v:.4f}"
            st.session_state.pop("six_manual", None)
            st.rerun()

        if btn_cols[1].button("Nạp mẫu nước ô nhiễm", help="Nạp chỉ số axit cao, độc chất và độ đục vượt ngưỡng"):
            for k, v in PRESET_SAMPLES["polluted"]["values"].items():
                st.session_state[f"six_{k}"] = f"{v:.4f}"
            st.session_state.pop("six_manual", None)
            st.rerun()

        btn_cols[2].button("Nạp mẫu minh họa", on_click=fill_example, args=(medians,), help="Nạp trung vị tập huấn luyện")
        btn_cols[3].button("Xóa ô nhập", on_click=clear_inputs, help="Xóa trắng 9 ô nhập liệu")

        st.write("")
        record = {}

        group_categories = [
            ("Tính chất lý hóa cơ bản", [c for c in FEATURE_COLUMNS if WATER_STANDARDS[c]["category"] == "Tính chất lý hóa cơ bản"]),
            ("Khử trùng & Khoáng chất", [c for c in FEATURE_COLUMNS if WATER_STANDARDS[c]["category"] == "Khử trùng & Khoáng chất"]),
            ("Độc chất vi lượng & Độ đục", [c for c in FEATURE_COLUMNS if WATER_STANDARDS[c]["category"] == "Độc chất vi lượng & Độ đục"]),
        ]

        group_cols = st.columns(3)
        for idx, (group_name, group_features) in enumerate(group_categories):
            with group_cols[idx]:
                st.html(f"<div class='feature-group-header'>{group_name}</div>")
                for name in group_features:
                    info = WATER_STANDARDS.get(name, {})
                    label_text = FEATURE_LABELS.get(name, name)
                    help_text = f"Cột CSV: {name}. Chuẩn WHO: {info.get('who_range', '-')}. {info.get('desc', '')}"
                    record[name] = st.text_input(
                        label_text,
                        value=st.session_state.get(f"six_{name}", f"{medians[FEATURE_COLUMNS.index(name)]:.4f}"),
                        key=f"six_{name}",
                        help=help_text,
                    )

        identity = tuple(record.values())
        if st.session_state.get("six_manual_identity") != identity:
            st.session_state.pop("six_manual", None)
            st.session_state["six_manual_identity"] = identity

        st.write("")
        col_btn, _ = st.columns([2, 5])
        with col_btn:
            submit = st.button("Dự đoán và so sánh", type="primary")

        if submit:
            st.session_state["six_manual"] = predict([record], models)

        if "six_manual" in st.session_state:
            result = st.session_state["six_manual"]
            primary = result[0]

            st.write("---")
            st.subheader("Kết quả phân loại")

            if primary["label"] is not None:
                is_potable = primary["label"] == 1
                card_class = "result-hero-potable" if is_potable else "result-hero-non-potable"
                status_title = (
                    "MÔ HÌNH CHÍNH (NAIVE BAYES): ĐẠT CHUẨN UỐNG ĐƯỢC (Nhãn 1)"
                    if is_potable
                    else "MÔ HÌNH CHÍNH (NAIVE BAYES): KHÔNG ĐẠT CHUẨN UỐNG ĐƯỢC (Nhãn 0)"
                )
                status_desc = (
                    "Mẫu nước có các đặc trưng phù hợp với nguồn nước uống được theo phân phối xác suất mô hình."
                    if is_potable
                    else "Mẫu nước có các đặc trưng tương đồng với nguồn nước không uống được theo bộ dữ liệu thực nghiệm."
                )
                score_val = primary["score_1"] if primary["score_1"] is not None else 0.5
                fill_class = "confidence-bar-fill-potable" if is_potable else "confidence-bar-fill-non-potable"
                bar_width = f"{score_val * 100:.1f}%" if is_potable else f"{(1 - score_val) * 100:.1f}%"

                hero_html = f"""
                <div class="{card_class}">
                    <div class="result-hero-title">{status_title}</div>
                    <div class="result-hero-desc">{status_desc}</div>
                    <div class="confidence-container">
                        <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600;">
                            <span>Độ tin cậy xác suất lớp 1 (Uống được):</span>
                            <span>{score_val * 100:.2f}% ({percent(score_val)})</span>
                        </div>
                        <div class="confidence-bar-bg">
                            <div class="{fill_class}" style="width: {bar_width};"></div>
                        </div>
                    </div>
                </div>
                """
                st.html(hero_html)
            else:
                st.error(f"Lỗi: {primary['error']}")

            board_html = render_consensus_board(result)
            if board_html:
                st.html(board_html)

            st.markdown("##### Chi tiết kết quả từng thuật toán:")
            show_results(result, "manual_download")

            with st.expander("Đối chiếu các chỉ số mẫu nước vừa nhập với chuẩn an toàn WHO"):
                safety_data = check_water_safety(record)
                table(("Chỉ tiêu đo", "Giá trị nhập", "Khoảng khuyến cáo WHO", "Đánh giá sơ bộ"), safety_data)

    with upload:
        st.subheader("Dự đoán nhiều mẫu từ CSV")
        st.write("Tải tệp UTF-8 có đủ chín cột chỉ tiêu. Mỗi dòng được dự đoán bằng cả sáu mô hình.")

        with st.expander("Định dạng tệp và dữ liệu mẫu", expanded=True):
            st.code(",".join(FEATURE_COLUMNS), language="text")
            st.caption("Có thể kèm sample_id và Potability. Potability không được sử dụng làm đầu vào dự đoán.")
            st.download_button("Tải CSV mẫu", example_csv(raw_records), "mau_du_doan.csv", "text/csv")

        st.write("")
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
            valid_rows = len(row_ids) - len(error_ids)

            nb_results = [r for r in csv_results if r["algorithm"] == "Naive Bayes" and r["label"] is not None]
            potable_nb = sum(1 for r in nb_results if r["label"] == 1)

            st.write("---")
            st.subheader("Thống kê kết quả phân loại tập tin CSV")
            st.html(f"""
            <div class="metric-grid">
                {render_metric_card("Tổng số dòng", str(len(row_ids)), "Mẫu dữ liệu đã xử lý")}
                {render_metric_card("Dòng hợp lệ", str(valid_rows), f"{valid_rows / max(len(row_ids), 1) * 100:.1f}% tổng số")}
                {render_metric_card("Dòng có lỗi", str(len(error_ids)), "Bị từ chối do thiếu")}
                {render_metric_card("Tỷ lệ uống được (NB)", f"{potable_nb / max(len(nb_results), 1) * 100:.1f}%", f"{potable_nb}/{len(nb_results)} mẫu hợp lệ")}
            </div>
            """)
            show_results(csv_results, "csv_download")

    with experiments:
        st.subheader("So sánh sáu thuật toán")
        st.caption("Cùng 2.620 mẫu phát triển, 656 mẫu kiểm tra và năm lượt kiểm định chéo. Naive Bayes được tô nổi bật trong bảng.")

        st.html(f"""
        <div class="metric-grid">
            {render_metric_card("Accuracy (KNN)", "64.79%", "Độ chính xác cao nhất")}
            {render_metric_card("Balanced Acc (KNN)", "60.72%", "Cân bằng giữa hai lớp")}
            {render_metric_card("ROC-AUC (KNN)", "0.6518", "Diện tích dưới đường cong")}
            {render_metric_card("Naive Bayes (Chính)", "60.37%", "Accuracy mô hình đề tài")}
        </div>
        """)

        rows = []
        chart = []
        for algorithm in ALGORITHMS:
            metric = summary["models"][algorithm]["test"]
            is_main = algorithm == "naive_bayes"
            algo_cell = (
                f"<span class='badge-model-primary'>{NAMES[algorithm]} [Chính]</span>"
                if is_main
                else f"<strong>{NAMES[algorithm]}</strong>"
            )
            rows.append((
                algo_cell,
                f"<code>{percent(metric['accuracy'])}</code>",
                f"<code>{percent(metric['balanced_accuracy'])}</code>",
                f"<code>{metric['macro_f1']:.4f}</code>",
                f"<code>{metric['roc_auc']:.4f}</code>",
            ))
            chart.append({
                "Thuật toán": NAMES[algorithm],
                "Accuracy (%)": metric["accuracy"] * 100,
                "Balanced Accuracy (%)": metric["balanced_accuracy"] * 100,
            })

        table(("Thuật toán", "Accuracy", "Balanced Accuracy", "Macro-F1", "ROC-AUC"), rows, primary_column=0)
        st.bar_chart(
            pd.DataFrame(chart).set_index("Thuật toán"),
            height=320,
            color=["#0284c7", "#0d9488"],
            stack=False,
            width="stretch",
        )
        st.caption("Tập kiểm tra đã được sử dụng trong phiên bản trước của đề tài. Kết quả mô tả các cấu hình đã chạy; chưa phải đánh giá trên dữ liệu ngoài độc lập.")

        st.write("---")
        st.subheader("Biểu đồ và đồ thị trực quan hóa thực nghiệm")

        fig_tab1, fig_tab2, fig_tab3, fig_tab4 = st.tabs(
            ("Đối chiếu hiệu năng", "Đường cong ROC", "Ma trận nhầm lẫn", "Phân phối dữ liệu (EDA)")
        )

        with fig_tab1:
            comparison_img = FOLDER / "figures" / "comparison.png"
            if comparison_img.is_file():
                st.image(str(comparison_img), caption="Biểu đồ so sánh bốn chỉ số đánh giá của sáu thuật toán", width="stretch")

        with fig_tab2:
            roc_img = FOLDER / "figures" / "roc_comparison.png"
            if roc_img.is_file():
                st.image(str(roc_img), caption="Đường cong ROC của sáu mô hình trên tập kiểm thử", width="stretch")

        with fig_tab3:
            confusions_img = FOLDER / "figures" / "confusions.png"
            if confusions_img.is_file():
                st.image(str(confusions_img), caption="Ma trận nhầm lẫn (Confusion Matrix) của từng mô hình", width="stretch")

        with fig_tab4:
            eda_cols = st.columns(2)
            hist_img = FOLDER / "figures" / "histograms.png"
            corr_img = FOLDER / "figures" / "correlation.png"
            if hist_img.is_file():
                eda_cols[0].image(str(hist_img), caption="Phân bố tần suất 9 chỉ tiêu hóa lý", width="stretch")
            if corr_img.is_file():
                eda_cols[1].image(str(corr_img), caption="Ma trận hệ số tương quan Pearson giữa các chỉ số", width="stretch")

    with raw_tab:
        st.subheader("Dữ liệu trước tiền xử lý")
        st.write("CSV Water Quality nguyên trạng, bao gồm ô thiếu và nhãn Potability. Bảng này chưa điền thiếu, chuẩn hóa hoặc chia khoảng.")
        stats = st.columns(4)
        stats[0].metric("Số mẫu", f"{len(raw_records):,}".replace(",", "."))
        stats[1].metric("Thuộc tính đầu vào", len(FEATURE_COLUMNS))
        stats[2].metric("Không uống được · 0", raw_info["classes"][0])
        stats[3].metric("Uống được · 1", raw_info["classes"][1])
        st.dataframe(
            pd.read_csv(io.BytesIO(raw_bytes)),
            width="stretch",
            height=390,
            column_config={
                **{name: st.column_config.NumberColumn(format="%.4f") for name in FEATURE_COLUMNS},
                "Potability": st.column_config.NumberColumn(format="%d"),
            },
        )
        st.caption("Bảng hiển thị bốn chữ số thập phân để dễ đọc; tệp tải xuống giữ nguyên độ chính xác của CSV.")
        st.download_button("Tải toàn bộ CSV gốc", raw_bytes, "water_potability.csv", "text/csv", type="primary")
        st.caption("Ô thiếu: " + " · ".join(f"{name}: {count}" for name, count in raw_info["missing"].items() if count))
        st.caption("Nguồn: Water Quality, Aditya Kadiwal, Kaggle, phiên bản 3. Tệp tải xuống giữ nguyên các byte của dữ liệu gốc.")

    with guide_tab:
        st.subheader("Cẩm nang ý nghĩa 9 chỉ tiêu chất lượng nước theo tiêu chuẩn WHO")
        st.markdown(
            "Bộ dữ liệu gồm chín đặc trưng đo lường tính chất hóa học và sinh học của nước. "
            "Dưới đây là định nghĩa, ý nghĩa môi trường y tế và ngưỡng tiêu chuẩn khuyến cáo bởi Tổ chức Y tế Thế giới (WHO):"
        )
        for col_name in FEATURE_COLUMNS:
            info = WATER_STANDARDS.get(col_name, {})
            with st.expander(f"{info.get('name', col_name)} — Khuyến nghị: {info.get('who_range', '-')}"):
                st.markdown(f"**Nhóm chỉ tiêu:** {info.get('category', '-')}")
                st.markdown(f"**Đơn vị đo:** `{info.get('unit', '')}`")
                st.markdown(f"**Ý nghĩa y tế và an toàn:** {info.get('desc', '')}")


if __name__ == "__main__":
    main()
