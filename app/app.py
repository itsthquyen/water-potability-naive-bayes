"""Demo tiếng Việt chỉ sử dụng pipeline đã học và API dự đoán chung với giao diện hiện đại."""

import hashlib
from html import escape
import json
from pathlib import Path
import re
import sys

import streamlit as st

# Streamlit chạy file theo đường dẫn nên cần gốc project để import src.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ui_theme import (
    CUSTOM_CSS,
    FEATURE_INFO,
    render_hero_header,
    render_metric_card,
)
from src.config import DEMO_RUN_ID, FEATURE_COLUMNS, POTABILITY_LABEL_NAMES, PROJECT_ROOT
from src.predict import predict_csv, predict_records, results_to_csv


@st.cache_resource(show_spinner=False)
def load_cached_pipeline(model_path, checksum, modified_ns, sidecar_checksum):
    """Nạp model; khóa cache gồm nội dung, mtime và sidecar chống trạng thái cũ.

    Các checksum là SHA-256 của bytes thực tế; load_pipeline tự kiểm tra sidecar.
    modified_ns giữ dấu vết thay file ngay cả khi nội dung không đổi.
    """
    from src.pipeline import load_pipeline

    return load_pipeline(Path(model_path), expected_sha256=checksum)


def verified_run_files(run_id, model_checksum):
    """Đọc manifest đúng run, chỉ trả các file nội bộ có checksum khớp.

    Không dùng run_id hoặc đường dẫn trong manifest để đọc ngoài results.
    Thiếu/sai manifest gây ValueError hoặc OSError và không hiển thị kết quả.
    """
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
        raise ValueError("Run ID không hợp lệ; không đọc kết quả thực nghiệm.")
    results_root = (PROJECT_ROOT / "results").resolve()
    manifest_path = (results_root / "runs" / f"{run_id}.json").resolve()
    if not manifest_path.is_relative_to(results_root):
        raise ValueError("Manifest nằm ngoài thư mục kết quả.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("run_id") != run_id or manifest.get("model_sha256") != model_checksum:
        raise ValueError("Manifest không khớp run ID hoặc checksum model đang chọn.")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Manifest chưa có danh sách checksum kết quả.")
    verified = {}
    allowed_roots = [(results_root / folder / run_id).resolve() for folder in ("metrics", "figures")]
    for relative_path, expected_checksum in files.items():
        if not isinstance(relative_path, str) or not isinstance(expected_checksum, str):
            raise ValueError("Danh sách checksum kết quả không hợp lệ.")
        path = (PROJECT_ROOT / relative_path).resolve()
        if not path.is_relative_to(results_root) or not any(path.is_relative_to(root) for root in allowed_roots):
            continue
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_checksum:
            raise ValueError(f"Checksum kết quả không khớp: {path.name}")
        verified[path] = content
    return verified


def show_experiment(run_id, model_checksum):
    """Hiển thị số liệu đã lưu, không tính lại metric hoặc tạo kết quả giả."""
    st.subheader("Kết quả thực nghiệm mô hình")
    try:
        files = verified_run_files(run_id, model_checksum)
        metrics_root = (PROJECT_ROOT / "results" / "metrics" / run_id).resolve()
        metrics_bytes = files.get(metrics_root / "test_metrics.json")
        if metrics_bytes is None:
            metrics_bytes = files.get(metrics_root / "metrics.json")
        if metrics_bytes is None:
            st.info("Chưa có kết quả thực nghiệm đã xác minh cho model này.")
            return
        metrics = json.loads(metrics_bytes.decode("utf-8"))
        if not isinstance(metrics, dict) or ("run_id" in metrics and metrics["run_id"] != run_id):
            raise ValueError("Run ID trong metrics không khớp model.")
        
        acc = metrics.get("accuracy")
        bal_acc = metrics.get("balanced_accuracy")
        f1 = metrics.get("macro_f1")
        auc = metrics.get("roc_auc")
        if acc is not None and bal_acc is not None:
            st.html(f"""
            <div class="metric-grid">
                {render_metric_card("Accuracy", f"{acc:.2%}", "Độ chính xác phân loại")}
                {render_metric_card("Balanced Accuracy", f"{bal_acc:.2%}", "Cân bằng hai lớp")}
                {render_metric_card("Macro-F1", f"{f1:.4f}" if f1 is not None else "-", "Trung bình F1 các lớp")}
                {render_metric_card("ROC-AUC", f"{auc:.4f}" if auc is not None else "-", "Diện tích dưới đường ROC")}
            </div>
            """)

        st.caption("Số liệu đã lưu từ thực nghiệm; không tính lại trên dữ liệu người dùng.")
        with st.expander("Dữ liệu JSON chi tiết của lần chạy"):
            st.json(metrics)
        figures = [(path, content) for path, content in files.items() if path.suffix.lower() == ".png"]
        if figures:
            with st.expander("Biểu đồ thực nghiệm và khám phá dữ liệu cùng lần chạy", expanded=True):
                for path, content in sorted(figures):
                    st.image(content, caption=path.stem, width="stretch")
    except FileNotFoundError:
        st.info("Chưa có kết quả thực nghiệm đã xác minh cho model này.")
    except (OSError, ValueError, TypeError) as error:
        st.warning(f"Không hiển thị kết quả chưa xác minh: {error}")


def show_prediction_results(results, download_key):
    """Trình bày nguyên kết quả API; dòng lỗi không suy diễn nhãn thay thế."""
    for result in results:
        if result["error"]:
            st.error(f"Dòng {result['input_row_id']}: {result['error']}")
        for warning in result["warnings"]:
            st.warning(f"Dòng {result['input_row_id']}: {warning}")

    if len(results) == 1 and results[0]["predicted_label"] is not None:
        predicted = results[0]["predicted_label"]
        label = POTABILITY_LABEL_NAMES[predicted]
        st.success(f"Mô hình dự đoán: {label} theo nhãn dữ liệu.")

        prob_1 = results[0].get("probability_1")
        is_potable = predicted == 1
        card_class = "result-hero-potable" if is_potable else "result-hero-non-potable"
        status_text = (
            "KẾT QUẢ: ĐẠT TIÊU CHUẨN UỐNG ĐƯỢC (Nhãn 1)"
            if is_potable
            else "KẾT QUẢ: KHÔNG ĐẠT TIÊU CHUẨN UỐNG ĐƯỢC (Nhãn 0)"
        )
        desc_text = (
            "Dựa trên 9 chỉ tiêu hóa lý đã nhập, mô hình Naive Bayes phân loại mẫu nước vào nhóm uống được."
            if is_potable
            else "Dựa trên 9 chỉ tiêu hóa lý đã nhập, mô hình Naive Bayes phân loại mẫu nước vào nhóm không uống được."
        )
        conf_pct = (prob_1 * 100) if prob_1 is not None else (100.0 if is_potable else 0.0)
        fill_pct = f"{conf_pct:.1f}%" if is_potable else f"{(100.0 - conf_pct):.1f}%"
        fill_class = "confidence-bar-fill-potable" if is_potable else "confidence-bar-fill-non-potable"

        st.html(f"""
        <div class="{card_class}">
            <div class="result-hero-title">{status_text}</div>
            <div class="result-hero-desc">{desc_text}</div>
            <div class="confidence-container">
                <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600;">
                    <span>Xác suất ước lượng lớp 1 (Uống được):</span>
                    <span>{conf_pct:.2f}%</span>
                </div>
                <div class="confidence-bar-bg">
                    <div class="{fill_class}" style="width: {fill_pct};"></div>
                </div>
            </div>
        </div>
        """)

    st.caption("probability_0/1 là xác suất do mô hình ước lượng, không phải tỷ lệ an toàn thực tế.")
    display_limit = 100
    fields = ("input_row_id", "sample_id", "predicted_label", "probability_0", "probability_1", "error")
    header = ("Dòng", "ID mẫu", "Nhãn dự đoán", "Xác suất lớp 0", "Xác suất lớp 1", "Lỗi")
    table = ["<div class='custom-table-wrapper' style='overflow-x:auto'><table class='custom-table'><thead><tr>"]
    table.extend(f"<th scope='col'>{title}</th>" for title in header)
    table.append("</tr></thead><tbody>")
    for result in results[:display_limit]:
        table.append("<tr>")
        for field in fields:
            value = result[field]
            if field == "predicted_label" and value is not None:
                value = f"{value} - {POTABILITY_LABEL_NAMES[value]}"
            if field.startswith("probability_") and value is not None:
                value = f"{value:.6f}"
            table.append(f"<td>{escape(str(value)) if value is not None else ''}</td>")
        table.append("</tr>")
    table.append("</tbody></table></div>")
    st.html("".join(table))
    if len(results) > display_limit:
        st.caption(f"Đang xem {display_limit}/{len(results)} dòng. Tệp CSV chứa toàn bộ kết quả.")
    st.download_button(
        "Tải kết quả CSV",
        data=results_to_csv(results).encode("utf-8-sig"),
        file_name="ket_qua_du_doan.csv",
        mime="text/csv",
        key=download_key,
    )


def main():
    """Khởi tạo giao diện; thiếu model thì dừng trước mọi lời gọi dự đoán."""
    st.set_page_config(
        page_title="Phân loại chất lượng nước",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Phân loại khả năng uống được của nước")
    st.warning("Demo học thuật; không thay thế kiểm nghiệm chất lượng nước.")
    st.write(
        "Naive Bayes đếm tần suất dự đoán nhãn Potability từ 9 chỉ tiêu đã chia khoảng. "
        "Nhãn 0: không uống được; nhãn 1: uống được theo bộ dữ liệu, không phải chứng nhận an toàn."
    )
    with st.expander("Giới thiệu và giới hạn"):
        st.write("Nhóm Quyền, Lộc, Séc | Trường Đại học Thủy Lợi.")
        st.markdown("Nguồn: [Water Potability, Aditya Kadiwal trên Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability), phiên bản 3.")
        st.write("Ứng dụng chỉ nạp model đã học, không huấn luyện khi dự đoán. Xác suất Naive Bayes có thể chưa được hiệu chỉnh tốt; dữ liệu thiếu hoặc ngoài phạm vi train làm kết quả có giới hạn.")

    with st.sidebar:
        st.header("Model đã học")
        selected_path = st.text_input(
            "Đường dẫn model",
            value=f"artifacts/{DEMO_RUN_ID}/pipeline.json",
            help="Đường dẫn tương đối từ gốc project hoặc đường dẫn cục bộ. Cần file .sha256 đi kèm.",
        )
        st.caption("Chọn tệp pipeline đã lưu từ thực nghiệm. Không có thao tác huấn luyện trong demo.")

    model_path = Path(selected_path).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not selected_path.strip() or not model_path.is_file():
        st.error("Chưa có model tại đường dẫn đã chọn. Không thể dự đoán; hãy chọn pipeline đã học.")
        return

    try:
        model_path = model_path.resolve()
        checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
        sidecar = Path(str(model_path) + ".sha256")
        sidecar_checksum = hashlib.sha256(sidecar.read_bytes()).hexdigest()
        pipeline = load_cached_pipeline(str(model_path), checksum, model_path.stat().st_mtime_ns, sidecar_checksum)
        state = pipeline.to_state()
        run_id = state["run_id"]
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        st.error(f"Không thể nạp model đã xác minh: {error}")
        return

    # Chỉ chèn CSS và giao diện cao cấp sau khi đã xác thực model thành công
    st.html(CUSTOM_CSS)

    st.sidebar.write(f"Run ID: {run_id or '(chưa có)'}")
    st.sidebar.caption(f"SHA-256: {checksum}")
    with st.sidebar.expander("Metadata model"):
        st.json(state.get("metadata", {}))

    model_identity = (str(model_path), checksum, sidecar_checksum, run_id)
    if st.session_state.get("model_identity") != model_identity:
        st.session_state.pop("manual_results", None)
        st.session_state.pop("csv_results", None)
        st.session_state["model_identity"] = model_identity

    manual_tab, csv_tab, experiment_tab, guide_tab = st.tabs(
        ["Nhập một mẫu", "Dự đoán CSV", "Thực nghiệm", "Thông tin 9 chỉ tiêu"]
    )

    with manual_tab:
        st.caption("Để trống hoặc nhập NaN cho ô thiếu. Dữ liệu được kiểm tra và điền bằng trạng thái đã học; mẫu thiếu cả 9 chỉ tiêu bị từ chối.")
        with st.form("manual_prediction"):
            columns = st.columns(3)
            record = {}
            for index, feature in enumerate(FEATURE_COLUMNS):
                info = FEATURE_INFO.get(feature, {})
                label_text = f"{feature} ({info.get('unit', '')})" if info.get("unit") else feature
                help_text = f"{info.get('name', feature)}. {info.get('desc', '')}"
                record[feature] = columns[index % 3].text_input(
                    label_text,
                    value="",
                    key=f"feature_{feature}",
                    help=help_text,
                )
            st.write("")
            submitted = st.form_submit_button("Dự đoán mẫu", type="primary")

        if submitted:
            st.session_state.pop("manual_results", None)
            try:
                st.session_state["manual_results"] = predict_records([record], pipeline, run_id=run_id)
            except (ValueError, ArithmeticError) as error:
                st.error(f"Không thể dự đoán: {error}")

        if "manual_results" in st.session_state:
            show_prediction_results(st.session_state["manual_results"], "download_manual")

    with csv_tab:
        st.caption("CSV UTF-8, dấu phẩy, đúng 9 tên cột (phân biệt hoa/thường). Có thể thêm sample_id và Potability; nhãn thật không được dùng để dự đoán.")
        st.code(",".join(FEATURE_COLUMNS), language="text")
        uploaded = st.file_uploader("Tải CSV đầu vào", type=["csv"], key="input_csv")
        uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
        upload_identity = hashlib.sha256(uploaded_bytes).hexdigest() if uploaded_bytes is not None else None
        if st.session_state.get("upload_identity") != upload_identity:
            st.session_state.pop("csv_results", None)
            st.session_state["upload_identity"] = upload_identity

        if st.button("Dự đoán CSV", disabled=uploaded is None, type="primary"):
            st.session_state.pop("csv_results", None)
            try:
                text = uploaded_bytes.decode("utf-8-sig")
                st.session_state["csv_results"] = predict_csv(text, pipeline, from_text=True, run_id=run_id)
            except (UnicodeError, ValueError, ArithmeticError) as error:
                st.error(f"CSV bị từ chối: {error}")

        if "csv_results" in st.session_state:
            csv_res = st.session_state["csv_results"]
            total = len(csv_res)
            err_count = sum(1 for r in csv_res if r.get("error"))
            valid_count = total - err_count
            potable_count = sum(1 for r in csv_res if r.get("predicted_label") == 1)

            st.html(f"""
            <div class="metric-grid">
                {render_metric_card("Tổng mẫu tải lên", str(total), "Dòng trong tệp CSV")}
                {render_metric_card("Mẫu hợp lệ", str(valid_count), f"{valid_count / max(total, 1) * 100:.1f}% tổng số")}
                {render_metric_card("Mẫu lỗi", str(err_count), "Không đủ dữ liệu")}
                {render_metric_card("Dự đoán uống được", str(potable_count), f"{potable_count / max(valid_count, 1) * 100:.1f}% mẫu hợp lệ")}
            </div>
            """)
            show_prediction_results(csv_res, "download_csv")

    with experiment_tab:
        show_experiment(run_id, checksum)

    with guide_tab:
        st.subheader("Thông tin các chỉ tiêu trong dữ liệu nguồn")
        for col_name in FEATURE_COLUMNS:
            info = FEATURE_INFO.get(col_name, {})
            with st.expander(f"{info.get('name', col_name)}"):
                st.markdown(f"**Nhóm chỉ tiêu:** {info.get('category', '-')}")
                st.markdown(f"**Đơn vị đo:** `{info.get('unit', '')}`")
                st.markdown(f"**Ý nghĩa môi trường và an toàn sức khỏe:** {info.get('desc', '')}")


if __name__ == "__main__":
    main()
