"""Demo tiếng Việt chỉ sử dụng pipeline đã học và API dự đoán chung."""

import hashlib
import json
from pathlib import Path
import re
import sys

import streamlit as st

# Streamlit chạy file theo đường dẫn nên cần gốc project để import src.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FEATURE_COLUMNS, PROJECT_ROOT
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
    st.subheader("Kết quả thực nghiệm")
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
        st.caption("Số liệu đã lưu từ thực nghiệm; không tính lại trên dữ liệu người dùng.")
        st.json(metrics)
        figures = [(path, content) for path, content in files.items() if path.suffix.lower() == ".png"]
        if figures:
            with st.expander("Biểu đồ thực nghiệm và khám phá dữ liệu cùng lần chạy"):
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
        label = "uống được" if results[0]["predicted_label"] == 1 else "không uống được"
        st.success(f"Mô hình dự đoán: {label} theo nhãn dữ liệu.")
    st.caption("probability_0/1 là xác suất do mô hình ước lượng, không phải tỷ lệ an toàn thực tế.")
    st.dataframe(results, hide_index=True, width="stretch")
    st.download_button(
        "Tải kết quả CSV", data=results_to_csv(results).encode("utf-8-sig"),
        file_name="ket_qua_du_doan.csv", mime="text/csv", key=download_key,
    )


def main():
    """Khởi tạo giao diện; thiếu model thì dừng trước mọi lời gọi dự đoán."""
    st.set_page_config(page_title="Phân loại chất lượng nước", page_icon=None, layout="wide")
    st.title("Phân loại khả năng uống được của nước")
    st.warning("Demo học thuật; không thay thế kiểm nghiệm chất lượng nước.")
    st.write(
        "Gaussian Naive Bayes tự cài đặt dự đoán nhãn Potability từ 9 chỉ tiêu. "
        "Nhãn 0: không uống được; nhãn 1: uống được theo bộ dữ liệu, không phải chứng nhận an toàn."
    )
    with st.expander("Giới thiệu và giới hạn"):
        st.write("Nhóm Quyền, Lộc, Séc | Trường Đại học Thủy Lợi.")
        st.markdown("Nguồn dự kiến: [Water Potability, Aditya Kadiwal trên Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability).")
        st.write("Ứng dụng chỉ nạp model đã học, không huấn luyện khi dự đoán. Xác suất Naive Bayes có thể chưa được hiệu chỉnh tốt; dữ liệu thiếu hoặc ngoài phạm vi train làm kết quả có giới hạn.")

    with st.sidebar:
        st.header("Model đã học")
        selected_path = st.text_input("Đường dẫn model", value="artifacts/pipeline.json", help="Đường dẫn tương đối từ gốc project hoặc đường dẫn cục bộ. Cần file .sha256 đi kèm.")
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
    st.sidebar.write(f"Run ID: {run_id or '(chưa có)'}")
    st.sidebar.caption(f"SHA-256: {checksum}")
    with st.sidebar.expander("Metadata model"):
        st.json(state.get("metadata", {}))
    model_identity = (str(model_path), checksum, sidecar_checksum, run_id)
    if st.session_state.get("model_identity") != model_identity:
        st.session_state.pop("manual_results", None)
        st.session_state.pop("csv_results", None)
        st.session_state["model_identity"] = model_identity

    manual_tab, csv_tab, experiment_tab = st.tabs(["Nhập một mẫu", "Dự đoán CSV", "Thực nghiệm"])
    with manual_tab:
        st.caption("Để trống hoặc nhập NaN cho ô thiếu. Dữ liệu được kiểm tra và điền bằng trạng thái đã học; mẫu thiếu cả 9 chỉ tiêu bị từ chối.")
        with st.form("manual_prediction"):
            columns = st.columns(3)
            record = {feature: columns[index % 3].text_input(feature, value="", key=f"feature_{feature}") for index, feature in enumerate(FEATURE_COLUMNS)}
            submitted = st.form_submit_button("Dự đoán mẫu")
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
        if st.button("Dự đoán CSV", disabled=uploaded is None):
            st.session_state.pop("csv_results", None)
            try:
                text = uploaded_bytes.decode("utf-8-sig")
                st.session_state["csv_results"] = predict_csv(text, pipeline, from_text=True, run_id=run_id)
            except (UnicodeError, ValueError, ArithmeticError) as error:
                st.error(f"CSV bị từ chối: {error}")
        if "csv_results" in st.session_state:
            show_prediction_results(st.session_state["csv_results"], "download_csv")
    with experiment_tab:
        show_experiment(run_id, checksum)


if __name__ == "__main__":
    main()
