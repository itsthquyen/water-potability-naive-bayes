"""Thành phần trình bày và mô tả thuộc tính; không chứa thuật toán hoặc ngưỡng phân lớp."""

FEATURE_INFO = {'ph': {'name': 'Độ pH',
        'desc': 'Chỉ tiêu thể hiện tính axit hoặc kiềm của mẫu nước.',
        'category': 'Tính chất lý hóa cơ bản'},
 'Hardness': {'name': 'Độ cứng (Hardness)',
              'desc': 'Chỉ tiêu liên quan đến hàm lượng các chất khoáng gây độ cứng.',
              'category': 'Tính chất lý hóa cơ bản'},
 'Solids': {'name': 'Tổng chất rắn hòa tan (TDS)',
            'desc': 'Chỉ tiêu tổng lượng chất rắn hòa tan trong mẫu.',
            'category': 'Tính chất lý hóa cơ bản'},
 'Chloramines': {'name': 'Chloramines',
                 'desc': 'Chỉ tiêu Chloramines được ghi trong dữ liệu nguồn.',
                 'category': 'Khử trùng & Khoáng chất'},
 'Sulfate': {'name': 'Sulfate (SO₄²⁻)',
             'desc': 'Chỉ tiêu hàm lượng sulfate trong mẫu.',
             'category': 'Khử trùng & Khoáng chất'},
 'Conductivity': {'name': 'Độ dẫn điện (EC)',
                  'desc': 'Chỉ tiêu khả năng dẫn điện của mẫu nước.',
                  'category': 'Khử trùng & Khoáng chất'},
 'Organic_carbon': {'name': 'Tổng cacbon hữu cơ (TOC)',
                    'desc': 'Chỉ tiêu carbon hữu cơ trong mẫu.',
                    'category': 'Độc chất vi lượng & Độ đục'},
 'Trihalomethanes': {'name': 'Trihalomethanes (THMs)',
                     'desc': 'Chỉ tiêu nhóm hợp chất Trihalomethanes trong dữ liệu nguồn.',
                     'category': 'Độc chất vi lượng & Độ đục'},
 'Turbidity': {'name': 'Độ đục (Turbidity)',
               'desc': 'Chỉ tiêu mô tả độ đục của mẫu nước.',
               'category': 'Độc chất vi lượng & Độ đục'}}

PRESET_SAMPLES = {
    "sample_a": {
        "label": "Bộ số thử A",
        "description": "Dữ liệu thử thao tác, chưa có nhãn kiểm nghiệm",
        "values": {
            "ph": 7.25,
            "Hardness": 185.0,
            "Solids": 8500.0,
            "Chloramines": 3.10,
            "Sulfate": 235.0,
            "Conductivity": 360.0,
            "Organic_carbon": 2.80,
            "Trihalomethanes": 45.0,
            "Turbidity": 2.10,
        },
    },
    "sample_b": {
        "label": "Bộ số thử B",
        "description": "Dữ liệu thử thao tác với các trị số khác bộ A",
        "values": {
            "ph": 4.10,
            "Hardness": 380.0,
            "Solids": 32000.0,
            "Chloramines": 8.50,
            "Sulfate": 460.0,
            "Conductivity": 610.0,
            "Organic_carbon": 16.20,
            "Trihalomethanes": 110.0,
            "Turbidity": 7.60,
        },
    },
}

CUSTOM_CSS = """
<style>
/* Font và cài đặt chung */
:root {
    --primary-color: #0284c7;
    --primary-hover: #0369a1;
    --bg-card: rgba(255, 255, 255, 0.85);
    --border-color: #e2e8f0;
    --success-bg: #ecfdf5;
    --success-border: #a7f3d0;
    --success-text: #065f46;
    --danger-bg: #fff1f2;
    --danger-border: #fecdd3;
    --danger-text: #9f1239;
    --neutral-bg: #f8fafc;
    --neutral-text: #334155;
}

/* Nâng cấp Hero Header */
.hero-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
    border-radius: 16px;
    padding: 28px 32px;
    color: #ffffff;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25), 0 8px 10px -6px rgba(15, 23, 42, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.hero-title {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0 0 8px 0;
    color: #f8fafc;
}

.hero-subtitle {
    font-size: 15px;
    color: #cbd5e1;
    margin-bottom: 16px;
    line-height: 1.5;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 9999px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 500;
    color: #f1f5f9;
}

/* Callout Warning Banner trang nhã */
.scientific-notice {
    background: #f8fafc;
    border-left: 4px solid #0284c7;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 20px;
    font-size: 13.5px;
    color: #475569;
    border-top: 1px solid #e2e8f0;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
}

/* Thẻ Metric / KPI Cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin: 16px 0 24px 0;
}

.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
}

.metric-title {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748b;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 26px;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
}

.metric-caption {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 4px;
}

/* Thẻ kết quả Hero Result Card */
.result-hero-potable {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 1.5px solid #6ee7b7;
    border-radius: 14px;
    padding: 22px 26px;
    margin: 18px 0;
    color: #065f46;
}

.result-hero-non-potable {
    background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
    border: 1.5px solid #fda4af;
    border-radius: 14px;
    padding: 22px 26px;
    margin: 18px 0;
    color: #9f1239;
}

.result-hero-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.result-hero-desc {
    font-size: 14px;
    line-height: 1.5;
    opacity: 0.9;
}

/* Thanh đo độ tin cậy / xác suất (Confidence Meter) */
.confidence-container {
    margin-top: 14px;
    background: rgba(255, 255, 255, 0.7);
    border-radius: 8px;
    padding: 10px 14px;
    border: 1px solid rgba(0, 0, 0, 0.06);
}

.confidence-bar-bg {
    width: 100%;
    height: 10px;
    background: #e2e8f0;
    border-radius: 9999px;
    overflow: hidden;
    margin-top: 6px;
}

.confidence-bar-fill-potable {
    height: 100%;
    background: linear-gradient(90deg, #10b981 0%, #059669 100%);
    border-radius: 9999px;
    transition: width 0.4s ease;
}

.confidence-bar-fill-non-potable {
    height: 100%;
    background: linear-gradient(90deg, #f43f5e 0%, #e11d48 100%);
    border-radius: 9999px;
    transition: width 0.4s ease;
}

/* Thẻ phân nhóm chỉ tiêu hóa lý */
.feature-group-header {
    font-size: 14px;
    font-weight: 600;
    color: #0284c7;
    border-bottom: 2px solid #e0f2fe;
    padding-bottom: 6px;
    margin-bottom: 12px;
    margin-top: 14px;
}

/* Bảng hiển thị chuẩn cao cấp */
.custom-table-wrapper {
    overflow-x: auto;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
    margin: 16px 0;
    background: #ffffff;
}

.custom-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13.5px;
    text-align: left;
}

.custom-table th {
    background-color: #f8fafc;
    color: #475569;
    font-weight: 600;
    padding: 12px 16px;
    border-bottom: 1px solid #e2e8f0;
    white-space: nowrap;
}

.custom-table td {
    padding: 12px 16px;
    border-bottom: 1px solid #f1f5f9;
    color: #1e293b;
    vertical-align: middle;
}

.custom-table tr:last-child td {
    border-bottom: none;
}

.custom-table tr:hover td {
    background-color: #f8fafc;
}

/* Badges phân loại */
.badge-potable {
    display: inline-block;
    background-color: #d1fae5;
    color: #065f46;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid #a7f3d0;
}

.badge-non-potable {
    display: inline-block;
    background-color: #ffe4e6;
    color: #9f1239;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid #fecdd3;
}

.badge-error {
    display: inline-block;
    background-color: #fef3c7;
    color: #92400e;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 500;
    border: 1px solid #fde68a;
}

.badge-model-primary {
    display: inline-block;
    background-color: #e0f2fe;
    color: #0369a1;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid #bae6fd;
}

/* Thống kê đồng thuận Consensus Cards */
.consensus-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 16px 0;
}

/* Clean up Streamlit elements */
div[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
}
</style>
"""


def render_hero_header(title, subtitle, badges=None):
    """Tạo mã HTML hero header hiện đại với badge thông tin."""
    badge_items = badges or []
    badge_html = "".join(f'<span class="hero-badge">{b}</span>' for b in badge_items)
    return f"""
    <div class="hero-container">
        <h1 class="hero-title">{title}</h1>
        <div class="hero-subtitle">{subtitle}</div>
        <div class="badge-row">{badge_html}</div>
    </div>
    """


def render_metric_card(title, value, caption=""):
    """Tạo HTML cho một thẻ KPI."""
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-caption">{caption}</div>' if caption else ''}
    </div>
    """


def render_consensus_board(results):
    """Tính toán và hiển thị sự đồng thuận giữa 6 mô hình cho một mẫu đơn lẻ."""
    valid_predictions = [r for r in results if r.get("label") is not None]
    if not valid_predictions:
        return ""
    potable_count = sum(1 for r in valid_predictions if r["label"] == 1)
    non_potable_count = sum(1 for r in valid_predictions if r["label"] == 0)
    total_valid = len(valid_predictions)
    
    consensus_pct = (max(potable_count, non_potable_count) / total_valid) * 100 if total_valid > 0 else 0
    majority_text = "Uống được (1)" if potable_count > non_potable_count else "Không uống được (0)"
    if potable_count == non_potable_count:
        majority_text = "Tỷ lệ hòa (3 vs 3)"

    return f"""
    <div class="consensus-card">
        <div style="font-weight: 600; font-size: 15px; color: #0f172a; margin-bottom: 8px;">
            Đồng thuận giữa các mô hình học máy: <span style="color: #0284c7;">{majority_text}</span>
        </div>
        <div style="font-size: 13.5px; color: #475569;">
            Có <strong>{potable_count}/{total_valid}</strong> mô hình dự đoán <em>Uống được</em> và 
            <strong>{non_potable_count}/{total_valid}</strong> mô hình dự đoán <em>Không uống được</em> 
            (Mức độ đồng thuận: <strong>{consensus_pct:.1f}%</strong>).
        </div>
    </div>
    """
