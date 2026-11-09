"""Cấu hình chung của đồ án; đường dẫn tính từ gốc project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]
CLASS_LABELS = [0, 1]
POTABILITY_LABEL_NAMES = {0: "không uống được", 1: "uống được"}
TARGET_COLUMN = "Potability"
SEED = 42
TEST_FRACTION = 0.2
N_FOLDS = 5
N_BINS = 3  # Chia theo vị trí trên train; bỏ điểm cắt nếu hai giá trị kề trùng.
MODEL_ALGORITHM = "naive_bayes_frequency"
MISSING_WARNING_COUNT = 3
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "water_potability.csv"
DEMO_RUN_ID = "20261109-NB-final"  # Bản dựng báo cáo và demo dùng cùng lần chạy.
