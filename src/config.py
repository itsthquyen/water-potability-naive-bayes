"""Cấu hình chung của đồ án; đường dẫn tính từ gốc project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]
CLASS_LABELS = [0, 1]
TARGET_COLUMN = "Potability"
SEED = 42
TEST_FRACTION = 0.2
N_FOLDS = 5
DEFAULT_VAR_SMOOTHING = 1e-9
VARIANCE_FLOOR = 1e-12
MISSING_WARNING_COUNT = 3
SMOOTHING_VALUES = [10.0 ** exponent for exponent in range(-12, -5)]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "water_potability.csv"
