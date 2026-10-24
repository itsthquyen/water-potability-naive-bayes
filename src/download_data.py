"""Tải bản CSV công khai từ đúng nguồn Kaggle, giữ nguyên bytes gốc."""

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

from src.config import PROJECT_ROOT, RAW_DATA_PATH

SOURCE_URL = "https://www.kaggle.com/datasets/adityakadiwal/water-potability"
METADATA_URL = "https://www.kaggle.com/api/v1/datasets/list?search=water-potability&user=adityakadiwal"
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/adityakadiwal/water-potability?datasetVersionNumber=3"


def download_dataset(destination=RAW_DATA_PATH):
    """Tải version 3 sau khi kiểm tra nguồn/CC0; không ghi đè CSV đã có.

    Trả provenance JSON chứa URL, ngày UTC, version, license và SHA256.
    Lỗi mạng/ZIP/metadata được báo thật, không thay nguồn bằng mirror.
    """
    destination = Path(destination)
    if destination.exists():
        raise ValueError("CSV đã tồn tại; không ghi đè dữ liệu gốc.")
    with urlopen(Request(METADATA_URL, headers={"User-Agent": "water-potability-academic/1.0"}), timeout=60) as response:
        metadata = json.load(response)
    matches = [item for item in metadata if item.get("ref") == "adityakadiwal/water-potability"]
    if len(matches) != 1 or matches[0].get("licenseName") != "CC0: Public Domain":
        raise ValueError("Không xác minh được nguồn và giấy phép CC0; dừng tải.")
    with urlopen(Request(DOWNLOAD_URL, headers={"User-Agent": "water-potability-academic/1.0"}), timeout=120) as response:
        archive_bytes = response.read()
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = [name for name in archive.namelist() if Path(name).name == "water_potability.csv"]
        if len(names) != 1:
            raise ValueError("ZIP không chứa đúng một water_potability.csv.")
        csv_bytes = archive.read(names[0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(csv_bytes)
    provenance = {
        "source_url": SOURCE_URL, "download_url": DOWNLOAD_URL,
        "metadata_url": METADATA_URL, "dataset_version": 3,
        "license": matches[0]["licenseName"],
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "origin_limitations": "Trang tổng hợp chưa xác minh được địa điểm/thời gian đo hoặc quy trình tạo nhãn; không coi là dữ liệu Việt Nam.",
    }
    (PROJECT_ROOT / "data" / "source.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return provenance


def main():
    """CLI tải dữ liệu; cần truy cập HTTPS, không cần lưu token trong project."""
    parser = argparse.ArgumentParser(description="Tải dữ liệu Kaggle công khai version 3")
    parser.parse_args()
    try:
        print(json.dumps(download_dataset(), ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Lỗi tải dữ liệu: {error}\n")


if __name__ == "__main__":
    main()
