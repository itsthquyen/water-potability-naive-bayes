"""Dựng báo cáo tiếng Việt từ manifest, số liệu tự tính và ảnh panel Weka."""

import argparse
import csv
import json
from bisect import bisect_left
from fractions import Fraction
from functools import partial
from html import escape
from math import prod
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from xml.sax.saxutils import quoteattr

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable, Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from src.config import DEMO_RUN_ID
from src.evaluate import _confined_path
from src.pipeline import sha256_file

REPORT = ROOT / "report"
WEKA = REPORT / "weka" / "evidence"
OUTPUT_DIRECTORY = ROOT / "output" / "pdf"
RUN = DEMO_RUN_ID
METRIC_DIR = ROOT / "results" / "metrics" / RUN
GROUP_NAME = "Nhóm 11"
CLASS_NAME = "65CNTT"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


DATA = read_json(ROOT / "data" / "manifest.json")
MANIFEST = read_json(ROOT / "results" / "runs" / f"{RUN}.json")
EVIDENCE = read_json(WEKA / "evidence_manifest.json")
if (MANIFEST["stage"] != "evaluated" or MANIFEST["run_id"] != RUN
        or EVIDENCE["run_id"] != RUN or EVIDENCE["model_sha256"] != MANIFEST["model_sha256"]
        or EVIDENCE["csv_sha256"] != MANIFEST["csv_sha256"]):
    raise ValueError("Model, kết quả và ảnh Weka phải thuộc cùng lần chạy đã đánh giá.")
if sha256_file(ROOT / "data" / "manifest.json") != MANIFEST["data_manifest_sha256"]:
    raise ValueError("Manifest dữ liệu không khớp lần chạy.")
for relative_path, expected_hash in MANIFEST["files"].items():
    if sha256_file(_confined_path(ROOT, relative_path)) != expected_hash:
        raise ValueError(f"Tệp kết quả không khớp SHA-256: {relative_path}")
MODEL = read_json(ROOT / MANIFEST["model_path"])
DEV = read_json(ROOT / "results" / "figures" / RUN / "eda_manifest.json")["statistics"]
CV = read_json(METRIC_DIR / "cv_results.json")
TEST = read_json(METRIC_DIR / "test_metrics.json")
BASELINE = read_json(METRIC_DIR / "baseline_test_metrics.json")
ERROR = read_json(METRIC_DIR / "error_analysis.json")

INK = colors.black
BLUE = colors.HexColor("#10216e")
LINE = colors.HexColor("#343434")
PALE = colors.HexColor("#f7f7f7")
W, H = letter
LEFT = 79
RIGHT = 51
WIDTH = W - LEFT - RIGHT - 12

FONTS = Path("C:/Windows/Fonts")
for family, filename in (("TimesVN", "times.ttf"), ("TimesVNB", "timesbd.ttf"),
                         ("TimesVNI", "timesi.ttf"), ("TimesVNBI", "timesbi.ttf"),
                         ("Mono", "consola.ttf")):
    pdfmetrics.registerFont(TTFont(family, str(FONTS / filename)))
pdfmetrics.registerFontFamily("TimesVN", normal="TimesVN", bold="TimesVNB",
                              italic="TimesVNI", boldItalic="TimesVNBI")

body = ParagraphStyle("Body", fontName="TimesVN", fontSize=12, leading=15.5,
                      textColor=INK, alignment=TA_JUSTIFY, spaceAfter=9)
small = ParagraphStyle("Small", parent=body, fontSize=10.8, leading=14.4,
                       alignment=0, spaceAfter=6)
lead = ParagraphStyle("Lead", parent=body, fontSize=12, leading=16.5, spaceAfter=5)
chapter = ParagraphStyle("Chapter", parent=body, fontName="TimesVNB", fontSize=13,
                         leading=17, alignment=0, leftIndent=18, spaceAfter=8)
title_style = ParagraphStyle("Title", parent=body, fontName="TimesVNB", fontSize=12,
                             leading=16, alignment=0, leftIndent=18, spaceAfter=10)
center_title = ParagraphStyle("CenterTitle", parent=body, fontName="TimesVNB", fontSize=16,
                              leading=19, alignment=TA_CENTER, spaceBefore=5, spaceAfter=14)
formal_body = ParagraphStyle("FormalBody", parent=body, firstLineIndent=29)
sub = ParagraphStyle("Sub", parent=body, fontName="TimesVNB", fontSize=12,
                     leading=16, alignment=0, spaceBefore=3, spaceAfter=6)
caption = ParagraphStyle("Caption", parent=small, alignment=TA_CENTER,
                         fontName="TimesVNI", fontSize=11, spaceBefore=7, spaceAfter=11)
code_style = ParagraphStyle("Code", fontName="Mono", fontSize=9.2, leading=13.4,
                            textColor=INK)
cell_style = ParagraphStyle("Cell", parent=small, fontSize=11, leading=14.4, spaceAfter=0)
head_style = ParagraphStyle("CellHead", parent=cell_style, fontName="TimesVNB", alignment=TA_CENTER)


def p(text, style=body):
    return Paragraph(text, style)


def heading(text):
    return p(text, sub)


def table(headers, rows, fractions):
    formatted = [[p(escape(str(value)), head_style) for value in headers]]
    formatted.extend([p(escape(str(value)), cell_style) for value in row] for row in rows)
    result = Table(formatted, colWidths=[WIDTH * fraction for fraction in fractions],
                   repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), .45, LINE),
    ]))
    return result


def inset(text, style=small):
    block = Table([[p(text, style)]], colWidths=[WIDTH], hAlign="LEFT")
    block.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), .35, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return KeepTogether([block, Spacer(1, 10)])


def code(text):
    block = Table([[Preformatted(text.strip(), code_style)]], colWidths=[WIDTH], hAlign="LEFT")
    block.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return KeepTogether([block, Spacer(1, 10)])


def fig(filename, legend):
    path = WEKA / f"{filename}.png"
    if sha256_file(path) != EVIDENCE["files"].get(path.name):
        raise ValueError(f"Ảnh Weka không khớp manifest: {path.name}")
    pixels_x, pixels_y = ImageReader(str(path)).getSize()
    return KeepTogether([Image(str(path), width=WIDTH, height=WIDTH * pixels_y / pixels_x),
                         p(legend, caption)])


def dec(number, digits=4):
    return f"{number:.{digits}f}".replace(".", ",")


def pct(number, digits=2):
    return dec(100 * number, digits) + "%"


def count(number):
    return f"{number:,}".replace(",", ".")


def link(url, label):
    return f'<link href={quoteattr(url)} color="#10216e">{escape(label)}</link>'


pages = []


def page(section, title, *contents):
    section_names = {
        "I. MÔ TẢ BÀI TOÁN": "I. Mô tả bài toán",
        "II. QUY TRÌNH KHAI PHÁ DỮ LIỆU": "II. Quy trình khai phá dữ liệu",
        "III. PHƯƠNG PHÁP PHÂN LỚP": "III. Phương pháp phân lớp và thuật toán Naive Bayes",
        "IV. TRIỂN KHAI VÀ THỰC NGHIỆM": "IV. Triển khai Python và kết quả thực nghiệm",
    }
    pages.append((section_names.get(section, section), title, contents))


page("LỜI CẢM ƠN", "",
     p(f"{GROUP_NAME} xin cảm ơn giảng viên học phần Khai phá dữ liệu đã hướng dẫn nhóm vận dụng lý thuyết phân lớp và thiết kế thực nghiệm. Bài tập giúp nhóm thực hành từ kiểm tra dữ liệu, xây dựng bộ phân lớp đến đọc kết quả và xác định giới hạn của mô hình.", formal_body),
     p("Nhóm xin chân thành cảm ơn!", formal_body))

class_counts = MODEL["model"]["class_counts"]
dev_total = sum(class_counts)
test_counts = [sum(row) for row in TEST["confusion_matrix"]]
test_total = sum(test_counts)
data_total = DATA["raw_count"]

page("I. MÔ TẢ BÀI TOÁN", "1. Mục tiêu và phạm vi",
     p("Bài toán nhận chín chỉ tiêu đo của một mẫu nước và dự đoán nhãn <i>Potability</i>. Nhãn 0 là không uống được và nhãn 1 là uống được theo bộ dữ liệu. Đây là bài toán phân lớp có giám sát vì tập huấn luyện cung cấp cả số đo và nhãn."),
     p("Mục tiêu của nhóm là tự triển khai Naive Bayes cho các thuộc tính đã rời rạc hóa và đánh giá trên dữ liệu giữ riêng. Phương pháp đối chiếu là bộ dự đoán luôn chọn lớp chiếm đa số trong tập học. Hai câu hỏi thực nghiệm là mô hình có nhận ra lớp 1 tốt hơn mốc đối chiếu hay không và đổi lại có bao nhiêu mẫu lớp 0 bị gán nhầm sang lớp 1."),
     fig("01-data-and-class", f"Hình 1. Phân bố nhãn của {count(data_total)} mẫu trong Weka Explorer."),
     p(f"Mở <i>raw.arff</i> tại <i>Explorer → Preprocess</i> và chọn <i>Potability</i>. "
       f"Lớp 0 có {count(class_counts[0] + test_counts[0])} mẫu, chiếm "
       f"{pct((class_counts[0] + test_counts[0]) / data_total)}. Vì vậy, accuracy cao hơn 50% tự nó chưa chứng minh khả năng nhận diện cả hai lớp."))

missing = DATA["audit"]["missing_counts"]
dictionary = [
    ("ph", "Chỉ số pH"),
    ("Hardness", "Độ cứng"),
    ("Solids", "Tổng chất rắn hòa tan (TDS)"),
    ("Chloramines", "Hàm lượng chloramine"),
    ("Sulfate", "Hàm lượng sulfate"),
    ("Conductivity", "Độ dẫn điện"),
    ("Organic_carbon", "Carbon hữu cơ (TOC)"),
    ("Trihalomethanes", "Hàm lượng trihalomethane (THM)"),
    ("Turbidity", "Độ đục"),
]
page("I. MÔ TẢ BÀI TOÁN", "2. Nguồn và cấu trúc dữ liệu",
     p(f"Tệp <i>water_potability.csv</i> được lấy từ bộ <i>Water Quality</i> của Aditya Kadiwal trên Kaggle [3], phiên bản {DATA['source']['dataset_version']}. Bản dùng trong thực nghiệm có {count(data_total)} bản ghi, 9 cột số và 1 cột nhãn."),
     table(["Cột", "Ý nghĩa", "Ô thiếu", "Tỷ lệ thiếu"], [
         (name, meaning, count(missing[name]), pct(missing[name] / data_total))
         for name, meaning in dictionary
     ] + [("Potability", "Nhãn mục tiêu, nhận 0 hoặc 1", "0", "0,00%")], (.30, .38, .14, .18)),
     Spacer(1, 12),
     p("Tỷ lệ thiếu ở bảng tính riêng theo từng cột. Một mẫu có thể thiếu nhiều chỉ tiêu. Nhóm giữ nguyên số đo trong CSV và không suy diễn đơn vị hoặc ngưỡng kiểm nghiệm từ phần mô tả chung trên trang đăng dữ liệu."),
     p("Nguồn công bố xác định được tệp và phiên bản. Metadata của tệp chưa đủ để xác minh địa điểm, thời gian lấy mẫu hay quy trình gán nhãn. Vì vậy, kết quả trong báo cáo chỉ dùng để đánh giá việc dự đoán nhãn của bộ dữ liệu này."))

page("II. QUY TRÌNH KHAI PHÁ DỮ LIỆU", "1. Làm sạch và phân tách mẫu",
     p("Trước khi chia tập, mã kiểm tra tên cột, kiểu dữ liệu và nhãn. Ô rỗng hoặc giá trị NaN được đánh dấu thiếu. Chuỗi không đọc được thành số, số vô hạn hoặc nhãn ngoài 0/1 khiến bước kiểm tra dừng để sửa dữ liệu."),
     p("Khóa nhận diện trùng gồm chín giá trị đầu vào, kể cả vị trí bị thiếu. Nếu trùng cả đầu vào và nhãn, giữ bản ghi đầu tiên. Nếu trùng đầu vào nhưng khác nhãn, dừng để tránh tự chọn một nhãn. Sau audit, dữ liệu giữ nguyên 3.276 mẫu vì không phát hiện bản ghi lỗi hoặc trùng."),
     heading("Phép chia phân tầng"),
     p("Nhóm xáo trộn các chỉ số trong từng lớp với seed 42, rồi lấy cho tập test số mẫu bằng phần nguyên của 0,2 × số mẫu lớp + 0,5. Chỉ số của hai tập được kiểm tra không giao nhau. Với lớp 0, phép tính cho 400 mẫu test. Với lớp 1, phép tính cho 256 mẫu."),
     table(["Tập dữ liệu", "Lớp 0", "Lớp 1", "Tổng"], [
         ("Phát triển (development)", *map(count, class_counts), count(dev_total)),
         ("Kiểm tra (test)", *map(count, test_counts), count(test_total)),
         ("Toàn bộ", *(count(a + b) for a, b in zip(class_counts, test_counts)), count(data_total)),
     ], (.40, .20, .20, .20)),
     Spacer(1, 12),
     p("Tập phát triển dùng cho kiểm định chéo và lần huấn luyện cuối. Tập kiểm tra chỉ được biến đổi bằng trung vị và ngưỡng đã học từ tập phát triển. Giữ ranh giới này giúp số liệu test phản ánh dự đoán trên mẫu chưa dùng để học."))

page("II. QUY TRÌNH KHAI PHÁ DỮ LIỆU", "2. Thiết kế kiểm định chéo",
     p("Tập phát triển được chia phân tầng thành 5 fold. Mã xáo trộn từng lớp rồi phân phối lần lượt các mẫu vào 5 fold. Ở lượt k, fold k dùng để kiểm định và bốn fold còn lại dùng để học. Nhờ vậy, mỗi mẫu phát triển được đánh giá đúng một lần trong 5 lượt."),
     table(["Bước trong một lượt", "Dữ liệu sử dụng"], [
         ("Tính trung vị, ngưỡng và bảng đếm", "Chỉ dùng bốn fold huấn luyện."),
         ("Xác định lớp đa số để đối chiếu", "Dựa vào nhãn của cùng bốn fold huấn luyện."),
         ("Mã hóa mẫu kiểm định", "Dùng lại trung vị và ngưỡng của lượt đó."),
         ("Tính chỉ số", "Dùng nhãn thật của fold kiểm định sau khi có dự đoán."),
     ], (.51, .49)),
     Spacer(1, 12),
     p("Cấu hình thực nghiệm được cố định trước khi tính điểm gồm tối đa ba khoảng mỗi cột, điền thiếu bằng trung vị và ước lượng tần suất trực tiếp. Kiểm định chéo đo mức dao động theo cách chia mẫu. Sau 5 lượt, nhóm học lại một mô hình trên toàn bộ tập phát triển và dùng mô hình đó cho test."),
     p("Một cách làm sai là điền khuyết hoặc chia khoảng trên toàn bộ tập phát triển trước khi chia fold. Khi đó, giá trị của mẫu kiểm định đã ảnh hưởng đến trung vị và điểm cắt. Trong mã, mỗi lượt khởi tạo một pipeline mới để tránh sự phụ thuộc này."))

page("II. QUY TRÌNH KHAI PHÁ DỮ LIỆU", "3. Xác định các giá trị thiếu",
     p("Trong Weka Explorer, mở <i>development_raw.arff</i> ở thẻ Preprocess, chọn cột Sulfate và đặt Potability làm lớp. Trường <i>Missing</i> xác nhận 616 ô thiếu. Weka làm tròn 616/2.620 thành 24% trên giao diện."),
     fig("02-development-missing", "Hình 2. Sulfate trước khi điền khuyết trên tập phát triển."),
     table(["Chỉ tiêu", "Thiếu trên toàn CSV", "Thiếu trên tập phát triển"], [
         (name, count(missing[name]), count(DEV["features"][name]["missing"]))
         for name in ("ph", "Sulfate", "Trihalomethanes")
     ], (.28, .34, .38)),
     Spacer(1, 9),
     p("Nhóm dùng trung vị vì đây là một giá trị đại diện ít bị ảnh hưởng hơn trung bình khi xuất hiện số đo cực trị. Mỗi cột có trung vị riêng và chỉ dùng giá trị quan sát trong tập học để tính. Tỷ lệ thiếu mô tả CSV và không xác định được nguyên nhân thiếu."))

cut_points = MODEL["preprocessor"]["cut_points"]
fill_values = MODEL["preprocessor"]["fill_values"]
cut_table = [
    (name, dec(cuts[0]) if cuts else "—", dec(cuts[1]) if len(cuts) == 2 else "—", dec(median))
    for (name, _), cuts, median in zip(dictionary, cut_points, fill_values)
]
page("II. QUY TRÌNH KHAI PHÁ DỮ LIỆU", "4. Học trung vị và điểm cắt",
     p("Với một cột, sắp tăng dần n giá trị quan sát của phần huấn luyện. Nếu n lẻ, trung vị là phần tử giữa. Nếu n chẵn, lấy trung bình của hai phần tử giữa. Điểm cắt được học từ cùng dãy quan sát, trước khi bổ sung trung vị vào ô thiếu."),
     p("Làm tròn lên n/3 và 2n/3 để lấy hai vị trí k. Với chỉ số bắt đầu từ 1, ngưỡng là trung điểm của giá trị thứ k và k + 1 nếu k &lt; n và hai giá trị khác nhau. Nếu chúng bằng nhau, bỏ vị trí cắt để cùng một số đo luôn nhận cùng mã. Khi trung điểm của hai số thực kề nhau bị làm tròn lên cận trên, mã dùng cận dưới. Cột hằng có một khoảng, còn cột toàn thiếu khiến lượt huấn luyện dừng."),
     p("Ví dụ, với dãy quan sát 1, 2, 3, 4, 5, 6, trung vị là 3,5. Hai vị trí k là 2 và 4, nên các điểm cắt là 2,5 và 4,5. Một ô thiếu được điền 3,5 và nhận mã 1. Giá trị đúng 4,5 cũng nhận mã 1 vì điểm cắt thuộc khoảng dưới."),
     table(["Chỉ tiêu", "Ngưỡng 1", "Ngưỡng 2", "Trung vị"], cut_table, (.32, .23, .23, .22)),
     Spacer(1, 10),
     p("Bảng ghi tham số của lần học cuối trên tập phát triển. Mã dùng toàn bộ chữ số lưu trong mô hình. Các giá trị làm tròn ở bảng chỉ phục vụ trình bày."))

page("II. QUY TRÌNH KHAI PHÁ DỮ LIỆU", "5. Kiểm tra dữ liệu đã biến đổi bằng Weka",
     p("Khi biến đổi mẫu, Python điền giá trị thiếu bằng trung vị đã học rồi xác định mã theo điểm cắt. Với hai ngưỡng t₁ &lt; t₂, mã 0 ứng với x ≤ t₁, mã 1 ứng với t₁ &lt; x ≤ t₂ và mã 2 ứng với x &gt; t₂. Phép biến đổi giữ thứ tự chín cột. Mỗi mã đại diện một khoảng và được xử lý như giá trị danh nghĩa."),
     fig("03-development-processed", "Hình 3. Sulfate sau biến đổi trong Weka, dạng Nominal và Missing = 0."),
     p("Mở <i>development_processed.arff</i> và chọn lại Sulfate. So với Hình 2, Weka ghi nhận không còn ô thiếu và cột có ba mã danh nghĩa. Nhóm thực hiện điền và rời rạc hóa bằng Python, sau đó mở kết quả bằng Weka Explorer [4]."),
     p("Mục đích chia khoảng là dùng được công thức tần suất của thuộc tính rời rạc. Một đánh đổi là các số đo khác nhau trong cùng khoảng tạo ra cùng một thừa số xác suất. Sau khi điền trung vị, khoảng giữa cũng có thể nhận nhiều mẫu hơn hai khoảng còn lại."))

ph_cut_1, ph_cut_2 = cut_points[0]
ph_median = fill_values[0]
ph_counts = [MODEL["model"]["feature_counts"][label][0] for label in (0, 1)]
page("II. QUY TRÌNH KHAI PHÁ DỮ LIỆU", "6. Đối chiếu ba khoảng pH",
     p(f"Với pH, điểm cắt là {dec(ph_cut_1)} và {dec(ph_cut_2)}. Trung vị {dec(ph_median)} nằm trong khoảng 1. "
       "Mở <i>development_processed.arff</i> rồi chọn ph để đối chiếu ba mã khoảng theo màu của lớp."),
     fig("08-ph-processed", "Hình 4. Ba mã khoảng pH và phân bố nhãn trong Weka."),
     table(["Lớp", "Khoảng 0", "Khoảng 1", "Khoảng 2", "Tổng"], [
         (label, *ph_counts[label], class_counts[label]) for label in (0, 1)
     ], (.12, .22, .22, .22, .22)),
     Spacer(1, 9),
     p(f"Chẳng hạn, xác suất pH thuộc khoảng 1 trong lớp 1 là {ph_counts[1][1]}/{class_counts[1]} "
       f"≈ {dec(ph_counts[1][1] / class_counts[1])}. Mẫu số là toàn bộ mẫu huấn luyện mang nhãn 1. "
       "Bảng đếm được lập sau khi điền thiếu nên tổng theo ba khoảng bằng số mẫu của từng lớp."))

page("III. PHƯƠNG PHÁP PHÂN LỚP", "1. Định lý Bayes và giả định độc lập",
     p("Gọi C là nhãn lớp, c nhận giá trị 0 hoặc 1, và x = (x₁, …, x₉) là vector mã khoảng của một mẫu. Gọi N là số mẫu dùng để học, n<sub>c</sub> là số mẫu lớp c, và n<sub>c,j,v</sub> là số mẫu lớp c có thuộc tính j mang mã v."),
     inset("<b>Định lý Bayes</b><br/>"
           "P(C = c | x) = P(C = c) · P(x | C = c) / P(x), với P(x) &gt; 0.<br/>"
           "P(x) = P(C = 0) · P(x | C = 0) + P(C = 1) · P(x | C = 1).", lead),
     p("Trong định lý Bayes, mẫu số P(x) chung cho cả hai lớp. Muốn tìm lớp có hậu nghiệm lớn nhất (MAP), chỉ cần so sánh tử số của hai lớp. Giá trị P(x | C = c) vẫn khó ước lượng trực tiếp vì một tổ hợp chín mã cụ thể có thể ít xuất hiện."),
     p("Naive Bayes giả định các thuộc tính độc lập với nhau <b>khi đã biết lớp</b>. Nhờ đó, xác suất đồng thời của chín mã trong một lớp được biểu diễn bằng tích chín xác suất riêng. Công thức phân rã và cách đếm cho thuộc tính rời rạc được trình bày trong [1, tr. 77-79] và [2, mục 8.3]."),
     inset("<b>Điểm dùng để so sánh</b><br/>"
           "S<sub>c</sub>(x) = P(C = c) · ∏<sub>j=1..9</sub> P(A<sub>j</sub> = x<sub>j</sub> | C = c).<br/>"
           "P(C = c) = n<sub>c</sub> / N.<br/>"
           "P(A<sub>j</sub> = v | C = c) = n<sub>c,j,v</sub> / n<sub>c</sub>.", lead),
     p("Ví dụ, với pH đã mã hóa và lớp 1, tử số là số mẫu lớp 1 rơi vào đúng khoảng của mẫu đang xét. Mẫu số là số mẫu lớp 1. Nếu xét thuộc tính Sulfate trong cùng lớp, mẫu số vẫn giữ nguyên nhưng tử số được lấy ở cột khác."),
     p("Giả định độc lập có điều kiện là giả định của mô hình. Vì vậy, hậu nghiệm là điểm xác suất ước lượng từ dữ liệu và cách chia khoảng của nhóm."))

page("III. PHƯƠNG PHÁP PHÂN LỚP", "2. Học bảng đếm và kiểm chứng tính tay",
     p("Với mỗi hàng huấn luyện đã mã hóa, tăng n<sub>c</sub> đúng một lần. Sau đó, ở từng cột j, tăng ô n<sub>c,j,v</sub> ứng với mã v của hàng. Tổng số đếm qua mọi mã của một cột phải bằng n<sub>c</sub>. Tổng này là điều kiện kiểm tra khi nạp mô hình."),
     p("Xét ví dụ hai thuộc tính A, B mang mã 0 hoặc 1. Tập học lớp 0 gồm (0,0), (0,1), (1,0). Tập học lớp 1 gồm (1,1), (1,0). Cần dự đoán x = (1,0)."),
     table(["Thừa số", "Lớp 0", "Lớp 1"], [
         ("Prior P(C = c)", "3/5", "2/5"),
         ("P(A = 1 | C = c)", "1/3", "2/2"),
         ("P(B = 0 | C = c)", "2/3", "1/2"),
         ("Điểm S của lớp", "(3/5)(1/3)(2/3) = 2/15", "(2/5)(2/2)(1/2) = 1/5"),
     ], (.38, .31, .31)),
     Spacer(1, 12),
     inset("Tổng hai điểm = 2/15 + 1/5 = 1/3.<br/>"
           "P(C = 0 | x) = (2/15)/(1/3) = 0,4.<br/>"
           "P(C = 1 | x) = (1/5)/(1/3) = 0,6. Chọn lớp 1.", lead),
     p("Mã kiểm thử dùng các phân số trên làm đáp án độc lập. Ví dụ có số mẫu hai lớp khác nhau để kiểm tra cả xác suất tiên nghiệm và việc chia số đếm cho đúng số mẫu của từng lớp."),
     p("Với x = (0,1) trong cùng ví dụ, lớp 1 có P(A = 0 | C = 1) = 0 nên toàn bộ S₁ bằng 0. S₀ dương, do đó mô hình chọn lớp 0 và trả hậu nghiệm [1, 0]."))

page("III. PHƯƠNG PHÁP PHÂN LỚP", "3. Tính hậu nghiệm và quy tắc dự đoán",
     p("Mã lưu điểm S<sub>c</sub> dưới dạng phân số a<sub>c</sub>/b<sub>c</sub>. Tử a<sub>c</sub> bắt đầu bằng n<sub>c</sub> rồi nhân với số đếm ứng với từng mã của mẫu. Mẫu b<sub>c</sub> bằng N · n<sub>c</sub><super>d</super> với d là số thuộc tính, do có một mẫu số N ở prior và d mẫu số n<sub>c</sub> ở các xác suất có điều kiện. Mẫu số được tính một lần khi học hoặc nạp mô hình."),
     inset("T<sub>0</sub> = a<sub>0</sub> · b<sub>1</sub> và T<sub>1</sub> = a<sub>1</sub> · b<sub>0</sub>.<br/>"
           "p<sub>0</sub> = T<sub>0</sub> / (T<sub>0</sub> + T<sub>1</sub>) và p<sub>1</sub> = T<sub>1</sub> / (T<sub>0</sub> + T<sub>1</sub>).", lead),
     p("Vì b₀ và b₁ đều dương, so sánh hai tích chéo tương đương so sánh S₀ và S₁. Số nguyên Python giữ chính xác thứ tự trong bước so sánh. Chỉ phép chia cuối cùng để trình bày hậu nghiệm mới trả về số thực."),
     table(["Điều kiện", "Kết quả xử lý"], [
         ("T₁ > T₀", "Dự đoán lớp 1."),
         ("T₀ > T₁", "Dự đoán lớp 0."),
         ("T₀ = T₁ > 0", "Hậu nghiệm đều bằng 0,5. Quy ước chọn lớp 0."),
         ("T₀ = T₁ = 0", "Hậu nghiệm không xác định. Báo lỗi dòng dự đoán."),
     ], (.33, .67)),
     Spacer(1, 11),
     p("Các xác suất được ước lượng trực tiếp từ số đếm và giữ nguyên số đếm 0. Lỗi hai điểm bằng 0 được trả theo dòng khi dùng demo hoặc CSV. Trong thực nghiệm, lỗi như vậy sẽ làm dừng lượt đánh giá để tránh công bố chỉ số trên một tập mẫu đã bị bỏ bớt."),
     p("Với N mẫu và d cột, việc học ngưỡng tốn O(dN log N) do sắp xếp. Lập bảng đếm tốn O(Nd) lượt cập nhật. Với hai lớp và tối đa ba mã mỗi cột, bảng đếm có tối đa 6d ô. Mỗi dự đoán dùng 2d lượt tra bảng."))

case = next(item for item in ERROR["examples"] if item["error_type"] == "FP" and item["features"][0] is None)
case_codes = [
    bisect_left(cuts, median if value is None else value)
    for value, median, cuts in zip(case["features"], fill_values, cut_points)
]
case_counts = [
    [MODEL["model"]["feature_counts"][label][column][value]
     for column, value in enumerate(case_codes)]
    for label in (0, 1)
]
case_scores = [
    Fraction(class_counts[label], dev_total)
    * prod(Fraction(value, class_counts[label]) for value in case_counts[label])
    for label in (0, 1)
]
case_posteriors = [float(score / sum(case_scores)) for score in case_scores]
case_prediction = int(case_scores[1] > case_scores[0])
with (METRIC_DIR / "test_predictions.csv").open(encoding="utf-8", newline="") as handle:
    prediction_row = next(row for row in csv.DictReader(handle) if int(row["row_id"]) == case["row_id"])
if (case_prediction != case["predicted"] or case_prediction != int(prediction_row["predicted"])
        or case["actual"] != int(prediction_row["actual"])
        or case_posteriors != [float(prediction_row["probability_0"]), float(prediction_row["probability_1"])]):
    raise ValueError("Ví dụ tính tay không khớp bản ghi dự đoán đã lưu.")

page("III. PHƯƠNG PHÁP PHÂN LỚP", "4. Tính đủ chín thuộc tính trên một mẫu test",
     p(f"Mẫu có mã dòng {case['row_id']} thuộc tập test. Ô pH thiếu được điền bằng {dec(ph_median)} và nhận mã 1. "
       "Bảng cho thấy từng giá trị đem so với ngưỡng, mã thu được và số đếm tra từ mô hình cuối. Hai cột đếm lần lượt có mẫu số là 1.598 và 1.022 khi đổi sang xác suất có điều kiện."),
     table(["Chỉ tiêu", "Giá trị đem chia", "Mã", "Đếm lớp 0", "Đếm lớp 1"], [
         (name, dec(fill_values[j] if case["features"][j] is None else case["features"][j]),
          case_codes[j], case_counts[0][j], case_counts[1][j])
         for j, (name, _) in enumerate(dictionary)
     ], (.31, .25, .10, .17, .17)),
     Spacer(1, 11),
     p("Với mỗi lớp, lấy prior nhân với chín thừa số lấy từ bảng. Chẳng hạn, thừa số của pH "
       f"là {case_counts[0][0]}/{class_counts[0]} ở lớp 0 và {case_counts[1][0]}/{class_counts[1]} ở lớp 1."),
     inset(f"S₀ = ({class_counts[0]}/{dev_total}) · ∏ (số đếm lớp 0 / {class_counts[0]}) "
           f"≈ {dec(float(case_scores[0]), 9)}.<br/>"
           f"S₁ = ({class_counts[1]}/{dev_total}) · ∏ (số đếm lớp 1 / {class_counts[1]}) "
           f"≈ {dec(float(case_scores[1]), 9)}.<br/>"
           f"P(C = 0 | x) ≈ {dec(case_posteriors[0], 6)} và P(C = 1 | x) ≈ {dec(case_posteriors[1], 6)}.", lead),
     p(f"Vì S₁ &gt; S₀, dự đoán là {case_prediction}. Nhãn thật là {case['actual']}, nên đây là một dương tính giả (FP). "
       "Ví dụ cho phép đối chiếu toàn bộ đường tính từ ô thiếu đến nhãn và ma trận nhầm lẫn."))

page("IV. TRIỂN KHAI VÀ THỰC NGHIỆM", "1. Tổ chức mã và kiểm chứng",
     p("Các phần huấn luyện, kiểm định và dự đoán gọi chung một pipeline. Lúc học, pipeline rời rạc hóa dữ liệu rồi lập bảng đếm. Lúc dự đoán, pipeline dùng lại tham số rời rạc hóa và tra bảng đã học. Weka đọc dữ liệu và kết quả được xuất sang ARFF."),
     table(["Tệp Python", "Công việc chính"], [
         ("data_validation.py, preprocessing.py", "Kiểm tra dữ liệu, xử lý trùng và chia tập phân tầng."),
         ("discretization.py", "Học trung vị và ngưỡng. Dùng lại tham số khi mã hóa."),
         ("naive_bayes.py, pipeline.py", "Lập bảng đếm, quy đồng điểm, chọn nhãn và lưu mô hình."),
         ("experiments.py, metrics.py", "Chạy kiểm định chéo và tự tính các chỉ số đánh giá."),
         ("train.py, evaluate.py, predict.py", "Tổ chức các lượt học, đánh giá và dự đoán cho dữ liệu mới."),
     ], (.42, .58)),
     Spacer(1, 10),
     heading("Lõi đếm và tính điểm"),
     code("""
# Học bảng đếm từ mỗi dòng huấn luyện
class_counts[label] += 1
for column, value in enumerate(row):
    feature_counts[label][column][value] += 1

# Tử số điểm của một lớp khi dự đoán
numerator = class_counts[label]
for column, value in enumerate(row):
    numerator *= feature_counts[label][column][value]
"""),
     p("Khi nạp mô hình, mã kiểm tra số chiều, số đếm từng ô, tổng đếm của từng lớp và sự phù hợp giữa số khoảng với bảng đếm. Tập kiểm thử bao gồm ví dụ tính tay, điểm cắt, cột hằng, cả hai điểm bằng 0 và khả năng dự đoán sau khi nạp mô hình."),
     p("Một phép kiểm tra độc lập dùng phân số chính xác để tính lại trung vị, ngưỡng, bảng đếm và hậu nghiệm từ CSV gốc theo cùng danh sách chỉ số chia tập. Toàn bộ 656 nhãn và cặp xác suất kiểm tra khớp với tệp kết quả. Huấn luyện lại cũng tái tạo bảng đếm và số liệu kiểm định chéo."))

page("IV. TRIỂN KHAI VÀ THỰC NGHIỆM", "2. Các thước đo đánh giá",
     p("Nhãn 1 được chọn làm lớp dương. Hàng của ma trận nhầm lẫn là nhãn thật và cột là nhãn dự đoán. TP và TN là các dự đoán đúng ở hai lớp. FP là mẫu lớp 0 bị dự đoán thành 1, còn FN là mẫu lớp 1 bị dự đoán thành 0."),
     table(["Chỉ số", "Cách tính và ý nghĩa"], [
         ("Accuracy", "(TP + TN) / N. Đo tỷ lệ đúng trên toàn tập."),
         ("Precision lớp 1", "TP / (TP + FP). Đo tỷ lệ đúng trong dự đoán lớp 1."),
         ("Recall lớp 1", "TP / (TP + FN). Đo tỷ lệ mẫu lớp 1 được nhận ra."),
         ("Recall lớp 0", "TN / (TN + FP). Đo tỷ lệ mẫu lớp 0 được nhận ra."),
         ("F1 lớp 1", "2TP / (2TP + FP + FN). Kết hợp precision và recall."),
         ("Balanced Accuracy", "(Recall lớp 0 + Recall lớp 1) / 2."),
         ("Macro-F1", "(F1 lớp 0 + F1 lớp 1) / 2."),
         ("ROC-AUC", "Diện tích dưới đường TPR theo FPR khi thay đổi ngưỡng."),
     ], (.31, .69)),
     Spacer(1, 10),
     p("Balanced Accuracy và macro-F1 cho hai lớp trọng số ngang nhau. Accuracy chịu ảnh hưởng trực tiếp của tỷ lệ mỗi lớp. Mốc đa số không dự đoán lớp 1 nên precision lớp 1 không xác định, còn F1 lớp 1 bằng 0 vì tập test có mẫu dương bị bỏ sót."),
     p("Để tính ROC, mã sắp xác suất lớp 1 theo thứ tự giảm dần. Mỗi mức điểm bổ sung tất cả mẫu đồng điểm trước khi ghi một cặp FPR = FP/(FP + TN), TPR = TP/(TP + FN). AUC là tổng diện tích hình thang giữa các cặp điểm liên tiếp. Gộp mẫu đồng điểm giúp kết quả không phụ thuộc thứ tự của chúng."))

cv_bayes = CV["naive_bayes"]
cv_baseline = CV["baseline"]
page("IV. TRIỂN KHAI VÀ THỰC NGHIỆM", "3. Kết quả kiểm định chéo",
     p("Hai phương pháp sử dụng cùng năm fold. Trong mỗi lượt, mốc đối chiếu luôn chọn lớp nhiều mẫu hơn trong phần huấn luyện. Cột BA và macro-F1 dưới đây được tính riêng trên fold kiểm định của lượt đó."),
     table(["Fold", "BA Naive Bayes", "BA mốc đa số", "Macro-F1 NB", "Macro-F1 mốc đa số"], [
         (index, dec(nb["balanced_accuracy"]), dec(ref["balanced_accuracy"]),
          dec(nb["macro_f1"]), dec(ref["macro_f1"]))
         for index, (nb, ref) in enumerate(zip(cv_bayes["fold_metrics"], cv_baseline["fold_metrics"]), 1)
     ] + [("TB ± SD", *(
         f"{dec(summary[metric]['mean'])} ± {dec(summary[metric]['std'])}"
         for metric in ("balanced_accuracy", "macro_f1")
         for summary in (cv_bayes["summary"], cv_baseline["summary"])
     ))], (.13, .21, .21, .21, .24)),
     Spacer(1, 13),
     p(f"BA trung bình của Naive Bayes là {dec(cv_bayes['summary']['balanced_accuracy']['mean'])}, "
       f"so với {dec(cv_baseline['summary']['balanced_accuracy']['mean'])} của mốc đa số. "
       f"Macro-F1 trung bình lần lượt là {dec(cv_bayes['summary']['macro_f1']['mean'])} "
       f"và {dec(cv_baseline['summary']['macro_f1']['mean'])}. "
       "Việc tăng macro-F1 cho thấy mô hình có đóng góp ở lớp mà mốc đa số bỏ qua."),
     p("Trung bình tính ngang nhau cho năm fold. Độ lệch chuẩn dùng mẫu số 5 và mô tả độ dao động giữa các fold của phép chia hiện tại. Giá trị BA vẫn gần 0,5 nên mức phân biệt hai lớp còn hạn chế. Kết quả này cũng chưa đủ để kết luận chênh lệch có ý nghĩa thống kê."))

tn, fp = TEST["confusion_matrix"][0]
fn, tp = TEST["confusion_matrix"][1]
page("IV. TRIỂN KHAI VÀ THỰC NGHIỆM", "4. Kết quả trên tập kiểm tra",
     p(f"Sau khi học lại trên {count(dev_total)} mẫu phát triển, nhóm đánh giá mô hình trên {count(test_total)} mẫu test. "
       "Ma trận nhầm lẫn dưới đây tính từ một dự đoán cho mỗi mẫu."),
     table(["Nhãn thật", "Dự đoán 0", "Dự đoán 1", "Tổng"], [
         ("0", f"{tn} (TN)", f"{fp} (FP)", tn + fp),
         ("1", f"{fn} (FN)", f"{tp} (TP)", fn + tp),
         ("Tổng", tn + fn, fp + tp, test_total),
     ], (.29, .24, .24, .23)),
     Spacer(1, 12),
     table(["Chỉ số", "Mốc đa số", "Naive Bayes"], [
         ("Accuracy", pct(BASELINE["accuracy"]), pct(TEST["accuracy"])),
         ("Balanced Accuracy", pct(BASELINE["balanced_accuracy"]), pct(TEST["balanced_accuracy"])),
         ("Macro-F1", dec(BASELINE["macro_f1"]), dec(TEST["macro_f1"])),
         ("Precision lớp 1", "Không xác định", pct(TEST["per_class"][1]["precision"])),
         ("Recall lớp 1", pct(BASELINE["per_class"][1]["recall"]), pct(TEST["per_class"][1]["recall"])),
         ("ROC-AUC lớp 1", dec(BASELINE["roc_auc"]), dec(TEST["roc_auc"])),
     ], (.43, .25, .32)),
     Spacer(1, 11),
     p(f"Accuracy = ({tn} + {tp})/{test_total} = {pct(TEST['accuracy'])}. "
       f"BA = ({tn}/{tn + fp} + {tp}/{tp + fn})/2 = {pct(TEST['balanced_accuracy'])}. "
       f"Trong {fp + tp} dự đoán lớp 1, có {tp} dự đoán đúng, nên precision lớp 1 = {tp}/{fp + tp} "
       f"= {pct(TEST['per_class'][1]['precision'])}."),
     p(f"So với mốc đa số, Naive Bayes nhận đúng thêm {tp} mẫu lớp 1 nhưng làm sai thêm {fp} mẫu lớp 0. "
       f"Vì thế số dự đoán đúng giảm từ {BASELINE['confusion_matrix'][0][0]} xuống {tn + tp}. "
       "Balanced Accuracy tăng vì công thức cân bằng đóng góp của hai lớp."))

page("IV. TRIỂN KHAI VÀ THỰC NGHIỆM", "5. Đường ROC trên Weka",
     p("Trong Weka Visualize, mở <i>roc.arff</i> và chọn FPR ở trục ngang, TPR ở trục dọc. Mỗi điểm đến từ một mức xác suất lớp 1 trong tệp dự đoán. Hai đầu đường cong tương ứng với không chọn mẫu nào và chọn tất cả mẫu là dương."),
     fig("05-roc", f"Hình 5. Đường ROC trên {count(test_total)} mẫu. AUC = {dec(TEST['roc_auc'])}."),
     p(f"Mốc đa số gán cùng một điểm cho mọi mẫu nên AUC = {dec(BASELINE['roc_auc'])}. "
       f"Với AUC {dec(TEST['roc_auc'])}, Naive Bayes đạt {pct(TEST['roc_auc'])} tổng điểm tối đa khi so mọi cặp gồm "
       "một mẫu thật lớp 1 và một mẫu thật lớp 0. Một cặp được một điểm nếu mẫu lớp 1 xếp cao hơn, nửa điểm nếu hòa và 0 điểm nếu xếp thấp hơn."),
     p(f"Tại quy tắc dự đoán đã chốt, điểm vận hành có FPR = {fp}/{tn + fp} = {pct(fp / (tn + fp))} "
       f"và TPR = {tp}/{tp + fn} = {pct(tp / (tp + fn))}. "
       "AUC mô tả thứ hạng trên nhiều ngưỡng, còn điểm vận hành dùng để diễn giải số sai tại ngưỡng hiện hành."))

false_negative = next(item for item in ERROR["examples"] if item["error_type"] == "FN")
page("IV. TRIỂN KHAI VÀ THỰC NGHIỆM", "6. Phân tích lỗi dự đoán",
     p("Mở <i>errors.arff</i> bằng Weka Visualize. Trục ngang là xác suất lớp 1, trục dọc là pH gốc. Màu cho biết kết cục TN, FP, FN hoặc TP. Mẫu thiếu pH có thể vắng trên đồ thị do thiếu tọa độ dọc nhưng vẫn được tính trong ma trận."),
     fig("06-errors", "Hình 6. Kết cục dự đoán theo xác suất lớp 1 và pH gốc."),
     table(["Dạng lỗi", "Mẫu số dùng để diễn giải", "Tỷ lệ"], [
         ("FP", f"{fp} mẫu sai trên {tn + fp} mẫu thật lớp 0", pct(fp / (tn + fp))),
         ("FP", f"{fp} mẫu sai trên {fp + tp} dự đoán lớp 1", pct(fp / (fp + tp), 3)),
         ("FN", f"{fn} mẫu sai trên {fn + tp} mẫu thật lớp 1", pct(fn / (fn + tp))),
     ], (.18, .59, .23)),
     Spacer(1, 9),
     p(f"Mẫu {case['row_id']} ở mục III.4 là FP với p₁ = {dec(case_posteriors[1])}. "
       f"Mẫu {false_negative['row_id']} là FN với p₁ = {dec(false_negative['probability_1'])}. "
       "Hai trường hợp cho thấy lỗi xảy ra theo cả hai hướng. Đồ thị chỉ đặt một biến gốc cạnh đầu ra của mô hình dùng chín biến, nên vị trí của một điểm chưa cho biết biến nào gây lỗi."))

page("KẾT LUẬN", "",
     p("Nhóm đã xây dựng bộ phân lớp Naive Bayes cho nhãn Potability. Quy trình học trung vị và điểm cắt trên tập huấn luyện, đếm tần suất theo lớp và chọn lớp bằng so sánh hậu nghiệm. Ví dụ tính tay và phép tính độc lập trên toàn bộ test kiểm tra được đường đi từ dữ liệu đầu vào đến kết quả dự đoán.", formal_body),
     p(f"Trên {count(test_total)} mẫu test, Balanced Accuracy đạt {pct(TEST['balanced_accuracy'])} "
       f"và AUC đạt {dec(TEST['roc_auc'])}. Accuracy {pct(TEST['accuracy'])} thấp hơn "
       f"mốc đa số {pct(BASELINE['accuracy'])}. Mô hình nhận ra {tp} trong {tp + fn} mẫu lớp 1 "
       f"và gán sai {fp} mẫu lớp 0 sang lớp 1. Vì vậy, khả năng phân biệt hai nhãn của cấu hình này còn hạn chế.", formal_body),
     p("Các giới hạn chính là giả định độc lập có điều kiện, độ chi tiết bị mất khi chia khoảng và thiếu thông tin về quy trình gán nhãn. Các chỉ số thu được từ một phép chia của cùng bộ dữ liệu. Chúng mô tả chất lượng dự đoán nhãn trong bài tập, chưa đủ để kết luận về chất lượng nước ngoài thực tế.", formal_body),
     Spacer(1, 14),
     p("TÀI LIỆU THAM KHẢO", center_title),
     p("[1] Trần Mạnh Tuấn, Hoàng Thị Minh Châu (chủ biên), Trần Thanh Đại, Vũ Mỹ Hạnh và Vũ Anh Tuấn. "
       "<i>Giáo trình Khai phá dữ liệu</i>. Khoa Công nghệ thông tin, Trường Đại học Kinh tế - Kỹ thuật Công nghiệp, Hà Nội, 2022. "
       "Chương 4, mục 4.2.1-4.2.3, tr. 77-80.", small),
     p(f"[2] Jiawei Han, Micheline Kamber và Jian Pei. {link('https://www.sciencedirect.com/science/article/pii/B9780123814791000083', 'Data Mining: Concepts and Techniques')}. "
       "Ấn bản 3. Morgan Kaufmann, 2012. Chương 8, mục 8.3 <i>Bayes Classification Methods</i>.", small),
     p(f"[3] Aditya Kadiwal. {link(DATA['source']['source_url'], 'Water Quality')}. Kaggle (adityakadiwal/water-potability), phiên bản {DATA['source']['dataset_version']}, "
       "tệp <i>water_potability.csv</i>. Nguồn dữ liệu thực nghiệm.", small),
     p(f"[4] University of Waikato. {link('https://ml.cms.waikato.ac.nz/weka/gui_explorer.html', 'Weka Knowledge Explorer')}. "
       "Tài liệu chính thức về các thẻ Preprocess và Visualize.", small))

class ContentsLine(Flowable):
    def __init__(self, label, number, indent=0):
        super().__init__()
        self.label = label
        self.number = number
        self.indent = indent

    def wrap(self, available_width, _available_height):
        self.width = available_width
        self.height = 16
        return self.width, self.height

    def draw(self):
        canvas = self.canv
        font = "TimesVN"
        size = 11.6
        number = str(self.number)
        canvas.setFont(font, size)
        canvas.drawString(self.indent, 4, self.label)
        canvas.drawRightString(self.width, 4, number)
        start = self.indent + pdfmetrics.stringWidth(self.label, font, size) + 4
        end = self.width - pdfmetrics.stringWidth(number, font, size) - 4
        if start < end:
            canvas.setLineWidth(.45)
            canvas.setDash(.6, 1.6)
            canvas.line(start, 4.8, end, 4.8)
        canvas.linkRect("", f"p{self.number}", (self.indent, 1, self.width, self.height), relative=1)


def draw_cover(canvas, students):
    left, right, bottom, top = 64, 576, 38, 752
    center = (left + right) / 2
    canvas.setStrokeColor(BLUE)
    for inset, stroke in ((0, 1.1), (3.4, 3.6), (7, .65)):
        canvas.setLineWidth(stroke)
        canvas.rect(left + inset, bottom + inset, right - left - 2 * inset,
                    top - bottom - 2 * inset)
    canvas.setLineWidth(.6)
    for x in (left + 8, right - 20):
        for y in (bottom + 8, top - 20):
            canvas.rect(x, y, 12, 12)
            canvas.rect(x + 2, y + 2, 8, 8)
            canvas.rect(x + 4, y + 4, 4, 4)
    canvas.setFillColor(INK)
    canvas.setFont("TimesVNB", 14)
    canvas.drawCentredString(center, 722, "TRƯỜNG ĐẠI HỌC THỦY LỢI")
    canvas.drawCentredString(center, 695.5, "KHOA CÔNG NGHỆ THÔNG TIN")
    canvas.drawImage(str(REPORT / "tlu-emblem.png"), center - 76, 552, width=152,
                     height=118, mask="auto")
    canvas.drawCentredString(center, 526, "BÀI TẬP LỚN KHAI PHÁ DỮ LIỆU")
    canvas.setFont("TimesVNBI", 13)
    canvas.drawString(85, 475.5, "Đề tài:")
    canvas.setFont("TimesVNB", 13)
    canvas.drawCentredString(center, 448, "PHÂN LOẠI CHẤT LƯỢNG NƯỚC UỐNG ĐƯỢC HAY KHÔNG UỐNG ĐƯỢC")
    canvas.drawCentredString(center, 429, "BẰNG THUẬT TOÁN NAIVE BAYES")
    canvas.drawString(312, 342, f"Sinh viên thực hiện: {GROUP_NAME}")
    canvas.setFont("TimesVN", 13)
    canvas.drawString(312, 324, f"Lớp: {CLASS_NAME}")
    for index, (name, student_id) in enumerate(students, start=1):
        canvas.drawString(312, 320 - 17 * index, f"{index}. {name} - {student_id}")
    canvas.setFont("TimesVNB", 13)
    canvas.drawCentredString(center, 63.1, "HÀ NỘI, 2026")


def draw_page(canvas, doc, students):
    canvas.setTitle("Bài tập lớn Khai phá dữ liệu - Naive Bayes")
    canvas.setAuthor(", ".join(name for name, _ in students) if students else GROUP_NAME)
    canvas.setSubject("Phân loại nhãn Potability; thực nghiệm và hình ảnh Weka")
    canvas.bookmarkPage(f"p{doc.page}")
    if doc.page == 1:
        draw_cover(canvas, students)
        return
    left, right = LEFT + 6, W - RIGHT - 6
    canvas.setFillColor(INK)
    canvas.setFont("TimesVNI", 10)
    if doc.page % 2 == 0:
        canvas.drawString(left, 745, "Khai phá dữ liệu")
    else:
        canvas.drawRightString(right, 745, GROUP_NAME)
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(.55)
    canvas.line(left, 740.8, right, 740.8)
    canvas.setFont("TimesVN", 11)
    canvas.drawRightString(right, 52.3, str(doc.page))


def build(output, students):
    output.parent.mkdir(parents=True, exist_ok=True)
    story = [Spacer(1, 1), PageBreak(), p("MỤC LỤC", center_title)]
    previous_section = ""
    for number, (section, title, _) in enumerate(pages, start=3):
        if section != previous_section:
            story.append(ContentsLine(section, number))
            previous_section = section
        if title:
            prefix = title.split(maxsplit=1)[0].rstrip(".")
            story.append(ContentsLine(title, number, indent=34 if "." in prefix else 18))
    story.append(ContentsLine("TÀI LIỆU THAM KHẢO", len(pages) + 2))
    previous_section = ""
    for section, title, contents in pages:
        story.append(PageBreak())
        if section != previous_section:
            story.append(p(escape(section), chapter if title else center_title))
            previous_section = section
        if title:
            story.append(p(escape(title), title_style))
        story.extend(contents)
    expected_pages = len(pages) + 2
    draw = partial(draw_page, students=students)
    with TemporaryDirectory(prefix="pdf-build-", dir=output.parent) as temp_dir:
        temp_pdf = Path(temp_dir) / output.name
        doc = SimpleDocTemplate(str(temp_pdf), pagesize=letter, leftMargin=LEFT, rightMargin=RIGHT,
                                topMargin=65, bottomMargin=68)
        doc.build(story, onFirstPage=draw, onLaterPages=draw)
        if doc.page != expected_pages:
            raise RuntimeError(f"Sai số trang dự kiến: {doc.page}, cần {expected_pages}.")
        temp_pdf.replace(output)
    print(f"{output}: {doc.page} pages")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--students", type=Path, help="Tệp JSON cục bộ gồm các cặp [họ tên, MSSV].")
    args = parser.parse_args()
    if args.students is None:
        build(OUTPUT_DIRECTORY / "BaiTapLon-cong-khai.pdf", [])
        return
    try:
        students = read_json(args.students)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not isinstance(students, list) or not students or any(
        not isinstance(row, list) or len(row) != 2
        or not isinstance(row[0], str) or not row[0].strip()
        or not isinstance(row[1], str) or not row[1].isascii() or not row[1].isdigit()
        for row in students
    ):
        parser.error("Danh sách thành viên phải gồm các cặp [họ tên, MSSV].")
    build(OUTPUT_DIRECTORY / "BaiTapLon.pdf", students)


if __name__ == "__main__":
    main()
