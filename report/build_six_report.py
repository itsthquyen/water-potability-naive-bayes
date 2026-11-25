"""Tạo báo cáo theo DOCX mẫu từ số liệu sáu thuật toán đã kiểm chứng."""

import argparse
from copy import deepcopy
import csv
from fractions import Fraction
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import sys

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import FEATURE_COLUMNS, RAW_DATA_PATH
from src.data_validation import read_csv_records, validate_records
from src.eda import summarize_features
from src.six_experiments import ALGORITHMS, NAMES, RUN_ID

FOLDER = ROOT / "results" / "comparison" / RUN_ID


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def number(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}".replace(".", ",")


def percent(value):
    return "N/A" if value is None else number(value * 100, 2) + "%"


def font(run, size=13, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.bold, run.italic = bold, italic
    run.font.color.rgb = RGBColor(0, 0, 0)


def build(template, students, public=False):
    source_hash = sha256(template.read_bytes()).hexdigest()
    manifest = read(FOLDER / "manifest.json")
    for relative, expected in manifest["files"].items():
        if sha256((FOLDER / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum không khớp {relative}")
    summary, weka = read(FOLDER / "summary.json"), read(FOLDER / "weka_summary.json")
    params = read(FOLDER / "preprocessing.json")
    results = summary["models"]
    raw = read_csv_records(RAW_DATA_PATH, training=True)
    validated = validate_records(raw, training=True)
    X, y = validated["X"], validated["y"]
    dev_ids, test_ids = summary["development_ids"], summary["test_ids"]
    X_dev = [X[i] for i in dev_ids]
    stats = summarize_features(X_dev, FEATURE_COLUMNS)
    doc = Document(template)
    original_paragraphs = list(doc.paragraphs)
    cover = [deepcopy(p._p) for p in original_paragraphs[:44]]
    cover_section = deepcopy(doc.sections[0]._sectPr)
    for child in list(doc.element.body):
        doc.element.body.remove(child)
    for node in cover:
        for old_section in node.xpath("./w:pPr/w:sectPr"):
            old_section.getparent().remove(old_section)
        doc.element.body.append(node)
    doc.element.body.append(cover_section)
    replacements = {15: "Phân loại chất lượng nước uống được hay không uống được",
                    16: "bằng thuật toán Naive Bayes và so sánh năm thuật toán học máy",
                    25: "Nhóm sinh viên thực hiện: Nhóm 11 — 65CNTT",
                    26: "", 27: "", 28: ""}
    for i, (name, student_id) in enumerate(students, 26):
        replacements[i] = f"{i - 25}. {name} - {student_id}"
    for i, text in replacements.items():
        paragraph = doc.paragraphs[i]
        paragraph.clear()
        run = paragraph.add_run(text)
        font(run, bold=i in (15, 16))
    for index, paragraph in enumerate(doc.paragraphs):
        paragraph.paragraph_format.first_line_indent = Cm(0)
        if not paragraph.text.strip():
            spacing = 19 if 2 <= index <= 10 else 15 if 29 <= index <= 42 else 11
            paragraph.paragraph_format.line_spacing = Pt(spacing)
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(0)
    section = doc.add_section(WD_SECTION_START.NEW_PAGE)
    section.left_margin = Cm(2.5)
    section.right_margin = section.top_margin = section.bottom_margin = Cm(2)
    section.different_first_page_header_footer = False
    for numbering in section._sectPr.findall(qn("w:pgNumType")):
        section._sectPr.remove(numbering)
    for name, size in (("Normal", 13), ("Heading 1", 14), ("Heading 2", 13), ("Heading 3", 13)):
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = name != "Normal"
        style.paragraph_format.line_spacing = 1.08
        style.paragraph_format.space_after = Pt(5)
    doc.styles["Normal"].paragraph_format.first_line_indent = Cm(0.8)
    doc.core_properties.author = ", ".join(name for name, _ in students) or "Nhóm 11"
    doc.core_properties.title = "Phân loại chất lượng nước bằng Naive Bayes và so sánh năm thuật toán"
    doc.core_properties.subject = "Bài tập lớn Khai phá dữ liệu — Nhóm 11 — 65CNTT"
    doc.core_properties.comments = ""
    doc.core_properties.last_modified_by = "Nhóm 11"

    def para(text, bold=False):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        font(p.add_run(text), bold=bold)
        return p

    def heading(text, level=2, new_page=False):
        p = doc.add_paragraph(style=f"Heading {level}")
        p.paragraph_format.page_break_before = new_page
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.first_line_indent = Cm(0)
        font(p.add_run(text), 14 if level == 1 else 13, True)
        return p

    def pseudocode(lines):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.line_spacing = 1
        p.paragraph_format.space_before = Pt(5)
        p.paragraph_format.space_after = Pt(7)
        p.paragraph_format.keep_together = True
        run = p.add_run(lines.strip())
        run.font.name = "Consolas"
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(0, 0, 0)

    def page(title, chapter=None):
        if chapter:
            heading(chapter, 1, True)
            heading(title)
        else:
            heading(title, 2, True)

    def table(headers, rows, widths=None):
        t = doc.add_table(rows=1, cols=len(headers))
        t.autofit = False
        for index, label in enumerate(headers):
            t.rows[0].cells[index].text = str(label)
        for values in rows:
            cells = t.add_row().cells
            for i, value in enumerate(values):
                cells[i].text = str(value)
        for i, row in enumerate(t.rows):
            for j, cell in enumerate(row.cells):
                if widths:
                    cell.width = Cm(widths[j])
                properties = cell._tc.get_or_add_tcPr()
                margins = OxmlElement("w:tcMar")
                for side in ("top", "left", "bottom", "right"):
                    el = OxmlElement("w:" + side); el.set(qn("w:w"), "70"); el.set(qn("w:type"), "dxa"); margins.append(el)
                properties.append(margins)
                borders = OxmlElement("w:tcBorders")
                for side in ("top", "left", "bottom", "right"):
                    el = OxmlElement("w:" + side); el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "4"); el.set(qn("w:color"), "444444"); borders.append(el)
                properties.append(borders)
                for p in cell.paragraphs:
                    p.paragraph_format.first_line_indent = Cm(0)
                    p.paragraph_format.space_after = Pt(1)
                    p.paragraph_format.line_spacing = 1
                    for run in p.runs:
                        font(run, 10.5 if len(headers) > 5 else 11.5, i == 0)
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        if widths:
            for col, width in zip(t.columns, widths):
                col.width = Cm(width)
        doc.add_paragraph().paragraph_format.space_after = Pt(0)
        return t

    figure_index = 0

    def figure(path, caption, width=15.5, max_height=11):
        nonlocal figure_index
        from PIL import Image
        with Image.open(path) as im:
            width = min(width, max_height * im.width / im.height)
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.keep_with_next = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Cm(width))
        figure_index += 1
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        font(p.add_run(f"Hình {figure_index}. {caption}"), 11.5, italic=True)

    toc_title = para("MỤC LỤC", True)
    toc_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    toc_title.paragraph_format.first_line_indent = Cm(0)
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), 'TOC \\o "1-2" \\h \\z \\u')
    p._p.append(field)
    heading("LỜI MỞ ĐẦU", 1, True)
    para("Phân loại mẫu nước theo các chỉ tiêu hóa lý là bài toán học có giám sát với dữ liệu số, giá trị thiếu và hai lớp có số lượng khác nhau. Đề tài giúp nhóm vận dụng định lý Bayes để xây dựng một mô hình có thể giải thích bằng số đếm và kiểm tra từng bước tính toán.")
    para("Nhóm chọn Naive Bayes làm thuật toán chính. Từ các bài giảng phân lớp và giáo trình được cung cấp, nhóm lựa chọn ID3, C4.5, KNN và Logistic Regression để đối chiếu, đồng thời bổ sung SVM tuyến tính. Sáu thuật toán được thực nghiệm trên cùng các mẫu phát triển, năm fold và tập kiểm tra, nhằm làm rõ thế mạnh và giới hạn của từng cách biểu diễn dữ liệu.")
    para("Báo cáo trình bày dữ liệu thô, quy trình tiền xử lý trên Python và Weka, cơ sở lý thuyết, thực nghiệm có số liệu đánh giá và ứng dụng dự đoán. Phần Naive Bayes gồm công thức, ví dụ tính tay và một mẫu test được tính đủ chín thuộc tính. Các số liệu được lấy từ lần chạy đã lưu, kèm dự đoán từng mẫu để kiểm chứng.")
    para("Kết quả thực nghiệm cho thấy Naive Bayes đếm tần suất còn khó nhận diện lớp uống được. KNN có kết quả tốt nhất trong sáu cấu hình cố định trên tập test của bài tập. Nhóm phân tích sự khác nhau này, thay vì chỉ dựa vào accuracy để đánh giá chất lượng mô hình.")

    page("1. Lý do chọn đề tài", "I. Mô tả bài toán")
    para("Bài toán phân loại nước cho phép kết nối kiến thức khai phá dữ liệu với một tập dữ liệu có cấu trúc rõ ràng. Mỗi mẫu được mô tả bằng chín chỉ tiêu số; nhãn đích biểu thị khả năng uống được theo cách gán nhãn của nguồn công bố. Các thách thức cụ thể gồm dữ liệu thiếu, thang đo khác nhau, quan hệ giữa các thuộc tính và tỷ lệ lớp không cân bằng.")
    para("Naive Bayes phù hợp cho mục tiêu học tập vì các đại lượng trong mô hình có thể giải thích bằng tần suất. Sau khi chia mỗi chỉ tiêu thành khoảng, nhóm có thể đếm số mẫu của từng lớp, đếm mỗi giá trị thuộc tính trong lớp và nhân các xác suất có điều kiện để ra quyết định. Việc tự cài giúp phân biệt lỗi cài đặt với hạn chế của giả định mô hình.")
    heading("2. Tổng quan bài toán")
    para("Đầu vào là vector chín thuộc tính theo thứ tự ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes và Turbidity. Đầu ra là Potability, với lớp 0 mang nhãn không uống được và lớp 1 mang nhãn uống được. Nhãn thật chỉ tham gia huấn luyện và đánh giá, không tham gia vector đầu vào.")
    para("Mục tiêu chính là kiểm chứng bộ phân lớp Naive Bayes tự cài. Mục tiêu đối chiếu là xem năm cách phân lớp khác có cải thiện khả năng nhận diện lớp 1 và thứ hạng ROC-AUC hay không. Web cuối cùng sử dụng lại trạng thái đã học để dự đoán một mẫu hoặc nhiều mẫu từ CSV.")
    para("Trong phạm vi bài tập, dự đoán là dự đoán nhãn của bộ dữ liệu. CSV công bố không cung cấp đủ metadata về lấy mẫu và kiểm nghiệm để suy rộng thành chứng nhận chất lượng nước. Giới hạn này được giữ xuyên suốt khi diễn giải đầu ra của ứng dụng.")

    page("3. Quy trình thực hiện")
    table(["Bước", "Hoạt động", "Đầu ra"], [
        (1,"Tải dữ liệu gốc và kiểm checksum","CSV nguyên trạng, thông tin nguồn"),
        (2,"Audit, kiểm tra trùng và chia mẫu","Development/test và ID từng fold"),
        (3,"Phân tích và học tiền xử lý từ train","Bảng thống kê, trung vị, điểm cắt, mean/std"),
        (4,"Học sáu thuật toán và kiểm định chéo","Điểm trung bình và dao động năm fold"),
        (5,"Học lại development và đánh giá test","Dự đoán, ma trận, bảng so sánh, ROC"),
        (6,"Thực hành và đối chiếu Weka","Log, dự đoán và ảnh panel Weka"),
        (7,"Phát triển ứng dụng","Web dự đoán và xuất CSV")], [1.3,8,6])
    para("Các bước được sắp theo ranh giới dữ liệu. Trước hết chia mẫu để tránh cho giá trị ở test ảnh hưởng đến trung vị hoặc điểm cắt. Trong kiểm định chéo, mỗi lượt lại học tiền xử lý từ phần train của lượt đó. Sau kiểm định, mô hình cuối học từ toàn development và dùng một lần cấu hình cố định để đánh giá trên test.")
    para("Bộ dữ liệu kiểm tra đã xuất hiện trong phiên bản trước của đề tài. Do đó, báo cáo gọi đây là phép đối chiếu trên holdout lịch sử, không coi là kiểm chứng trên dữ liệu ngoài độc lập. Nhóm giữ cấu hình định trước và không thay seed hoặc chọn ngưỡng dựa vào kết quả test.")

    page("4. Phân tích dữ liệu thô")
    para("Nguồn là bộ Water Quality do Aditya Kadiwal công bố trên Kaggle, phiên bản 3 [4]. Nhóm tải lại từ đường dẫn của nhà phát hành ngày 24/11/2026 và đối chiếu nguyên byte với CSV đã lưu. Tệp có 3.276 mẫu và 10 cột, trong đó Potability là nhãn.")
    para("Dữ liệu thô trong dự án là bản CSV do nguồn công bố, trước khi áp dụng các bước điền thiếu, chuẩn hóa hoặc rời rạc hóa của nhóm. Bảng dưới trích các cột của năm dòng đầu; ô trống được ghi là thiếu. Các giá trị được làm tròn chỉ khi trình bày.")
    table(["Dòng", "pH", "Hardness", "Sulfate", "THM", "Nhãn"], [
        (i, *("Thiếu" if X[i][j] is None else number(X[i][j],2) for j in (0,1,4,7)),y[i]) for i in range(5)], [1.1,2.2,3.1,3.1,3.1,1.7])
    table(["Đại lượng", "Giá trị"],[("Số mẫu",3276),("Số thuộc tính đầu vào",9),("Số lớp",2),("Lớp 0",1998),("Lớp 1",1278),("pH thiếu",491),("Sulfate thiếu",781),("Trihalomethanes thiếu",162)], [8.5,6.8])
    para("Tỷ lệ lớp 0 là 60,99% và lớp 1 là 39,01% trên toàn CSV. Nếu chỉ đánh giá bằng accuracy, mô hình luôn đoán lớp 0 vẫn có điểm khá cao. Vì vậy, phần thực nghiệm bổ sung recall từng lớp, Balanced Accuracy, macro-F1 và ROC-AUC.")

    page("1.1. Kiểm tra dữ liệu trên Python", "II. Quy trình khai phá dữ liệu")
    para("Trước khi học, nhóm kiểm tra đủ tên cột, thứ tự cột và nhãn hợp lệ. Các giá trị số phải hữu hạn. Ô rỗng hoặc NaN được biểu diễn nội bộ bằng None để phân biệt với số 0 thực sự. Chuỗi không chuyển thành số được và số vô hạn bị báo lỗi, không tự biến thành ô thiếu.")
    para("Khóa nhận diện trùng gồm chín giá trị đầu vào sau khi chuẩn hóa kiểu biểu diễn thiếu. Nếu hai hàng có cùng đầu vào và cùng nhãn, giữ hàng đầu để tránh một mẫu xuất hiện ở cả train và test. Nếu trùng đầu vào nhưng nhãn khác nhau, dừng để kiểm tra mâu thuẫn. Với CSV hiện tại, không phát hiện dòng lỗi hoặc nhóm trùng nên giữ 3.276 mẫu.")
    table(["Bước", "Số mẫu trước", "Số mẫu sau"],[("Đọc CSV",3276,3276),("Kiểm tra kiểu và nhãn",3276,3276),("Kiểm tra trùng",3276,3276),("Chia development/test",3276,"2.620 / 656")], [7.3,4,4])
    para("Nhóm giữ cả chín chỉ tiêu vì chúng là các cột mô tả mẫu được nguồn cung cấp. Potability được tách ra làm nhãn. ID dòng chỉ dùng để nối kết quả với dữ liệu gốc. Không phát sinh một thuộc tính mới từ nhãn hoặc từ kết quả kiểm tra để tăng điểm dự đoán.")
    para("Đối với ngoại lệ, nhóm dùng IQR và histogram để mô tả, không xóa quan sát chỉ vì nằm xa trung vị. Thiếu thông tin quy trình đo khiến một số đo cực trị chưa đủ để kết luận là lỗi. Cách giữ lại này được áp dụng nhất quán cho mọi mô hình.")

    page("1.1.1. Điền thiếu và chia khoảng cho Naive Bayes")
    para("Với từng cột trong train, nhóm sắp tăng dần các ô quan sát. Nếu có n giá trị, trung vị là phần tử giữa khi n lẻ và trung bình hai phần tử giữa khi n chẵn. Các ô thiếu ở train, validation hoặc test đều dùng cùng trung vị đã học từ train tương ứng.")
    para("Để dùng công thức Naive Bayes đếm tần suất, nhóm chuyển các chỉ tiêu số sang tối đa ba khoảng. Hai vị trí chia là kết quả làm tròn lên của n/3 và 2n/3, trên các ô quan sát trước điền. Nếu hai giá trị kề vị trí chia khác nhau, lấy trung điểm làm ngưỡng. Nếu bằng nhau, bỏ điểm cắt để cùng một số luôn nhận cùng mã.")
    table(["Chỉ tiêu", "Trung vị", "Ngưỡng 1", "Ngưỡng 2"],[
        (name,number(median),*(number(cut) for cut in cuts)) for name,median,cuts in zip(FEATURE_COLUMNS,params["nominal"]["fill_values"],params["nominal"]["cut_points"])], [4.8,3.5,3.5,3.5])
    para("Với hai ngưỡng t1 < t2, mã 0 ứng với x ≤ t1, mã 1 ứng với t1 < x ≤ t2 và mã 2 ứng với x > t2. Cột hằng có một mã. Cột toàn thiếu trong train khiến lượt học dừng. ID3 và C4.5 tự cài cũng nhận cùng biểu diễn danh nghĩa này để việc so sánh ba mô hình kiểm soát được phần chia khoảng.")

    page("1.1.2. Chuẩn hóa cho các mô hình số")
    para("KNN tính khoảng cách giữa các mẫu nên chỉ tiêu có trị số lớn có thể chi phối khoảng cách nếu giữ nguyên thang. Logistic Regression và SVM tối ưu hệ số cũng chịu ảnh hưởng của thang đo. Vì vậy, ba mô hình này dùng dữ liệu điền trung vị rồi chuẩn hóa z-score.")
    para("Trên train đã điền thiếu, mean là tổng giá trị chia số mẫu. Phương sai tổng thể bằng tổng bình phương độ lệch so với mean chia số mẫu. Độ lệch chuẩn là căn bậc hai phương sai. Biến đổi z = (x − mean)/std. Cột có std bằng 0 dùng scale 1. Mã tự viết các bước cộng, chia và căn; không gọi hàm thống kê học máy có sẵn.")
    table(["Chỉ tiêu", "Mean train", "Std train"],[(name,number(mean),number(scale)) for name,mean,scale in zip(FEATURE_COLUMNS,params["numeric"]["means"],params["numeric"]["scales"])],[6.3,4.5,4.5])
    para("Ví dụ, một giá trị đúng mean của cột được đổi thành 0. Khi nhận mẫu mới ngoài min/max train, web vẫn tính được z-score và áp dụng mô hình, nhưng giá trị đó có thể thuộc vùng ít được đại diện trong dữ liệu học. Việc chuẩn hóa giữ thứ tự chín cột và không sử dụng nhãn.")
    para("Những thống kê trong bảng thuộc lần học cuối trên 2.620 mẫu development. Trong mỗi lượt CV, mean/std và trung vị được tính lại từ bốn fold train; bảng cuối không được dùng để chuẩn bị trước dữ liệu validation của các lượt đó.")

    for title, filename, caption, text in [
        ("1.2.1. Dữ liệu trước xử lý trên Weka", "01-weka-raw.png", "Sulfate trên development trước tiền xử lý trong Weka.",
         "Trong Weka Explorer, chọn Preprocess rồi Open file và mở development_raw.arff. Đặt Class là Potability và chọn Sulfate. Giao diện cho thấy 2.620 mẫu, 10 thuộc tính và 616 ô Sulfate thiếu. Các vạch histogram có màu theo lớp giúp quan sát phân bố trước khi điền.") ,
        ("1.2.2. Xử lý thiếu bằng Weka", "02-weka-mean.png", "Kết quả bộ lọc ReplaceMissingValues của Weka.",
         "Trong Preprocess, chọn Filter → unsupervised → attribute → ReplaceMissingValues rồi Apply trên development. Bộ lọc thực hiện điền giá trị số bằng trung bình học từ development. Sau bước này, Missing của Sulfate bằng 0. Kết quả được lưu thành weka_mean_filled.arff để đối chiếu."),
        ("1.2.3. Rời rạc hóa bằng Weka", "03-weka-discretize.png", "Sulfate sau bộ lọc Discretize của Weka.",
         "Từ dữ liệu development đã điền, chọn bộ lọc Discretize, đặt bins = 3 và useEqualFrequency = True rồi Apply. Weka đổi các cột số thành các mức danh nghĩa. Tệp weka_discretized.arff lưu trạng thái sau biến đổi. Nhóm đối chiếu kiểu thuộc tính, số khoảng và số mẫu để kiểm tra thao tác.")]:
        page(title)
        para(text)
        figure(FOLDER/"weka"/filename,caption)
        para("Phần này minh họa bộ lọc Weka thật [5]. ReplaceMissingValues của Weka dùng mean, còn pipeline Python của nhóm dùng median. Weka học ngưỡng sau bước điền, còn Python học điểm cắt từ các ô quan sát trước điền. Vì vậy hai trạng thái đã xử lý có thể khác nhau.")
        para("Các hình minh họa chỉ dùng development. Bảng đánh giá các bộ phân lớp Weka ở phần III dùng các ARFF theo cùng ID test và được biến đổi bằng tham số train của Python, nhằm kiểm soát khác biệt đầu vào. Nhóm không lấy tệp đã lọc toàn bộ CSV làm đầu vào cho cross-validation.")

    page("2.1. Thống kê và chất lượng dữ liệu")
    figure(FOLDER/"figures"/"class_and_missing.png", "Phân bố lớp và giá trị thiếu trên development.")
    table(["Cột", "Quan sát", "Thiếu", "Min", "Trung vị", "Max"],[(name,stats[name]["count"],stats[name]["missing"],number(stats[name]["min"],2),number(stats[name]["median"],2),number(stats[name]["max"],2)) for name in FEATURE_COLUMNS],[4,1.7,1.4,2.7,2.8,2.7])
    para("Sulfate thiếu nhiều nhất, tiếp đến pH và Trihalomethanes. Khi thay giá trị thiếu bằng trung vị, khoảng giữa của các cột này có thể tăng số mẫu đáng kể. Vì vậy, ba khoảng gần cân bằng theo ô quan sát không có nghĩa là số mẫu sau điền cân bằng hoàn toàn.")
    para("Bảng dùng các giá trị quan sát trong development trước điền. Min/max và tỷ lệ thiếu mô tả chất lượng dữ liệu, không được biến thành ngưỡng đạt chuẩn nước uống khi chưa có metadata đo và quy trình gán nhãn.")

    page("2.2. Phân bố các chỉ tiêu và tương quan")
    figure(FOLDER/"figures"/"histograms.png", "Histogram các chỉ tiêu số trên development.",max_height=10)
    para("Hình cho thấy các chỉ tiêu có thang đo và hình dạng phân bố khác nhau. Việc chia khoảng giảm số mức cần ước lượng trong Naive Bayes, nhưng làm mất sự khác nhau giữa các số đo cùng khoảng. Ngược lại, KNN và hai mô hình tuyến tính giữ độ lớn tương đối qua dữ liệu đã chuẩn hóa.")
    figure(FOLDER/"figures"/"correlation.png", "Tương quan Pearson tính theo các cặp ô quan sát.",width=10,max_height=7)
    para("Pearson được tính từ các cặp ô không thiếu. Một hệ số nhỏ chỉ mô tả liên hệ tuyến tính giữa hai cột; nó không chứng minh độc lập có điều kiện theo lớp. Do đó, biểu đồ tương quan hỗ trợ phân tích dữ liệu và không được dùng làm bằng chứng rằng giả định Naive Bayes hoàn toàn thỏa mãn.")

    page("3.1. Lựa chọn sáu thuật toán học máy")
    table(["Thuật toán", "Nguồn ưu tiên", "Vai trò"],[("Naive Bayes","Bài 5 trang 40–45; giáo trình 4.2","Chính"),("ID3","Bài 5 trang 25–33; giáo trình 4.3","Đối chiếu"),("C4.5","Bài 5 trang 34","Đối chiếu"),("KNN","Bài 5 trang 46–49; giáo trình 4.4","Đối chiếu"),("Logistic Regression","Giáo trình mục 4.5.4","Đối chiếu"),("SVM tuyến tính","Tài liệu học máy Cornell [3]","Bổ sung")],[4.5,7.3,3.5])
    para("Các bộ phân lớp được lựa chọn để có những cách suy luận khác nhau trên cùng nhãn nhị phân. Naive Bayes dùng xác suất sinh dữ liệu theo lớp. ID3 và C4.5 chia không gian theo thuộc tính. KNN quyết định theo các mẫu gần nhau. Logistic Regression và SVM tuyến tính xây dựng một mặt phân cách trong không gian đã chuẩn hóa.")
    para("Đề tài triển khai tường minh các công thức và quy tắc quyết định bằng Python. Các phần đọc file, trình bày đồ thị và giao diện sử dụng thư viện nền. Các mô hình Weka là phần tham chiếu riêng, có thể khác cách xử lý nhánh chưa thấy, cắt tỉa, tối ưu và điểm đầu ra. Báo cáo công bố khác biệt thay vì gộp kết quả của hai môi trường.")
    para("C4.5 tự cài trong bài tập tập trung vào lõi gain ratio trên dữ liệu danh nghĩa đã chia khoảng và giữ cây chưa cắt tỉa. Đây là phạm vi triển khai cụ thể. Bảng Weka dùng J48 chưa cắt tỉa để so sánh một cấu hình gần với phạm vi đó.")

    page("3.2.1. Cơ sở lý thuyết Naive Bayes")
    para("Gọi C là lớp, nhận 0 hoặc 1, và x = (x1,…,x9) là vector mã khoảng. Theo định lý Bayes, P(C=c | x) = P(C=c)P(x | C=c)/P(x), với P(x) > 0. Khi so sánh hai lớp, mẫu số P(x) chung có thể bỏ khỏi bước chọn nhãn.")
    para("Naive Bayes giả định các thuộc tính độc lập với nhau khi đã biết lớp [1, 2]. Khi đó, P(x | C=c) được thay bằng tích của chín xác suất P(Aj=xj | C=c). Điểm lớp là Sc = P(C=c) × tích_j P(Aj=xj | C=c). Mô hình chọn lớp có điểm lớn hơn và chuẩn hóa hai điểm để hiển thị hậu nghiệm.")
    table(["Đại lượng", "Ước lượng từ train"],[("N","Tổng số mẫu train"),("nc","Số mẫu thuộc lớp c"),("nc,j,v","Số mẫu lớp c có cột j mang mã v"),("P(C=c)","nc / N"),("P(Aj=v | C=c)","nc,j,v / nc"),("P(C=c | x)","Sc / (S0 + S1), khi tổng dương")],[5.4,9.9])
    para("Tần suất 0 được giữ nguyên. Nếu một thừa số của lớp bằng 0, điểm lớp đó bằng 0. Nếu cả hai điểm đều bằng 0, hậu nghiệm không xác định và chương trình báo lỗi dòng đó. Nếu hai điểm dương bằng nhau, quy ước chọn lớp 0. Trong lần chạy này, tất cả 656 mẫu test đều có thể dự đoán.")
    para("Naive Bayes cần ước lượng từng thuộc tính trong mỗi lớp, thay vì đếm tất cả tổ hợp chín thuộc tính cùng lúc. Điều này giúp giảm số lượng bảng tần suất cần học. Đổi lại, các quan hệ giữa thuộc tính trong cùng lớp và thông tin bị mất khi chia khoảng có thể làm điểm hậu nghiệm thiếu chính xác.")

    page("3.2.2. Xây dựng thuật toán Naive Bayes")
    para("Ở bước fit, mỗi hàng mã khoảng làm tăng nc của lớp đúng một lần. Sau đó chương trình đi qua chín cột và tăng ô đếm tương ứng [lớp][cột][mã]. Tổng đếm của mỗi cột trong một lớp phải bằng số mẫu lớp. Mã lưu trung vị, ngưỡng, thứ tự cột và bảng đếm cùng mô hình.")
    table(["Bước", "Nội dung thực hiện"],[(1,"Kiểm tra X, y và miền mã của từng cột"),(2,"Khởi tạo hai số đếm lớp và các ô tần suất"),(3,"Duyệt các hàng train, cập nhật lớp và từng thuộc tính"),(4,"Với mẫu mới, biến đổi bằng thống kê train đã lưu"),(5,"Nhân prior và chín thừa số của mỗi lớp"),(6,"So sánh hai điểm rồi chuẩn hóa hậu nghiệm")],[1.6,13.7])
    para("Để tránh nhân những số thực rất nhỏ ở bước so sánh, chương trình giữ Sc = ac/bc. Tử ac bắt đầu bằng nc rồi nhân các số đếm của mẫu. Mẫu bc bằng N × nc^d với d=9 và được tính một lần khi fit hoặc nạp model. So sánh a0b1 với a1b0 tương đương so sánh S0 với S1 vì các mẫu số đều dương.")
    para("Sau khi quy đồng, p1 = a1b0/(a0b1+a1b0) và p0 = 1 − p1 theo cùng công thức. Quy đồng là phương pháp số học để tính đúng mô hình đã chọn, giữ được nhãn khi các xác suất hiển thị bị làm tròn. JSON lưu số đếm, không lưu mô hình thư viện.")
    para("Độ phức tạp lập bảng đếm là O(Nd), sau chi phí sắp xếp học điểm cắt O(dN log N). Với hai lớp và ba mã mỗi cột, bảng đếm có tối đa 54 ô. Mỗi mẫu dự đoán đọc 18 ô đếm, vì phải xét chín thuộc tính ở hai lớp.")
    pseudocode("""for row, label in training:
    class_counts[label] += 1
    for column, value in enumerate(row):
        feature_counts[label][column][value] += 1
# Sau khi tinh hai diem bang so nguyen:
label = 1 if score_1 > score_0 else 0
probability_1 = score_1 / (score_0 + score_1)""")

    page("3.2.3. Ví dụ kiểm chứng tính tay")
    para("Xét dữ liệu tổng hợp hai thuộc tính A, B. Lớp 0 gồm (0,0), (0,1), (1,0), còn lớp 1 gồm (1,1), (1,0). Dự đoán x=(1,0). Ví dụ này dùng để kiểm tra công thức, tách khỏi số liệu thực nghiệm của dữ liệu nước.")
    table(["Thừa số", "Lớp 0", "Lớp 1"],[("Prior","3/5","2/5"),("P(A=1 | C)","1/3","2/2"),("P(B=0 | C)","2/3","1/2"),("Điểm S","2/15","1/5"),("Hậu nghiệm","0,4","0,6")],[5.3,5,5])
    para("Tổng hai điểm là 2/15 + 1/5 = 1/3. Do đó P(0|x)=(2/15)/(1/3)=0,4 và P(1|x)=(1/5)/(1/3)=0,6. Bộ phân lớp chọn lớp 1. Mẫu số của xác suất điều kiện ở từng lớp lần lượt là 3 và 2, không dùng tổng 5 làm mẫu số cho các xác suất này.")
    para("Nếu x=(0,1), lớp 1 có tần suất A=0 bằng 0 nên S1=0, trong khi S0 dương. Kết quả là lớp 0 với hậu nghiệm [1,0]. Trong một bộ dữ liệu mà mỗi lớp đều có ít nhất một thừa số bằng 0 cho cùng mẫu, tổng hai điểm bằng 0 và mẫu không có hậu nghiệm theo mô hình.")
    para("Các kiểm thử kiểm tra đáp án phân số độc lập, tổng đếm từng cột, cách xử lý hòa và tần suất 0. Thêm các trường hợp điểm cắt, giá trị thiếu, cột hằng và nạp lại mô hình để kiểm tra đường tính từ đầu vào đến dự đoán.")

    nb=read(FOLDER/"models"/"naive_bayes.json")
    row_id=27
    codes=[]
    for value,median,cuts in zip(X[row_id],nb["preprocessor"]["fill_values"],nb["preprocessor"]["cut_points"]):
        used=median if value is None else value
        codes.append(sum(used>cut for cut in cuts))
    counts=[[nb["model"]["feature_counts"][c][j][v] for j,v in enumerate(codes)] for c in (0,1)]
    totals=nb["model"]["class_counts"]
    scores=[Fraction(totals[c],sum(totals))*prod(Fraction(v,totals[c]) for v in counts[c]) for c in (0,1)]
    page("3.2.4. Đối chiếu một mẫu nước thực tế")
    para("Mẫu test có ID dòng 27 thiếu pH. Pipeline điền pH bằng trung vị 7,0402 từ development. Từng số đo được mã hóa bằng điểm cắt đã học, rồi tra bảng tần suất như dưới đây. Bảng làm tròn số đo để đọc; mô hình dùng giá trị đầy đủ.")
    table(["Chỉ tiêu","Giá trị dùng","Mã","Đếm lớp 0","Đếm lớp 1"],[(name,number(nb["preprocessor"]["fill_values"][j] if X[row_id][j] is None else X[row_id][j]),codes[j],counts[0][j],counts[1][j]) for j,name in enumerate(FEATURE_COLUMNS)],[4.4,3.5,1.3,3,3.1])
    para(f"Prior lớp 0 là {totals[0]}/2620 và prior lớp 1 là {totals[1]}/2620. Với từng lớp, nhân prior với chín phân số lấy số đếm ở bảng chia cho số mẫu lớp đó. Kết quả S0 ≈ {number(float(scores[0]),9)} và S1 ≈ {number(float(scores[1]),9)}.")
    para(f"Sau chuẩn hóa, p0 ≈ {number(float(scores[0]/sum(scores)),6)} và p1 ≈ {number(float(scores[1]/sum(scores)),6)}. Do p1 lớn hơn p0, mô hình dự đoán 1. Nhãn thật của dòng là 0 nên đây là FP. Ví dụ thể hiện một đường tính đúng vẫn có thể cho dự đoán sai do hạn chế của mô hình và dữ liệu.")

    page("3.3. Thuật toán ID3")
    para("ID3 xây dựng cây quyết định từ trên xuống, chọn thuộc tính có độ lợi thông tin lớn nhất tại mỗi nút [1, 2]. Với tỷ lệ lớp pc, entropy H(S)=−tổng_c pc log2(pc). Giá trị 0 không tham gia log. Nếu nút chỉ có một lớp thì entropy bằng 0.")
    para("Với thuộc tính A, chia S thành các nhóm Sv theo mã v. Entropy sau chia là tổng_v |Sv|/|S| × H(Sv). Gain(S,A)=H(S) trừ entropy sau chia. ID3 chọn A có gain lớn nhất; khi hòa chọn cột đứng trước để kết quả lặp lại được.")
    para("Trong ví dụ bốn mẫu (A,B,y)=(0,0,0),(0,1,0),(1,0,1),(1,1,1), entropy ban đầu bằng 1. Chia theo A tạo hai nút thuần nên entropy sau chia bằng 0 và gain bằng 1. Chia theo B tạo hai nhóm đều chứa cả hai lớp, gain bằng 0. Vì vậy A được chọn làm nút gốc.")
    para("Mã tự cài nhận cùng chín mã khoảng như Naive Bayes. Một thuộc tính được bỏ khỏi danh sách ở đường đi sau khi đã dùng. Cây dừng khi nút thuần, hết thuộc tính hoặc các thuộc tính còn lại không tạo được ít nhất hai nhóm. Lá lưu số đếm hai lớp để lấy nhãn đa số và điểm lớp 1.")
    para("Khi mẫu mới đi đến một mã chưa xuất hiện tại nút, bản Python dùng phân bố của nút đang xét. Cơ chế này giúp mọi mẫu có kết quả dự đoán nhưng có thể khác ID3 giáo dục của Weka. Phần III ghi nhận cụ thể sai khác và số mẫu Weka không gán được nhãn.")
    para("Cây chưa cắt tỉa có thể mô tả chi tiết các kết hợp mã trong train. Việc chia khoảng cũng làm mất chi tiết số đo. Vì vậy, cần đọc kết quả CV và test để đánh giá mức tổng quát hóa, thay vì coi cây học được nhiều nhánh là dấu hiệu chất lượng tốt hơn.")
    pseudocode("""grow(rows, attributes):
    if pure(rows) or attributes is empty: return leaf(counts)
    best = attribute_with_largest_information_gain(rows)
    for value in observed_values(best):
        child = grow(rows_with_value(best, value), attributes - best)
    return node(best, children, counts)""")

    page("3.4. Thuật toán C4.5 theo gain ratio")
    para("C4.5 phát triển từ cây quyết định dùng gain, bổ sung tỷ lệ độ lợi thông tin để giảm thiên lệch về phép chia có nhiều nhánh. SplitInfo(A)=−tổng_v (|Sv|/|S|)log2(|Sv|/|S|). GainRatio(A)=Gain(A)/SplitInfo(A), với SplitInfo dương.")
    para("Bản tự cài trước hết tính gain của các phép chia hợp lệ tại nút, rồi lấy gain trung bình. Chỉ xét thuộc tính có gain dương và không thấp hơn trung bình, sau đó chọn gain ratio lớn nhất. Điều kiện này tránh chọn một phép chia có SplitInfo rất nhỏ nhưng gain thực tế quá ít.")
    para("Với ví dụ ID3 ở trên, SplitInfo của A bằng 1 và gain bằng 1 nên gain ratio bằng 1. Thuộc tính B có gain 0 nên bị loại. Cả hai tiêu chí chọn A ở ví dụ đơn giản, nhưng trên dữ liệu có phân bố nhóm khác nhau chúng có thể chọn các cột khác nhau.")
    para("Phạm vi bài tập là cây danh nghĩa chưa cắt tỉa trên chín mã khoảng đã học từ train. Mỗi thuộc tính dùng một lần trên một đường đi, lá lưu phân bố lớp và nhánh chưa gặp dùng phân bố nút. Phạm vi này không triển khai toàn bộ xử lý thuộc tính liên tục, missing phân nhánh trọng số hoặc cắt tỉa của các bản C4.5 đầy đủ.")
    para("Weka dùng J48 chưa cắt tỉa, đặt minNumObj=1 và collapseTree=False trên cùng ARFF danh nghĩa. Dù gần với phạm vi đó, J48 vẫn có chi tiết lựa chọn phép chia và xử lý dự đoán riêng. Báo cáo giữ hai kết quả để nhìn rõ ảnh hưởng của cài đặt và cấu hình.")
    pseudocode("""for attribute in valid_splits:
    gain = entropy_before - weighted_entropy_after
    ratio = gain / split_information
average_gain = sum(gains) / number_of_valid_splits
eligible = [a for a in valid_splits if gain[a] > 0
            and gain[a] >= average_gain]
best = attribute_with_largest_ratio(eligible)""")

    page("3.5. Thuật toán KNN")
    para("KNN giữ các mẫu train đã điền trung vị và chuẩn hóa. Khi dự đoán, tính khoảng cách Euclid từ mẫu mới đến từng mẫu học, chọn k mẫu gần nhất và bỏ phiếu theo nhãn [1, 2]. Bài tập cố định k=5 và mọi láng giềng có cùng trọng số.")
    para("Bình phương khoảng cách d²(x,z)=tổng_j (xj−zj)² đủ để xếp hạng vì căn bậc hai bảo toàn thứ tự. Khi hòa khoảng cách, chương trình xét chỉ số mẫu train để kết quả xác định. Điểm lớp 1 là số phiếu lớp 1 chia k, còn nhãn là lớp có nhiều phiếu hơn.")
    para("Ví dụ một chiều đã chuẩn hóa gồm các điểm 0,2,5,8 với nhãn 0,1,1,0. Với mẫu 3 và k=3, các khoảng cách bình phương lần lượt là 9,1,4,25. Ba láng giềng gần nhất là 2,5,0, cho hai phiếu lớp 1 và một phiếu lớp 0. Mẫu được dự đoán là 1 với điểm 2/3.")
    para("Ưu điểm của KNN là có thể tận dụng cấu trúc cục bộ không tuyến tính. Hạn chế là mỗi lần dự đoán phải so với nhiều mẫu train và chất lượng phụ thuộc vào cách đo khoảng cách. Chuẩn hóa train-only giúp hạn chế cột Solids hoặc Conductivity chi phối các cột có trị số nhỏ hơn.")
    para("Thời gian học của KNN chủ yếu là lưu dữ liệu; thời gian dự đoán mới chứa phần tính khoảng cách. Vì vậy, khi so chi phí giữa các mô hình, nhóm ghi riêng fit và predict để tránh kết luận KNN nhanh chỉ từ thời gian fit.")
    pseudocode("""distances = []
for index, train_row in enumerate(train):
    d2 = sum((x[j] - train_row[j])**2 for j in columns)
    distances.append((d2, index))
neighbors = sorted(distances)[:5]
votes = sum(train_labels[i] for distance, i in neighbors)
return int(2 * votes > 5), votes / 5""")

    page("3.6. Logistic Regression")
    para("Logistic Regression mô hình hóa trực tiếp xác suất lớp 1 bằng sigmoid của tổ hợp tuyến tính. Đặt z=w·x+b và p1=1/(1+exp(−z)). Bản tự cài tính sigmoid theo hai phía của 0 để tránh tràn số khi z rất lớn hoặc rất nhỏ [3].")
    para("Hàm mục tiêu là cross-entropy trung bình cộng với L2: J=−mean[y log(p)+(1−y)log(1−p)] + λ||w||²/2. Gradient theo wj là mean[(p−y)xj]+λwj. Gradient theo b là mean[p−y]. Bias không bị phạt L2.")
    para("Mã khởi tạo w=0, b=0, tính gradient trên toàn bộ train ở mỗi epoch rồi cập nhật đồng thời w và b. Cấu hình định trước là 400 epoch, learning rate 0,1, λ=0,001. Ngưỡng nhãn là p1 > 0,5; hòa chọn 0. Không thử lại tham số dựa vào test.")
    para("Ví dụ hai mẫu x=−1,y=0 và x=1,y=1. Ở khởi tạo p=0,5 cho cả hai, gradient w bằng −0,5 và gradient b bằng 0. Nếu learning rate=1 và λ=0, sau một bước w=0,5,b=0. Đây là đáp án tính tay dùng trong kiểm thử bước cập nhật.")
    para("Trong kết quả hiện tại, Logistic dự đoán toàn bộ test là lớp 0. Điều đó phản ánh ngưỡng quyết định và khả năng tách tuyến tính của cấu hình đã dùng. Nó vẫn có các điểm xác suất khác nhau nên ROC-AUC không nhất thiết bằng 0,5. Báo cáo phân biệt rõ nhãn tại một ngưỡng với thứ hạng trên nhiều ngưỡng.")
    pseudocode("""for epoch in range(400):
    gradient_w, gradient_b = zeros(), 0
    for x, y in train:
        error = sigmoid(dot(w, x) + b) - y
        gradient_w += error * x / N
        gradient_b += error / N
    w -= 0.1 * (gradient_w + 0.001 * w)
    b -= 0.1 * gradient_b""")

    page("3.7. SVM tuyến tính")
    para("SVM tuyến tính tìm một hàm điểm f(x)=w·x+b và dùng nhãn mã hóa y' thuộc {−1,+1}. Hàm hinge loss là max(0,1−y'f(x)). Mục tiêu của bản tự cài là λ||w||²/2 + mean hinge loss, được tối ưu bằng SGD [3].")
    para("Ở một mẫu, trước hết tính margin=y'f(x). Mỗi bước co w theo hệ số 1−ηλ. Nếu margin < 1, cộng ηy'x vào w và cộng ηy' vào b. Nếu margin ≥ 1, chỉ còn bước co w. Bias không bị co. Chương trình xáo trộn train bằng seed 42 ở từng epoch.")
    para("Cấu hình cố định gồm 100 epoch, learning rate đầu 0,01, λ=0,01 và tốc độ giảm ηepoch=0,01/(1+0,1×epoch). Mẫu được gán lớp 1 khi f(x)>0, ngược lại là lớp 0. Điểm f(x) dùng để tính ROC-AUC và được hiển thị là biên quyết định trên web.")
    para("Ví dụ một mẫu x=2,y'=+1, khởi tạo w=b=0 và η=0,1. Margin=0 nên sau một bước w=0,2,b=0,1. Với x=2, điểm mới là 0,5. Đây là đáp án độc lập kiểm tra chiều cập nhật của hinge loss.")
    para("Bản Weka SMO dùng kernel tuyến tính và C=1, giải bài toán bằng cơ chế tối ưu khác. Giá trị C không được coi là tương đương trực tiếp với λ của SGD nếu chưa quy đổi theo cách chuẩn hóa hàm mục tiêu. Vì vậy bảng tham chiếu Weka có thể khác đáng kể về nhãn và accuracy.")
    pseudocode("""for epoch in range(100):
    shuffle(training_indices, seeded_generator)
    eta = 0.01 / (1 + 0.1 * epoch)
    for x, y in shuffled_train:
        sign = 2 * y - 1
        margin = sign * (dot(w, x) + b)
        w *= 1 - eta * 0.01
        if margin < 1:
            w += eta * sign * x
            b += eta * sign""")

    page("1. Thiết kế đánh giá", "III. Thực nghiệm và phân tích kết quả")
    table(["Phần dữ liệu","Lớp 0","Lớp 1","Tổng"],[("Development",1598,1022,2620),("Test",400,256,656),("Toàn bộ",1998,1278,3276)],[6.3,3,3,3])
    para("Sáu thuật toán dùng cùng danh sách ID của năm lượt kiểm định trong development. Mỗi mẫu development xuất hiện đúng một lần ở validation. Pipeline mới ở từng fold học lại tiền xử lý để ngăn giá trị validation đi vào thống kê train.")
    table(["Chỉ số","Cách tính"],[("Accuracy","(TP+TN)/N"),("Precision lớp 1","TP/(TP+FP)"),("Recall lớp 1","TP/(TP+FN)"),("F1 lớp 1","2TP/(2TP+FP+FN)"),("Balanced Accuracy","(Recall lớp 0 + Recall lớp 1)/2"),("Macro-F1","(F1 lớp 0 + F1 lớp 1)/2"),("ROC-AUC","Diện tích dưới đường TPR theo FPR")],[5.4,9.9])
    para("Nhãn 1 được chọn làm dương. Ma trận dùng hàng là nhãn thật, cột là nhãn dự đoán theo thứ tự [0,1]. Chỉ số có mẫu số 0 được ghi N/A. Baseline luôn chọn lớp đa số trong train chỉ là mốc tham chiếu thêm và không tính vào sáu thuật toán.")
    para("Các cấu hình cố định được công bố trước khi chạy. Những kết quả ở đây mô tả sáu cấu hình đã triển khai, không khẳng định thuật toán nào luôn tốt nhất sau khi tối ưu mọi tham số. Độ lệch chuẩn qua năm fold mô tả dao động của phép chia, không phải khoảng tin cậy.")

    page("2. Kết quả kiểm định chéo năm fold")
    table(["Thuật toán","Accuracy TB","BA TB ± SD","Macro-F1 TB","AUC TB"],[
        (NAMES[a],percent(results[a]["cv"]["accuracy"]["mean"]),number(results[a]["cv"]["balanced_accuracy"]["mean"])+" ± "+number(results[a]["cv"]["balanced_accuracy"]["std"]),number(results[a]["cv"]["macro_f1"]["mean"]),number(results[a]["cv"]["roc_auc"]["mean"])) for a in ALGORITHMS],[4.1,2.7,3.4,2.7,2.4])
    table(["Fold","BA Naive Bayes","BA KNN","AUC Naive Bayes","AUC KNN"],[(i+1,number(results["naive_bayes"]["folds"][i]["balanced_accuracy"]),number(results["knn"]["folds"][i]["balanced_accuracy"]),number(results["naive_bayes"]["folds"][i]["roc_auc"]),number(results["knn"]["folds"][i]["roc_auc"])) for i in range(5)],[1.3,3.5,3.5,3.5,3.5])
    para("KNN có BA trung bình cao nhất trong sáu cấu hình. ID3 và C4.5 cũng cao hơn NB theo BA ở CV, cho thấy việc kết hợp các mã bằng cây có thể tận dụng quan hệ mà tích xác suất độc lập bỏ qua. Tuy nhiên, mức BA của các phương pháp vẫn thấp, nên kết quả cần được đọc cùng test.")
    para("AUC trung bình của NB cao hơn BA của nó về trị số nhưng hai chỉ số có ý nghĩa khác nhau. AUC đo thứ hạng qua các mức điểm, còn BA đo nhãn ở ngưỡng hiện hành. Không so trực tiếp trị số AUC và BA để nói một chỉ số tốt hơn chỉ số kia.")

    page("3. Kết quả kiểm tra và so sánh sáu mô hình")
    table(["Thuật toán","Accuracy","BA","Macro-F1","Recall 1","AUC"],[
        (NAMES[a],percent(results[a]["test"]["accuracy"]),percent(results[a]["test"]["balanced_accuracy"]),number(results[a]["test"]["macro_f1"]),percent(results[a]["test"]["per_class"][1]["recall"]),number(results[a]["test"]["roc_auc"])) for a in ALGORITHMS],[4.1,2.3,2.3,2.3,2.3,2])
    figure(FOLDER/"figures"/"comparison.png", "Accuracy và Balanced Accuracy trên cùng 656 mẫu test.")
    para("KNN đạt accuracy 64,79%, cao hơn mốc đa số 60,98% khoảng 3,81 điểm phần trăm. BA của KNN là 60,72% và nhận đúng 108/256 mẫu lớp 1. NB đạt accuracy 60,37%, BA 51,61% và chỉ nhận đúng 30/256 mẫu lớp 1. Chênh lệch cho thấy cách khai thác lân cận hữu ích hơn công thức tần suất độc lập ở cấu hình này.")
    para("Logistic và SVM Python dự đoán toàn lớp 0 nên accuracy bằng mốc đa số. ID3 và C4.5 nhận ra nhiều mẫu lớp 1 hơn NB, nhưng tạo nhiều FP hơn, khiến accuracy thấp hơn. Kết luận cần dựa vào mục tiêu và đánh đổi giữa hai dạng lỗi.")

    page("4. Ma trận nhầm lẫn")
    figure(FOLDER/"figures"/"confusions.png", "Sáu ma trận nhầm lẫn, hàng là thật và cột là dự đoán.",max_height=13)
    para("Đối với Naive Bayes, TN=366, FP=34, FN=226, TP=30. Accuracy=(366+30)/656=60,37%. Con số 30/256 là recall của riêng lớp 1, không phải accuracy trên toàn test. Phần lớn dự đoán đúng của NB đến từ 366 mẫu lớp 0.")
    para("KNN chuyển một phần mẫu bị bỏ sót sang dự đoán đúng lớp 1, đồng thời tăng FP. ID3/C4.5 có xu hướng dự đoán lớp 1 thường xuyên hơn NB nên giảm FN nhưng tăng FP mạnh hơn. Hai mô hình tuyến tính Python có hàng dự đoán lớp 1 bằng 0 ở cấu hình này.")

    page("5. Đường ROC và điểm đầu ra")
    figure(FOLDER/"figures"/"roc_comparison.png", "ROC của sáu mô hình Python trên test.",max_height=12)
    para("Mã sắp điểm lớp 1 giảm dần, xử lý tất cả các mẫu đồng điểm cùng một bước và tính diện tích bằng quy tắc hình thang. Naive Bayes dùng hậu nghiệm; cây dùng tỷ lệ lớp ở lá; KNN dùng tỷ lệ phiếu; Logistic dùng sigmoid; SVM dùng f(x) trực tiếp.")
    para("KNN có AUC 0,6518, cao hơn NB 0,5218. AUC gần 0,5 của NB phản ánh khả năng xếp hạng còn yếu trên tập này. Logistic và SVM có điểm thay đổi giữa các mẫu nên AUC hơi khác 0,5 dù nhãn ở ngưỡng hiện tại đều bằng 0. Các điểm xác suất chưa được kiểm tra hiệu chỉnh nên không diễn giải là mức an toàn nước thực tế.")

    page("6. Phân tích kết quả của Naive Bayes")
    metric=results["naive_bayes"]["test"]
    table(["Cách đọc lỗi","Phép tính","Kết quả"],[("Mẫu lớp 0 bị gán nhầm 1","34/400","8,50%"),("Dự đoán 1 bị sai","34/64","53,125%"),("Mẫu lớp 1 bị bỏ sót","226/256","88,28%"),("Mẫu lớp 1 được nhận đúng","30/256","11,72%")],[7.3,4,4])
    para("Các mẫu số ở bảng trả lời những câu hỏi khác nhau. Trong 400 mẫu thật lớp 0, NB gán nhầm 34 mẫu thành 1. Nhưng trong 64 lần NB dự đoán lớp 1, có 34 lần sai. Vì vậy, cả khi dự đoán uống được, mô hình vẫn cần được diễn giải với mức chính xác dương thấp.")
    para("Có ba cơ chế có thể góp phần vào kết quả. Thứ nhất, mỗi khoảng số gom các giá trị khác nhau nên mất chi tiết có ích. Thứ hai, điền trung vị tạo thêm khối lượng ở khoảng giữa, nhất là Sulfate và pH. Thứ ba, giả định độc lập có điều kiện không mô hình hóa được tương tác giữa các chỉ tiêu. Đây là các giới hạn của thiết kế, chưa phải chứng minh nhân quả cho từng lỗi.")
    para("So sánh trên cùng biểu diễn danh nghĩa cho thấy cây có thể nhận ra thêm lớp 1, dù làm tăng FP. So với biểu diễn liên tục chuẩn hóa của KNN, NB còn mất thông tin trong khoảng. Vì vậy, chênh lệch KNN–NB bao gồm cả mô hình và biểu diễn tiền xử lý. Báo cáo công khai yếu tố này để tránh gán toàn bộ khác biệt cho một công thức học.")
    para("Một kết quả thấp vẫn có giá trị học thuật khi đường tính đúng và phân tích đầy đủ. Thực nghiệm không đặt mục tiêu đạt một mức accuracy bằng cách thay seed, xóa mẫu khó hoặc gộp thêm dữ liệu không rõ nguồn gốc.")

    for a in ALGORITHMS:
        page("7."+str(ALGORITHMS.index(a)+1)+". Kết quả "+NAMES[a]+" trên Weka")
        figure(FOLDER/"weka"/f"result-{a}.png", f"Panel Classify của Weka hiển thị kết quả {NAMES[a]}.",max_height=12)
        para(f"Tập học gồm 2.620 mẫu và test gồm 656 mẫu theo cùng ID của Python. Accuracy Weka là {percent(weka[a]['accuracy'])}. Có {weka[a]['unclassified']} mẫu không được gán nhãn. Kết quả được lấy từ lớp Evaluation của Weka và đối chiếu với dự đoán từng mẫu đã xuất.")
        explanations={
            "naive_bayes":"Bộ phân lớp DirectFrequencyNB được viết cho Weka để dùng đúng tần suất trực tiếp và không cộng giả mẫu. Đầu vào là các mã khoảng Python đã học trên development. Nhãn và hậu nghiệm của cả 656 mẫu khớp bản Python trong sai số số học đã kiểm tra. NB mặc định của Weka có cơ chế ước lượng khác nên không dùng thay số liệu này.",
            "id3":"ID3 lấy từ gói simpleEducationalLearningSchemes chính thức của Weka. Khi gặp nhánh chưa có mẫu học, phiên bản giáo dục có thể trả phân bố rỗng và không gán nhãn. Có 53 mẫu như vậy, gồm 28 mẫu lớp 0 và 25 mẫu lớp 1. Accuracy dùng toàn bộ 656 mẫu làm mẫu số, không bỏ các mẫu chưa gán để làm tăng điểm.",
            "c45":"Weka dùng J48 với cây chưa cắt tỉa, minNumObj=1 và collapseTree=False. Bản Python dùng gain ratio trên các thuộc tính đã mã hóa. Các chi tiết lựa chọn và dự đoán của J48 vẫn có thể khác. Điểm accuracy bằng nhau chưa chứng minh mọi nhãn hoặc mọi nhánh đều giống nhau.",
            "knn":"Weka dùng IBk k=5, khoảng cách Euclid và tắt normalize nội bộ vì đầu vào đã được chuẩn hóa bằng thống kê train. Nhãn trên test khớp Python. Weka có quy tắc trả phân bố và xử lý hòa khoảng cách riêng, nên điểm xác suất có thể khác dù nhãn và accuracy trùng.",
            "logistic":"Weka Logistic dùng ridge=0,001 và tối đa 400 lượt tối ưu. Cơ chế tối ưu và cách quy mô hóa penalty khác với gradient toàn tập của Python. Cả hai đều dự đoán toàn bộ test là lớp 0 ở cấu hình này. Không xem cùng giá trị ridge và λ là bằng chứng hai hàm mục tiêu hoàn toàn trùng.",
            "svm":"Weka SMO dùng kernel đa thức bậc 1 tương ứng kernel tuyến tính, C=1 và không chuẩn hóa nội bộ. Bản Python tối ưu hinge loss với SGD và λ=0,01. Hai cách đặt regularization và tối ưu không tương đương trực tiếp. Bản SMO chưa hiệu chỉnh xác suất nên điểm phân bố của nó không được so với biên Python như cùng một loại xác suất."
        }
        para(explanations[a])

    page("8. So sánh Python với Weka")
    table(["Thuật toán","Accuracy Python","Accuracy Weka","Chưa gán Weka"],[(NAMES[a],percent(results[a]["test"]["accuracy"]),percent(weka[a]["accuracy"]),weka[a]["unclassified"]) for a in ALGORITHMS],[4.5,3.9,3.9,3])
    para("NB có cùng công thức, cùng bảng biểu diễn và cho cùng dự đoán. KNN có cùng k, khoảng cách và mẫu học nên cho cùng nhãn. C4.5 và Logistic bằng accuracy trong lần chạy này, nhưng điều đó không đủ để khẳng định cài đặt hoặc toàn bộ điểm đầu ra trùng nhau.")
    para("ID3 Python dùng phân bố tại nút khi thiếu nhánh, còn Weka ID3 giáo dục không gán 53 mẫu. SVM khác đáng kể vì cách đặt regularization và tối ưu khác nhau. Các khác biệt có nguyên nhân được xác định trong cấu hình, không được che bằng cách trộn số liệu Weka vào bảng kết quả Python.")
    para("Các mô hình tham chiếu dùng ARFF đã được biến đổi từ development train của Python. Thí nghiệm bộ lọc Weka ở phần II có mục đích trình bày quá trình thao tác riêng. Sự tách bạch này cho phép đánh giá phần bộ phân lớp trong Weka mà không để khác biệt mean–median trở thành một yếu tố ẩn.")

    page("1. Mục tiêu và luồng sử dụng", "IV. Phát triển ứng dụng dự đoán")
    para("Ứng dụng Streamlit chạy cục bộ tại http://127.0.0.1:8501. Trang chính ghi rõ Naive Bayes là phương pháp chính và có ba tab cho nhập một mẫu, dự đoán CSV và xem kết quả thực nghiệm. Giao diện dùng tiếng Việt và giữ chín tên cột theo dữ liệu để tránh nhầm thứ tự.")
    para("Người dùng nhập các chỉ tiêu hoặc giữ ví dụ trung vị train đang hiển thị. Ô thiếu có thể để trống. Sau khi nhấn Dự đoán và so sánh, web kiểm dữ liệu, dùng lại tiền xử lý đã học và trả nhãn của NB trước. Bảng phía dưới hiển thị thêm năm phương pháp đối chiếu và loại điểm đầu ra của từng mô hình.")
    figure(FOLDER/"web-metrics.png", "Web đang chạy với tab kết quả thực nghiệm.",max_height=15)
    para("Các số liệu trong tab thực nghiệm được đọc từ lần chạy 20261124-six-v1 có manifest. Trang không huấn luyện khi người dùng bấm dự đoán. Điều này giữ kết quả có thể lặp lại và tránh làm đầu vào mới ảnh hưởng trạng thái đã học.")

    page("2. Kết quả dự đoán và xử lý dữ liệu CSV")
    figure(FOLDER/"web-prediction.png", "Kết quả dự đoán thật của sáu mô hình trên web.",max_height=15)
    para("Bảng phân biệt điểm xác suất ước lượng của NB, cây, KNN và Logistic với biên quyết định của SVM. Biên SVM có thể âm hoặc dương và không nằm trong [0,1]. Người dùng có thể tải kết quả CSV để giữ mã dòng, mã mẫu, nhãn, điểm và lỗi.")
    para("CSV phải có chín tên cột và mã hóa UTF-8. Cột sample_id được giữ để đối chiếu. Nếu có Potability trong tệp dự đoán, nhãn đó không đi vào vector đầu vào. Dòng lỗi vẫn có bản ghi lỗi theo từng mô hình và không được trình bày như một dự đoán hợp lệ.")

    page("3. Kiểm thử ứng dụng và khả năng mở rộng")
    table(["Tình huống kiểm tra","Kết quả mong đợi"],[("Nhập một mẫu hợp lệ","Hiển thị NB và đủ năm mô hình đối chiếu"),("CSV nhiều dòng","Giữ thứ tự, ID mẫu, kết quả đủ sáu mô hình"),("Có ô thiếu","Dùng trung vị train đã lưu"),("Chuỗi không phải số","Báo lỗi dòng và không tự gán nhãn"),("Tất cả chỉ tiêu trống","Từ chối dự đoán"),("Model bị thay đổi byte","Dừng nạp khi hash không khớp"),("Nhấn dự đoán nhiều lần","Dùng cùng trạng thái đã học")],[7.3,8])
    para("Ứng dụng đã được kiểm tra qua AppTest và qua giao diện trình duyệt đang chạy. Lượt nhập một mẫu hiển thị đủ sáu dòng, cho phép tải CSV và đối chiếu các số liệu trong tab thực nghiệm. Mã kiểm thử thuật toán gồm entropy, lựa chọn nút, khoảng cách, gradient Logistic một bước, cập nhật hinge loss và lưu/nạp pipeline.")
    para("Giới hạn hiện tại gồm nguồn dữ liệu chưa có metadata đo đầy đủ, một phép chia holdout lịch sử và các cấu hình cố định. Ứng dụng phù hợp để minh họa cách vận hành mô hình trong bài tập. Muốn sử dụng trên dữ liệu thực tế, cần kiểm chứng đầu vào, nhãn và kết quả trên một tập mẫu độc lập có nguồn gốc rõ ràng.")
    para("Các hướng phát triển hợp lý gồm thu thập dữ liệu có quy trình gán nhãn rõ ràng, kiểm tra biểu diễn thuộc tính bằng CV và đánh giá độ hiệu chỉnh xác suất. Mỗi thay đổi cần chốt trên development và giữ dữ liệu ngoài cho đánh giá cuối, thay vì điều chỉnh theo các mẫu đã xem.")

    heading("KẾT LUẬN",1,True)
    para("Nhóm đã thực hiện quy trình phân loại Potability từ CSV thô, phân tích dữ liệu, tiền xử lý trên Python và Weka, tự cài sáu thuật toán và xây dựng web dự đoán. Naive Bayes giữ vai trò chính, được giải thích bằng công thức, ví dụ nhỏ và bảng đếm đủ chín thuộc tính trên mẫu nước thật.")
    para("Trên cùng 656 mẫu kiểm tra, NB đạt accuracy 60,37%, BA 51,61% và ROC-AUC 0,5218. KNN đạt kết quả cao nhất trong sáu cấu hình với accuracy 64,79%, BA 60,72% và AUC 0,6518. Kết quả giúp làm rõ rằng một cài đặt đúng vẫn có thể phân loại kém nếu giả định và cách biểu diễn chưa phù hợp với cấu trúc dữ liệu.")
    para("Thực nghiệm Weka xác nhận công thức NB tự cài cho cùng dự đoán trên biểu diễn tương ứng. Đồng thời, ID3 và SVM minh họa ảnh hưởng của xử lý nhánh thiếu và cấu hình tối ưu đến kết quả. Các mẫu chưa gán nhãn của Weka được ghi đầy đủ và không bị bỏ để nâng chỉ số.")
    para("Nhóm kết luận rằng dữ liệu, biểu diễn, quy tắc đánh giá và cách đọc lỗi phải được xem cùng với accuracy. Phạm vi hiện tại là bài tập dự đoán nhãn theo bộ dữ liệu công bố, chưa đủ để kết luận an toàn nước ngoài thực tế.")
    heading("TÀI LIỆU THAM KHẢO",1,True)
    for text in [
        "[1] Trần Mạnh Tuấn. Bài giảng Khai phá dữ liệu, Bài 2 Tiền xử lý dữ liệu và Bài 5 Phân lớp dữ liệu. Khoa Công nghệ thông tin, Trường Đại học Thủy Lợi. Bản bài giảng được cung cấp trong học phần.",
        "[2] Trần Mạnh Tuấn, Hoàng Thị Minh Châu (chủ biên), Trần Thanh Đại, Vũ Mỹ Hạnh, Vũ Anh Tuấn. Giáo trình Khai phá dữ liệu. Trường Đại học Kinh tế - Kỹ thuật Công nghiệp, 2022. Chương 2 và 4, đặc biệt mục 4.2, 4.3, 4.4, 4.5.4 và 4.6.",
        "[3] Cornell University. CS 4780/5780 Machine Learning, Lecture notes on Logistic Regression and Support Vector Machines. https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/ . Truy cập 24/11/2026.",
        "[4] Aditya Kadiwal. Water Quality, Kaggle, phiên bản 3, water_potability.csv. https://www.kaggle.com/datasets/adityakadiwal/water-potability . Bản tải đối chiếu ngày 24/11/2026.",
        "[5] University of Waikato. Weka Knowledge Explorer và tài liệu API Weka 3.8 về J48, IBk, Logistic, SMO, ReplaceMissingValues và Discretize. https://ml.cms.waikato.ac.nz/weka/gui_explorer.html và https://weka.sourceforge.io/doc.stable-3-8/ . Truy cập 24/11/2026.",
        "[6] Weka team. simpleEducationalLearningSchemes 1.0.1, ID3 educational implementation. https://weka.sourceforge.io/packageMetaData/simpleEducationalLearningSchemes/1.0.1.html . Truy cập 24/11/2026."]:
        para(text)
    # Xóa quan hệ ảnh thân báo cáo mẫu đã được thay thế, giữ logo/bìa đang dùng.
    used_relationships=set(doc.element.xpath('//@r:embed')) | set(doc.element.xpath('//@r:id'))
    for identifier, rel in list(doc.part.rels.items()):
        if "image" in rel.reltype and identifier not in used_relationships:
            doc.part.drop_rel(identifier)
    output=ROOT/"output"/"docx"/("BaiTapLon-cong-khai.docx" if public else "BaiTapLon.docx")
    output.parent.mkdir(parents=True,exist_ok=True)
    doc.save(output)
    if sha256(template.read_bytes()).hexdigest()!=source_hash:
        raise ValueError("Tệp mẫu bị thay đổi.")
    print(output)
    return output


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template",type=Path,required=True)
    parser.add_argument("--students",type=Path,default=ROOT/"report"/"students.local.json")
    parser.add_argument("--public", action="store_true", help="Tạo bản công khai với bìa ghi tên nhóm.")
    args=parser.parse_args()
    build(args.template,[] if args.public else read(args.students),public=args.public)
