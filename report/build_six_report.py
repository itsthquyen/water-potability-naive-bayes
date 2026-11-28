"""Tạo báo cáo theo DOCX mẫu từ số liệu sáu thuật toán đã kiểm chứng."""

import argparse
from copy import deepcopy
import csv
from fractions import Fraction
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import re
import sys

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT, WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import FEATURE_COLUMNS, RAW_DATA_PATH
from src.data_processing.data_validation import read_csv_records, validate_records
from src.eda import summarize_features
from src.six_experiments import ALGORITHMS, NAMES, RUN_ID

FOLDER = ROOT / "results" / "comparison" / RUN_ID
EVIDENCE = ROOT / "report" / "evidence" / "20261128"
WEB_EVIDENCE = ROOT / "report" / "evidence" / "20261128-c45-full"
REPORT_TITLE = ("Phân loại chất lượng nước uống được hay không uống được "
                "bằng thuật toán Naive Bayes và so sánh với các thuật toán khác")
LECTURER = "TS. Trần Mạnh Tuấn"
TABLE_CAPTIONS = (
    "Các bước thực hiện đề tài",
    "Trích năm mẫu đầu của dữ liệu thô",
    "Quy mô dữ liệu và số ô thiếu trước tiền xử lý",
    "Số mẫu qua các bước kiểm tra và chia dữ liệu",
    "Trung vị và ngưỡng chia khoảng học từ development",
    "Trung bình và độ lệch chuẩn học từ development",
    "Thống kê các chỉ tiêu trên development",
    "Nguồn tài liệu và vai trò của sáu thuật toán",
    "Các đại lượng trong công thức Naive Bayes",
    "Trình tự huấn luyện và dự đoán bằng Naive Bayes",
    "Ví dụ tính tay xác suất cho hai thuộc tính",
    "Bảng đếm Naive Bayes cho mẫu có ID dòng 27",
    "Phân bố hai lớp trong development và test",
    "Các chỉ số đánh giá mô hình phân loại",
    "Kết quả kiểm định chéo năm fold của sáu thuật toán",
    "Balanced Accuracy và AUC của Naive Bayes và KNN theo fold",
    "So sánh sáu mô hình Python trên cùng tập test",
    "Phân tích các dạng sai số của Naive Bayes",
    "Đối chiếu kết quả Python và Weka",
    "Kiểm chứng C4.5 giữa Python và J48 theo tập dữ liệu",
    "Các tình huống kiểm thử ứng dụng dự đoán",
    "Trích dữ liệu thô với đầy đủ chín chỉ tiêu và nhãn",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def number(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}".replace(".", ",")


def percent(value):
    return "N/A" if value is None else number(value * 100, 2) + "%"


def font(run, size=13, bold=False, italic=False):
    run.font.name = "Times New Roman"
    for script in ("ascii", "hAnsi", "eastAsia", "cs"):
        run._r.get_or_add_rPr().rFonts.set(qn("w:" + script), "Times New Roman")
    run.font.size = Pt(size)
    run.bold, run.italic = bold, italic
    run.font.color.rgb = RGBColor(0, 0, 0)


def build(template, students, public=False):
    source_hash = sha256(template.read_bytes()).hexdigest()
    manifest = read(FOLDER / "manifest.json")
    for relative, expected in manifest["files"].items():
        if sha256((FOLDER / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum không khớp {relative}")
    for evidence_folder in (EVIDENCE, WEB_EVIDENCE):
        for filename, expected in read(evidence_folder / "manifest.json")["files"].items():
            if sha256((evidence_folder / filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Ảnh minh chứng bị thay đổi: {filename}")
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
    # Giữ nguyên nhóm hình khung bìa và biểu trưng của tài liệu mẫu.
    cover = deepcopy(doc.paragraphs[0]._p)
    cover_section = deepcopy(doc.sections[0]._sectPr)
    for child in list(doc.element.body):
        doc.element.body.remove(child)
    for old_section in cover.xpath("./w:pPr/w:sectPr"):
        old_section.getparent().remove(old_section)
    doc.element.body.append(cover)
    doc.element.body.append(cover_section)

    def cover_paragraph(text, size=13, bold=False, before=0, after=0, left=0, style=None):
        p = doc.add_paragraph(style=style)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if left else WD_ALIGN_PARAGRAPH.CENTER
        fmt = p.paragraph_format
        fmt.left_indent, fmt.right_indent, fmt.first_line_indent = Cm(left), Cm(0), Cm(0)
        fmt.space_before, fmt.space_after = Pt(before), Pt(after)
        fmt.line_spacing = 1.15
        fmt.keep_together = True
        font(p.add_run(text), size, bold)
        return p

    school = doc.paragraphs[0]
    for run in school.runs:
        if run.text:
            run.text = ""
    school.alignment = WD_ALIGN_PARAGRAPH.CENTER
    school.paragraph_format.left_indent = Cm(0)
    school.paragraph_format.right_indent = Cm(0)
    school.paragraph_format.first_line_indent = Cm(0)
    school.paragraph_format.space_before = Pt(0)
    school.paragraph_format.space_after = Pt(3)
    school.paragraph_format.line_spacing = 1.15
    font(school.add_run("TRƯỜNG ĐẠI HỌC THỦY LỢI"), 14, True)
    cover_paragraph("KHOA CÔNG NGHỆ THÔNG TIN", 14, True)
    cover_paragraph("BÀI TẬP LỚN", 16, True, before=175)
    cover_paragraph("HỌC PHẦN KHAI PHÁ DỮ LIỆU", 13, True, before=5)
    cover_paragraph("Đề tài", 13, True, before=14, after=7)
    cover_paragraph(REPORT_TITLE.replace(" bằng thuật toán", "\nbằng thuật toán", 1),
                    14, True, style="Title")
    cover_paragraph("Giảng viên hướng dẫn: " + LECTURER, before=36, left=2.4)
    cover_paragraph("Nhóm thực hiện: Nhóm 11", bold=True, before=12, left=2.4)
    cover_paragraph("Lớp: 65CNTT", after=6, left=2.4)
    for index, (name, student_id) in enumerate(students, 1):
        cover_paragraph(f"{index}. {name} - {student_id}", before=3, left=2.4)
    cover_paragraph("HÀ NỘI, 2026", 13, True, before=170 if students else 222)
    section = doc.add_section(WD_SECTION_START.NEW_PAGE)
    section.left_margin = Cm(2.5)
    section.right_margin = section.top_margin = section.bottom_margin = Cm(2)
    section.different_first_page_header_footer = False
    numbering = section._sectPr.find(qn("w:pgNumType"))
    if numbering is None:
        numbering = OxmlElement("w:pgNumType")
        section._sectPr.append(numbering)
    numbering.set(qn("w:start"), "1")
    for name, size in (("Normal", 13), ("Heading 1", 14), ("Heading 2", 13),
                       ("Heading 3", 13), ("Heading 4", 13), ("Title", 14), ("Caption", 11.5)):
        style = (doc.styles[name] if name in doc.styles
                 else doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH))
        if name == "Caption":
            style.base_style = doc.styles["Normal"]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = name not in ("Normal", "Caption")
        style.font.italic = name in ("Heading 4", "Caption")
        style.paragraph_format.line_spacing = 1.15
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.widow_control = True
        fonts = style.element.get_or_add_rPr().rFonts
        for attribute in list(fonts.attrib):
            if "theme" in attribute.lower():
                del fonts.attrib[attribute]
        for script in ("ascii", "hAnsi", "eastAsia", "cs"):
            fonts.set(qn("w:" + script), "Times New Roman")
    doc.styles["Normal"].paragraph_format.first_line_indent = Cm(0.8)
    for name in ("Heading 1", "Heading 2", "Heading 3", "Heading 4"):
        doc.styles[name].paragraph_format.space_before = Pt(12)
        doc.styles[name].paragraph_format.keep_with_next = True
    doc.styles["Title"].paragraph_format.first_line_indent = Cm(0)
    front_style = doc.styles.add_style("Front Matter Title", WD_STYLE_TYPE.PARAGRAPH)
    front_style.base_style = doc.styles["Heading 1"]
    front_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), "9")
    front_style.element.get_or_add_pPr().append(outline)
    for name, indent in (("TOC 1", 0), ("TOC 2", 12), ("TOC 3", 24), ("Table of Figures", 0)):
        style = (doc.styles[name] if name in doc.styles
                 else doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH))
        style.base_style = doc.styles["Normal"]
        style.font.name = "Times New Roman"
        style.font.size = Pt(12 if name == "Table of Figures" else 10.5)
        style.font.bold = name == "TOC 1"
        style.font.color.rgb = RGBColor(0, 0, 0)
        fmt = style.paragraph_format
        fmt.first_line_indent, fmt.left_indent, fmt.right_indent = Pt(0), Pt(indent), Pt(0)
        fmt.space_before, fmt.space_after = Pt(0), Pt(4 if name == "Table of Figures" else 0)
        fmt.line_spacing = 1
        fmt.tab_stops.clear_all()
        fmt.tab_stops.add_tab_stop(Cm(16.5), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
    for footer_paragraph in section.footer.paragraphs:
        footer_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        footer_paragraph.paragraph_format.first_line_indent = Cm(0)
        for run in footer_paragraph.runs:
            font(run, 11)
    doc.core_properties.author = ", ".join(name for name, _ in students) or "Nhóm 11"
    doc.core_properties.title = REPORT_TITLE
    doc.core_properties.subject = "Bài tập lớn Khai phá dữ liệu — Nhóm 11 — 65CNTT"
    doc.core_properties.comments = ""
    doc.core_properties.last_modified_by = "Nhóm 11"
    update_fields = doc.settings.element.find(qn("w:updateFields"))
    if update_fields is None:
        update_fields = OxmlElement("w:updateFields")
        doc.settings.element.append(update_fields)
    update_fields.set(qn("w:val"), "true")

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
        p.paragraph_format.keep_together = True
        font(p.add_run(text), 14 if level == 1 else 13, True, level == 4)
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
        parents = {
            "1.1. Kiểm tra dữ liệu trên Python": "1. Tiền xử lý dữ liệu",
            "2.1. Thống kê và chất lượng dữ liệu": "2. Phân tích dữ liệu",
            "3.1. Lựa chọn sáu thuật toán học máy": "3. Xây dựng sáu thuật toán học máy",
            "3.2.1. Cơ sở lý thuyết Naive Bayes": "3.2. Thuật toán Naive Bayes",
        }
        replacements = {
            "1.1.1. Điền thiếu và chia khoảng cho Naive Bayes": "1.2. Điền thiếu và chia khoảng cho Naive Bayes",
            "1.1.2. Chuẩn hóa cho các mô hình số": "1.3. Chuẩn hóa cho các mô hình số",
            "1.2.1. Dữ liệu trước xử lý trên Weka": "1.4.2. Mở tập phát triển trước xử lý",
            "1.2.2. Xử lý thiếu bằng Weka": "1.4.3. Xử lý thiếu bằng Weka",
            "1.2.3. Rời rạc hóa bằng Weka": "1.4.4. Rời rạc hóa bằng Weka",
        }
        if title in parents:
            parent = parents[title]
            heading(parent, min(4, len(re.match(r"[\d.]+", parent)[0].strip(".").split(".")) + 1),
                    parent == "3.2. Thuật toán Naive Bayes")
        title = replacements.get(title, title)
        depth = len(re.match(r"[\d.]+", title)[0].strip(".").split("."))
        # Mỗi thuật toán đối chiếu là một khối đủ lớn cho một trang lý thuyết.
        algorithm_page = any(title.startswith(prefix) for prefix in
                             ("3.2.2.", "3.2.3.", "3.2.4.", "3.3.", "3.4.", "3.5.", "3.6.", "3.7."))
        heading(title, min(4, depth + 1), algorithm_page)

    def sequence_caption(label, index, text, above=False):
        p = doc.add_paragraph(style="Caption")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(7 if above else 3)
        p.paragraph_format.space_after = Pt(4 if above else 8)
        p.paragraph_format.keep_with_next = above
        p.paragraph_format.keep_together = True
        font(p.add_run(label + " "), 11.5, italic=True)
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), f" SEQ {label} \\* ARABIC ")
        r, value = OxmlElement("w:r"), OxmlElement("w:t")
        value.text = str(index)
        r.append(value)
        field.append(r)
        p._p.append(field)
        font(p.add_run(". " + text.rstrip(".")), 11.5, italic=True)
        return p

    table_index = 0

    def table(headers, rows, widths=None, font_size=None, compact=False):
        nonlocal table_index
        table_index += 1
        t = doc.add_table(rows=1, cols=len(headers))
        t.autofit = False
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for index, label in enumerate(headers):
            t.rows[0].cells[index].text = str(label)
        for values in rows:
            cells = t.add_row().cells
            for i, value in enumerate(values):
                cells[i].text = str(value)
        for i, row in enumerate(t.rows):
            for j, cell in enumerate(row.cells):
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                if widths:
                    cell.width = Cm(widths[j])
                properties = cell._tc.get_or_add_tcPr()
                margins = OxmlElement("w:tcMar")
                for side in ("top", "left", "bottom", "right"):
                    el = OxmlElement("w:" + side); el.set(qn("w:w"), "60" if compact else "100"); el.set(qn("w:type"), "dxa"); margins.append(el)
                properties.append(margins)
                borders = OxmlElement("w:tcBorders")
                for side in ("top", "left", "bottom", "right"):
                    el = OxmlElement("w:" + side); el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "4"); el.set(qn("w:color"), "D9D9D9"); borders.append(el)
                properties.append(borders)
                if i == 0:
                    shade = OxmlElement("w:shd")
                    shade.set(qn("w:fill"), "E8EDF2")
                    properties.append(shade)
                for p in cell.paragraphs:
                    p.paragraph_format.first_line_indent = Cm(0)
                    p.paragraph_format.space_after = Pt(1 if compact else 2)
                    p.paragraph_format.space_before = Pt(1 if compact else 2)
                    p.paragraph_format.line_spacing = 1.05
                    p.alignment = (WD_ALIGN_PARAGRAPH.CENTER if i == 0 or re.fullmatch(r"[-+0-9.,/%± ()]+|N/A", p.text)
                                   else WD_ALIGN_PARAGRAPH.LEFT)
                    for run in p.runs:
                        font(run, font_size or (10.5 if len(headers) > 5 else 11.5), i == 0)
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.keep_with_next = True
        t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        if widths:
            for col, width in zip(t.columns, widths):
                col.width = Cm(width)
        sequence_caption("Bảng", table_index, TABLE_CAPTIONS[table_index - 1])
        spacer = doc.add_paragraph()
        spacer.paragraph_format.space_after = Pt(0)
        spacer.paragraph_format.space_before = Pt(0)
        spacer.paragraph_format.line_spacing = Pt(3)
        font(spacer.add_run(""), 3)
        return t

    figure_index = 0

    def figure(path, caption, width=15.5, max_height=11):
        nonlocal figure_index
        from PIL import Image
        # Giữ đoạn giới thiệu với ảnh; tránh tiêu đề và lời dẫn ở trang trước.
        if doc.paragraphs and doc.paragraphs[-1].text.strip():
            doc.paragraphs[-1].paragraph_format.keep_with_next = True
            doc.paragraphs[-1].paragraph_format.keep_together = True
        with Image.open(path) as im:
            width = min(width, max_height * im.width / im.height)
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.keep_with_next = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Cm(width))
        figure_index += 1
        sequence_caption("Hình", figure_index, caption)

    def contents(title, instruction, new_page=False):
        p = doc.add_paragraph(style="Front Matter Title")
        p.paragraph_format.page_break_before = new_page
        p.paragraph_format.first_line_indent = Cm(0)
        font(p.add_run(title), 14, True)
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), instruction)
        p._p.append(field)

    contents("MỤC LỤC", 'TOC \\o "1-3" \\h \\z')
    contents("DANH MỤC BẢNG", 'TOC \\h \\z \\c "Bảng"', True)
    contents("DANH MỤC HÌNH", 'TOC \\h \\z \\c "Hình"', True)
    heading("LỜI MỞ ĐẦU", 1, True)
    para("Chất lượng nước được mô tả qua nhiều chỉ tiêu, mỗi chỉ tiêu phản ánh một đặc điểm của mẫu nước. Khi các số đo được lưu trữ cùng nhãn đánh giá, chúng tạo thành nguồn dữ liệu để nghiên cứu mối liên hệ giữa đặc điểm của mẫu và kết quả phân loại. Việc khai thác những dữ liệu này là một hướng vận dụng bài toán phân lớp trong khai phá dữ liệu, từ khâu tìm hiểu số liệu đến xây dựng mô hình và đánh giá khả năng dự đoán.")
    para("Trong học phần Khai phá dữ liệu, nhóm chúng em lựa chọn đề tài “" + REPORT_TITLE + "”. Đề tài sử dụng các chỉ tiêu hóa lý để dự đoán nhãn uống được hoặc không uống được của mẫu nước trong bộ dữ liệu Water Quality. Naive Bayes được lựa chọn làm thuật toán chính để nhóm tìm hiểu cách ước lượng xác suất từ dữ liệu và vận dụng định lý Bayes vào quá trình phân lớp.")
    para("Để thực hiện đề tài, nhóm tiến hành tìm hiểu dữ liệu, kiểm tra và xử lý các giá trị thiếu, phân tích thống kê và xây dựng chương trình bằng Python. Thuật toán Naive Bayes được cài đặt tường minh theo cách đếm tần suất. Nhóm đồng thời xây dựng ID3, C4.5, KNN, Logistic Regression và SVM tuyến tính để so sánh trên cùng cách chia dữ liệu, thực hành đối chiếu bằng Weka và phân tích những trường hợp các mô hình dự đoán sai.")
    para("Thông qua bài tập lớn, nhóm mong muốn củng cố kiến thức về phân lớp và rèn luyện cách tổ chức một thực nghiệm có thể kiểm tra lại. Việc phân tích kết quả giúp nhóm hiểu rõ hơn ưu điểm, hạn chế của Naive Bayes trên dữ liệu đã chọn, cũng như ảnh hưởng của bước tiền xử lý và cách đánh giá. Từ các mô hình đã huấn luyện, nhóm phát triển một ứng dụng web đơn giản để nhập thông số, dự đoán mẫu nước và xem kết quả của các thuật toán.")

    page("1. Lý do chọn đề tài", "I. Mô tả bài toán")
    para("Một mẫu nước có thể được mô tả bằng nhiều số đo như pH, độ cứng, chất rắn hòa tan hoặc độ đục. Các chỉ tiêu có phạm vi giá trị khác nhau, vì vậy cần xem xét chúng trong cùng một quy trình xử lý và phân tích dữ liệu. Bộ dữ liệu Water Quality còn có những ô thiếu và số mẫu giữa hai lớp chênh lệch. Những đặc điểm này đặt ra các vấn đề cụ thể để nhóm thực hành kiến thức đã học, từ kiểm tra chất lượng dữ liệu đến lựa chọn cách đánh giá một mô hình phân lớp.")
    para("Nhóm lựa chọn Naive Bayes vì quá trình huấn luyện và dự đoán có thể giải thích qua xác suất của từng lớp và tần suất thuộc tính trong lớp. Việc tự xây dựng bảng đếm, tính xác suất và kiểm tra bằng ví dụ nhỏ giúp nhóm đối chiếu lý thuyết với chương trình. Để đánh giá kết quả có cơ sở, nhóm so sánh Naive Bayes với năm thuật toán khác trên cùng tập dữ liệu, đồng thời phát triển ứng dụng web để thể hiện cách sử dụng mô hình sau huấn luyện. Qua đó, đề tài gắn nội dung phân lớp của học phần với một bài toán có dữ liệu và sản phẩm thực hành cụ thể.")
    heading("2. Tổng quan bài toán")
    para("Bài toán sử dụng bộ dữ liệu Water Quality gồm 3.276 mẫu nước. Mỗi mẫu có chín chỉ tiêu đầu vào: pH, độ cứng, tổng chất rắn hòa tan, chloramines, sulfate, độ dẫn điện, carbon hữu cơ, trihalomethanes và độ đục. Cột Potability là nhãn phân loại, trong đó 0 là không uống được và 1 là uống được theo dữ liệu công bố. Từ các mẫu đã có nhãn, mô hình học cách dự đoán nhãn cho một bộ thông số đầu vào; cột nhãn và mã dòng không được dùng làm thuộc tính dự đoán.")
    para("Trên tập dữ liệu này, nhóm thực hiện quy trình khai phá dữ liệu từ kiểm tra dữ liệu thô, tiền xử lý, phân tích đến xây dựng và đánh giá mô hình. Naive Bayes giữ vai trò chính; ID3, C4.5, KNN, Logistic Regression và SVM tuyến tính được sử dụng để so sánh. Cụ thể, đề tài tập trung vào các nội dung sau:")
    for text in (
        "Tìm hiểu và tiền xử lý dữ liệu: xác định ý nghĩa các thuộc tính, kiểm tra giá trị thiếu và bản ghi trùng, phân tích phân bố lớp; học tham số điền thiếu, chia khoảng hoặc chuẩn hóa từ phần dữ liệu huấn luyện.",
        "Xây dựng thuật toán Naive Bayes: trình bày cơ sở lý thuyết, tự cài đặt bước đếm tần suất và tính xác suất, kiểm chứng bằng ví dụ tính tay và áp dụng cho dữ liệu mẫu nước.",
        "Thực nghiệm và so sánh: đánh giá sáu thuật toán trên cùng cách chia dữ liệu, đối chiếu kết quả với Weka, phân tích ma trận nhầm lẫn và khả năng nhận diện từng lớp thay vì chỉ xét tỷ lệ dự đoán đúng chung.",
        "Phát triển ứng dụng: xây dựng trang web cho phép nhập một mẫu hoặc tải tệp CSV, hiển thị dự đoán của Naive Bayes cùng năm thuật toán đối chiếu và tải kết quả để kiểm tra lại.",
    ):
        item = para("- " + text)
        item.paragraph_format.left_indent = Cm(0.5)
        item.paragraph_format.first_line_indent = Cm(-0.5)
    para("Kết quả của đề tài gồm chương trình tự cài đặt thuật toán, các bảng và biểu đồ đánh giá, minh chứng thực hành Weka và ứng dụng dự đoán. Phạm vi nghiên cứu là phân loại nhãn trên bộ dữ liệu đã công bố. Do nguồn dữ liệu chưa mô tả đầy đủ địa điểm, thời gian lấy mẫu và quy trình kiểm nghiệm, đầu ra của ứng dụng chỉ phục vụ học tập, chưa dùng để chứng nhận chất lượng nước uống.")

    page("3. Quy trình thực hiện")
    table(["Bước", "Hoạt động", "Đầu ra"], [
        (1,"Tải dữ liệu gốc và kiểm checksum","CSV nguyên trạng, thông tin nguồn"),
        (2,"Audit, kiểm tra trùng và chia mẫu","Development/test và ID từng fold"),
        (3,"Phân tích và học tiền xử lý từ train","Bảng thống kê, trung vị, điểm cắt, mean/std"),
        (4,"Học sáu thuật toán và kiểm định chéo","Điểm trung bình và dao động năm fold"),
        (5,"Học lại development và đánh giá test","Dự đoán, ma trận, bảng so sánh, ROC"),
        (6,"Thực hành và đối chiếu Weka","Log, dự đoán và ảnh panel Weka"),
        (7,"Phát triển ứng dụng","Web dự đoán và xuất CSV")], [1.3,8,6])
    para("Dữ liệu được chia thành tập phát triển (development) và tập kiểm tra (test). Tập phát triển dùng cho huấn luyện và kiểm định chéo năm phần. Ở mỗi lượt, trung vị, tham số chuẩn hóa và điểm cắt chỉ được học từ phần huấn luyện. Sau đó mô hình được huấn luyện lại trên toàn bộ tập phát triển và đánh giá trên tập kiểm tra.")
    para("Tập kiểm tra này đã được sử dụng trong phiên bản trước của đề tài, nên kết quả hiện tại là phép đánh giá trên một tập đã từng được xem, không phải dữ liệu ngoài độc lập. Các mô hình sử dụng cấu hình đã chốt; tập kiểm tra không được dùng để lựa chọn tham số hoặc ngưỡng phân lớp.")

    page("4. Phân tích dữ liệu thô")
    para("Bộ Water Quality do Aditya Kadiwal công bố trên Kaggle, phiên bản 3 [4]. Bản tải lại ngày 24/11/2026 có nội dung trùng khớp theo byte với CSV đang lưu trong dự án. Dữ liệu gồm 3.276 mẫu, chín thuộc tính đầu vào và cột nhãn Potability.")
    para("CSV gốc được giữ nguyên, bao gồm cả các ô thiếu. Những bước điền thiếu, chuẩn hóa và chia khoảng được thực hiện sau khi chia tập dữ liệu. Bảng 2 trích năm dòng đầu của CSV; giá trị số chỉ được làm tròn khi trình bày.")
    table(["Dòng", "pH", "Hardness", "Sulfate", "THM", "Nhãn"], [
        (i, *("Thiếu" if X[i][j] is None else number(X[i][j],2) for j in (0,1,4,7)),y[i]) for i in range(5)], [1.6,2.2,3.1,3.1,3.1,1.7])
    table(["Đại lượng", "Giá trị"],[("Số mẫu",3276),("Số thuộc tính đầu vào",9),("Số lớp",2),("Lớp 0",1998),("Lớp 1",1278),("pH thiếu",491),("Sulfate thiếu",781),("Trihalomethanes thiếu",162)], [8.5,6.8])
    para("Lớp 0 chiếm 60,99% và lớp 1 chiếm 39,01% toàn bộ dữ liệu. Do đó, một mô hình luôn chọn lớp 0 cũng đạt accuracy khoảng 61%. Việc so sánh cần xét thêm recall của lớp 1 và các chỉ số đánh giá cả hai lớp.")
    para("Phụ lục A trình bày thêm các bản ghi với đầy đủ mười cột của CSV, bao gồm những ô thiếu. Toàn bộ 3.276 dòng được cung cấp trong tệp water_potability.csv đi kèm báo cáo; tệp này giữ nguyên độ chính xác và nội dung của dữ liệu tải từ nguồn [4].")
    figure(EVIDENCE / "weka-raw-rows.png", "Các bản ghi CSV gốc trong cửa sổ Viewer của Weka, các ô trống chưa được điền.", max_height=11.5)

    page("1.1. Kiểm tra dữ liệu trên Python", "II. Quy trình khai phá dữ liệu")
    para("Dữ liệu được kiểm tra tên cột, thứ tự thuộc tính, kiểu số và nhãn 0/1. Ô rỗng hoặc NaN được biểu diễn bằng None; số 0 vẫn được giữ là một giá trị quan sát. Chuỗi không chuyển được thành số và giá trị vô hạn được ghi nhận là lỗi đầu vào.")
    para("Khóa nhận diện trùng gồm chín giá trị đầu vào sau khi chuẩn hóa kiểu biểu diễn thiếu. Nếu hai hàng có cùng đầu vào và cùng nhãn, giữ hàng đầu để tránh một mẫu xuất hiện ở cả train và test. Nếu trùng đầu vào nhưng nhãn khác nhau, dừng để kiểm tra mâu thuẫn. Với CSV hiện tại, không phát hiện dòng lỗi hoặc nhóm trùng nên giữ 3.276 mẫu.")
    table(["Bước", "Số mẫu trước", "Số mẫu sau"],[("Đọc CSV",3276,3276),("Kiểm tra kiểu và nhãn",3276,3276),("Kiểm tra trùng",3276,3276),("Chia development/test",3276,"2.620 / 656")], [7.3,4,4])
    para("Cả chín chỉ tiêu được giữ làm thuộc tính đầu vào. Potability được tách thành nhãn, còn ID dòng chỉ dùng để đối chiếu mẫu gốc với kết quả dự đoán. Hai cột này không tham gia tính toán đặc trưng.")
    para("Các giá trị ngoại lệ được quan sát qua khoảng tứ phân vị (IQR) và biểu đồ phân bố. Báo cáo giữ lại các mẫu này vì chưa có thông tin đo đạc đủ để xác định chúng là lỗi. Quy tắc xử lý được áp dụng như nhau cho sáu thuật toán.")

    page("1.1.1. Điền thiếu và chia khoảng cho Naive Bayes")
    para("Trung vị của mỗi cột được tính từ các giá trị quan sát trong tập huấn luyện. Với n giá trị đã sắp tăng dần, trung vị là phần tử giữa nếu n lẻ hoặc trung bình hai phần tử giữa nếu n chẵn. Trung vị này được dùng để điền ô thiếu ở cả phần huấn luyện, phần kiểm định và tập kiểm tra tương ứng.")
    para("Để dùng công thức Naive Bayes đếm tần suất, nhóm chuyển các chỉ tiêu số sang tối đa ba khoảng. Hai vị trí chia là kết quả làm tròn lên của n/3 và 2n/3, trên các ô quan sát trước điền. Nếu hai giá trị kề vị trí chia khác nhau, lấy trung điểm làm ngưỡng. Nếu bằng nhau, bỏ điểm cắt để cùng một số luôn nhận cùng mã.")
    table(["Chỉ tiêu", "Trung vị", "Ngưỡng 1", "Ngưỡng 2"],[
        (name,number(median),*(number(cut) for cut in cuts)) for name,median,cuts in zip(FEATURE_COLUMNS,params["nominal"]["fill_values"],params["nominal"]["cut_points"])], [4.8,3.5,3.5,3.5])
    para("Với hai ngưỡng t1 < t2, giá trị được gán mã 0 khi x ≤ t1, mã 1 khi t1 < x ≤ t2 và mã 2 khi x > t2. Cột hằng chỉ có một mã; nếu một cột không có giá trị quan sát trong tập huấn luyện thì chương trình dừng. ID3 sử dụng cùng bộ mã khoảng với Naive Bayes. C4.5 nhận các số đo ban đầu và giữ ô thiếu để xử lý bằng trọng số nhánh; không dùng mã khoảng hay điền trung vị cho mô hình này.")

    page("1.1.2. Chuẩn hóa cho các mô hình số")
    para("Nếu giữ nguyên thang đo, các cột có trị số lớn như Solids có thể chi phối khoảng cách trong KNN. Thang đo cũng ảnh hưởng đến việc cập nhật hệ số của Logistic Regression và SVM. Ba mô hình này sử dụng dữ liệu đã điền trung vị và chuẩn hóa z-score.")
    para("Trên tập huấn luyện đã điền thiếu, giá trị trung bình (mean) bằng tổng các giá trị chia số mẫu. Phương sai tổng thể bằng tổng bình phương độ lệch so với mean chia số mẫu; độ lệch chuẩn (std) là căn bậc hai của phương sai. Giá trị chuẩn hóa được tính bằng z = (x − mean)/std. Với cột có std bằng 0, chương trình dùng hệ số chia bằng 1.")
    table(["Chỉ tiêu", "Trung bình", "Độ lệch chuẩn"],[(name,number(mean),number(scale)) for name,mean,scale in zip(FEATURE_COLUMNS,params["numeric"]["means"],params["numeric"]["scales"])],[6.3,4.5,4.5])
    para("Một giá trị bằng mean được biến đổi thành 0. Đối với mẫu mới nằm ngoài khoảng min/max của dữ liệu huấn luyện, chương trình vẫn tính z-score theo các tham số đã lưu. Tuy nhiên, vùng giá trị này có thể ít được đại diện trong tập huấn luyện.")
    para("Bảng 6 trình bày tham số của lần huấn luyện cuối trên 2.620 mẫu development. Trong kiểm định chéo, mỗi lượt tính lại trung vị, mean và std từ bốn phần huấn luyện; phần còn lại chỉ dùng để đánh giá.")

    heading("1.4. Tiền xử lý dữ liệu trên Weka", 3, True)
    heading("1.4.1. Mở CSV gốc và thiết lập nhãn", 4)
    para("Mở Weka Explorer, chọn Preprocess → Open file, đổi Files of Type sang CSV data files rồi mở water_potability.csv. Chọn ph trong danh sách Attributes để kiểm tra 3.276 mẫu, 10 cột và 491 giá trị thiếu. Chọn Sulfate để kiểm tra 781 ô thiếu. Đây là dữ liệu đầy đủ trước khi chia tập, chỉ được mở để quan sát và kiểm tra nguồn vào.")
    figure(EVIDENCE / "weka-raw-ph.png", "CSV gốc trên Weka: 3.276 mẫu và 491 ô pH thiếu.")
    figure(EVIDENCE / "weka-raw-sulfate.png", "Kiểm tra 781 ô Sulfate thiếu trên dữ liệu gốc.")
    para("Khi đọc CSV, Weka nhận Potability là Numeric vì cột chỉ chứa số 0 và 1. Để dùng nhãn phân loại, chọn Filter → unsupervised → attribute → NumericToNominal, mở phần tham số và đặt attributeIndices = last. Giữ invertSelection = False, nhấn OK rồi Apply. Bộ lọc chỉ đổi kiểu cột cuối; chín chỉ tiêu đầu vào vẫn là Numeric và các ô thiếu vẫn được giữ nguyên.")
    figure(EVIDENCE / "weka-class-options.png", "Tham số NumericToNominal chỉ áp dụng cho cột nhãn cuối.", width=11, max_height=7)
    para("Đặt Class = Potability rồi chọn thuộc tính này. Weka hiển thị nhãn Nominal với 1.998 mẫu lớp 0 và 1.278 mẫu lớp 1. Việc đổi kiểu nhãn không điền thiếu, không chuẩn hóa và không học tham số từ toàn bộ dữ liệu. Sau bước kiểm tra này, các thao tác học tiền xử lý chỉ dùng tập development.")
    figure(EVIDENCE / "weka-class-counts.png", "Potability ở dạng danh nghĩa với số mẫu của hai lớp trên CSV gốc.")
    for title, filename, caption, text in [
        ("1.2.1. Dữ liệu trước xử lý trên Weka", "01-weka-raw.png", "Sulfate trên development trước tiền xử lý trong Weka.",
         "Trong Weka Explorer, chọn Preprocess rồi Open file và mở development_raw.arff. Đặt Class là Potability và chọn Sulfate. Giao diện cho thấy 2.620 mẫu, 10 thuộc tính và 616 ô Sulfate thiếu. Các vạch histogram có màu theo lớp giúp quan sát phân bố trước khi điền.") ,
        ("1.2.2. Xử lý thiếu bằng Weka", "02-weka-mean.png", "Kết quả bộ lọc ReplaceMissingValues của Weka.",
         "Trong Preprocess, chọn Filter → unsupervised → attribute → ReplaceMissingValues rồi Apply trên development. Bộ lọc thực hiện điền giá trị số bằng trung bình học từ development. Sau bước này, Missing của Sulfate bằng 0. Kết quả được lưu thành weka_mean_filled.arff để đối chiếu."),
        ("1.2.3. Rời rạc hóa bằng Weka", "03-weka-discretize.png", "Sulfate sau bộ lọc Discretize của Weka.",
         "Từ dữ liệu development đã điền, chọn bộ lọc Discretize, đặt bins = 3 và useEqualFrequency = True rồi Apply. Weka đổi các cột số thành các mức danh nghĩa. Tệp weka_discretized.arff lưu trạng thái sau biến đổi. Nhóm đối chiếu kiểu thuộc tính, số khoảng và số mẫu để kiểm tra thao tác.")]:
        page(title)
        para(text)
        if filename == "03-weka-discretize.png":
            figure(EVIDENCE / "weka-discretize-options.png", "Thiết lập Discretize: bins = 3, useEqualFrequency = True, ignoreClass = False.", max_height=10.5)
            para("Ảnh tham số ghi lại cấu hình rời rạc hóa trên tập phát triển; ảnh tiếp theo thể hiện kết quả sau khi điền thiếu và áp dụng bộ lọc trong lần thực nghiệm đã lưu.")
        figure(FOLDER/"weka"/filename,caption)
    para("Ba thao tác trên được thực hiện trong Weka Explorer với tập development [5]. ReplaceMissingValues điền bằng giá trị trung bình, còn chương trình Python điền bằng trung vị. Weka chia khoảng sau khi điền thiếu; Python tính điểm cắt từ các giá trị quan sát trước bước điền. Vì vậy, dữ liệu sau xử lý của hai quy trình có thể khác nhau.")
    para("Để so sánh các bộ phân lớp ở phần III, Weka giữ cùng danh sách ID huấn luyện, kiểm tra. NB/ID3 nhận mã khoảng và KNN/Logistic/SVM nhận số chuẩn hóa theo tham số train của Python. C4.5/J48 nhận số đo thô còn ô thiếu. Các tệp lọc minh họa ở phần này không thay đầu vào của phép kiểm định chéo.")

    page("2.1. Thống kê và chất lượng dữ liệu")
    figure(FOLDER/"figures"/"class_and_missing.png", "Phân bố lớp và giá trị thiếu trên development.")
    table(["Cột", "Quan sát", "Thiếu", "Min", "Trung vị", "Max"],[(name,stats[name]["count"],stats[name]["missing"],number(stats[name]["min"],2),number(stats[name]["median"],2),number(stats[name]["max"],2)) for name in FEATURE_COLUMNS],[4,1.7,1.4,2.7,2.8,2.7])
    para("Sulfate có nhiều ô thiếu nhất, tiếp theo là pH và Trihalomethanes. Điền trung vị làm tăng số mẫu ở khoảng chứa trung vị của từng cột. Vì vậy, các khoảng được chia gần cân bằng trên giá trị quan sát có thể có số mẫu khác nhau sau bước điền thiếu.")
    para("Bảng 7 được tính từ các giá trị quan sát trong development trước khi điền thiếu. Min/max trong bảng là phạm vi của dữ liệu nghiên cứu, không phải giới hạn cho phép của các chỉ tiêu chất lượng nước.")

    page("2.2. Phân bố các chỉ tiêu và tương quan")
    figure(FOLDER/"figures"/"histograms.png", "Histogram các chỉ tiêu số trên development.",max_height=10)
    para("Các histogram cho thấy chín chỉ tiêu có phạm vi giá trị khác nhau. Naive Bayes và ID3 dùng mã khoảng nên các số đo trong cùng một khoảng được xem là giống nhau. C4.5 tự tìm ngưỡng nhị phân trên các số đo, đồng thời giữ ô thiếu trong quá trình xây cây. KNN, Logistic Regression và SVM giữ giá trị số sau chuẩn hóa. Khác biệt này cần được xét khi so sánh kết quả.")
    figure(FOLDER/"figures"/"correlation.png", "Tương quan Pearson tính theo các cặp ô quan sát.",width=10,max_height=7)
    para("Hệ số Pearson được tính trên các cặp giá trị không thiếu. Biểu đồ cho biết mức liên hệ tuyến tính giữa từng cặp thuộc tính. Tương quan nhỏ giữa hai cột chưa đủ để kết luận chúng độc lập khi biết lớp, nên biểu đồ tương quan chỉ được dùng để mô tả dữ liệu.")

    page("3.1. Lựa chọn sáu thuật toán học máy")
    table(["Thuật toán", "Tài liệu lý thuyết", "Vai trò"],[("Naive Bayes","Giáo trình và bài giảng [1, 2]","Chính"),("ID3","Giáo trình và bài giảng [1, 2]","Đối chiếu"),("C4.5","Bài giảng Khai phá dữ liệu [2]","Đối chiếu"),("KNN","Giáo trình và bài giảng [1, 2]","Đối chiếu"),("Logistic Regression","Giáo trình Khai phá dữ liệu [1]","Đối chiếu"),("SVM tuyến tính","Tài liệu học máy Cornell [3]","Bổ sung")],[4.5,7.3,3.5])
    para("Các bộ phân lớp được lựa chọn để có những cách suy luận khác nhau trên cùng nhãn nhị phân. Naive Bayes dùng xác suất sinh dữ liệu theo lớp. ID3 và C4.5 chia không gian theo thuộc tính. KNN quyết định theo các mẫu gần nhau. Logistic Regression và SVM tuyến tính xây dựng một mặt phân cách trong không gian đã chuẩn hóa.")
    para("Mã Python tự cài đặt bước huấn luyện và dự đoán của sáu thuật toán. Thư viện được sử dụng cho đọc ghi tệp, vẽ biểu đồ, kiểm thử và giao diện. Weka được dùng làm môi trường đối chiếu; cấu hình và kết quả của Weka được trình bày riêng với Python.")
    para("C4.5 được cài đặt đủ các bước dựng cây và cắt tỉa cho bài toán hai lớp: thuộc tính số và danh nghĩa, hiệu chỉnh gain khi có giá trị thiếu, phân phối trọng số, MDL cho ngưỡng số, collapse, thay cây bằng lá và nâng cây con. J48 dùng cùng cấu hình để đối chiếu [5, 7].")

    page("3.2.1. Cơ sở lý thuyết Naive Bayes")
    para("Gọi C là lớp, nhận 0 hoặc 1, và x = (x1,…,x9) là vector mã khoảng. Theo định lý Bayes, P(C=c | x) = P(C=c)P(x | C=c)/P(x), với P(x) > 0. Khi so sánh hai lớp, mẫu số P(x) chung có thể bỏ khỏi bước chọn nhãn.")
    para("Naive Bayes giả định các thuộc tính độc lập với nhau khi đã biết lớp [1, 2]. Khi đó, P(x | C=c) được thay bằng tích của chín xác suất P(Aj=xj | C=c). Điểm lớp là Sc = P(C=c) × tích_j P(Aj=xj | C=c). Mô hình chọn lớp có điểm lớn hơn và chuẩn hóa hai điểm để hiển thị hậu nghiệm.")
    table(["Đại lượng", "Ước lượng từ tập huấn luyện"],[("N","Tổng số mẫu huấn luyện"),("nc","Số mẫu thuộc lớp c"),("nc,j,v","Số mẫu lớp c có cột j mang mã v"),("P(C=c)","nc / N"),("P(Aj=v | C=c)","nc,j,v / nc"),("P(C=c | x)","Sc / (S0 + S1), khi tổng dương")],[5.4,9.9])
    para("Tần suất 0 được giữ nguyên. Nếu một thừa số của lớp bằng 0, điểm lớp đó bằng 0. Nếu cả hai điểm đều bằng 0, hậu nghiệm không xác định và chương trình báo lỗi dòng đó. Nếu hai điểm dương bằng nhau, quy ước chọn lớp 0. Trong lần chạy này, tất cả 656 mẫu test đều có thể dự đoán.")
    para("Naive Bayes cần ước lượng từng thuộc tính trong mỗi lớp, thay vì đếm tất cả tổ hợp chín thuộc tính cùng lúc. Điều này giúp giảm số lượng bảng tần suất cần học. Đổi lại, các quan hệ giữa thuộc tính trong cùng lớp và thông tin bị mất khi chia khoảng có thể làm điểm hậu nghiệm thiếu chính xác.")

    page("3.2.2. Xây dựng thuật toán Naive Bayes")
    para("Ở bước huấn luyện, mỗi mẫu làm tăng nc của lớp tương ứng một đơn vị. Chương trình duyệt qua chín thuộc tính và cập nhật ô đếm [lớp][cột][mã]. Với mỗi cột, tổng các ô đếm trong một lớp bằng số mẫu của lớp đó. Trung vị, ngưỡng chia khoảng, thứ tự cột và bảng đếm được lưu cùng mô hình.")
    table(["Bước", "Nội dung thực hiện"],[(1,"Kiểm tra X, y và miền mã của từng cột"),(2,"Khởi tạo hai số đếm lớp và các ô tần suất"),(3,"Duyệt các hàng train, cập nhật lớp và từng thuộc tính"),(4,"Với mẫu mới, biến đổi bằng thống kê train đã lưu"),(5,"Nhân prior và chín thừa số của mỗi lớp"),(6,"So sánh hai điểm rồi chuẩn hóa hậu nghiệm")],[1.6,13.7])
    para("Để tránh nhân những số thực rất nhỏ ở bước so sánh, chương trình giữ Sc = ac/bc. Tử ac bắt đầu bằng nc rồi nhân các số đếm của mẫu. Mẫu bc bằng N × nc^d với d=9 và được tính một lần khi fit hoặc nạp model. So sánh a0b1 với a1b0 tương đương so sánh S0 với S1 vì các mẫu số đều dương.")
    para("Sau khi quy đồng, p1 = a1b0/(a0b1+a1b0) và p0 = 1 − p1. Cách tính bằng số nguyên giúp so sánh hai điểm trước khi chuyển sang số thực để hiển thị xác suất. Mô hình lưu các số đếm trong tệp JSON để có thể nạp lại và dự đoán.")
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
    para("Kết quả của ví dụ được dùng làm đáp án kiểm thử cho xác suất và nhãn dự đoán. Các trường hợp khác được kiểm tra gồm tần suất bằng 0, hai điểm bằng nhau, giá trị thiếu, cột hằng, điểm cắt trùng và dự đoán sau khi lưu rồi nạp mô hình.")

    nb=read(FOLDER/"models"/"naive_bayes.json")
    row_id=27
    codes=[]
    for value,median,cuts in zip(X[row_id],nb["preprocessor"]["fill_values"],nb["preprocessor"]["cut_points"]):
        used=median if value is None else value
        codes.append(sum(used>cut for cut in cuts))
    counts=[[nb["model"]["feature_counts"][c][j][v] for j,v in enumerate(codes)] for c in (0,1)]
    totals=nb["model"]["class_counts"]
    scores=[Fraction(totals[c],sum(totals))*prod(Fraction(v,totals[c]) for v in counts[c]) for c in (0,1)]
    page("3.2.4. Tính toán cho một mẫu trong tập kiểm tra")
    para("Mẫu trong tập kiểm tra có ID dòng 27 bị thiếu pH. Giá trị này được điền bằng trung vị 7,0402 học từ development. Sau đó, chín chỉ tiêu được gán mã khoảng và tra bảng tần suất. Các số đo trong Bảng 12 được làm tròn khi trình bày; phép tính sử dụng giá trị đầy đủ.")
    table(["Chỉ tiêu","Giá trị dùng","Mã","Đếm lớp 0","Đếm lớp 1"],[(name,number(nb["preprocessor"]["fill_values"][j] if X[row_id][j] is None else X[row_id][j]),codes[j],counts[0][j],counts[1][j]) for j,name in enumerate(FEATURE_COLUMNS)],[4.4,3.5,1.3,3,3.1])
    para(f"Prior lớp 0 là {totals[0]}/2620 và prior lớp 1 là {totals[1]}/2620. Với từng lớp, nhân prior với chín phân số lấy số đếm ở bảng chia cho số mẫu lớp đó. Kết quả S0 ≈ {number(float(scores[0]),9)} và S1 ≈ {number(float(scores[1]),9)}.")
    para(f"Sau chuẩn hóa, p0 ≈ {number(float(scores[0]/sum(scores)),6)} và p1 ≈ {number(float(scores[1]/sum(scores)),6)}. Mô hình chọn lớp 1 vì p1 lớn hơn p0. Nhãn gốc của dòng 27 là 0, nên mẫu này được tính vào số dự đoán dương sai (FP).")

    page("3.3. Thuật toán ID3")
    para("ID3 xây dựng cây quyết định từ trên xuống, chọn thuộc tính có độ lợi thông tin lớn nhất tại mỗi nút [1, 2]. Với tỷ lệ lớp pc, entropy H(S)=−tổng_c pc log2(pc). Giá trị 0 không tham gia log. Nếu nút chỉ có một lớp thì entropy bằng 0.")
    para("Với thuộc tính A, chia S thành các nhóm Sv theo mã v. Entropy sau chia là tổng_v |Sv|/|S| × H(Sv). Gain(S,A)=H(S) trừ entropy sau chia. ID3 chọn A có gain lớn nhất; khi hòa chọn cột đứng trước để kết quả lặp lại được.")
    para("Trong ví dụ bốn mẫu (A,B,y)=(0,0,0),(0,1,0),(1,0,1),(1,1,1), entropy ban đầu bằng 1. Chia theo A tạo hai nút thuần nên entropy sau chia bằng 0 và gain bằng 1. Chia theo B tạo hai nhóm đều chứa cả hai lớp, gain bằng 0. Vì vậy A được chọn làm nút gốc.")
    para("Mã tự cài nhận cùng chín mã khoảng như Naive Bayes. Một thuộc tính được bỏ khỏi danh sách ở đường đi sau khi đã dùng. Cây dừng khi nút thuần, hết thuộc tính hoặc các thuộc tính còn lại không tạo được ít nhất hai nhóm. Lá lưu số đếm hai lớp để lấy nhãn đa số và điểm lớp 1.")
    para("Khi mẫu mới đi đến một mã chưa xuất hiện tại nút, bản Python dùng phân bố của nút đang xét. Cơ chế này giúp mọi mẫu có kết quả dự đoán nhưng có thể khác ID3 giáo dục của Weka. Phần III ghi nhận cụ thể sai khác và số mẫu Weka không gán được nhãn.")
    para("Cây chưa cắt tỉa có thể tạo nhiều nhánh để mô tả các mẫu huấn luyện. Số nhánh lớn không bảo đảm dự đoán tốt trên dữ liệu mới. Khả năng tổng quát hóa của cây được đánh giá bằng kiểm định chéo và kết quả trên tập kiểm tra.")
    pseudocode("""grow(rows, attributes):
    if pure(rows) or attributes is empty: return leaf(counts)
    best = attribute_with_largest_information_gain(rows)
    for value in observed_values(best):
        child = grow(rows_with_value(best, value), attributes - best)
    return node(best, children, counts)""")

    page("3.4. Thuật toán C4.5")
    para("C4.5 phát triển từ ID3, bổ sung xử lý thuộc tính số, giá trị thiếu và cắt tỉa để kiểm soát độ phức tạp của cây [7]. Trong đề tài, cây phân loại hai nhãn Potability 0/1. Mã Python tự thực hiện toàn bộ các bước; J48 của Weka chỉ dùng để đối chiếu, không tham gia hàm học hoặc dự đoán Python.")
    heading("3.4.1. Chọn thuộc tính và ngưỡng số", 4)
    para("Với thuộc tính danh nghĩa, mỗi giá trị tạo một nhánh. Với thuộc tính số, chương trình sắp các giá trị quan sát tăng dần rồi duyệt vị trí giữa hai số khác nhau. Tại mỗi vị trí, bảng đếm trọng số hai lớp được chuyển dần từ nhánh phải sang nhánh trái. Nhờ vậy, không cần đếm lại toàn bộ dữ liệu cho từng ngưỡng.")
    para("Gọi N là tổng trọng số tại nút, K là phần có giá trị quan sát của thuộc tính. Gain(A)=(K/N)×[H(S_known)−tổng_b (K_b/K)H(S_b)]. SplitInfo tính trên các tỷ trọng K_b/N và thêm tỷ trọng thiếu (N−K)/N. GainRatio(A)=Gain(A)/SplitInfo(A). Khi không có dữ liệu thiếu, biểu thức trở về độ lợi thông tin thông thường.")
    para("Với cột số có m ngưỡng hợp lệ, hiệu chỉnh MDL trừ log2(m)/N khỏi gain tốt nhất của cột. Mỗi nhánh cần đủ trọng số theo min_samples_leaf; ngưỡng tối thiểu thích nghi bằng max(2, min(25, 0,1K/2)) trong cấu hình đang dùng. Các cột được lọc theo gain trung bình với dung sai 0,001 rồi chọn gain ratio lớn nhất. Cột danh nghĩa quá nhiều giá trị không chi phối trung bình khi vẫn có cột số hoặc cột ít giá trị, theo cách đối chiếu của J48.")
    para("Ví dụ tám số 1 đến 8, bốn số đầu có nhãn 0 và bốn số cuối nhãn 1. Chia tại giữa 4 và 5 cho entropy hai nhánh bằng 0 và gain ban đầu bằng 1. Có bảy vị trí hợp lệ khi min_samples_leaf=1, nên gain sau MDL bằng 1−log2(7)/8≈0,6491. Ngưỡng được lưu tại giá trị train 4 khi bật cách đặt ngưỡng theo J48. Thuộc tính số có thể tiếp tục được dùng ở nút sâu hơn.")
    pseudocode("""for column in feature_columns:
    if numeric(column):
        scan_sorted_values_with_prefix_class_counts()
        gain[column] -= log2(valid_threshold_count) / total_weight
    else:
        make_one_branch_per_nominal_value()
average_gain = average_of_eligible_column_gains()
split = largest_gain_ratio_with_gain_above_average()
build_children(split)""")

    page("3.4.2. Xử lý giá trị thiếu bằng trọng số")
    para("C4.5 giữ ô thiếu trong ma trận huấn luyện. Sau khi chọn cột chia, tính tỷ trọng nhánh q_b=K_b/K từ các mẫu có giá trị quan sát. Một mẫu thiếu với trọng số w được đưa vào mỗi nhánh b bằng trọng số w×q_b. Nhãn thật của mẫu được giữ nguyên ở mọi phần phân số. Tổng trọng số hai lớp sau khi chia phải bằng tổng tại nút cha.")
    para("Xét năm mẫu một thuộc tính: (0,0), (0,0), (10,1), (10,1), (thiếu,1), trong đó phần tử thứ hai là nhãn. N=5, K=4, gain đã giảm theo phần quan sát bằng 4/5. Hai nhánh có tỷ trọng 1/2. Mẫu thiếu nhãn 1 được chia trọng số 0,5 vào mỗi nhánh: nhánh trái có số đếm [2;0,5], nhánh phải [0;2,5]. Tổng vẫn là [2;3], khớp số đếm tại gốc.")
    para("Khi dự đoán mẫu thiếu ở cột chia, mô hình tính p1(x)=tổng_b q_b×p1_b(x). Trong ví dụ trên, p1 bên trái là 0,5/2,5=0,2 và bên phải là 1; do đó p1 của mẫu thiếu bằng 0,5×0,2+0,5×1=0,6. Đây là phép trộn theo nhánh, khác với thay giá trị thiếu bằng trung vị.")
    para("Mã cũng hỗ trợ trọng số đầu vào không âm và thuộc tính danh nghĩa dạng số hoặc chuỗi. Giá trị danh nghĩa mới được xử lý như chưa biết; nhánh đã khai báo nhưng rỗng dùng phân bố ở nút cha. Cột toàn thiếu hoặc cột hằng không tạo phép chia. Chương trình dừng ở lá nếu không tìm được độ lợi phù hợp.")
    pseudocode("""known_weight[b] = sum(weights_with_observed_value_in_branch_b)
branch_fraction[b] = known_weight[b] / sum(known_weight)
for row in rows_with_missing_split_value:
    for branch in branches:
        child_weight = row.weight * branch_fraction[branch]
        send(row, child_weight, branch)
prediction_if_missing = sum(branch_fraction[b] * child_probability[b])""")

    page("3.4.3. Cắt tỉa cây và nâng cây con")
    para("Sau khi dựng cây, bước collapse thu gọn các nhánh không làm giảm lỗi huấn luyện. Sau đó, mô hình duyệt cây từ dưới lên và so sánh ba phương án: giữ cây con hiện tại; thay toàn bộ nút bằng một lá; hoặc nâng nhánh có trọng số lớn nhất lên thay nút cha. Các so sánh dùng lỗi ước lượng bi quan với confidence factor CF=0,25 [5, 7].")
    para("Tại một lá, N là tổng trọng số và E là trọng số không thuộc lớp đa số. Với E=0, phần lỗi cộng thêm bằng N×(1−CF^(1/N)); với 0<E<1, nội suy tuyến tính giữa trường hợp lỗi 0 và 1. Nếu E+0,5≥N, giới hạn lỗi không vượt N. Các trường hợp này giúp xử lý cả số đếm phân số do ô thiếu.")
    para("Ở phần còn lại, đặt f=(E+0,5)/N và z là phân vị chuẩn 1−CF. Cận tỷ lệ lỗi r=[f+z²/(2N)+z×sqrt(f(1−f)/N+z²/(4N²))]/[1+z²/N]. Lỗi ước lượng của lá bằng N×r. Chương trình tự tính phân vị bằng xấp xỉ phân thức, không gọi hàm phân phối có sẵn. Phép tính này chỉ phục vụ cắt tỉa C4.5, không thay Naive Bayes đếm tần suất.")
    para("Lỗi cây bằng tổng lỗi ước lượng các lá. Khi xét nâng cây con, toàn bộ mẫu tại nút cha được phân phối lại theo cấu trúc nhánh được nâng; các bảng đếm và tỷ trọng được tính lại. Nếu lá có lỗi không lớn hơn hai phương án còn lại quá 0,1 thì chọn lá. Nếu không, nhánh lớn nhất được nâng khi lỗi của nó không lớn hơn lỗi cây quá 0,1. Sau nâng, tiếp tục kiểm tra cắt tỉa.")
    pseudocode("""collapse_if_training_error_does_not_increase(tree)
for node in postorder(tree):
    error_tree = sum(pessimistic_error(leaf) for leaf in subtree)
    error_leaf = pessimistic_error(as_single_leaf(node))
    error_raise = evaluate_largest_branch_on_all_parent_rows()
    if error_leaf <= min(error_tree, error_raise) + 0.1:
        replace_node_with_leaf()
    elif error_raise <= error_tree + 0.1:
        raise_largest_branch_and_recompute_distributions()
        prune_again()""")

    page("3.4.4. Cài đặt và kiểm chứng C4.5")
    para("Lớp C45 nằm riêng trong src/algorithms/c45.py, cạnh năm thuật toán còn lại. Các hàm chọn phép chia, chia trọng số, dựng cây, collapse, ước lượng lỗi và nâng cây con được tách theo trách nhiệm. File tiền xử lý chứa kiểm tra ma trận, chia phân tầng, trung vị và chuẩn hóa; C4.5 đi qua bước kiểm tra đầu vào nhưng giữ nguyên số đo và các ô thiếu.")
    para("Cấu hình thực nghiệm chốt trước khi chạy gồm min_samples_leaf=2, CF=0,25, bật MDL, collapse, cắt tỉa bi quan và nâng cây con. Ngưỡng số được đưa về giá trị train gần nhất không vượt trung điểm. J48 dùng cùng cấu hình mặc định tương ứng, nhận cùng ARFF số có ký hiệu ? cho ô thiếu, không qua ReplaceMissingValues hoặc Discretize trong phép so sánh C4.5.")
    para("Các kiểm thử tính tay kiểm tra gain, MDL, bảo toàn trọng số, xác suất trộn và cận lỗi. Trên 24 bộ kiểm tra tổng hợp với 2.009 lượt dự đoán, số nút và nhãn đều trùng J48; sai số xác suất lớn nhất dưới 4,45×10^−16. Dữ liệu tổng hợp chỉ kiểm tra cài đặt, không được dùng tính các chỉ số phân loại nước.")
    pruning = results['c45']['pruning']
    para(f"Trên toàn bộ development, cây trước cắt tỉa có {pruning['before']['nodes']} nút và {pruning['before']['leaves']} lá. Sau xử lý còn {pruning['after']['nodes']} nút, {pruning['after']['leaves']} lá, độ sâu {pruning['after']['depth']}. Có {pruning['collapsed']} lần collapse, {pruning['replaced_by_leaf']} lần thay bằng lá và {pruning['raised']} lần nâng cây con. Các số này được đọc từ trạng thái đã lưu của lần chạy {RUN_ID}.")
    para("Cấu trúc cây, số đếm trọng số, tỷ trọng nhánh và cấu hình được lưu bằng JSON. Khi nạp lại, chương trình kiểm tra miền ngưỡng, số chiều và tổng trọng số lớp của các nút con. Dữ liệu huấn luyện tạm trong quá trình cắt tỉa được loại khỏi cấu trúc dùng dự đoán. Kết quả đối chiếu trên tập nước và từng fold được trình bày ở phần III.")

    page("3.5. Thuật toán KNN")
    para("KNN giữ các mẫu train đã điền trung vị và chuẩn hóa. Khi dự đoán, tính khoảng cách Euclid từ mẫu mới đến từng mẫu học, chọn k mẫu gần nhất và bỏ phiếu theo nhãn [1, 2]. Bài tập cố định k=5 và mọi láng giềng có cùng trọng số.")
    para("Bình phương khoảng cách d²(x,z)=tổng_j (xj−zj)² đủ để xếp hạng vì căn bậc hai bảo toàn thứ tự. Khi hòa khoảng cách, chương trình xét chỉ số mẫu train để kết quả xác định. Điểm lớp 1 là số phiếu lớp 1 chia k, còn nhãn là lớp có nhiều phiếu hơn.")
    para("Ví dụ một chiều đã chuẩn hóa gồm các điểm 0,2,5,8 với nhãn 0,1,1,0. Với mẫu 3 và k=3, các khoảng cách bình phương lần lượt là 9,1,4,25. Ba láng giềng gần nhất là 2,5,0, cho hai phiếu lớp 1 và một phiếu lớp 0. Mẫu được dự đoán là 1 với điểm 2/3.")
    para("Ưu điểm của KNN là có thể tận dụng cấu trúc cục bộ không tuyến tính. Hạn chế là mỗi lần dự đoán phải so với nhiều mẫu train và chất lượng phụ thuộc vào cách đo khoảng cách. Chuẩn hóa train-only giúp hạn chế cột Solids hoặc Conductivity chi phối các cột có trị số nhỏ hơn.")
    para("KNN lưu các mẫu đã xử lý ở bước huấn luyện và tính khoảng cách khi có yêu cầu dự đoán. Vì vậy, chi phí dự đoán tăng theo số mẫu huấn luyện. Thời gian huấn luyện và thời gian dự đoán được ghi riêng trong kết quả thực nghiệm.")
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
    para("Ở cấu hình đang xét, Logistic Regression dự đoán toàn bộ tập kiểm tra là lớp 0. Xác suất đầu ra vẫn khác nhau giữa các mẫu, nên ROC-AUC có thể khác 0,5 dù nhãn dự đoán chỉ thuộc một lớp. Kết quả này chỉ phản ánh cấu hình đã chạy, chưa đủ để kết luận về mọi cấu hình của Logistic Regression.")
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
    para("Sáu thuật toán sử dụng cùng năm phần dữ liệu trong development. Mỗi lượt dùng bốn phần để huấn luyện và một phần để kiểm định (validation); mỗi mẫu xuất hiện đúng một lần ở phần kiểm định. Tiền xử lý và mô hình được học lại ở từng lượt.")
    table(["Chỉ số","Cách tính"],[("Accuracy","(TP+TN)/N"),("Precision lớp 1","TP/(TP+FP)"),("Recall lớp 1","TP/(TP+FN)"),("F1 lớp 1","2TP/(2TP+FP+FN)"),("Balanced Accuracy","(Recall lớp 0 + Recall lớp 1)/2"),("Macro-F1","(F1 lớp 0 + F1 lớp 1)/2"),("ROC-AUC","Diện tích dưới đường TPR theo FPR")],[5.4,9.9])
    para("Nhãn 1 được chọn làm lớp dương. Trong ma trận nhầm lẫn, hàng là nhãn thực tế và cột là nhãn dự đoán, theo thứ tự [0,1]. Chỉ số có mẫu số bằng 0 được ghi N/A. Mô hình luôn chọn lớp đa số của tập huấn luyện được dùng làm mốc so sánh bổ sung.")
    para("Trong các bảng kết quả, BA là Balanced Accuracy, AUC là diện tích dưới đường ROC, TB là trung bình và SD là độ lệch chuẩn qua năm lượt kiểm định. Precision, recall và F1 của lớp 1 được tính với nhãn uống được là lớp dương.")
    para("Các tham số được chốt trước khi chạy thực nghiệm. Kết quả so sánh trong báo cáo ứng với những cấu hình này, chưa bao gồm việc tìm tham số tối ưu cho từng thuật toán. Độ lệch chuẩn qua năm lượt kiểm định mô tả mức dao động theo cách chia dữ liệu.")

    page("2. Kết quả kiểm định chéo năm fold")
    table(["Thuật toán","Accuracy TB","BA TB ± SD","Macro-F1 TB","AUC TB"],[
        (NAMES[a],percent(results[a]["cv"]["accuracy"]["mean"]),number(results[a]["cv"]["balanced_accuracy"]["mean"])+" ± "+number(results[a]["cv"]["balanced_accuracy"]["std"]),number(results[a]["cv"]["macro_f1"]["mean"]),number(results[a]["cv"]["roc_auc"]["mean"])) for a in ALGORITHMS],[4.1,2.7,3.4,2.7,2.4])
    table(["Fold","BA Naive Bayes","BA KNN","AUC Naive Bayes","AUC KNN"],[(i+1,number(results["naive_bayes"]["folds"][i]["balanced_accuracy"]),number(results["knn"]["folds"][i]["balanced_accuracy"]),number(results["naive_bayes"]["folds"][i]["roc_auc"]),number(results["knn"]["folds"][i]["roc_auc"])) for i in range(5)],[1.3,3.5,3.5,3.5,3.5])
    para("Trong Bảng 15, KNN có BA trung bình cao nhất, đạt 0,5685. ID3 đạt 0,5614 và C4.5 đạt 0,5489, đều cao hơn mức 0,5200 của Naive Bayes. Đây là kết quả kiểm định trong development; phần tiếp theo sử dụng tập kiểm tra để xem các mô hình có duy trì được mức chênh lệch này hay không.")
    para("Với Naive Bayes, BA giữa các lượt nằm trong khoảng từ 0,5101 đến 0,5331; AUC từ 0,5242 đến 0,6013. BA được tính từ nhãn tại ngưỡng phân lớp đã chọn, còn AUC đánh giá thứ tự các mẫu theo điểm đầu ra. Hai chỉ số này bổ sung cho nhau khi đọc kết quả.")

    page("3. Kết quả kiểm tra và so sánh sáu mô hình")
    table(["Thuật toán","Accuracy","BA","Macro-F1","Recall 1","AUC"],[
        (NAMES[a],percent(results[a]["test"]["accuracy"]),percent(results[a]["test"]["balanced_accuracy"]),number(results[a]["test"]["macro_f1"]),percent(results[a]["test"]["per_class"][1]["recall"]),number(results[a]["test"]["roc_auc"])) for a in ALGORITHMS],[4.1,2.3,2.3,2.3,2.3,2])
    figure(FOLDER/"figures"/"comparison.png", "Accuracy và Balanced Accuracy trên cùng 656 mẫu test.")
    para("KNN đạt accuracy 64,79%, cao hơn mốc luôn đoán lớp 0 là 60,98% khoảng 3,81 điểm phần trăm. KNN nhận đúng 108/256 mẫu lớp 1 và có BA 60,72%. Naive Bayes đạt accuracy 60,37%, BA 51,61% và nhận đúng 30/256 mẫu lớp 1. Với tập dữ liệu và cách tiền xử lý đang dùng, KNN cho kết quả tốt hơn Naive Bayes ở các chỉ số này.")
    para("Logistic Regression và SVM Python dự đoán toàn bộ mẫu là lớp 0, nên accuracy bằng mốc đa số và recall lớp 1 bằng 0. ID3 nhận diện được nhiều mẫu lớp 1 hơn Naive Bayes nhưng có nhiều FP, nên accuracy thấp hơn. C4.5 mới có TN=364, FP=36, FN=214, TP=42; accuracy đạt 61,89%, BA 53,70% và recall lớp 1 bằng 16,41%. C4.5 nhận thêm 12 mẫu lớp 1 so với NB và phát sinh thêm hai FP, nhưng vẫn bỏ sót phần lớn mẫu lớp 1.")

    page("4. Ma trận nhầm lẫn")
    figure(FOLDER/"figures"/"confusions.png", "Sáu ma trận nhầm lẫn, hàng là thật và cột là dự đoán.",max_height=13)
    para("Đối với Naive Bayes, TN=366, FP=34, FN=226, TP=30. Accuracy=(366+30)/656=60,37%. Con số 30/256 là recall của riêng lớp 1, không phải accuracy trên toàn test. Phần lớn dự đoán đúng của NB đến từ 366 mẫu lớp 0.")
    para("KNN chuyển một phần mẫu bị bỏ sót sang dự đoán đúng lớp 1, đồng thời tăng FP. ID3 tăng số dự đoán lớp 1 mạnh hơn NB. C4.5 sau cắt tỉa giảm FN từ 226 của NB xuống 214 và tăng FP từ 34 lên 36. Hai mô hình tuyến tính Python có hàng dự đoán lớp 1 bằng 0 ở cấu hình này.")

    page("5. Đường ROC và điểm đầu ra")
    figure(FOLDER/"figures"/"roc_comparison.png", "ROC của sáu mô hình Python trên test.",max_height=12)
    para("Mã sắp điểm lớp 1 giảm dần, xử lý tất cả các mẫu đồng điểm cùng một bước và tính diện tích bằng quy tắc hình thang. Naive Bayes dùng hậu nghiệm; ID3 dùng tỷ lệ lớp ở lá; C4.5 trộn các lá theo trọng số khi thiếu thuộc tính; KNN dùng tỷ lệ phiếu; Logistic dùng sigmoid; SVM dùng f(x) trực tiếp.")
    para("KNN có AUC 0,6518, cao hơn NB 0,5218. AUC gần 0,5 của NB phản ánh khả năng xếp hạng còn yếu trên tập này. Logistic và SVM có điểm thay đổi giữa các mẫu nên AUC hơi khác 0,5 dù nhãn ở ngưỡng hiện tại đều bằng 0. Các điểm xác suất chưa được kiểm tra hiệu chỉnh nên không diễn giải là mức an toàn nước thực tế.")

    page("6. Phân tích kết quả của Naive Bayes")
    metric=results["naive_bayes"]["test"]
    table(["Cách đọc lỗi","Phép tính","Kết quả"],[("Mẫu lớp 0 bị gán nhầm 1","34/400","8,50%"),("Dự đoán 1 bị sai","34/64","53,125%"),("Mẫu lớp 1 bị bỏ sót","226/256","88,28%"),("Mẫu lớp 1 được nhận đúng","30/256","11,72%")],[7.3,4,4])
    para("Trong 400 mẫu thực tế thuộc lớp 0, Naive Bayes gán nhầm 34 mẫu sang lớp 1, tương ứng 8,50%. Nếu xét riêng 64 mẫu được dự đoán là lớp 1 thì có 34 mẫu sai. Hai tỷ lệ có mẫu số khác nhau và lần lượt mô tả tỷ lệ báo dương sai trên lớp 0 và tỷ lệ dự đoán sai trong nhóm được gán nhãn uống được.")
    para("Việc chia khoảng làm mất khác biệt giữa các số đo nằm trong cùng một khoảng. Bước điền trung vị còn tập trung thêm mẫu ở khoảng giữa, nhất là với Sulfate và pH có nhiều ô thiếu. Bên cạnh đó, Naive Bayes không mô hình hóa tương tác giữa các chỉ tiêu trong cùng lớp. Những đặc điểm này là hạn chế của cách biểu diễn và giả định mô hình; thực nghiệm hiện tại chưa tách riêng mức ảnh hưởng của từng yếu tố.")
    para("ID3 sử dụng cùng mã khoảng với Naive Bayes, còn C4.5 tự chọn ngưỡng từ số đo thô và giữ ô thiếu. Cả hai có recall lớp 1 cao hơn NB trong lần chạy này; mức tăng FP của C4.5 nhỏ hơn ID3. KNN sử dụng giá trị số đã chuẩn hóa nên khác cả mô hình lẫn cách biểu diễn đầu vào. Do đó, chênh lệch giữa KNN và Naive Bayes không thể quy hoàn toàn cho thuật toán phân lớp.")
    para("Accuracy của Naive Bayes thấp hơn mốc đa số, trong khi recall lớp 1 chỉ đạt 11,72%. Kết quả cho thấy cấu hình này chưa đáp ứng tốt việc nhận diện mẫu uống được. Các cải tiến về biểu diễn thuộc tính hoặc tham số cần được đánh giá trong development trước khi thử trên một tập kiểm tra độc lập.")

    heading("7. Thực nghiệm trên Weka", 2, True)
    para("Các bộ phân lớp trong Weka được học trên 2.620 mẫu và đánh giá trên 656 mẫu kiểm tra, theo cùng ID với Python. Kết quả được lấy từ lớp Evaluation của Weka và đối chiếu với tệp dự đoán từng mẫu. Các hình sau chụp màn hình Classify với chế độ Supplied test set.")
    for a in ALGORITHMS:
        page("7."+str(ALGORITHMS.index(a)+1)+". Kết quả "+NAMES[a]+" trên Weka")
        figure(FOLDER/"weka"/f"result-{a}.png", f"Kết quả {NAMES[a]} trong Weka Explorer.",max_height=12)
        matrix = weka[a]["matrix_with_unclassified"]
        correct = matrix[0][0] + matrix[1][1]
        unclassified = weka[a]["unclassified"]
        para(f"{NAMES[a]} phân loại đúng {correct}/656 mẫu, đạt accuracy {percent(weka[a]['accuracy'])}. "
             + (f"Có {unclassified} mẫu không được gán nhãn." if unclassified
                else "Tất cả mẫu kiểm tra đều được gán nhãn."))
        explanations={
            "naive_bayes":"Bộ phân lớp DirectFrequencyNB sử dụng phép đếm tần suất trực tiếp, không làm trơn. Đầu vào là các mã khoảng học từ development của Python. Nhãn của 656 mẫu trùng với kết quả Python; xác suất hậu nghiệm cũng khớp trong sai số số học. Bộ NaiveBayes mặc định của Weka dùng cách ước lượng khác nên phép đối chiếu này sử dụng DirectFrequencyNB.",
            "id3":"ID3 được lấy từ gói simpleEducationalLearningSchemes của Weka [6]. Khi gặp nhánh chưa có mẫu huấn luyện, phiên bản này có thể trả phân bố rỗng. Trong 53 mẫu chưa được gán nhãn có 28 mẫu lớp 0 và 25 mẫu lớp 1. Accuracy được tính trên toàn bộ 656 mẫu, bao gồm các trường hợp chưa gán nhãn.",
            "c45":"J48 dùng -C 0.25 -M 2, bật collapse và subtree raising, có MDL và không làm trơn Laplace. Đầu vào là ARFF số còn ô thiếu, cùng ID với Python. Cả hai có 21 nút và 11 lá. Nhãn 656 mẫu test trùng hoàn toàn; sai số xác suất lớp 1 lớn nhất dưới 1,56×10^−14. Năm fold cũng được đối chiếu riêng theo cùng ID validation.",
            "knn":"IBk dùng k=5 và khoảng cách Euclid. Chức năng chuẩn hóa trong Weka được tắt vì đầu vào đã được chuẩn hóa theo tham số huấn luyện. Nhãn dự đoán trùng với Python trên tập kiểm tra. Quy tắc trả phân bố và xử lý các láng giềng đồng khoảng cách của Weka có thể làm điểm đầu ra khác nhau dù nhãn trùng nhau.",
            "logistic":"Logistic của Weka đặt ridge=0,001 và tối đa 400 lượt tối ưu. Cả Weka và Python đều dự đoán toàn bộ tập kiểm tra là lớp 0. Hai chương trình khác cách tối ưu và cách tính hệ số điều chuẩn, nên ridge của Weka và λ trong mã Python không được xem là hai tham số tương đương chỉ vì có cùng trị số.",
            "svm":"SMO dùng kernel đa thức bậc 1, tương ứng kernel tuyến tính, với C=1 và tắt chuẩn hóa nội bộ. Python tối ưu hinge loss bằng SGD với λ=0,01. Cách điều chuẩn và tối ưu của hai cấu hình khác nhau. SMO trong thí nghiệm chưa hiệu chỉnh xác suất; phân bố đầu ra của Weka và biên quyết định của Python được xem xét riêng."
        }
        para(explanations[a])

    page("8. So sánh Python với Weka")
    table(["Thuật toán","Accuracy Python","Accuracy Weka","Chưa gán Weka"],[(NAMES[a],percent(results[a]["test"]["accuracy"]),percent(weka[a]["accuracy"]),weka[a]["unclassified"]) for a in ALGORITHMS],[4.5,3.9,3.9,3])
    para("NB có cùng công thức, cùng bảng biểu diễn và cho cùng dự đoán. KNN có cùng k, khoảng cách và mẫu học nên cho cùng nhãn. C4.5 cũng trùng nhãn J48 trên test và cả năm fold, được kiểm tra trực tiếp từ dự đoán từng mẫu. Logistic có cùng accuracy giữa hai môi trường nhưng điểm đầu ra và cơ chế tối ưu không được xem là đồng nhất.")
    para("ID3 Python dùng phân bố tại nút khi thiếu nhánh, còn ID3 của Weka không gán nhãn cho 53 mẫu. SVM có kết quả chênh lệch lớn giữa hai môi trường, cùng với sự khác nhau về tham số điều chuẩn và phương pháp tối ưu. Vì vậy, kết quả Weka được dùng để đối chiếu từng cấu hình và không thay thế số liệu Python.")
    para("Trong phép so sánh bộ phân lớp, NB/ID3 nhận ARFF mã khoảng và KNN/Logistic/SVM nhận ARFF chuẩn hóa theo tham số development của Python. Riêng C4.5/J48 nhận ARFF số còn ô thiếu. Các thao tác bộ lọc ở phần II được trình bày riêng để mô tả cách dùng Weka. Nhờ dùng cùng đầu vào trong phép đối chiếu, khác biệt giữa điền trung bình và điền trung vị không ảnh hưởng đến bảng so sánh bộ phân lớp.")

    page("8.1. Kiểm chứng C4.5 theo từng mẫu và từng fold")
    agreement = weka['c45']['python_agreement']
    checks = [('Test', agreement['compared'], agreement['different_labels'], f"{agreement['max_score_difference']:.2e}")]
    checks += [(f"Fold {row['fold']+1}", row['compared'], row['different_labels'], f"{row['max_score_difference']:.2e}")
               for row in weka['c45']['fold_agreement']]
    table(['Tập dữ liệu', 'Số mẫu', 'Khác nhãn', 'Sai số điểm lớn nhất'], checks, [4.3,2.5,3.5,5])
    para("Các giá trị ở bảng được tính từ hai tệp dự đoán theo cùng thứ tự ID. Toàn bộ 656 mẫu test và 2.620 lượt validation không có nhãn khác giữa C4.5 Python và J48. Sai số xác suất nhỏ phát sinh từ thứ tự cộng số thực và xấp xỉ phân vị trong cắt tỉa; không làm thay nhãn trên các tập đã chạy.")
    para("So với bản cây gain ratio danh nghĩa trước đây, C4.5 mới thay cả cách biểu diễn lẫn cắt tỉa. Accuracy tăng từ 55,03% lên 61,89%, nhưng recall lớp 1 giảm từ 45,70% xuống 16,41% và macro-F1 giảm từ 0,5328 xuống 0,4979. BA chỉ tăng từ 53,35% lên 53,70%. Vì vậy, accuracy cao hơn không có nghĩa mọi khía cạnh phân loại đều tốt hơn.")
    para("Kết quả trên xác nhận mức phù hợp giữa hai cài đặt ở cấu hình đã công bố, không chứng minh mô hình đã đủ tốt cho kiểm nghiệm thực tế. Trên tập nước, C4.5 vẫn thiên về lớp 0. Các cấu hình khác cần được đánh giá trong development; không điều chỉnh CF hoặc số mẫu tối thiểu theo kết quả test đã xem.")
    figure(WEB_EVIDENCE / 'weka-c45-settings.png', 'Cấu hình J48 trong Weka: confidence 0,25, minNumObj 2, cắt tỉa và nâng cây con.', max_height=13)
    figure(WEB_EVIDENCE / 'weka-c45-result.png', 'Kết quả thao tác J48 trên Weka với Supplied test set, 656 mẫu thô còn ô thiếu.', max_height=13)

    page("1. Mục tiêu và luồng sử dụng", "IV. Phát triển ứng dụng dự đoán")
    para("Ứng dụng sử dụng Streamlit và chạy cục bộ tại http://127.0.0.1:8501. Phiên bản được khảo sát sử dụng lần chạy 20261128-c45-full-v1 với C4.5 đầy đủ các bước dựng và cắt tỉa cây. Giao diện có năm tab: Dự đoán một mẫu, Dự đoán CSV, Kết quả thực nghiệm, Dữ liệu gốc và Thông tin 9 chỉ tiêu. Phần đầu trang giới thiệu đề tài, nhóm thực hiện, số mẫu và các thuật toán. Naive Bayes được đánh dấu là mô hình chính.")
    para("Ứng dụng nạp mô hình từ lần thực nghiệm 20261128-c45-full-v1. Bộ dữ liệu và ID chia tập được giữ như trước; sáu thuật toán được chạy lại, trong đó C4.5 dùng số liên tục, xử lý thiếu bằng trọng số và cắt tỉa cây. Bảng và hình ở phần III được cập nhật từ lần chạy này. Khi dự đoán, ứng dụng kiểm tra đầu vào, áp dụng tham số tiền xử lý đã học và gọi mô hình; người dùng không cần huấn luyện lại.")
    figure(WEB_EVIDENCE / "web-overview.png", "Trang ứng dụng với phần giới thiệu và năm tab chức năng.", max_height=13.5)
    para("Các ô nhập được chia thành ba nhóm: tính chất lý hóa cơ bản; khử trùng và khoáng chất; độc chất vi lượng và độ đục. Mỗi nhóm gồm ba chỉ tiêu, giữ đủ chín cột đầu vào. Người dùng nhập số thập phân bằng dấu chấm và có thể để trống chỉ tiêu chưa có. Tên cột gốc được giữ cùng nhãn tiếng Việt để đối chiếu với CSV.")
    figure(WEB_EVIDENCE / "web-form.png", "Chín ô nhập theo nhóm chỉ tiêu và các nút nạp mẫu thử.", max_height=13.5)
    para("Nút Nạp mẫu minh họa điền trung vị từ tập huấn luyện. Hai nút Nạp bộ số thử A và Nạp bộ số thử B điền các bộ số được cấu hình sẵn trong mã giao diện; chúng là dữ liệu thử thao tác, không có nhãn kiểm nghiệm độc lập. Nút Xóa ô nhập làm trống các chỉ tiêu. Khi đầu vào thay đổi, kết quả trước đó được ẩn để tránh đọc nhầm dự đoán.")

    page("2. Dự đoán một mẫu và mức đồng thuận")
    para("Sau khi nhập số đo, chọn Dự đoán và so sánh. Kết quả Naive Bayes được đặt trong khung nổi bật, dùng màu đỏ cho nhãn 0 và màu xanh cho nhãn 1. Bảng chi tiết bên dưới trình bày đủ sáu thuật toán, nhãn, điểm lớp 1 hoặc biên quyết định, loại điểm và lỗi. Xác suất được hiển thị cả số thập phân và phần trăm; SVM giữ điểm biên có dấu.")
    para("Với bộ trung vị được làm tròn trên biểu mẫu, Naive Bayes dự đoán lớp 0 và cho điểm ước lượng lớp 1 là 0,2479, tương ứng 24,79%. Sáu mô hình đều dự đoán lớp 0. Màn hình ghi mức đồng thuận 100%, được tính bằng số mô hình chọn nhãn chiếm đa số chia số mô hình có kết quả hợp lệ. Chỉ số này mô tả sự giống nhau giữa các đầu ra, không phải accuracy hoặc xác suất dự đoán đúng.")
    para("Bảng đồng thuận chỉ hỗ trợ quan sát; ứng dụng vẫn giữ Naive Bayes làm kết quả chính, không dùng biểu quyết để thay nhãn của mô hình. Dòng có lỗi không được tự gán nhãn. Các điểm xác suất chưa được hiệu chỉnh, còn biên SVM có thể âm hoặc dương và không nằm trong [0,1]. Tệp tải xuống giữ mã dòng, sample_id, nhãn, điểm chưa làm tròn và thông báo lỗi.")
    para("Tiêu đề kết quả ghi nhãn theo dữ liệu Potability. Thanh điểm thể hiện xác suất ước lượng lớp 1 ở cả hai trường hợp nhãn, không đảo thanh sang lớp 0. Mẫu thử A và B không có nhãn kiểm nghiệm và không được dùng làm đáp án để đánh giá mô hình.")

    page("3. Dự đoán CSV và tra cứu dữ liệu gốc")
    para("CSV phải có đủ chín tên cột và mã hóa UTF-8. Người dùng mở Định dạng tệp và dữ liệu mẫu để tải ví dụ, chọn Upload rồi nhấn Dự đoán CSV. Cột sample_id được giữ để đối chiếu; Potability nếu có trong tệp không tham gia vector đầu vào. Mỗi dòng được xử lý bằng cả sáu mô hình. Kết quả lỗi vẫn được lưu theo dòng và thuật toán.")
    para("Tệp minh họa trích các ID gốc 0, 27 và 100, bỏ cột nhãn và giữ sample_id. Kiểm tra bằng cùng luồng dự đoán CSV xử lý ba dòng hợp lệ, không có lỗi đầu vào và tạo 18 bản ghi kết quả. Naive Bayes dự đoán nhãn 1 cho một trong ba mẫu, nên thẻ Tỷ lệ uống được (NB) hiển thị 33,3%. Đây là tỷ lệ nhãn dự đoán trong tệp, không phải độ chính xác trên ba mẫu.")
    para("ID dòng trong bảng bắt đầu từ 0 theo vị trí ở tệp tải lên. Khi cần truy lại dữ liệu ban đầu, dùng sample_id trong CSV kết quả. Màn hình giới hạn 120 bản ghi kết quả để thuận tiện quan sát; nút Tải toàn bộ kết quả CSV xuất đầy đủ các dòng đã xử lý.")
    heading("3.1. Tra cứu và tải dữ liệu gốc", 3)
    para("Tab Dữ liệu gốc hiển thị toàn bộ 3.276 bản ghi trước tiền xử lý, gồm chín chỉ tiêu và Potability. Người dùng có thể cuộn bảng để xem đủ các cột, tra cứu những ô thiếu và tải tệp qua nút Tải toàn bộ CSV gốc. Số mẫu hai lớp và số ô thiếu được trình bày ngay trên màn hình. Bảng chỉ làm tròn số khi hiển thị; tệp tải xuống giữ nguyên byte của CSV nguồn.")

    page("4. Xem kết quả thực nghiệm trên ứng dụng")
    para("Tab Kết quả thực nghiệm trình bày các thẻ chỉ số, bảng sáu thuật toán và biểu đồ cột Accuracy, Balanced Accuracy. Những thẻ đầu trang nhấn mạnh cấu hình KNN có điểm cao nhất trong lần chạy và vẫn ghi riêng accuracy của Naive Bayes. Số liệu này được đọc từ hồ sơ đã lưu, không tính từ mẫu người dùng vừa nhập.")
    para("Bên dưới có bốn tab biểu đồ: Đối chiếu hiệu năng, Đường cong ROC, Ma trận nhầm lẫn và Phân phối dữ liệu (EDA). Các hình sử dụng cùng hồ sơ với phần III, giúp người dùng xem lại kết quả mà không phải mở từng tệp. Biểu đồ đối chiếu hiệu năng hiển thị hai chỉ số Accuracy và Balanced Accuracy; ROC dùng điểm đầu ra, ma trận nhầm lẫn dùng nhãn tại ngưỡng cố định.")

    page("5. Thông tin chỉ tiêu và phạm vi sử dụng")
    para("Tab Thông tin 9 chỉ tiêu giải thích tên cột và nhóm thuộc tính, giúp người dùng đối chiếu số đo với CSV. Ứng dụng giữ thang đo dữ liệu nguồn. Không dùng các khoảng tự đặt trong giao diện để suy ra nhãn, và không trình bày nhãn mô hình là chứng nhận đạt tiêu chuẩn nước uống.")
    para("Nguồn công bố CSV chưa xác minh đầy đủ đơn vị của từng cột, địa điểm, thời gian lấy mẫu và quy trình tạo nhãn. Vì vậy, người dùng cần giữ cách biểu diễn trong tệp nguồn khi thử ứng dụng. Các bộ số thử chỉ giúp thao tác nhập dữ liệu; tỷ lệ đồng thuận giữa sáu mô hình không chứng minh mẫu an toàn.")
    para("Các bước tiền xử lý được áp dụng theo từng mô hình đã lưu. NB và ID3 dùng trung vị và mã khoảng; KNN, Logistic và SVM dùng trung vị, mean và std. C4.5 giữ ô thiếu để phối hợp các nhánh theo trọng số. Không có ngưỡng hiển thị nào được đưa thêm vào bước quyết định nhãn.")

    page("6. Kiểm thử ứng dụng và khả năng mở rộng")
    table(["Tình huống kiểm tra","Kết quả mong đợi"],[("Nhập một mẫu hợp lệ","Hiển thị NB và đủ năm mô hình đối chiếu"),("CSV nhiều dòng","Giữ thứ tự, ID mẫu, kết quả đủ sáu mô hình"),("Có ô thiếu","Điền trung vị theo mô hình; C4.5 chia trọng số nhánh"),("Chuỗi không phải số","Báo lỗi dòng và không tự gán nhãn"),("Tất cả chỉ tiêu trống","Từ chối dự đoán"),("Model bị thay đổi byte","Dừng nạp khi hash không khớp"),("Nhấn dự đoán nhiều lần","Dùng cùng trạng thái đã học")],[7.3,8])
    para("Các chức năng được kiểm tra bằng AppTest; giao diện nhập một mẫu được kiểm tra thêm trên trình duyệt. Bộ kiểm thử tại phiên bản khảo sát có 328 ca đạt. Với mẫu trung vị, ứng dụng hiển thị đủ sáu kết quả và mức đồng thuận 6/6 cho lớp 0. Đã kiểm tra nạp trung vị, CSV ba mẫu và đối chiếu CSV gốc với nguồn. Tệp dữ liệu tải xuống khớp nguyên byte với nguồn. Bảng 21 liệt kê các tình huống và kết quả mong đợi khi kiểm tra ứng dụng.")
    para("Ứng dụng hiện phục vụ thực hành phân lớp trong bài tập. Dữ liệu chưa có đầy đủ thông tin lấy mẫu và kiểm nghiệm; tập kiểm tra cũng đã từng được sử dụng ở phiên bản trước. Vì vậy, cần có dữ liệu độc lập và nhãn được xác định rõ trước khi đánh giá khả năng sử dụng ngoài phạm vi bài tập.")
    para("Hướng phát triển tiếp theo là bổ sung dữ liệu có nguồn và quy trình gán nhãn rõ ràng, thử cách biểu diễn thuộc tính khác trong kiểm định chéo, đồng thời đánh giá độ hiệu chỉnh của xác suất. Cấu hình được lựa chọn từ development cần được giữ nguyên khi đánh giá trên tập dữ liệu mới.")

    heading("KẾT LUẬN",1,True)
    para("Nhóm đã xây dựng quy trình phân loại Potability từ CSV gốc, gồm kiểm tra và tiền xử lý dữ liệu, cài đặt sáu thuật toán, thực nghiệm với Weka và phát triển ứng dụng web. Naive Bayes là thuật toán chính, được trình bày bằng công thức, ví dụ tính tay và bảng đếm cho một mẫu trong tập kiểm tra.")
    para("Trên 656 mẫu kiểm tra, Naive Bayes đạt accuracy 60,37%, BA 51,61% và ROC-AUC 0,5218. KNN có kết quả cao nhất trong sáu cấu hình, với accuracy 64,79%, BA 60,72% và AUC 0,6518. Điểm hạn chế nổi bật của Naive Bayes là chỉ nhận đúng 30/256 mẫu uống được, dù accuracy trên toàn tập đạt hơn 60%.")
    para("Bộ phân lớp Naive Bayes đếm tần suất trên Weka cho cùng nhãn dự đoán với Python trên 656 mẫu. C4.5 đầy đủ cũng khớp J48 trên toàn bộ test và năm fold; cây sau cắt tỉa có 21 nút, 11 lá. Kết quả ID3 và SVM khác giữa hai môi trường do cách xử lý nhánh chưa gặp, tham số và cơ chế tối ưu. Riêng 53 mẫu không được ID3 của Weka gán nhãn vẫn được giữ trong mẫu số đánh giá.")
    para("Kết quả hiện tại cho thấy cần cải thiện khả năng nhận diện lớp 1 của Naive Bayes. Việc bổ sung dữ liệu hoặc thay đổi tiền xử lý phải được kiểm tra bằng thực nghiệm trên dữ liệu có nguồn gốc phù hợp. Ứng dụng chỉ dự đoán nhãn theo bộ dữ liệu nghiên cứu và chưa thay thế được việc kiểm nghiệm chất lượng nước.")
    heading("LỜI CẢM ƠN", 1, True)
    para("Nhóm chúng em xin gửi lời cảm ơn đến TS. Trần Mạnh Tuấn, giảng viên học phần Khai phá dữ liệu. Những kiến thức và tài liệu của học phần là cơ sở để nhóm tìm hiểu bài toán phân lớp, xây dựng thuật toán Naive Bayes và tổ chức thực nghiệm so sánh trong bài tập lớn này.")
    para("Chúng em cảm ơn Khoa Công nghệ thông tin, Trường Đại học Thủy Lợi đã tạo điều kiện học tập và thực hành. Trong quá trình thực hiện, các thành viên đã cùng kiểm tra dữ liệu, trao đổi cách cài đặt thuật toán và đối chiếu kết quả để hoàn thiện báo cáo cũng như ứng dụng dự đoán.")
    para("Do kiến thức và kinh nghiệm thực nghiệm còn hạn chế, bài làm có thể còn thiếu sót. Nhóm chúng em mong nhận được góp ý của thầy để hiểu rõ hơn những hạn chế của phương pháp và cải thiện cách giải quyết bài toán.")
    thanks = para("Nhóm 11")
    thanks.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    heading("TÀI LIỆU THAM KHẢO",1,True)
    for text in [
        "[1]. Giáo trình Khai phá dữ liệu, TS. Trần Mạnh Tuấn.",
        "[2]. Bài giảng Khai phá dữ liệu, TS. Trần Mạnh Tuấn.",
        "[3]. Cornell University. CS 4780/5780 Machine Learning, Lecture Notes. https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/ . Truy cập 24/11/2026.",
        "[4]. Aditya Kadiwal. Water Quality, Kaggle, phiên bản 3, water_potability.csv. https://www.kaggle.com/datasets/adityakadiwal/water-potability . Bản tải đối chiếu ngày 24/11/2026.",
        "[5]. University of Waikato. Weka Knowledge Explorer và tài liệu API Weka 3.8. https://ml.cms.waikato.ac.nz/weka/gui_explorer.html và https://weka.sourceforge.io/doc.stable-3-8/ . Truy cập 24/11/2026.",
        "[6]. Weka team. simpleEducationalLearningSchemes 1.0.1. https://weka.sourceforge.io/packageMetaData/simpleEducationalLearningSchemes/1.0.1.html . Truy cập 24/11/2026.",
        "[7]. J. R. Quinlan. C4.5: Programs for Machine Learning. Morgan Kaufmann, 1993. Mô tả thuật toán và cấu hình đối chiếu được dẫn trong tài liệu J48 của University of Waikato [5]."]:
        reference = para(text)
        reference.alignment = WD_ALIGN_PARAGRAPH.LEFT
        reference.paragraph_format.left_indent = Cm(0.75)
        reference.paragraph_format.first_line_indent = Cm(-0.75)
    appendix = doc.add_section(WD_SECTION_START.NEW_PAGE)
    appendix.orientation = WD_ORIENT.LANDSCAPE
    appendix.page_width, appendix.page_height = Cm(29.7), Cm(21)
    appendix.left_margin, appendix.right_margin = Cm(2.5), Cm(2)
    appendix.top_margin = appendix.bottom_margin = Cm(2)
    # Tiếp tục số trang, không lặp lại số 1 của phần thân.
    appendix_numbering = appendix._sectPr.find(qn("w:pgNumType"))
    if appendix_numbering is not None:
        appendix._sectPr.remove(appendix_numbering)
    heading("PHỤ LỤC A. DỮ LIỆU THÔ", 1)
    para("Bảng dưới trích 13 bản ghi, gồm 12 dòng đầu và dòng có ID 27 dùng trong ví dụ tính Naive Bayes. ID bắt đầu từ 0, không phải cột của CSV. Ô rỗng là dữ liệu thiếu; số chỉ được làm tròn bốn chữ số thập phân trong bảng. Nhãn 0/1 được giữ nguyên. Toàn bộ dữ liệu nằm trong tệp water_potability.csv đi kèm báo cáo.")
    table(["ID", "ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_\ncarbon", "Trihalo-\nmethanes", "Turbidity", "Potability"],
          [(i, *("" if X[i][j] is None else number(X[i][j]) for j in range(9)), y[i])
           for i in (*range(12), 27)],
          [0.8, 1.4, 2.5, 2.9, 2.5, 2.3, 2.9, 2.5, 2.8, 2.3, 2.0], font_size=9, compact=True)
    note = para("Nguồn: Water Quality, Aditya Kadiwal, Kaggle, phiên bản 3 [4]. CSV gốc có 3.276 dòng dữ liệu và 10 cột; 491 ô ph, 781 ô Sulfate và 162 ô Trihalomethanes bị thiếu. Không có thao tác điền thiếu, chuẩn hóa hoặc chia khoảng trong tệp này.")
    note.paragraph_format.first_line_indent = Cm(0)
    for run in note.runs:
        font(run, 11)
    # Xóa quan hệ ảnh thân báo cáo mẫu đã được thay thế, giữ logo/bìa đang dùng.
    assert table_index == len(TABLE_CAPTIONS)
    used_relationships=set(doc.element.xpath('//@r:embed')) | set(doc.element.xpath('//@r:id'))
    for identifier, rel in list(doc.part.rels.items()):
        if "image" in rel.reltype and identifier not in used_relationships:
            doc.part.drop_rel(identifier)
    output=ROOT/"output"/"docx"/("BaiTapLon-cong-khai.docx" if public else "BaiTapLon-C45.docx")
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
