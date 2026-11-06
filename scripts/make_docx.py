"""Markdown -> DOCX converter for the manuscript (headings, bold/italic, bullets,
pipe tables, images with captions)."""
import re, sys, os
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
sys.stdout.reconfigure(encoding="utf-8")

if len(sys.argv) < 3:
    raise SystemExit("usage: make_docx.py INPUT.md OUTPUT.docx")
SRC, DST = sys.argv[1], sys.argv[2]
BASE = os.path.dirname(SRC)


def set_font(run, name="Times New Roman", size=11, bold=None, italic=None):
    run.font.name = name
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rf)
    rf.set(qn("w:eastAsia"), "SimSun")


def add_runs(par, text, size=11, base_bold=False):
    """Handle **bold**, *italic*, `code`."""
    tokens = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            r = par.add_run(tok[2:-2]); set_font(r, size=size, bold=True)
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = par.add_run(tok[1:-1]); set_font(r, size=size, italic=True, bold=base_bold)
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            r = par.add_run(tok[1:-1]); set_font(r, name="Consolas", size=size - 1, bold=base_bold)
        else:
            r = par.add_run(tok); set_font(r, size=size, bold=base_bold)


doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
pf = st.paragraph_format
pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
pf.space_after = Pt(6)

for s in doc.sections:
    s.top_margin = Cm(2.54); s.bottom_margin = Cm(2.54)
    s.left_margin = Cm(2.54); s.right_margin = Cm(2.54)

# page-number footer
from docx.oxml import OxmlElement
for s in doc.sections:
    para = s.footer.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run()
    for el, attrs, text in [("w:fldChar", {"w:fldCharType": "begin"}, None),
                            ("w:instrText", {"xml:space": "preserve"}, " PAGE "),
                            ("w:fldChar", {"w:fldCharType": "end"}, None)]:
        e = OxmlElement(el)
        for k, v in attrs.items():
            e.set(qn(k), v)
        if text:
            e.text = text
        run._r.append(e)
    set_font(run, size=9)

lines = open(SRC, encoding="utf-8").read().split("\n")
i = 0
n_tab = 0
while i < len(lines):
    ln = lines[i].rstrip()
    if not ln.strip():
        i += 1
        continue

    # ---- tables
    if ln.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
        header = [c.strip() for c in ln.strip().strip("|").split("|")]
        i += 2
        body = []
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            body.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
            i += 1
        t = doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = True
        fsz = 9 if len(header) <= 8 else (8 if len(header) <= 10 else 7)
        for j, h in enumerate(header):
            cell = t.rows[0].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.paragraph_format.space_after = Pt(0)
            add_runs(p, h, size=fsz, base_bold=True)
        for row in body:
            cells = t.add_row().cells
            for j, c in enumerate(row[:len(header)]):
                cells[j].text = ""
                p = cells[j].paragraphs[0]
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                p.paragraph_format.space_after = Pt(0)
                add_runs(p, c, size=fsz)
        n_tab += 1
        doc.add_paragraph()
        continue

    # ---- images
    m = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", ln)
    if m:
        alt, path = m.group(1), m.group(2)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        full = path if os.path.isabs(path) else os.path.join(BASE, path)
        if os.path.exists(full):
            p.add_run().add_picture(full, width=Cm(15.4))
        else:
            add_runs(p, f"[missing figure: {path}]", size=9)
        i += 1
        continue

    # ---- headings
    m = re.match(r"^(#{1,4})\s+(.*)$", ln)
    if m:
        level, txt = len(m.group(1)), m.group(2).strip()
        p = doc.add_paragraph(style=f"Heading {min(level, 3)}")
        p.paragraph_format.space_before = Pt(12 if level <= 2 else 8)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        r = p.add_run(txt)
        set_font(r, size={1: 15, 2: 13, 3: 11.5, 4: 11}.get(level, 11), bold=True)
        r.font.color.rgb = RGBColor(0, 0, 0)
        i += 1
        continue

    # ---- bullets / numbered
    m = re.match(r"^\s*[-*]\s+(.*)$", ln)
    if m:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        add_runs(p, m.group(1))
        i += 1
        continue
    m = re.match(r"^\s*(\d+)[.)]\s+(.*)$", ln)
    if m:
        p = doc.add_paragraph(style="List Number")
        p.paragraph_format.space_after = Pt(3)
        add_runs(p, m.group(2))
        i += 1
        continue

    # ---- horizontal rule / page break marker
    if ln.strip() in ("---", "***"):
        i += 1
        continue

    # ---- paragraph (join consecutive non-empty lines)
    buf = [ln.strip()]
    i += 1
    while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,4}\s|!\[|\s*[-*]\s|\s*\d+[.)]\s|\|)", lines[i]):
        buf.append(lines[i].strip())
        i += 1
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0.0)
    add_runs(p, " ".join(buf))

doc.save(DST)
print("wrote", DST, "tables:", n_tab)
