"""Render paper.md into a journal-style one-column PDF (peer-review / preprint format)."""

import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table,
    TableStyle, Image, ListFlowable, ListItem, KeepTogether, HRFlowable,
)

BASE = r'C:\Users\dhans\Desktop\research\tech'
MD_PATH = os.path.join(BASE, 'paper.md')
RES = os.path.join(BASE, 'results')
OUT_PATH = os.path.join(BASE, 'paper.pdf')

# Figure number -> result image file (order of appearance == numbering)
FIGURE_IMAGES = {
    1: 'fig1_performance.png',
    2: 'fig2_heatmap.png',
    3: 'fig3_key_isolation.png',
    4: 'fig7_family_shift.png',
    5: 'fig4_degradation.png',
    6: 'fig11_wasserstein_vs_f1drop.png',
    7: 'fig12_threshold_sweep.png',
    8: 'fig10_rocauc.png',
}

SHORT_TITLE = 'Evaluating Intrusion Detection Models Under Distribution Shift'

# ---------------------------------------------------------------- styles
BODY_FONT = 'Times-Roman'
BOLD_FONT = 'Times-Bold'
ITAL_FONT = 'Times-Italic'
BOLD_ITAL = 'Times-BoldItalic'
MONO_FONT = 'Courier'
HFONT = 'Helvetica'
HBOLD = 'Helvetica-Bold'

st_meta = ParagraphStyle('meta', fontName=BODY_FONT, fontSize=10, leading=13.5,
                         alignment=TA_CENTER, spaceAfter=2)
st_title = ParagraphStyle('title', fontName=BOLD_FONT, fontSize=17, leading=20,
                          alignment=TA_CENTER, spaceAfter=10)
st_h1 = ParagraphStyle('h1', fontName=BOLD_FONT, fontSize=12.5, leading=16,
                       spaceBefore=14, spaceAfter=6, keepWithNext=1)
st_h2 = ParagraphStyle('h2', fontName=BOLD_FONT, fontSize=11, leading=14,
                       spaceBefore=10, spaceAfter=4, keepWithNext=1)
st_body = ParagraphStyle('body', fontName=BODY_FONT, fontSize=10, leading=13.2,
                         alignment=TA_JUSTIFY, spaceAfter=6)
st_body_prev = ParagraphStyle('body2', parent=st_body, spaceBefore=2)
st_caption = ParagraphStyle('caption', fontName=ITAL_FONT, fontSize=8.5, leading=11,
                            alignment=TA_LEFT, spaceBefore=4, spaceAfter=8)
st_caption_noimg = ParagraphStyle('caption_noimg', parent=st_caption)
st_cell = ParagraphStyle('cell', fontName=BODY_FONT, fontSize=8, leading=10)
st_cell_h = ParagraphStyle('cellh', fontName=BOLD_FONT, fontSize=8, leading=10)
st_cellc = ParagraphStyle('cellc', fontName=BODY_FONT, fontSize=8, leading=10, alignment=TA_CENTER)
st_bullet = ParagraphStyle('bullet', fontName=BODY_FONT, fontSize=9.5, leading=12.5)
st_section_sub = ParagraphStyle('labelsub', fontName=ITAL_FONT, fontSize=10, leading=13.2,
                                alignment=TA_JUSTIFY, spaceAfter=6, leftIndent=0)
st_kw = ParagraphStyle('kw', fontName=BODY_FONT, fontSize=9.5, leading=12.5, spaceAfter=8)

# ---------------------------------------------------------------- inline text
TOKEN_RE = re.compile(r'(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)')

def inline(text):
    def repl(m):
        t = m.group(0)
        if t.startswith('**'):
            return '<font face="%s">%s</font>' % (BOLD_FONT, t[2:-2])
        if t.startswith('`'):
            return '<font face="%s">%s</font>' % (MONO_FONT, t[1:-1])
        return '<font face="%s">%s</font>' % (ITAL_FONT, t[1:-1])
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = text.replace('---', '&#8212;')
    text = text.replace('--', '&#8211;')
    text = re.sub(r'([0-9])\s*[xX]\s*([0-9])', r'\1x\2', text)
    return TOKEN_RE.sub(repl, text)

def is_caption(par):
    return bool(re.match(r'^[*_]{0,2}(Table|Figure)\s+\d+\s*:', par.strip()))

def parse_table(lines, i):
    """Parse a pipe table starting at lines[i]; returns (Table flowable, next_index)."""
    rows = []
    align = []
    while i < len(lines) and lines[i].strip().startswith('|'):
        cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
        rows.append(cells)
        i += 1
    header = rows[0]
    if len(rows) >= 2 and re.match(r'^[\s:|-]+$', '|'.join(rows[1]).replace('|', '')):
        sep = rows[1]
        align = [('RIGHT' if s.rstrip().endswith(':') and not s.lstrip().startswith(':')
                  else 'LEFT') if ':' in s else 'LEFT' for s in sep]
        rows = [header] + rows[2:]
    else:
        align = ['LEFT'] * len(header)
        sep = None
    ncols = len(header)
    table_data = []
    for r in rows:
        if len(r) < ncols:
            r = r + [''] * (ncols - len(r))
        table_data.append([Paragraph(inline(c), st_cell_h if r is header else st_cellc)
                           for c in r])
    avail = 451.0
    widths = [avail / ncols] * ncols
    t = Table(table_data, colWidths=widths, repeatRows=1, hAlign='CENTER')
    style = [
        ('FONT', (0, 0), (-1, 0), BOLD_FONT, 8),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8e8e8')),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#999999')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(style))
    return t, i

def add_figure(num):
    fn = FIGURE_IMAGES[num]
    img = Image(os.path.join(RES, fn))
    target = 451.0
    if num in (2, 5):
        target = 380.0
    if num == 8:
        target = 400.0
    scale = min(target / img.imageWidth, 1.0)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    img.hAlign = 'CENTER'
    return img

def render_caption(text):
    m = re.match(r'^[*_]{0,2}(Table|Figure)\s+(\d+)\s*:(.*?)[*_]*$', text.strip())
    label = '%s %s:' % (m.group(1), m.group(2))
    rest = m.group(3)
    style = st_caption
    p = Paragraph('<b><i>%s</i></b>%s' % (label, inline(rest)), style)
    return p

# ---------------------------------------------------------------- doc
class PaperDoc(BaseDocTemplate):
    def __init__(self, fn, **kw):
        super().__init__(fn, **kw)

def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFont(HFONT, 7.5)
    canvas.setFillColor(colors.HexColor('#555555'))
    canvas.drawString(1 * inch, h - 0.55 * inch, SHORT_TITLE[:95])
    canvas.drawRightString(w - 1 * inch, h - 0.55 * inch, 'Dhansika.R')
    canvas.setFont(HFONT, 8.5)
    canvas.setFillColor(colors.black)
    canvas.drawCentredString(w / 2.0, 0.55 * inch, str(canvas.getPageNumber()))
    canvas.restoreState()

doc = PaperDoc(OUT_PATH, pagesize=A4, leftMargin=1 * inch, rightMargin=1 * inch,
               topMargin=0.7 * inch, bottomMargin=0.75 * inch,
               title=SHORT_TITLE, author='Dhansika.R')
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='main')
doc.addPageTemplates([PageTemplate(id='pt', frames=[frame], onPage=on_page)])

lines = open(MD_PATH, 'r', encoding='utf-8').read().split('\n')

story = []
i = 0
_pending = []          # items awaiting a table caption (table + its caption stay together)

# ---- title block
while i < len(lines) and lines[i].strip() == '':
    i += 1
if lines[i].startswith('# '):
    story.append(Paragraph(inline(lines[i][2:].strip()), st_title))
    i += 1
if i < len(lines) and lines[i].strip() == '---':
    i += 1
while i < len(lines) and lines[i].strip() != '' and not lines[i].startswith('#'):
    line = lines[i].strip()
    if line.startswith('**'):
        story.append(Paragraph(inline(line), st_meta))
    else:
        story.append(Paragraph(inline(line), st_meta))
    i += 1

def flush_pending():
    global _pending
    if _pending:
        story.append(KeepTogether(_pending))
        _pending = []

def other_content(item):
    flush_pending()
    story.append(item)

while i < len(lines):
    line = lines[i].rstrip()
    if line.strip() == '':
        i += 1
        continue
    if line.lstrip().startswith('```'):
        i += 1
        continue
    if line.strip() == '---':
        i += 1
        other_content(Spacer(1, 8))
        continue
    if line.startswith('## ') or line.startswith('### '):
        level = '##' if line.startswith('##') else '###'
        text = line.strip().lstrip('#').strip()
        other_content(Paragraph(inline(text), st_h1 if level == '##' else st_h2))
        i += 1
        continue
    if line.startswith('|'):
        t, i = parse_table(lines, i)
        _pending.append(Spacer(1, 3))
        _pending.append(t)
        _pending.append(Spacer(1, 2))
        continue
    if re.match(r'^[-*]\s+', line):
        items = []
        while i < len(lines) and re.match(r'^[-*]\s+', lines[i].strip()):
            items.append(Paragraph(inline(re.sub(r'^[-*]\s+', '', lines[i].strip())), st_bullet))
            i += 1
        other_content(ListFlowable(items, bulletType='bullet', start='circle',
                                   bulletFontSize=4, bulletOffsetY=0))
        other_content(Spacer(1, 4))
        continue
    if re.match(r'^\d+\.\s', line.strip()):
        items = []
        while i < len(lines) and re.match(r'^\d+\.\s', lines[i].strip()):
            items.append(Paragraph(inline(re.sub(r'^\d+\.\s', '', lines[i].strip())), st_bullet))
            i += 1
        other_content(ListFlowable(items, bulletType='1', bulletFontSize=9, leftIndent=18))
        other_content(Spacer(1, 4))
        continue
    # paragraph / caption
    text = line.strip()
    if is_caption(text):
        m = re.match(r'^[*_]{0,2}(Table|Figure)\s+(\d+)\s*:', text.strip())
        kind = m.group(1)
        num = int(m.group(2))
        if kind == 'Figure':
            flush_pending()
            img = add_figure(num)
            style2 = ParagraphStyle('capfig', parent=st_caption, spaceBefore=10, spaceAfter=2)
            cap = render_caption(text)
            cap.style = style2
            story.append(KeepTogether([img, cap]))
        else:
            _pending.append(render_caption(text))
            flush_pending()
        i += 1
        continue
    # merge continuation lines of the same paragraph
    para = text
    i += 1
    while i < len(lines) and lines[i].strip() != '' and not lines[i].startswith('#') \
            and not lines[i].startswith('|') and not is_caption(lines[i].strip()):
        para += ' ' + lines[i].strip()
        i += 1
    other_content(Paragraph(inline(para), st_body))

story.extend(_pending) if _pending else None
doc.build(story)
print('PDF written:', OUT_PATH)