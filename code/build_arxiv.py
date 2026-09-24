# -*- coding: utf-8 -*-
"""Convert paper.md -> arXiv-ready LaTeX (main.tex) + copy figures."""
import re, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'paper.md')
OUT = os.path.join(HERE, 'arxiv')
RESULTS = os.path.join(HERE, 'results')
FIGMAP = {
    1: 'fig1_performance.png',
    2: 'fig2_heatmap.png',
    3: 'fig3_key_isolation.png',
    4: 'fig7_family_shift.png',
    5: 'fig4_degradation.png',
    6: 'fig11_wasserstein_vs_f1drop.png',
    7: 'fig12_threshold_sweep.png',
    8: 'fig10_rocauc.png',
}

PRE = r"""\documentclass[10pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage[hidelinks]{hyperref}
\emergencystretch=3em
\newcommand{\ra}[1]{\renewcommand{\arraystretch}{#1}}
\title{%s}
\author{Dhansika.R \\ \small Independent Researcher%s}
\date{%s}
\begin{document}
\maketitle
\noindent\textit{Preprint note: %s}\par
\begin{abstract}
%s
\end{abstract}
\noindent\textbf{Keywords:} %s
"""


def esc(s):
    return (s.replace('\\', r'\textbackslash{}')
             .replace('%', r'\%').replace('&', r'\&').replace('#', r'\#')
             .replace('_', r'\_').replace('$', r'\$')
             .replace('{', r'\{').replace('}', r'\}')
             .replace('~', r'\textasciitilde{}').replace('^', r'\textasciicircum{}'))


def paired_quotes(s):
    out, opn = [], True
    for ch in s:
        if ch == '"':
            out.append('``' if opn else "''")
            opn = not opn
        else:
            out.append(ch)
    return ''.join(out)


def inline(s):
    s = s.replace('P(Y|X)', r'$P(Y \mid X)$')
    s = s.replace('W1(f) = W(P_train(f), P_test(f))',
                  r'$W_1(f) = W(P_{\mathrm{train}}(f),\, P_{\mathrm{test}}(f))$')
    s = s.replace('[-10^6, 10^6]', r'$[-10^{6},\ 10^{6}]$')

    math_tok = {}
    def keepmath(m):
        k = len(math_tok) + 1
        math_tok[k] = m.group(0)
        return '\x01%d\x01' % k
    s = re.sub(r'\$[^$]+\$', keepmath, s)

    s = re.sub(r'`([^`]+)`', lambda m: r'\texttt{' + m.group(1).replace('_', r'\_') + '}', s)

    code_tok = {}
    def prot(m):
        k = len(code_tok) + 1
        code_tok[k] = m.group(0)
        return '\x02%d\x02' % k
    s = re.sub(r'\\texttt\{[^}]*\}', prot, s)

    s = re.sub(r'\*\*(.+?)\*\*', lambda m: r'\textbf{' + esc(m.group(1)) + '}', s)
    s = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', lambda m: r'\textit{' + esc(m.group(1)) + '}', s)
    s = re.sub(r'\\(textbf|textit)\{[^}]*\}', prot, s)

    s = paired_quotes(s)
    s = esc(s)

    for k, v in code_tok.items():
        s = s.replace('\x02%d\x02' % k, v)
    for k, v in math_tok.items():
        s = s.replace('\x01%d\x01' % k, v)
    return s


def heading_no(t):
    return re.sub(r'^[\d.]+\s+', '', t).strip()


def col_spec(cells):
    specs = []
    for i, h in enumerate(cells):
        if i == 0:
            specs.append('l')
        elif 'top shifted' in h.lower():
            specs.append('p{4.5cm}')
        else:
            specs.append('r')
    return ' '.join(specs)


def table_tex(rows, caption):
    header = [c.strip() for c in rows[0].split('|')]
    if header and header[0] == '':
        header.pop(0)
    if header and header[-1] == '':
        header.pop()
    data = []
    for r in rows[1:]:
        cells = [c.strip() for c in r.split('|')]
        if cells and cells[0] == '':
            cells.pop(0)
        if cells and cells[-1] == '':
            cells.pop()
        if cells:
            data.append(cells)
    out = []
    out.append(r'\begin{table}[ht]')
    out.append(r'\centering')
    out.append(r'\small')
    out.append(r'\setlength{\tabcolsep}{5pt}')
    out.append(r'\caption{%s}' % inline(caption))
    out.append(r'\begin{tabular}{%s}' % col_spec(header))
    out.append(r'\toprule')
    out.append(' & '.join(inline(c) for c in header) + r' \\')
    out.append(r'\midrule')
    for a in data:
        out.append(' & '.join(inline(c) for c in a) + r' \\')
    out.append(r'\bottomrule')
    out.append(r'\end{tabular}')
    out.append(r'\end{table}')
    return '\n'.join(out)


def figure_tex(num, caption):
    out = []
    out.append(r'\begin{figure}[ht]')
    out.append(r'\centering')
    out.append(r'\includegraphics[width=0.9\textwidth]{fig%d.png}' % num)
    out.append(r'\caption{%s}' % inline(caption))
    out.append(r'\label{fig:%d}' % num)
    out.append(r'\end{figure}')
    return '\n'.join(out)


def main():
    lines = open(SRC, 'r', encoding='utf-8').read().split('\n')

    title = author = affiliation = date_s = email = ''
    preprint = ''
    abstract = []
    keywords = ''

    out = []
    pending_rows = []

    in_abstract = False
    saw_keywords = False
    highlights = []
    in_highlights = False
    in_refs = False
    in_declarations = False
    ref_n = 0
    in_list = None
    list_pre = None
    n_tables = 0
    n_figs = 0
    n_sections = 0

    def flush_list():
        nonlocal in_list
        if in_list:
            out.append(r'\end{' + in_list + '}')
            in_list = None

    def flush_table():
        nonlocal pending_rows
        if pending_rows:
            out.append(table_tex(pending_rows, ''))
            pending_rows = []

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        i += 1
        if not line or line == '---' or line == '***':
            continue

        m = re.match(r'^\*\*Author:\*\*\s*(.*)$', line)
        if m:
            author = m.group(1).strip()
            continue
        m = re.match(r'^\*\*Affiliation:\*\*\s*(.*)$', line)
        if m:
            affiliation = m.group(1).strip()
            continue
        m = re.match(r'^\*\*Date:\*\*\s*(.*)$', line)
        if m:
            date_s = m.group(1).strip()
            continue
        m = re.match(r'^\*\*Correspondence:\*\*\s*(.*)$', line)
        if m:
            email = m.group(1).strip()
            continue
        m = re.match(r'^\*\*Preprint note:\*\*\s*(.*)$', line)
        if m:
            preprint = m.group(1).strip()
            continue

        # headings
        if line.startswith('#'):
            flush_list()
            flush_table()
            if line.startswith('# '):
                title = heading_no(line[2:].strip())
                continue
            h = line.lstrip('#').strip()
            if line.startswith('## Abstract'):
                in_abstract = True
                continue
            if h == 'Keywords' or h.startswith('**Keywords**'):
                continue
            if line.startswith('## References'):
                in_abstract = False
                in_refs = True
                in_declarations = False
                out.append(r'\section*{References}')
                out.append(r'\begin{thebibliography}{99}')
                continue
            if line.startswith('## Declarations'):
                in_abstract = False
                in_declarations = True
                out.append(r'\section*{Declarations}')
                continue
            is_sub = line.startswith('###')
            hn = heading_no(h)
            if is_sub or re.match(r'^\d+\.\s', h) or re.match(r'^\d+\.\d+\.\s', h):
                command = r'\subsection' if is_sub else r'\section'
            else:
                command = r'\section'
            if not is_sub:
                n_sections += 1
            out.append(command + '{' + inline(hn) + '}')
            continue

        # keywords line
        m = re.match(r'^\*\*Keywords:\*\*\s*(.*)$', line)
        if m:
            keywords = m.group(1).strip()
            saw_keywords = True
            in_abstract = False
            continue

        # highlights header
        if line == '**Highlights**':
            in_highlights = True
            continue

        # figure caption
        m = re.match(r'^\*Figure (\d+):\s*(.*)\*$', line)
        if m:
            flush_list()
            flush_table()
            out.append(figure_tex(int(m.group(1)), m.group(2)))
            n_figs += 1
            continue

        # table caption with pending rows
        m = re.match(r'^\*Table (\d+):\s*(.*)\*$', line)
        if m:
            if pending_rows:
                out.append(table_tex(pending_rows, m.group(2)))
                pending_rows = []
                n_tables += 1
            continue

        if line.startswith('|'):
            pending_rows.append(line)
            continue
        flush_table()

        if in_refs:
            ref_n += 1
            out.append(r'\bibitem{ref:%d} %s' % (ref_n, inline(line)))
            continue

        if in_abstract:
            abstract.append(inline(line))
            continue

        if in_highlights:
            if line.startswith('- '):
                highlights.append(r'\item ' + inline(line[2:]))
                continue
            else:
                in_highlights = False

        if line.startswith('- '):
            if in_list is None:
                if list_pre:
                    out.append(list_pre)
                    list_pre = None
                else:
                    out.append(r'\begin{itemize}')
                in_list = 'itemize'
            out.append(r'\item ' + inline(line[2:]))
            continue

        m = re.match(r'^(\d+)\.\s+(.*)$', line)
        if m:
            if in_list is None:
                out.append(r'\begin{enumerate}')
                in_list = 'enumerate'
            out.append(r'\item ' + inline(m.group(2)))
            continue

        if in_list:
            flush_list()

        if in_declarations or not in_refs:
            out.append(inline(line))
            out.append('')

    if in_list:
        out.append(r'\end{' + in_list + '}')

    hl = ''.join(highlights)
    highlights_tex = ''
    if highlights:
        highlights_tex = ('\\noindent\\textbf{Highlights}\n'
                          '\\begin{itemize}\n'
                          + '\n'.join(highlights) +
                          '\n\\end{itemize}\n')

    title_tex = esc(title)
    abstract_tex = '\n\n'.join(abstract)
    email_note = (' \\thanks{Corresponding author: %s; ORCID: 0009-0005-5894-4578}' % esc(email)) if email else ''

    doc = (PRE % (title_tex, email_note, date_s, esc(preprint), abstract_tex, esc(keywords)))
    doc += '\n' + highlights_tex + '\n'

    body = '\n'.join(line for line in out if line != '')
    doc += body + '\n'
    doc += r'\end{thebibliography}' if in_refs else ''
    doc += '\n\\end{document}\n'

    if not os.path.exists(OUT):
        os.makedirs(OUT)
    texp = os.path.join(OUT, 'main.tex')
    open(texp, 'w', encoding='utf-8').write(doc)

    for n, fn in FIGMAP.items():
        shutil.copy2(os.path.join(RESULTS, fn), os.path.join(OUT, 'fig%d.png' % n))

    print('WROTE', texp)
    print('chars(before \\end{document}):', len(doc.split(r'\end{document}')[0]))
    print('tables:', n_tables, 'figures:', n_figs, 'sections:', n_sections, 'refs:', ref_n)
    print('absorbed figures from:', ', '.join(FIGMAP.values()))
    return doc


if __name__ == '__main__':
    main()