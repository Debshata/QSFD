"""Static consistency checks for the LaTeX package (no TeX engine required): \\input targets, labels vs refs,
citations vs .bib, macro definitions vs usage, figure files, brace balance. This is NOT a substitute for compiling."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "report"
bib = (ROOT / "common/bibliography/references.bib").read_text(encoding="utf-8")
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
macros = set(re.findall(r"\\newcommand\{\\(\w+)\}", (ROOT / "common/generated/results_macros.tex").read_text(encoding="utf-8")))
builtin = set("""figw placeholder section subsection paragraph textbf texttt emph cite ref eqref label input includegraphics caption
begin end item centering small chapter tableofcontents listoffigures listoftables bibliographystyle bibliography maketitle
title author institution email keywords and titlerunning authorrunning IEEEauthorblockN IEEEauthorblockA textit hline
toprule midrule bottomrule endfirsthead endhead multicolumn midrule frac sum otimes cdots langle rangle mathrm text tfrac Vert
log hat Delta theta psi leq le times approx cdot vspace hspace noindent usepackage documentclass graphicspath newcommand
onehalfspacing appendix IEEEkeywords abstract titlepage equation itemize longtable tabular table figure url Z X
""".split())
problems = []


def expand(path, base=None, seen=None):
    base = base or path.parent
    seen = seen or set()
    txt = path.read_text(encoding="utf-8")
    out = [(path, txt)]
    for m in re.finditer(r"\\input\{([^}]+)\}", txt):
        tgt = (base / m.group(1)).resolve()
        if not tgt.suffix:
            tgt = tgt.with_suffix(".tex")
        if not tgt.exists():
            problems.append(f"{path.name}: missing input {m.group(1)}")
        elif tgt not in seen and "results_macros" not in tgt.name:
            seen.add(tgt)
            out += expand(tgt, base, seen)
    return out


for root in ("university", "ieee", "springer"):
    files = expand(ROOT / root / "main.tex")
    text = "\n".join(t for _, t in files)
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text))
    for r in sorted(refs - labels):
        problems.append(f"{root}: undefined ref {r}")
    cites = {c.strip() for m in re.findall(r"\\cite\{([^}]+)\}", text) for c in m.split(",")}
    for c in sorted(cites - keys):
        problems.append(f"{root}: undefined citation {c}")
    used = set(re.findall(r"\\([A-Za-z]+)\{\}", text))
    for u in sorted(used - macros - builtin):
        problems.append(f"{root}: macro used but not generated: \\{u}")
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        if not (ROOT / "common/figures" / (g + ".png")).exists():
            problems.append(f"{root}: missing figure {g}")
    for p, t in files:
        if t.count("{") != t.count("}") and "\\begin{abstract}" not in t:
            problems.append(f"{p.name}: unbalanced braces ({t.count('{')} vs {t.count('}')})")
    print(f"{root}: inputs={len(files)} labels={len(labels)} refs={len(refs)} cites={len(cites)} macros_used={len(used & macros)}")
print("PROBLEMS:", len(problems))
for p in problems:
    print(" -", p)
sys.exit(1 if problems else 0)
