"""Flattens the university report's LaTeX to plain prose (macros substituted) for plagiarism / AI-detector checking.
Output: report/university/plain_text.txt (not committed)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "report"
macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (ROOT / "common/generated/results_macros.tex").read_text(encoding="utf-8")))
B = chr(92)


def load(p, base=ROOT / "university", seen=None):
    t = p.read_text(encoding="utf-8")
    def inc(m):
        q = (base / m.group(1)).resolve()
        if q.suffix == "":
            q = q.with_suffix(".tex")
        if "results_macros" in q.name or not q.exists():
            return ""
        return load(q, base)
    return re.sub(r"\\input\{([^}]+)\}", inc, t)


text = load(ROOT / "university/main.tex")
text = text.split(B + "begin{document}", 1)[1]
for k, v in macros.items():
    text = text.replace(B + k + "{}", v).replace(B + k + " ", v + " ")
text = re.sub(r"\\begin\{(table|longtable|figure|tabular|equation|IEEEkeywords)\*?\}.*?\\end\{\1\*?\}", " ", text, flags=re.S)
text = re.sub(r"\$[^$]*\$", " ", text)
text = re.sub(r"\\(?:chapter|section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}", r"\n\n\1.\n", text)
text = re.sub(r"\\(?:emph|textbf|textit|texttt|path|url|placeholder)\{([^{}]*)\}", r"\1", text)
text = re.sub(r"\\(?:cite|ref|label|eqref)\{[^}]*\}", "", text)
text = re.sub(r"\\(?:item|centering|small|noindent|vspace|hspace|tableofcontents|listoffigures|listoftables|bibliographystyle|bibliography|appendix|clearpage)\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", text)
text = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?", " ", text)
text = text.replace("~", " ").replace("--", "-").replace("{", "").replace("}", "").replace("\\%", "%").replace("\\&", "&").replace("\\_", "_").replace("\\", " ")
text = re.sub(r"[ \t]+", " ", text)
text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()
out = ROOT / "university/plain_text.txt"
out.write_text(text, encoding="utf-8")
print(len(text.split()), "words ->", out)
