"""One-off LaTeX hygiene: breakable \\path{} for file paths, itemize with short bold leads in the evidence map, bib sort keys."""
import pathlib
import re

B = chr(92)
ROOT = pathlib.Path(__file__).resolve().parents[2] / "report"
files = list((ROOT / "common/paper").glob("*.tex")) + list((ROOT / "university/sections").glob("*.tex"))
n = 0


def f(m):
    global n
    inner = m.group(1)
    if "/" in inner or "_" in inner:
        n += 1
        return B + "path{" + inner.replace(B + "_", "_") + "}"
    return m.group(0)


TEXTTT = re.compile(B * 2 + "texttt" + B + "{([^{}]*)}")
ITEM = re.compile(B * 2 + "item" + B + "[(C" + B + "d) -- (.*?)" + B + "." + B + "]")
for p in files:
    s = p.read_text(encoding="utf-8")
    s = TEXTTT.sub(f, s)
    if p.name == "contributions.tex":
        s = s.replace(B + "begin{description}", B + "begin{itemize}").replace(B + "end{description}", B + "end{itemize}")
        s = ITEM.sub(lambda m: B + "item " + B + "textbf{" + m.group(1) + " -- " + m.group(2) + ".}", s)
    p.write_text(s, encoding="utf-8")
print("paths converted:", n)
bib = ROOT / "common/bibliography/references.bib"
b = bib.read_text(encoding="utf-8")
for k in ("pqcrypto", "liboqs"):
    if f"key          = {{{k}}}" not in b:
        b = b.replace("@misc{%s,\n" % k, "@misc{%s,\n  key          = {%s},\n" % (k, k))
bib.write_text(b, encoding="utf-8")
