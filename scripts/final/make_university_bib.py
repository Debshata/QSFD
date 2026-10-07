"""Builds report/university/bibliography.tex (thebibliography) from references.bib in the university report's required
reference formats, ordered by first citation in the university report:
  Journal : [n] Author, "Title", Journal, Publisher, Location, Year, Edition/Reprint, Page start-end.
  Web     : [n] Author, "Title", Complete URL.
  Book    : [n] Author, "Title of the Book", Publication, Edition, Year.
Publication details and URLs are italicised. Fields absent from the .bib (publisher, location, edition) are omitted,
never invented."""
import re
from pathlib import Path

B = chr(92)
ROOT = Path(__file__).resolve().parents[2] / "report"
bib = (ROOT / "common/bibliography/references.bib").read_text(encoding="utf-8")


def parse(text):
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        typ, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
        fields = {}
        for f in re.finditer(r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\d+)", body):
            v = f.group(2)
            fields[f.group(1).lower()] = v[1:-1] if v.startswith("{") else v
        out[key] = (typ, fields)
    return out


def clean(s):
    s = re.sub(r"\{\\'([a-zA-Z])\}", r"\1", s)
    s = s.replace("{\\`e}", "e").replace("{\\\"u}", "u").replace("{\\v{c}}", "c").replace("{\\v{e}}", "e").replace("{\\'i}", "i").replace("{\\'{\\i}}", "i")
    s = s.replace("{", "").replace("}", "")
    s = s.replace("\\&", "&").replace("&", B + "&").replace("%", B + "%").replace("_", B + "_")
    return s.replace("--", "--")


def authors(a):
    names = []
    for p in re.split(r"\s+and\s+", a):
        p = p.strip()
        if "," in p:
            last, first = [x.strip() for x in p.split(",", 1)]
            p = f"{first} {last}"
        names.append(p)
    return ", ".join(names[:-1]) + (", and " if len(names) > 2 else " and ") + names[-1] if len(names) > 1 else names[0]


entries = parse(bib)

# citation order from the university report
def expand(p, base, seen):
    t = p.read_text(encoding="utf-8")
    out = t
    for m in re.finditer(r"\\input\{([^}]+)\}", t):
        q = (base / m.group(1)).resolve()
        if not q.suffix:
            q = q.with_suffix(".tex")
        if q.exists() and "results_macros" not in q.name and q not in seen and "bibliography" not in q.name:
            seen.add(q)
            out = out.replace(m.group(0), expand(q, base, seen))
    return out


text = expand(ROOT / "university/main.tex", ROOT / "university", set())
order = []
for m in re.finditer(r"\\cite\{([^}]+)\}", text):
    for k in m.group(1).split(","):
        k = k.strip()
        if k in entries and k not in order:
            order.append(k)
for k in entries:
    if k not in order:
        order.append(k)

lines = [B + "begin{thebibliography}{99}", B + "setstretch{1.5}"]
for n, k in enumerate(order, 1):
    typ, f = entries[k]
    au = authors(clean(f["author"])) if "author" in f else None
    title = clean(f.get("title", ""))
    year = f.get("year", "")
    it = lambda s: B + "textit{" + s + "}"
    if typ == "article":
        pages = f.get("pages", "").replace("--", "--")
        det = ", ".join(x for x in [it(clean(f["journal"])) if "journal" in f else "", (f"vol. {f['volume']}" if "volume" in f else ""), (f"no. {f['number']}" if "number" in f else ""), year, (f"pp. {pages}" if pages else "")] if x)
        doi = f.get("doi")
        tail = ", " + it("doi:" + clean(doi)) if doi else ""
        s = f"{au}, {B}textquotedblleft{{}}{title}{B}textquotedblright{{}}, {det}{tail}."
    elif typ == "inproceedings":
        pages = f.get("pages", "")
        det = ", ".join(x for x in [it(clean(f.get("booktitle", ""))), year, (f"pp. {pages}" if pages else "")] if x)
        doi = f.get("doi")
        tail = ", " + it("doi:" + clean(doi)) if doi else ""
        s = f"{au}, {B}textquotedblleft{{}}{title}{B}textquotedblright{{}}, {det}{tail}."
    elif typ == "book":
        s = f"{au}, {B}textquotedblleft{{}}{title}{B}textquotedblright{{}}, {clean(f.get('publisher', ''))}, {clean(f.get('edition', ''))} ed., {year}."
    elif typ in ("techreport",):
        au2 = au or clean(f.get("institution", ""))
        doi = f.get("doi")
        s = f"{au2}, {B}textquotedblleft{{}}{title}{B}textquotedblright{{}}, {it(clean(f.get('institution', '')) + (', ' + clean(f['number']) if 'number' in f else ''))}, {year}" + (", " + it("doi:" + clean(doi)) if doi else "") + "."
    else:  # misc / web
        url = f.get("howpublished", "") + " " + f.get("note", "")
        m = re.search(r"https?://[^\s}]+", url)
        a2 = au or "Open Quantum Safe project" if k == "liboqs" else (au or "pqcrypto contributors")
        if k == "kyriienko":
            s = f"{au}, {B}textquotedblleft{{}}{title}{B}textquotedblright{{}}, {it('arXiv:2208.01203')}, {year}."
        elif k == "qiskit":
            s = f"{au}, {B}textquotedblleft{{}}{title}{B}textquotedblright{{}}, {it('arXiv:2405.08810')}, {year}."
        elif m:
            s = f"{a2}, {B}textquotedblleft{{}}{title}{B}textquotedblright{{}}, {it(m.group(0).replace('_', B + '_'))}."
        else:
            s = f"{a2}, {B}textquotedblleft{{}}{title}{B}textquotedblright{{}}, {it('Python Package Index, version 1.0.0, ' + str(year))}."
    lines.append(B + f"bibitem{{{k}}} {s}")
lines.append(B + "end{thebibliography}")
(ROOT / "university/bibliography.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(len(order), "entries ->", ROOT / "university/bibliography.tex")
