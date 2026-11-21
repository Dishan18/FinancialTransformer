"""Write the paper abstract as plain text (for the arXiv metadata form) to zenodo/arxiv_abstract.txt."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
defs = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", (ROOT / "paper/numbers.tex").read_text(encoding="utf-8")))
s = (ROOT / "paper/body.tex").read_text(encoding="utf-8")
a = s[s.index("\\begin{abstract}") + len("\\begin{abstract}"):s.index("\\end{abstract}")].strip()
for k in sorted(defs, key=len, reverse=True):
    a = a.replace("\\" + k + "{}", defs[k]).replace("\\" + k, defs[k])
a = a.replace("\\releasestatement{}", "Code: https://github.com/Dishan18/FinancialTransformer")
for x, y in (("S\\&P~500", "S&P 500"), ("\\,", " "), ("$-$", "-"), ("$", ""), ("{,}", ","), ("--", "-"), ("~", " ")):
    a = a.replace(x, y)
(ROOT / "zenodo/arxiv_abstract.txt").write_text(a, encoding="utf-8")
print(len(a))
print(a)
