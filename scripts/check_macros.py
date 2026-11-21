"""Every macro used by the paper must be defined (numbers.tex / wrappers) or be a standard command."""
import re
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "paper"
defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (D / "numbers.tex").read_text(encoding="utf-8")))
defined.add("releasestatement")
used = {}
for f in ["body.tex", "main_blind.tex", "main_arxiv.tex", "refs.tex", "tab_aapl.tex", "tab_base.tex",
          "tab_meta.tex", "tab_system.tex"]:
    s = re.sub(r"(?<!\\)%.*", "", (D / f).read_text(encoding="utf-8"))
    for m in re.findall(r"\\([A-Za-z]+)", s):
        used.setdefault(m, f)
std = set("""documentclass usepackage input newcommand begin end title author IEEEauthorblockN IEEEauthorblockA
maketitle section label ref cite emph textbf textit texttt url appendices item caption centering scriptsize
footnotesize setlength tabcolsep toprule midrule bottomrule cmidrule multicolumn includegraphics columnwidth
textwidth operatorname sum sqrt top hat partial le quad qquad mathbb mathrm sigma phi psi theta pi eta ell
Delta textstyle Big frac in times ldots bibitem itemsep and tilde i infty mid pm""".split())
bad = sorted((u, f) for u, f in used.items() if u not in defined and u not in std)
print("undefined or unexpected:", bad if bad else "none")
sys.exit(1 if bad else 0)
