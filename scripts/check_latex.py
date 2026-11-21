"""Static checks for research.tex (no TeX installation required).

* every \\input{...} file exists; every \\includegraphics file exists
* braces balance (ignoring escaped braces and comments) in every file
* \\begin{env} / \\end{env} nest correctly
* every \\ref/\\eqref label is defined, and no label is defined twice
* every \\cite key has a \\bibitem, and every \\bibitem is cited
* every macro used that comes from numbers.tex is defined there
* no leftover placeholders (TODO, XXX, ??)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
main = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "research.tex")
BASE = main.parent
errors: list[str] = []


def strip_comments(s: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in s.splitlines())


def load(path: Path) -> str:
    return strip_comments(path.read_text(encoding="utf-8"))


src = load(main)
files = {main.name: src}
for inc in re.findall(r"\\input\{([^}]+)\}", src):
    p = BASE / (inc if inc.endswith(".tex") else inc + ".tex")
    if not p.exists():
        errors.append(f"missing \\input file {p}")
    else:
        files[p.name] = load(p)
for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src):
    if not (BASE / g).exists():
        errors.append(f"missing graphic {g}")

for name, s in files.items():
    t = s.replace("\\{", "").replace("\\}", "")
    depth = 0
    for i, ch in enumerate(t):
        depth += ch == "{"
        depth -= ch == "}"
        if depth < 0:
            errors.append(f"{name}: unmatched '}}' near: {t[max(0, i - 60):i + 10]!r}")
            depth = 0
    if depth != 0:
        errors.append(f"{name}: {depth} unclosed '{{'")
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", s):
        kind, env = m.groups()
        if kind == "begin":
            stack.append(env)
        elif not stack or stack[-1] != env:
            errors.append(f"{name}: \\end{{{env}}} does not match {stack[-1] if stack else 'nothing'}")
        else:
            stack.pop()
    if stack and name == main.name:
        errors.append(f"{name}: unclosed environments {stack}")
    for bad in ("TODO", "XXX", "??"):
        if bad in s:
            errors.append(f"{name}: contains placeholder {bad}")

all_src = "\n".join(files.values())
labels = re.findall(r"\\label\{([^}]+)\}", all_src)
dups = {l for l in labels if labels.count(l) > 1}
if dups:
    errors.append(f"duplicate labels {sorted(dups)}")
for r in re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", all_src):
    if r not in labels:
        errors.append(f"undefined reference {r}")
cites = set(k.strip() for c in re.findall(r"\\cite\{([^}]+)\}", all_src) for k in c.split(","))
bibs = set(re.findall(r"\\bibitem\{([^}]+)\}", all_src))
for c in sorted(cites - bibs):
    errors.append(f"citation without bibitem: {c}")
for b in sorted(bibs - cites):
    errors.append(f"bibitem never cited: {b}")

numbers = BASE / "numbers.tex"
defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", numbers.read_text(encoding="utf-8"))) if numbers.exists() else set()
defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", all_src))
used = set(re.findall(r"\\([A-Za-z]+)", all_src))
# macros that look like generated number macros
generated_prefix = ("m", "p", "w", "dz", "hpo", "ma", "g", "cs", "rho", "n", "date", "up", "bh", "tr")
known_latex = set("""documentclass usepackage begin end section subsection subsubsection label ref eqref cite
textbf textit emph item caption centering footnotesize scriptsize small normalsize toprule midrule bottomrule
multicolumn includegraphics newcommand input title author maketitle IEEEauthorblockN IEEEauthorblockA
IEEEoverridecommandlockouts IEEEkeywords left right frac sum sqrt mathbb mathrm mathcal operatorname sigma
hat bar top in quad qquad cdot cdots dots ldots le ge leq geq neq approx times mid infty log exp max min
arg sgn softmax text texttt url hspace vspace noindent par bibitem thebibliography setlength itemsep parsep
linewidth columnwidth textwidth resizebox tabular table figure equation align cases mathbf boldsymbol
alpha beta gamma delta epsilon varepsilon theta lambda mu pi rho tau phi varphi psi omega Delta Sigma
odot otimes oplus partial nabla prime lVert rVert lvert rvert Vert vert langle rangle tilde widehat
widetilde overline underline smallskip medskip bigskip newline linebreak pagebreak clearpage ensuremath
mathit footnote thanks hfill vfill rule renewcommand arraystretch tabcolsep cline cmidrule addlinespace
subfigure subcaption phantom mbox makebox raisebox scalebox definecolor color textcolor href hline
IEEEPARstart rm sc kern lower hbox BibTeX def AtBeginDocument abovedisplayskip belowdisplayskip
abovedisplayshortskip belowdisplayshortskip plus minus columnsep newcolumntype arraybackslash
X C l c r p ell sim propto forall exists to mapsto Leftarrow Rightarrow leftarrow rightarrow
bigl bigr Bigl Bigr big Big biggl biggr displaystyle textstyle scriptstyle limits nolimits
iff land lor neg emptyset subset subseteq cup cap setminus colon vdots ddots ldotp
textsuperscript textsubscript textsc textrm textsf mathsf mathtt S P quad ast star circ bullet
REQUIRE ENSURE STATE RETURN FOR ENDFOR IF ENDIF algorithmic algorithm floatname
sloppy fussy allowbreak nobreak protect relax string the value number arabic roman Roman
textendash textemdash textasciitilde textbackslash textunderscore ldots textquotedbl dag ddag
footnotemark footnotetext tablefootnote enspace thinspace negthinspace nolinebreak
""".split())
for u in sorted(used):
    if u in defined or u in known_latex:
        continue
    if u.startswith(generated_prefix) and any(ch.isupper() for ch in u[1:]):
        errors.append(f"macro \\{u} used but not defined in numbers.tex")

print("\n".join(errors) if errors else "OK: no problems found")
sys.exit(1 if errors else 0)
