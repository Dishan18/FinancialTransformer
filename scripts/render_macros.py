"""Print paper/body.tex with every numbers.tex macro expanded (for proofreading)."""
import re
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "paper"
defs = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", (D / "numbers.tex").read_text(encoding="utf-8")))
s = (D / "body.tex").read_text(encoding="utf-8")
for k in sorted(defs, key=len, reverse=True):
    s = s.replace("\\" + k + "{}", defs[k]).replace("\\" + k, defs[k])
s = s.replace("$-$", "-")
print(s)
