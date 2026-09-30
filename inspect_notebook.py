import json
from pathlib import Path

p = Path("start-here-a-gentle-introduction.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
for idx, cell in enumerate(nb.get("cells", [])):
    src = "".join(cell.get("source", []))
    if "poly_features" in src or "Imputer" in src or "SimpleImputer" in src:
        print("CELL", idx, cell.get("cell_type"))
        print(src)
        print("---")
