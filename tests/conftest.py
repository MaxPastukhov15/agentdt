import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"

for p in [str(ROOT), str(APP)]:
    if p not in sys.path:
        sys.path.insert(0, p)
