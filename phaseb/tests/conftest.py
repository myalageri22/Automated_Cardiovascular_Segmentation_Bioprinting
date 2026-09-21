import sys
from pathlib import Path


PHASEB_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PHASEB_SRC))
sys.modules.pop("phaseb", None)
