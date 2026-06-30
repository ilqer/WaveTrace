import pathlib
import sys

_root = pathlib.Path(__file__).resolve().parent.parent
# Make the repo root importable so `import fixtures.SyntheticCsi` resolves (namespace package).
sys.path.insert(0, str(_root))
# Make scripts/ importable with bare names (scripts import each other and tests import scripts).
sys.path.insert(0, str(_root / "scripts"))
