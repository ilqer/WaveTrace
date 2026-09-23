import pathlib
import sys

_root = pathlib.Path(__file__).resolve().parent.parent
# Repo root on the path so `import fixtures.SyntheticCsi` resolves (namespace package).
sys.path.insert(0, str(_root))
# scripts/ on the path: scripts import each other by bare name, and tests import scripts.
sys.path.insert(0, str(_root / "scripts"))
