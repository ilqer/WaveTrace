import pathlib
import sys

# Make the repo root importable so `import fixtures.SyntheticCsi` resolves (namespace package).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
