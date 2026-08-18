import sys
from pathlib import Path

# Ensure project root is on sys.path so tests can import the src package.
root = Path(__file__).parent.resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
