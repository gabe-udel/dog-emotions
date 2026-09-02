"""Put src/ on the path for the test session only.

Library modules under src/ deliberately do no sys.path manipulation of their own - this
is test infrastructure, and the one place it is allowed.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
