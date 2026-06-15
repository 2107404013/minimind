from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Historical entry name kept for compatibility. The implementation prepares
# black-box candidate-level preference data; it does not run token-level KL.
from math_tutor.distill import main


if __name__ == "__main__":
    main()
