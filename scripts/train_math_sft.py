from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from math_tutor.train import load_yaml, run_or_print, sft_command


def main() -> None:
    config = load_yaml(ROOT / "configs" / "math_tutor.yaml")
    command = sft_command(config, "math_sft")
    run_or_print(command, run="--run" in sys.argv, cuda_visible_devices=config["environment"]["local_cuda_visible_devices"])


if __name__ == "__main__":
    main()
