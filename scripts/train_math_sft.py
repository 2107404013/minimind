from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from math_tutor.train import load_yaml, run_diagnostics, run_math_sft, run_math_sft_overfit_debug, run_official_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or dry-run MiniMind official SFT workflows.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "math_tutor.yaml"))
    parser.add_argument("--mode", choices=["official_sft", "math_sft"], default="official_sft")
    parser.add_argument("--dry_run", action="store_true", help="Print the official MiniMind command without training.")
    parser.add_argument("--run", action="store_true", help="Execute training. Omit this for dry run.")
    parser.add_argument("--diagnose", action="store_true", help="Print checkpoint, data, tokenizer, and loss-mask diagnostics.")
    parser.add_argument("--overfit_debug", action="store_true", help="Run a 100-sample math SFT overfit debug workflow.")
    args = parser.parse_args()

    if args.dry_run and args.run:
        parser.error("--dry_run and --run cannot be used together")
    if args.overfit_debug and args.mode != "math_sft":
        parser.error("--overfit_debug is only supported with --mode math_sft")

    config = load_yaml(args.config)
    if args.diagnose:
        run_diagnostics(config, mode=args.mode)
        return
    if args.overfit_debug:
        run_math_sft_overfit_debug(config, run=not args.dry_run)
        return
    if args.mode == "math_sft":
        run_math_sft(config, run=args.run)
    else:
        run_official_sft(config, mode=args.mode, run=args.run)


if __name__ == "__main__":
    main()
