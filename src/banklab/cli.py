"""CLI entry points for dataset preparation and reproducible experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import UCI_DOI, download_uci, load_data, make_synthetic_demo
from .experiment import run_experiment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bank Marketing Decision Lab")
    sub = parser.add_subparsers(dest="command", required=True)
    p_download = sub.add_parser("download", help="Download original UCI Bank Marketing dataset")
    p_download.add_argument("--output", default="data/raw/bank-additional-full.csv")
    p_download.add_argument("--replace", action="store_true")
    p_train = sub.add_parser("train", help="Train and evaluate on real UCI data")
    p_train.add_argument("--data", default="data/raw/bank-additional-full.csv")
    p_train.add_argument("--output", default="reports/latest")
    p_train.add_argument("--fast", action="store_true", help="Smaller forest for quick local iteration")
    p_demo = sub.add_parser("demo", help="Run a SYNTHETIC demonstration (NOT research findings)")
    p_demo.add_argument("--rows", type=int, default=1600)
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.add_argument("--output", default="reports/demo")
    args = parser.parse_args(argv)
    if args.command == "download":
        result = download_uci(Path(args.output), replace=args.replace)
        print(f"Official UCI data saved to {result}. Citation: {UCI_DOI}")
        return
    if args.command == "train":
        data = load_data(args.data)
        if len(data) != 41_188:
            raise ValueError("Real-data mode expects the complete 41,188-row bank-additional-full.csv")
        source = f"UCI Bank Marketing bank-additional-full.csv ({UCI_DOI})"
        fast = args.fast
    else:
        data = make_synthetic_demo(args.rows, args.seed)
        source = "SYNTHETIC DEMO — NOT EMPIRICAL EVIDENCE"
        fast = True
    result = run_experiment(data, args.output, source=source, fast=fast)
    print(f"Completed: {args.output} | champion: {result['champion']}")
    print("Read RESULTS.md and metrics.json in the output directory.")


if __name__ == "__main__":
    main()
