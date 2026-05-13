"""
main.py
-------
Phase 1 Entry Point — Facial Expression Recognition Project
-----------------------------------------------------------
Runs the full Phase 1 pipeline:
  1. Discover the FER2013 train dataset
  2. Select a random batch
  3. Run image quality checks
  4. Preprocess all usable images
  5. Generate analysis plots and save a quality report (CSV)

Usage:
    python main.py [--batch-size N] [--seed S] [--data-dir PATH]

Defaults:
    --batch-size  210   (30 images per class × 7 classes)
    --seed         42   (reproducible random selection)
    --data-dir   data/train
"""

import sys
import time
import argparse
from pathlib import Path

# ── Ensure src/ is on the path ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.batch_selector  import discover_dataset, select_random_batch
from src.quality_checker import run_quality_checks
from src.preprocessor    import preprocess_batch
from src.visualizer      import generate_all_plots


# ── ANSI colour helpers ──────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

BANNER = f"""{CYAN}{BOLD}
+==============================================================+
|    Facial Expression Recognition - Phase 1: Preprocessing   |
+==============================================================+{RESET}
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="FER2013 Phase 1 — Batch Selection & Preprocessing"
    )
    parser.add_argument("--batch-size", type=int,   default=210,
                        help="Total images to sample (default: 210)")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--data-dir",   type=str,   default="data/train",
                        help="Path to the FER2013 train folder (default: data/train)")
    parser.add_argument("--no-save",    action="store_true",
                        help="Skip saving preprocessed images to disk")
    return parser.parse_args()


def step(n: int, title: str):
    print(f"\n{CYAN}{BOLD}[Step {n}] {title}{RESET}")
    print("-" * 60)


def main():
    print(BANNER)
    args = parse_args()
    t0 = time.time()

    # ── Step 1: Discover dataset ─────────────────────────────────────────────
    step(1, "Discovering Dataset")
    full_df = discover_dataset(args.data_dir)

    # ── Step 2: Select random batch ──────────────────────────────────────────
    step(2, f"Selecting Random Batch  (size={args.batch_size}, seed={args.seed})")
    batch_df = select_random_batch(full_df, batch_size=args.batch_size, seed=args.seed)

    # ── Step 3: Quality checks ───────────────────────────────────────────────
    step(3, "Running Quality Checks")
    batch_df = run_quality_checks(batch_df)

    # Save quality report
    report_dir = Path("output/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "quality_report.csv"
    batch_df.to_csv(report_path, index=False)
    print(f"\n  Quality report saved -> {report_path.resolve()}")

    # ── Step 4: Preprocessing ────────────────────────────────────────────────
    step(4, "Preprocessing Images")
    X, y, label_list = preprocess_batch(
        batch_df,
        output_dir="output/batch_samples",
        save_images=not args.no_save,
    )

    # ── Step 5: Visualizations ───────────────────────────────────────────────
    step(5, "Generating Analysis Plots")
    plot_paths = generate_all_plots(batch_df, X, y, label_list, output_dir="output/reports")

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    total    = len(batch_df)
    usable   = int(batch_df["is_usable"].sum())

    print(f"\n{GREEN}{BOLD}{'='*60}")
    print("  Phase 1 Complete!")
    print(f"{'='*60}{RESET}")
    print(f"  Batch size       : {total} images")
    print(f"  Usable images    : {usable}  ({usable/total*100:.1f}%)")
    print(f"  Preprocessed arr : shape={X.shape}, dtype={X.dtype}")
    print(f"  Time elapsed     : {elapsed:.1f}s")
    print(f"\n  Output files saved to: {Path('output').resolve()}")
    print(f"    +-- batch_samples/   (preprocessed images by class)")
    print(f"    +-- reports/")
    print(f"        +-- quality_report.csv")
    for name, path in plot_paths.items():
        print(f"        +-- {Path(path).name}")
    print(f"\n{YELLOW}  Tip: Open output/reports/ to view all analysis plots.{RESET}\n")


if __name__ == "__main__":
    main()
