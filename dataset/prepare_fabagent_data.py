"""
prepare_fabagent_data.py
------------------------
End-to-end FabAgent dataset preparation and model training pipeline.

Steps:
  Step 1  copy      - Copy processed MatterGen dataset to FabAgent data directory
  Step 2  features  - Extract composition (CBFV/Oliynyk) and CIF structure features
                      and save as sparse CSR matrices (.npz)
  Step 3  train     - Train Random Forest models for all six property targets:
                      pce, voc, jsc, ff, dft_band_gap, energy_above_hull

Usage:
  # Run all steps (feature extraction + training)
  python prepare_fabagent_data.py

  # Run only feature extraction (skip training)
  python prepare_fabagent_data.py --steps copy features

  # Train with hyperparameter optimisation (slower, better accuracy)
  python prepare_fabagent_data.py --steps train --hyperopt --n-trials 50

  # Train only specific targets
  python prepare_fabagent_data.py --steps train --targets pce dft_band_gap

  # Verify existing outputs
  python prepare_fabagent_data.py --verify
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent              # dataset/
ROOT = HERE.parent                                  # repo root

# MatterGen cleaned dataset (preferred) or raw output
MATTERGEN_CLEANED_CSV  = HERE / "mattergen" / "mattergen_dataset_cleaned" / "full" / "full.csv"
MATTERGEN_FULL_CSV     = HERE / "mattergen" / "mattergen_dataset" / "full_dataset.csv"

FABAGENT_DIR  = ROOT / "mcp" / "fab_agent" / "Perovskite_PI_Multi"
FABAGENT_RAW  = FABAGENT_DIR / "data" / "raw"
FABAGENT_CSR  = FABAGENT_DIR / "data" / "csr"
FABAGENT_MODEL = FABAGENT_DIR / "data" / "model" / "single_target"

PROCESS_SCRIPT = FABAGENT_DIR / "process.py"
TRAIN_SCRIPT   = FABAGENT_DIR / "main_single_target.py"

ALL_TARGETS  = ["pce", "voc", "jsc", "ff", "dft_band_gap", "energy_above_hull"]
ALL_STEPS    = ["copy", "features", "train"]

# Feature modes produced by process.py
FEATURE_MODES = [
    ("comp_only", "Composition-based CBFV features (Oliynyk, 264-dim)"),
    ("cif_only",  "CIF crystal structure features (pymatgen, 9-dim)"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(text: str):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def _run(cmd: list, cwd: Path | None = None):
    print(f"[RUN] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise RuntimeError(f"Command exited with code {result.returncode}")


def _source_csv() -> Path:
    """Return the best available MatterGen dataset CSV."""
    if MATTERGEN_CLEANED_CSV.exists():
        return MATTERGEN_CLEANED_CSV
    if MATTERGEN_FULL_CSV.exists():
        return MATTERGEN_FULL_CSV
    return None


# ---------------------------------------------------------------------------
# Step 1 — Copy dataset
# ---------------------------------------------------------------------------

def step_copy():
    _banner("STEP 1 — Copy MatterGen dataset → FabAgent data/raw/")

    src = _source_csv()
    if src is None:
        print(
            "[ERROR] No MatterGen dataset found at:\n"
            f"  {MATTERGEN_CLEANED_CSV}\n"
            f"  {MATTERGEN_FULL_CSV}\n"
            "Run dataset/mattergen/prepare_mattergen_data.py first."
        )
        sys.exit(1)

    FABAGENT_RAW.mkdir(parents=True, exist_ok=True)
    dest = FABAGENT_RAW / "full_dataset.csv"

    print(f"Source : {src.relative_to(ROOT)}")
    print(f"Dest   : {dest.relative_to(ROOT)}")
    shutil.copy2(src, dest)

    import pandas as pd
    n = len(pd.read_csv(dest))
    print(f"[DONE] Copied {n:,} records")


# ---------------------------------------------------------------------------
# Step 2 — Feature extraction
# ---------------------------------------------------------------------------

def step_features():
    _banner("STEP 2 — Extract composition and CIF structure features")

    raw_csv = FABAGENT_RAW / "full_dataset.csv"
    if not raw_csv.exists():
        raise RuntimeError(
            f"{raw_csv.relative_to(ROOT)} not found. Run 'copy' step first."
        )

    if not PROCESS_SCRIPT.exists():
        raise RuntimeError(f"Feature extraction script not found: {PROCESS_SCRIPT}")

    print("Computing features for modes: comp_only, cif_only")
    print("(cif_comp mode = concatenation of both, computed at training time)")

    # process.py reads its settings from settings2_train.py in the same directory.
    # We generate each mode separately by overriding USE_X via a small helper call.
    for mode, desc in FEATURE_MODES:
        csr_out = FABAGENT_CSR / f"{mode}_sp1_oliynyk_zero_csr.npz"
        if csr_out.exists():
            import scipy.sparse as sp
            m = sp.load_npz(str(csr_out))
            print(f"[SKIP] {mode} features already exist  ({m.shape[0]} samples × {m.shape[1]} features)")
            continue

        print(f"\nGenerating {mode} features: {desc}")
        _run(
            [sys.executable, str(PROCESS_SCRIPT), f"--use-x={mode}"],
            cwd=FABAGENT_DIR,
        )

    # Verify outputs
    FABAGENT_CSR.mkdir(parents=True, exist_ok=True)
    import scipy.sparse as sp
    for mode, _ in FEATURE_MODES:
        csr_path = FABAGENT_CSR / f"{mode}_sp1_oliynyk_zero_csr.npz"
        col_path = FABAGENT_CSR / f"{mode}_sp1_oliynyk_zero_columns.npy"
        if csr_path.exists():
            m = sp.load_npz(str(csr_path))
            print(f"[OK] {csr_path.name}  shape={m.shape}")
        else:
            print(f"[WARN] {csr_path.name} not created — check process.py output")


# ---------------------------------------------------------------------------
# Step 3 — Model training
# ---------------------------------------------------------------------------

def step_train(targets: list[str], do_hyperopt: bool, n_trials: int, model_name: str):
    _banner("STEP 3 — Train property prediction models")

    raw_csv = FABAGENT_RAW / "full_dataset.csv"
    if not raw_csv.exists():
        raise RuntimeError(
            f"{raw_csv.relative_to(ROOT)} not found. Run 'copy' step first."
        )

    if not TRAIN_SCRIPT.exists():
        raise RuntimeError(f"Training script not found: {TRAIN_SCRIPT}")

    print(f"Model      : {model_name}")
    print(f"Targets    : {', '.join(targets)}")
    print(f"Hyperopt   : {'yes' if do_hyperopt else 'no'}")
    if do_hyperopt:
        print(f"Trials     : {n_trials}")

    FABAGENT_MODEL.mkdir(parents=True, exist_ok=True)

    for target in targets:
        print(f"\n--- Training {model_name} for target: {target} ---")

        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--target", target,
            "--feature", "cif_comp",
            "--model", model_name,
            "--raw-file", str(raw_csv),
        ]
        if do_hyperopt:
            cmd += ["--hyperopt", "--n-trials", str(n_trials)]

        _run(cmd, cwd=FABAGENT_DIR)

    # Report trained models
    print("\nTrained model files:")
    for f in sorted(FABAGENT_MODEL.glob("*.joblib")):
        size_kb = f.stat().st_size // 1024
        print(f"  [OK] {f.name}  ({size_kb:,} KB)")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify():
    _banner("VERIFICATION")

    import pandas as pd, scipy.sparse as sp
    all_ok = True

    def check(path, desc, loader=None):
        nonlocal all_ok
        if not path.exists():
            print(f"[MISSING] {path.relative_to(ROOT):<55}  {desc}")
            all_ok = False
            return
        extra = ""
        if loader:
            try:
                extra = loader(path)
            except Exception:
                pass
        print(f"[OK]      {str(path.relative_to(ROOT)):<55}  {desc}{extra}")

    check(FABAGENT_RAW / "full_dataset.csv",
          "raw dataset",
          lambda p: f"  ({len(pd.read_csv(p)):,} rows)")

    for mode, _ in FEATURE_MODES:
        check(FABAGENT_CSR / f"{mode}_sp1_oliynyk_zero_csr.npz",
              f"{mode} features",
              lambda p: f"  shape={sp.load_npz(str(p)).shape}")
        check(FABAGENT_CSR / f"{mode}_sp1_oliynyk_zero_columns.npy",
              f"{mode} column names")

    for t in ALL_TARGETS:
        # Accept any model name suffix
        matches = list(FABAGENT_MODEL.glob(f"*RF*{t}*.joblib")) + \
                  list(FABAGENT_MODEL.glob(f"*{t}*RF*.joblib"))
        if matches:
            p = matches[0]
            size_kb = p.stat().st_size // 1024
            print(f"[OK]      {str(p.relative_to(ROOT)):<55}  RF model for {t}  ({size_kb:,} KB)")
        else:
            print(f"[MISSING] model for {t:<20}  (run 'train' step)")
            all_ok = False

    print()
    if all_ok:
        print("All FabAgent files verified.")
    else:
        print("Some files are missing — see steps above.")
    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FabAgent dataset preparation and model training pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS + ["all"],
        default=["all"],
        metavar="STEP",
        help=f"Steps to run (default: all). Choices: {', '.join(ALL_STEPS + ['all'])}",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=ALL_TARGETS,
        choices=ALL_TARGETS,
        metavar="TARGET",
        help="Property targets to train models for (default: all six).",
    )
    parser.add_argument(
        "--model",
        default="RF",
        choices=["RF", "GBDT", "NN"],
        help="ML model type (default: RF — Random Forest).",
    )
    parser.add_argument(
        "--hyperopt",
        action="store_true",
        help="Run Optuna hyperparameter optimisation before training.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50, used only with --hyperopt).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing outputs and exit.",
    )
    args = parser.parse_args()

    if args.verify:
        ok = verify()
        sys.exit(0 if ok else 1)

    steps = ALL_STEPS if "all" in args.steps else args.steps

    print("=" * 60)
    print("  FabAgent Data Preparation & Training Pipeline")
    print("=" * 60)
    print(f"Steps  : {' → '.join(steps)}")
    print(f"Model  : {args.model}")
    print(f"Targets: {', '.join(args.targets)}")

    step_fn = {
        "copy":     step_copy,
        "features": step_features,
        "train":    lambda: step_train(args.targets, args.hyperopt, args.n_trials, args.model),
    }

    failed = []
    for step in steps:
        try:
            step_fn[step]()
        except Exception as exc:
            print(f"\n[FAILED] {step}: {exc}")
            failed.append(step)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"Pipeline finished with errors in: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("Pipeline completed successfully.")
        verify()


if __name__ == "__main__":
    main()
