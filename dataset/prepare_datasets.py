"""
prepare_datasets.py
-------------------
Top-level orchestrator for all PeroMAS dataset preparation pipelines.

Delegates to three component scripts (each can also be run independently):

  dataset/mattergen/prepare_mattergen_data.py   — Full MatterGen pipeline (4 sub-steps)
      build      Query Materials Project API, build CSV dataset
      clean      Generate cleaned subsets (pce_only, high_pce, multi_target, ...)
      register   Register custom PSC properties in MatterGen globals.py
      preprocess Convert CSV splits to MatterGen binary cache; update average_density

  dataset/CSLLM/generate_perovskite_finetune_datasets.py  — CSLLM instruction datasets
      Generates dataset_synthesis.json, dataset_method.json, dataset_precursor.json

  dataset/prepare_fabagent_data.py   — FabAgent feature extraction + model training
      copy       Copy MatterGen dataset to FabAgent data/raw/
      features   Extract CBFV composition + CIF structure features (CSR matrices)
      train      Train Random Forest models for 6 property targets

Usage:
  # Run all pipelines (requires Materials Project API key for MatterGen)
  python prepare_datasets.py --mp-api-key YOUR_KEY

  # Run only CSLLM and FabAgent (no API key needed)
  python prepare_datasets.py --steps csllm fabagent

  # Run only the MatterGen pipeline (all 4 sub-steps)
  python prepare_datasets.py --mp-api-key YOUR_KEY --steps mattergen

  # Skip model training in the FabAgent step
  python prepare_datasets.py --steps fabagent --no-train

  # Verify all outputs without reprocessing
  python prepare_datasets.py --verify
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_DIR   = Path(__file__).resolve().parent     # dataset/
ROOT          = DATASET_DIR.parent                  # repo root

SOURCE_CSV    = DATASET_DIR / "Perovskite_database_content_all_data.csv"

# Component orchestration scripts
MATTERGEN_SCRIPT = DATASET_DIR / "mattergen" / "prepare_mattergen_data.py"
CSLLM_SCRIPT     = DATASET_DIR / "CSLLM" / "generate_perovskite_finetune_datasets.py"
FABAGENT_SCRIPT  = DATASET_DIR / "prepare_fabagent_data.py"

# Expected outputs for verification
MATTERGEN_OUT    = DATASET_DIR / "mattergen" / "mattergen_dataset"
MATTERGEN_CLEAN  = DATASET_DIR / "mattergen" / "mattergen_dataset_cleaned"
CSLLM_OUT        = DATASET_DIR / "CSLLM" / "finetune_datasets"
FABAGENT_DIR     = ROOT / "mcp" / "fab_agent" / "Perovskite_PI_Multi"
FABAGENT_RAW     = FABAGENT_DIR / "data" / "raw"
FABAGENT_CSR     = FABAGENT_DIR / "data" / "csr"
FABAGENT_MODEL   = FABAGENT_DIR / "data" / "model" / "single_target"

ALL_STEPS = ["mattergen", "csllm", "fabagent"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list, cwd: Path | None = None, env: dict | None = None):
    print(f"\n[RUN] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env or os.environ.copy(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command exited with code {result.returncode}")


def _check_source_csv():
    if not SOURCE_CSV.exists():
        print(
            f"[ERROR] Source database not found: {SOURCE_CSV}\n"
            "Download from Zenodo (see dataset/README.md) and place it in dataset/."
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1 — Full MatterGen pipeline (delegates to prepare_mattergen_data.py)
# ---------------------------------------------------------------------------

def step_mattergen(mp_api_key: str):
    print("\n" + "=" * 60)
    print("PIPELINE: MatterGen (build → clean → register → preprocess)")
    print("=" * 60)
    print("Delegating to: dataset/mattergen/prepare_mattergen_data.py")

    _check_source_csv()

    if not mp_api_key:
        print(
            "[ERROR] --mp-api-key is required for the MatterGen pipeline.\n"
            "Get a free key at: https://materialsproject.org/api"
        )
        sys.exit(1)

    if not MATTERGEN_SCRIPT.exists():
        print(f"[ERROR] Script not found: {MATTERGEN_SCRIPT}")
        sys.exit(1)

    env = os.environ.copy()
    env["MP_API_KEY"] = mp_api_key

    _run(
        [sys.executable, str(MATTERGEN_SCRIPT), "--mp-api-key", mp_api_key],
        cwd=MATTERGEN_SCRIPT.parent,
        env=env,
    )

    # Quick summary
    import pandas as pd
    for name, path in [
        ("full dataset", MATTERGEN_OUT / "full_dataset.csv"),
        ("train split",  MATTERGEN_OUT / "train.csv"),
        ("val split",    MATTERGEN_OUT / "val.csv"),
    ]:
        if path.exists():
            n = len(pd.read_csv(path))
            print(f"[OK] {name:<15}: {n:,} rows")


# ---------------------------------------------------------------------------
# Step 2 — CSLLM fine-tuning datasets
# ---------------------------------------------------------------------------

def step_csllm():
    print("\n" + "=" * 60)
    print("PIPELINE: CSLLM fine-tuning datasets (synthesis / method / precursor)")
    print("=" * 60)
    print("Delegating to: dataset/CSLLM/generate_perovskite_finetune_datasets.py")

    _check_source_csv()

    if not CSLLM_SCRIPT.exists():
        print(f"[ERROR] Script not found: {CSLLM_SCRIPT}")
        sys.exit(1)

    CSLLM_OUT.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, str(CSLLM_SCRIPT)], cwd=CSLLM_SCRIPT.parent)

    for fname in ["dataset_synthesis.json", "dataset_method.json", "dataset_precursor.json"]:
        fpath = CSLLM_OUT / fname
        if fpath.exists():
            with open(fpath) as f:
                n = len(json.load(f))
            print(f"[OK] {fpath.relative_to(ROOT)}  ({n:,} examples)")


# ---------------------------------------------------------------------------
# Step 3 — FabAgent feature extraction + model training
# ---------------------------------------------------------------------------

def step_fabagent(skip_train: bool):
    print("\n" + "=" * 60)
    print("PIPELINE: FabAgent (copy → features → train)")
    print("=" * 60)
    print("Delegating to: dataset/prepare_fabagent_data.py")

    if not FABAGENT_SCRIPT.exists():
        print(f"[ERROR] Script not found: {FABAGENT_SCRIPT}")
        sys.exit(1)

    cmd = [sys.executable, str(FABAGENT_SCRIPT)]
    if skip_train:
        cmd += ["--steps", "copy", "features"]
        print("[INFO] Skipping model training (--no-train flag set)")

    _run(cmd, cwd=DATASET_DIR)

    # Report feature matrices
    import scipy.sparse as sp
    for pattern in ["*_csr.npz"]:
        for fpath in sorted(FABAGENT_CSR.glob(pattern)):
            if fpath.exists():
                m = sp.load_npz(str(fpath))
                print(f"[OK] {fpath.relative_to(ROOT)}  shape={m.shape}")

    if not skip_train:
        models = list(FABAGENT_MODEL.glob("*.joblib"))
        print(f"[OK] Trained models: {len(models)} files in {FABAGENT_MODEL.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify():
    print("\n" + "=" * 60)
    print("VERIFYING ALL DATASET OUTPUTS")
    print("=" * 60)

    import pandas as pd

    all_ok = True

    def check(path, desc, min_rows=None):
        nonlocal all_ok
        if not path.exists():
            print(f"[MISSING] {str(path.relative_to(ROOT)):<58}  {desc}")
            all_ok = False
            return
        extra = ""
        if path.suffix == ".csv" and min_rows is not None:
            try:
                n = len(pd.read_csv(path))
                extra = f"  ({n:,} rows)"
                if n < min_rows:
                    print(f"[WARN]    {str(path.relative_to(ROOT)):<58}  {desc}{extra}  < {min_rows:,} expected")
                    return
            except Exception as e:
                extra = f"  (read error: {e})"
        elif path.suffix == ".json":
            try:
                with open(path) as f:
                    n = len(json.load(f))
                extra = f"  ({n:,} examples)"
            except Exception:
                pass
        elif path.suffix == ".npz":
            try:
                import scipy.sparse as sp
                m = sp.load_npz(str(path))
                extra = f"  shape={m.shape}"
            except Exception:
                extra = f"  ({path.stat().st_size // 1024:,} KB)"
        print(f"[OK]      {str(path.relative_to(ROOT)):<58}  {desc}{extra}")

    # --- Source ---
    check(SOURCE_CSV, "Perovskite Database (source)", 40000)

    # --- MatterGen ---
    print()
    check(MATTERGEN_OUT / "full_dataset.csv", "MatterGen full dataset",  100)
    check(MATTERGEN_OUT / "train.csv",        "MatterGen train split",    50)
    check(MATTERGEN_OUT / "val.csv",          "MatterGen val split",      10)
    check(MATTERGEN_OUT / "dataset_stats.json", "MatterGen statistics")
    for subset in ["pce_only", "pce_bandgap", "high_pce", "multi_target"]:
        check(MATTERGEN_CLEAN / subset / "train.csv", f"cleaned subset: {subset}")

    mattergen_cache = MATTERGEN_OUT / "cache"
    if mattergen_cache.exists():
        n = sum(1 for _ in mattergen_cache.rglob("*") if _.is_file())
        print(f"[OK]      {str(mattergen_cache.relative_to(ROOT)):<58}  MatterGen binary cache  ({n} files)")
    else:
        print(f"[INFO]    {str(mattergen_cache.relative_to(ROOT)):<58}  (binary cache — run 'mattergen' preprocess sub-step)")

    # --- CSLLM ---
    print()
    check(CSLLM_OUT / "dataset_synthesis.json", "CSLLM synthesis dataset")
    check(CSLLM_OUT / "dataset_method.json",    "CSLLM method dataset")
    check(CSLLM_OUT / "dataset_precursor.json", "CSLLM precursor dataset")

    # --- FabAgent ---
    print()
    check(FABAGENT_RAW / "full_dataset.csv", "FabAgent raw data", 100)
    check(FABAGENT_CSR / "comp_only_sp1_oliynyk_zero_csr.npz", "FabAgent composition features")
    check(FABAGENT_CSR / "cif_only_sp1_oliynyk_zero_csr.npz",  "FabAgent CIF features")
    for t in ["pce", "voc", "jsc", "ff", "dft_band_gap", "energy_above_hull"]:
        matches = list(FABAGENT_MODEL.glob(f"*{t}*.joblib"))
        if matches:
            size_kb = matches[0].stat().st_size // 1024
            print(f"[OK]      {str(matches[0].relative_to(ROOT)):<58}  RF model: {t}  ({size_kb:,} KB)")
        else:
            print(f"[INFO]    model for {t:<20} not trained yet  (run 'fabagent' step)")

    print()
    if all_ok:
        print("All required dataset files verified successfully.")
    else:
        print("Some files are missing — run the corresponding pipeline steps.")
    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Top-level PeroMAS dataset preparation orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mp-api-key",
        default=os.environ.get("MP_API_KEY", ""),
        help="Materials Project API key (required for 'mattergen' pipeline). "
             "Alternatively set the MP_API_KEY environment variable.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS + ["all"],
        default=["all"],
        metavar="STEP",
        help=f"Which pipelines to run (default: all). Choices: {', '.join(ALL_STEPS)}",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip FabAgent model training (feature extraction only).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing outputs and exit without reprocessing.",
    )
    args = parser.parse_args()

    if args.verify:
        ok = verify()
        sys.exit(0 if ok else 1)

    steps = ALL_STEPS if "all" in args.steps else args.steps

    print("=" * 60)
    print("  PeroMAS — Dataset Preparation Orchestrator")
    print("=" * 60)
    print(f"Pipelines : {' → '.join(steps)}")
    print(f"Repo root : {ROOT}")
    if args.mp_api_key:
        masked = "*" * max(0, len(args.mp_api_key) - 4) + args.mp_api_key[-4:]
        print(f"MP key    : {masked}")
    else:
        print("MP key    : (not set — required for 'mattergen' pipeline)")
    if args.no_train:
        print("Training  : skipped (--no-train)")

    step_map = {
        "mattergen": lambda: step_mattergen(args.mp_api_key),
        "csllm":     step_csllm,
        "fabagent":  lambda: step_fabagent(args.no_train),
    }

    failed = []
    for step in steps:
        try:
            step_map[step]()
        except Exception as exc:
            print(f"\n[FAILED] Pipeline '{step}': {exc}")
            failed.append(step)

    print("\n" + "=" * 60)
    if failed:
        print(f"Completed with errors in: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All pipelines completed successfully.")
        verify()


if __name__ == "__main__":
    main()
