"""
prepare_mattergen_data.py
-------------------------
End-to-end MatterGen dataset preparation pipeline.

Runs all four steps in order:
  Step 1  build     - Query Materials Project API and build CSV dataset from
                      Perovskite_database_content_all_data.csv
  Step 2  clean     - Generate cleaned subsets (pce_only, pce_bandgap, high_pce, ...)
  Step 3  register  - Register custom PSC properties (pce, stability_retention, ...)
                      in MatterGen's globals.py and copy YAML embedding configs
  Step 4  preprocess - Convert CSV splits to MatterGen binary cache format and
                       update average_density in the data-module config

Usage:
  python prepare_mattergen_data.py --mp-api-key YOUR_KEY
  python prepare_mattergen_data.py --mp-api-key YOUR_KEY --steps build clean
  python prepare_mattergen_data.py --steps register preprocess   # no API key needed
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent             # dataset/mattergen/
ROOT = HERE.parent.parent                          # repo root

SOURCE_CSV = HERE.parent / "Perovskite_database_content_all_data.csv"

SCRIPTS = {
    "build":      HERE / "build_mattergen_dataset.py",
    "clean":      HERE / "clean_mattergen_dataset.py",
    "register":   HERE / "register_mattergen_properties.py",
    "preprocess": HERE / "preprocess_mattergen_dataset.py",
}

ALL_STEPS = ["build", "clean", "register", "preprocess"]

STEP_DESCRIPTIONS = {
    "build":      "Query Materials Project API and build CSV dataset",
    "clean":      "Generate cleaned subsets for conditional generation",
    "register":   "Register custom PSC properties in MatterGen config",
    "preprocess": "Convert CSV to binary cache + update average_density",
}

STEP_REQUIRES_API = {"build"}
STEP_DEPS = {
    "clean":      ["build"],
    "register":   [],
    "preprocess": ["build"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(step: str):
    desc = STEP_DESCRIPTIONS[step]
    print(f"\n{'=' * 60}")
    print(f"  STEP: {step.upper()}  —  {desc}")
    print(f"{'=' * 60}")


def _run_script(script: Path, cwd: Path, env: dict | None = None):
    print(f"[RUN] python {script.name}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(cwd),
        env=env or os.environ.copy(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"{script.name} exited with code {result.returncode}")


def _check_dep(step: str, completed: set[str]):
    deps = STEP_DEPS.get(step, [])
    missing = [d for d in deps if d not in completed]
    if missing:
        raise RuntimeError(
            f"Step '{step}' requires {missing} to run first. "
            "Add them to --steps or run in default order."
        )


def _check_output(path: Path, description: str) -> bool:
    if path.exists():
        print(f"[OK] {path.relative_to(ROOT)}  ({description})")
        return True
    print(f"[MISSING] {path.relative_to(ROOT)}  ({description})")
    return False


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------

def step_build(mp_api_key: str):
    _banner("build")

    if not SOURCE_CSV.exists():
        print(
            f"[ERROR] Source database not found:\n  {SOURCE_CSV}\n"
            "Download from Zenodo (see dataset/README.md) and place it in dataset/."
        )
        sys.exit(1)

    env = os.environ.copy()
    env["MP_API_KEY"] = mp_api_key

    _run_script(SCRIPTS["build"], cwd=HERE, env=env)

    # Quick verification
    out = HERE / "mattergen_dataset" / "full_dataset.csv"
    if out.exists():
        import pandas as pd
        n = len(pd.read_csv(out))
        print(f"\n[DONE] full_dataset.csv: {n:,} records")
    else:
        print("[WARN] full_dataset.csv not found — check build_dataset.log for errors")


def step_clean():
    _banner("clean")

    full_csv = HERE / "mattergen_dataset" / "full_dataset.csv"
    if not full_csv.exists():
        raise RuntimeError("mattergen_dataset/full_dataset.csv not found. Run 'build' first.")

    _run_script(SCRIPTS["clean"], cwd=HERE)

    # Report subsets
    import json
    summary_file = HERE / "mattergen_dataset_cleaned" / "subsets_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        print("\nCleaned subsets:")
        for name, info in summary.items():
            if isinstance(info, dict):
                train_n = info.get("train_size", "?")
                val_n = info.get("val_size", "?")
                print(f"  {name:<22} train={train_n:>5}  val={val_n:>4}")


def step_register():
    _banner("register")

    mattergen_root = ROOT / "mcp" / "design_agent" / "mattergen"
    if not mattergen_root.exists():
        print(
            f"[WARN] MatterGen installation not found at:\n  {mattergen_root}\n"
            "Skipping property registration. You can run this step later once\n"
            "MatterGen is installed via:\n"
            "  cd mcp/design_agent/mattergen && pip install -e ."
        )
        return

    _run_script(SCRIPTS["register"], cwd=HERE)
    print("[DONE] Custom properties registered in MatterGen globals.py")


def step_preprocess():
    _banner("preprocess")

    train_csv = HERE / "mattergen_dataset" / "train.csv"
    if not train_csv.exists():
        raise RuntimeError("mattergen_dataset/train.csv not found. Run 'build' first.")

    _run_script(SCRIPTS["preprocess"], cwd=HERE)

    cache_dir = HERE / "mattergen_dataset" / "cache"
    if cache_dir.exists():
        n_cache_files = sum(1 for _ in cache_dir.rglob("*") if _.is_file())
        print(f"[DONE] Binary cache: {n_cache_files} files in {cache_dir.relative_to(ROOT)}")
    else:
        print(
            "[INFO] Binary cache directory not created — this is expected if MatterGen\n"
            "       is not installed. The CSV files are sufficient for evaluation;\n"
            "       the cache is only needed for training with mattergen-finetune."
        )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify():
    print(f"\n{'=' * 60}")
    print("  VERIFICATION")
    print(f"{'=' * 60}")

    all_ok = True
    checks = [
        (SOURCE_CSV,                                             "source Perovskite Database"),
        (HERE / "mattergen_dataset" / "full_dataset.csv",       "MatterGen full dataset"),
        (HERE / "mattergen_dataset" / "train.csv",              "MatterGen train split"),
        (HERE / "mattergen_dataset" / "val.csv",                "MatterGen val split"),
        (HERE / "mattergen_dataset" / "dataset_stats.json",     "dataset statistics"),
        (HERE / "mattergen_dataset_cleaned" / "pce_only" / "train.csv",    "pce_only train"),
        (HERE / "mattergen_dataset_cleaned" / "high_pce" / "train.csv",    "high_pce train"),
        (HERE / "mattergen_dataset_cleaned" / "multi_target" / "train.csv","multi_target train"),
    ]

    import pandas as pd
    for path, desc in checks:
        if path.suffix == ".csv" and path.exists():
            n = len(pd.read_csv(path))
            ok = _check_output(path, f"{desc}  {n:,} rows")
        else:
            ok = _check_output(path, desc)
        all_ok = all_ok and ok

    cache_dir = HERE / "mattergen_dataset" / "cache"
    if cache_dir.exists():
        n = sum(1 for _ in cache_dir.rglob("*") if _.is_file())
        print(f"[OK] {cache_dir.relative_to(ROOT)}  (binary cache, {n} files)")
    else:
        print(f"[INFO] {cache_dir.relative_to(ROOT)}  (not built — run 'preprocess' step)")

    return all_ok


# ---------------------------------------------------------------------------
# Next-step hints after successful run
# ---------------------------------------------------------------------------

def _print_next_steps():
    print(f"""
{'=' * 60}
  NEXT STEPS — Fine-tuning MatterGen
{'=' * 60}

1. Enter the MatterGen directory:
   cd mcp/design_agent/mattergen

2. Single-property fine-tuning (PCE only):
   mattergen-finetune \\
       adapter.pretrained_name=mattergen_base \\
       data_module=psc_perovskite \\
       data_module.properties=["pce"] \\
       +lightning_module/diffusion_module/model/property_embeddings\\
@adapter.adapter.property_embeddings_adapt.pce=pce \\
       trainer.devices=1 \\
       ~trainer.logger

3. Multi-property fine-tuning (recommended):
   mattergen-finetune \\
       adapter.pretrained_name=mattergen_base \\
       data_module=psc_perovskite \\
       data_module.properties=["pce","dft_band_gap","energy_above_hull"] \\
       +lightning_module/diffusion_module/model/property_embeddings\\
@adapter.adapter.property_embeddings_adapt.pce=pce \\
       +lightning_module/diffusion_module/model/property_embeddings\\
@adapter.adapter.property_embeddings_adapt.dft_band_gap=dft_band_gap \\
       +lightning_module/diffusion_module/model/property_embeddings\\
@adapter.adapter.property_embeddings_adapt.energy_above_hull=energy_above_hull \\
       trainer.devices=1 \\
       ~trainer.logger

4. Submit to a GPU cluster (Slurm):
   sbatch train_mattergen.slurm
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end MatterGen dataset preparation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mp-api-key",
        default=os.environ.get("MP_API_KEY", ""),
        help="Materials Project API key (required for 'build' step). "
             "Alternatively set the MP_API_KEY environment variable.",
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
        "--verify",
        action="store_true",
        help="Verify existing outputs without reprocessing.",
    )
    args = parser.parse_args()

    if args.verify:
        ok = verify()
        sys.exit(0 if ok else 1)

    steps = ALL_STEPS if "all" in args.steps else args.steps

    # Validate API key requirement
    if "build" in steps and not args.mp_api_key:
        print(
            "[ERROR] --mp-api-key is required for the 'build' step.\n"
            "Get a free key at: https://materialsproject.org/api"
        )
        sys.exit(1)

    print("=" * 60)
    print("  MatterGen Data Preparation Pipeline")
    print("=" * 60)
    print(f"Steps    : {' → '.join(steps)}")
    print(f"Directory: {HERE}")
    if args.mp_api_key and "build" in steps:
        masked = "*" * (len(args.mp_api_key) - 4) + args.mp_api_key[-4:]
        print(f"MP key   : {masked}")

    step_fn = {
        "build":      lambda: step_build(args.mp_api_key),
        "clean":      step_clean,
        "register":   step_register,
        "preprocess": step_preprocess,
    }

    completed: set[str] = set()
    failed = []

    for step in steps:
        try:
            _check_dep(step, completed)
            step_fn[step]()
            completed.add(step)
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
        if set(steps) >= {"build", "preprocess"}:
            _print_next_steps()


if __name__ == "__main__":
    main()
