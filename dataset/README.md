# Dataset Documentation — PeroMAS

[![arXiv](https://img.shields.io/badge/arXiv-2602.13312-b31b1b.svg)](https://arxiv.org/abs/2602.13312)
[![Zenodo](https://img.shields.io/badge/Zenodo-datasets-blue.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Paper:** *PeroMAS: A Multi-agent System of Perovskite Material Discovery*
> Yishu Wang, Wei Liu, Yifan Li, Shengxiang Xu, Xujie Yuan, Ran Li, Yuyu Luo, Jia Zhu, Shimin Di, Min-Ling Zhang, Guixiang Li
> arXiv preprint: [arXiv:2602.13312](https://arxiv.org/abs/2602.13312)

This document describes all datasets used in **PeroMAS** — an autonomous multi-agent system for accelerated perovskite solar cell (PSC) material discovery. It covers dataset acquisition, processing pipelines, and instructions to reproduce every training and evaluation split from scratch.

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Accessing Datasets on Zenodo](#2-accessing-datasets-on-zenodo)
3. [Source Database](#3-source-database)
4. [MatterGen Training Dataset](#4-mattergen-training-dataset)
5. [CSLLM Fine-tuning Datasets](#5-csllm-fine-tuning-datasets)
6. [FabAgent Prediction Dataset](#6-fabagent-prediction-dataset)
7. [Reproducing All Datasets from Scratch](#7-reproducing-all-datasets-from-scratch)
8. [Dependencies](#8-dependencies)

---

## 1. Dataset Overview

PeroMAS uses three categories of datasets, each serving a distinct AI component:

| Dataset | Used by | Size | Format |
|---|---|---|---|
| Perovskite Database | All agents (source) | 43,412 records, 738 columns | CSV |
| MatterGen training set | DesignAgent (MatterGen) | ~27,440 records | CSV + binary cache |
| CSLLM fine-tuning sets | DesignAgent (CSLLM) | ~43K instruction pairs × 3 tasks | JSON (Alpaca) |
| FabAgent prediction set | FabAgent (Random Forest) | ~700 records | CSV + CSR sparse matrices |

All processed datasets (training splits, evaluation splits, model checkpoints) are archived on **Zenodo** and can be reproduced locally using the scripts in this directory.

---

## 2. Accessing Datasets on Zenodo

All datasets and pre-trained model weights are publicly available:

> **Zenodo DOI:** `https://doi.org/10.5281/zenodo.XXXXXXX` *(will be updated upon acceptance)*

The Zenodo archive contains the following items:

```
zenodo_archive/
├── Perovskite_database_content_all_data.csv   # Raw source database (87 MB)
├── mattergen_dataset/
│   ├── full_dataset.csv                        # Complete MatterGen dataset
│   ├── train.csv                               # 90% training split
│   └── val.csv                                 # 10% validation split
├── csllm_finetune_datasets/
│   ├── dataset_synthesis.json                  # Synthesis feasibility task
│   ├── dataset_method.json                     # Synthesis method task
│   └── dataset_precursor.json                  # Precursor recommendation task
├── fabagent_data/
│   ├── full_dataset.csv                        # FabAgent prediction dataset
│   ├── cif_only_sp1_oliynyk_zero_csr.npz      # CIF-based feature matrix
│   └── comp_only_sp1_oliynyk_zero_csr.npz     # Composition-based feature matrix
└── model_checkpoints/
    ├── csllm/                                  # CSLLM LoRA adapter weights
    └── fabagent/                               # Trained RF/GBDT models (.joblib)
```

### Quick Download

```bash
# Install zenodo_get
pip install zenodo_get

# Download the full archive
zenodo_get 10.5281/zenodo.XXXXXXX

# Or download individual files via the Zenodo web interface
```

---

## 3. Source Database

### 3.1 Description

The foundation of all datasets is the **Perovskite Database Project**, a community-curated compilation of published PSC experimental results.

- **Source:** [perovskitedatabase.com](https://www.perovskitedatabase.com) / [doi:10.1038/s41597-022-01383-2](https://doi.org/10.1038/s41597-022-01383-2)
- **File:** `Perovskite_database_content_all_data.csv`
- **Size:** 87 MB, 43,412 device records, 738 columns

### 3.2 Key Column Groups

| Group | Example Columns | Description |
|---|---|---|
| Reference | `Ref_DOI_number`, `Ref_publication_date` | Literature metadata |
| Composition | `Perovskite_composition_short_form`, `Perovskite_composition_a_ions` | ABC₃ perovskite ions and coefficients |
| Performance (JV) | `JV_default_PCE`, `JV_default_Voc`, `JV_default_Jsc`, `JV_default_FF` | Solar cell figures of merit |
| Stability | `Stability_PCE_T80`, `Stability_PCE_T95` | Lifetime metrics |
| Deposition | `Perovskite_deposition_procedure`, `Perovskite_deposition_solvents` | Fabrication conditions |
| Device Stack | `Cell_stack_sequence`, `Cell_architecture` | Layer structure (nip/pin) |
| ETL/HTL | `ETL_stack_sequence`, `HTL_stack_sequence` | Transport layer info |

### 3.3 Performance Column Priority

When extracting PCE, Voc, Jsc, and FF, the following priority order is used:

```
JV_default_* > JV_reverse_scan_* > JV_forward_scan_*
```

### 3.4 Data Validation Ranges

Physical plausibility filters applied during processing:

| Property | Valid Range | Unit |
|---|---|---|
| PCE | 0 – 35 | % |
| Voc | 0 – 2.0 | V |
| Jsc | 0 – 35 | mA/cm² |
| FF | 0 – 100 | % |
| Band gap | 0.5 – 4.0 | eV |
| Stability T80 | > 0 | hours |
| Film thickness | 50 – 2000 | nm |
| Annealing temperature | 20 – 500 | °C |

---

## 4. MatterGen Training Dataset

### 4.1 Description

This dataset is used to fine-tune [MatterGen](https://github.com/microsoft/mattergen) — a diffusion-based crystal structure generative model — for conditional generation of perovskite structures targeting specified PCE, band gap, and stability.

- **Script:** `dataset/mattergen/build_mattergen_dataset.py`
- **Output directory:** `dataset/mattergen/mattergen_dataset/`
- **Final size:** ~700 unique material entries (after deduplication)
- **Train / Val split:** 90% / 10% (random seed 42)

### 4.2 Dataset Schema

| Column | Type | Description |
|---|---|---|
| `material_id` | string | Unique ID, format: `{ref_id}_{mp_id}` |
| `cif` | string | Crystal structure in CIF format (from Materials Project) |
| `mp_id` | string | Materials Project ID (e.g., `mp-35909`) |
| `composition` | string | Full chemical formula (long form) |
| `composition_short` | string | Short formula (e.g., `CsSnI3`) |
| `pce` | float | Power conversion efficiency (%) |
| `dft_band_gap` | float | DFT-calculated band gap from Materials Project (eV) |
| `energy_above_hull` | float | Thermodynamic stability from Materials Project (eV/atom) |
| `voc` | float | Open-circuit voltage (V) |
| `jsc` | float | Short-circuit current density (mA/cm²) |
| `ff` | float | Fill factor (%) |
| `stability_t80` | float | T80 lifetime (hours) |
| `cell_architecture` | string | Device configuration (`nip` or `pin`) |

### 4.3 Property Statistics

| Property | Min | Max | Mean | Median |
|---|---|---|---|---|
| PCE (%) | 0.00 | 34.80 | 9.50 | 9.18 |
| Band gap (eV) | 0.44 | 4.14 | 1.67 | — |
| Energy above hull (eV/atom) | 0.0001 | 0.3727 | — | — |
| Stability T80 (h) | 13.6 | 3400.0 | 603.2 | — |

### 4.4 Build Pipeline

The `build_mattergen_dataset.py` script executes the following steps:

```
Perovskite_database_content_all_data.csv  (43,412 records)
    │
    ├─ 1. Parse ABC₃ composition (A, B, X ions + coefficients)
    │       Organic cation mapping: MA → CH₃NH₃, FA → CH(NH₂)₂, ...
    │
    ├─ 2. Query Materials Project API
    │       Strategy 1: exact formula match (e.g., CH₃NH₃PbI₃)
    │       Strategy 2: chemical system match (e.g., Cs-Pb-I)
    │       Structure selection: lowest energy_above_hull, ≤ 20 atoms
    │
    ├─ 3. Merge performance data
    │       PCE, Voc, Jsc, FF from Perovskite Database
    │       DFT band gap, energy_above_hull from Materials Project
    │
    ├─ 4. Data cleaning
    │       Physical plausibility validation
    │       Remove compositions containing 'nan' or pipe-separated mixtures
    │
    ├─ 5. Deduplicate by (composition, mp_id), keep highest PCE record
    │
    └─ 6. 90/10 train/val split (random_state=42)

Output: train.csv, val.csv, full_dataset.csv
```

**Match rate:** ~75.2% of records successfully matched to Materials Project structures. Unmatched records (primarily fully organic perovskites not in MP) are saved to `unmatched_entries.csv` for analysis.

### 4.5 Cleaned Subsets

After building the full dataset, `clean_mattergen_dataset.py` generates targeted subsets in `mattergen_dataset_cleaned/`:

| Subset | Description | Condition |
|---|---|---|
| `pce_only` | PCE-conditioned generation | PCE available |
| `pce_bandgap` | PCE + band gap conditioning | PCE + band gap available |
| `pce_stability` | PCE + stability conditioning | PCE + T80 available |
| `multi_target` | All three properties | PCE + band gap + T80 |
| `high_quality` | High-confidence records | Low uncertainty |
| `high_pce` | Top-performing materials | PCE > 15% |

### 4.6 Requirements

- **Materials Project API key** (free): [https://materialsproject.org/api](https://materialsproject.org/api)
- Package: `mp-api` (see [Dependencies](#8-dependencies))

---

## 5. CSLLM Fine-tuning Datasets

### 5.1 Description

Three instruction-following datasets in **Alpaca format** for fine-tuning LLaMA-3-8B (CSLLM) to predict perovskite synthesis feasibility, method, and precursors.

- **Script:** `dataset/CSLLM/generate_perovskite_finetune_datasets.py`
- **Output directory:** `dataset/CSLLM/finetune_datasets/`

### 5.2 Dataset Tasks

#### Task 1 — Synthesis Feasibility (`dataset_synthesis.json`)

Predicts whether a perovskite composition can be experimentally synthesized.

```json
{
  "instruction": "Based on the perovskite composition, determine if this material can be synthesized.",
  "input": "MAPbI3",
  "output": "yes"
}
```

#### Task 2 — Synthesis Method (`dataset_method.json`)

Classifies the optimal deposition/synthesis method for a given composition.

Supported methods: `spin_coating`, `two_step`, `blade_coating`, `slot_die`, `inkjet`, `spray`, `evaporation`, `cvd`, `sputtering`, `ald`

```json
{
  "instruction": "What synthesis method is most appropriate for this perovskite?",
  "input": "FAPbI3",
  "output": "spin_coating"
}
```

#### Task 3 — Precursor Recommendation (`dataset_precursor.json`)

Recommends precursor chemicals and solvents for a target composition.

```json
{
  "instruction": "List the precursor materials needed to synthesize this perovskite.",
  "input": "MAPbI3",
  "output": "MAI (methylammonium iodide), PbI2 (lead iodide); Solvent: DMF/DMSO"
}
```

### 5.3 Ion Coverage

| Site | Ions Covered |
|---|---|
| A-site cations | MA, FA, Cs, Rb, K, Ba, PEA, GU, EA, DMA, and others |
| B-site metals | Pb, Sn, Ge, Bi, Sb, Cu, Ag, In, Tl |
| X-site halides | I, Br, Cl, F |

### 5.4 CSLLM Model Checkpoints

Pre-trained LoRA adapter weights are available on Zenodo:

| Model | Base | Fine-tuning Steps | Task |
|---|---|---|---|
| `synthesis_llm_llama3` | LLaMA-3-8B | ~196K | Synthesis feasibility |
| `method_llm_llama3` | LLaMA-3-8B | ~31K | Synthesis method |
| `precursor_llm_llama3` | LLaMA-3-8B | ~20K | Precursor recommendation |

---

## 6. FabAgent Prediction Dataset

### 6.1 Description

Dataset and pre-computed feature matrices for training the Random Forest / Gradient Boosting models that predict six PSC properties.

- **Source:** Derived from `mattergen_dataset/full_dataset.csv`
- **Script:** `mcp/fab_agent/Perovskite_PI_Multi/process.py`
- **Output:** `mcp/fab_agent/Perovskite_PI_Multi/data/`
- **Size:** ~700 records (after quality filtering)
- **Train / Test split:** 80% / 20%

### 6.2 Prediction Targets

| Target | Symbol | Unit | Description |
|---|---|---|---|
| `pce` | PCE | % | Power conversion efficiency |
| `voc` | Voc | V | Open-circuit voltage |
| `jsc` | Jsc | mA/cm² | Short-circuit current density |
| `ff` | FF | % | Fill factor |
| `dft_band_gap` | Eg | eV | DFT-calculated band gap |
| `energy_above_hull` | E_hull | eV/atom | Thermodynamic stability |

### 6.3 Feature Engineering

Two complementary feature sets are computed and merged:

**Composition Features (CBFV — Oliynyk element properties, 264 features)**

Generated using composition-based feature vectors from elemental property tables. Organic cation abbreviations (MA, FA, etc.) are expanded to full molecular formulas before feature computation.

**Crystal Structure Features (from CIF, 9 features)**

Extracted from Materials Project CIF files using `pymatgen`:

| Feature | Description |
|---|---|
| `density` | Atomic density (atoms/Å³) |
| `volume` | Unit cell volume (Å³) |
| `a`, `b`, `c` | Lattice parameters (Å) |
| `alpha`, `beta`, `gamma` | Lattice angles (°) |
| `nsites` | Number of atomic sites |

Features are stored as sparse CSR matrices (`.npz`) for efficiency.

### 6.4 Trained Models

Pre-trained model files are available on Zenodo:

| File | Algorithm | Notes |
|---|---|---|
| `model_RF_pce.joblib` | Random Forest | 270 estimators |
| `model_RF_voc.joblib` | Random Forest | 270 estimators |
| `model_RF_jsc.joblib` | Random Forest | 270 estimators |
| `model_RF_ff.joblib` | Random Forest | 270 estimators |
| `model_RF_dft_band_gap.joblib` | Random Forest | 270 estimators |
| `model_RF_energy_above_hull.joblib` | Random Forest | 270 estimators |

---

## 7. Reproducing All Datasets from Scratch

### 7.1 Prerequisites

```bash
# Clone the repository
git clone https://github.com/your-org/PSC_Agents.git
cd PSC_Agents

# Install dependencies
pip install -r requirements.txt

# Set your Materials Project API key (required for MatterGen dataset)
export MP_API_KEY="your_api_key_here"
# Get a free key at: https://materialsproject.org/api
```

### 7.2 Download the Source Database

Download the Perovskite Database CSV from Zenodo or the official source:

```bash
# Option A: from Zenodo (recommended)
zenodo_get 10.5281/zenodo.XXXXXXX --record=Perovskite_database_content_all_data.csv
mv Perovskite_database_content_all_data.csv dataset/

# Option B: from perovskitedatabase.com
# Visit https://www.perovskitedatabase.com and download the full CSV
```

### 7.3 One-Command Reproduction

A unified script `prepare_datasets.py` orchestrates all processing steps:

```bash
cd dataset
python prepare_datasets.py --mp-api-key YOUR_MP_API_KEY [--steps all]
```

**Available steps (can be run individually with `--steps`):**

| Step | Flag | Description |
|---|---|---|
| 1 | `mattergen` | Build MatterGen training dataset |
| 2 | `mattergen_clean` | Generate cleaned subsets |
| 3 | `csllm` | Generate CSLLM fine-tuning datasets |
| 4 | `fabagent` | Compute FabAgent feature matrices |
| all | `all` | Run all steps in order (default) |

**Examples:**

```bash
# Run everything
python prepare_datasets.py --mp-api-key YOUR_KEY

# Run only the CSLLM dataset generation (no API key needed)
python prepare_datasets.py --steps csllm

# Run MatterGen + cleaning steps only
python prepare_datasets.py --mp-api-key YOUR_KEY --steps mattergen mattergen_clean
```

### 7.4 Step-by-Step Manual Execution

If you prefer to run each step individually:

```bash
# Step 1: Build MatterGen dataset (requires Materials Project API key)
cd dataset/mattergen
python build_mattergen_dataset.py
# Output: mattergen_dataset/train.csv, val.csv, full_dataset.csv

# Step 2: Generate cleaned subsets
python clean_mattergen_dataset.py
# Output: mattergen_dataset_cleaned/{pce_only,pce_bandgap,...}/

# Step 3: Generate CSLLM fine-tuning datasets
cd ../CSLLM
python generate_perovskite_finetune_datasets.py
# Output: finetune_datasets/{dataset_synthesis,dataset_method,dataset_precursor}.json

# Step 4: Process FabAgent features
cd ../../mcp/fab_agent/Perovskite_PI_Multi
# First, copy the MatterGen dataset
cp ../../../dataset/mattergen/mattergen_dataset/full_dataset.csv data/raw/
python process.py
# Output: data/csr/*.npz
```

### 7.5 Expected Runtime

| Step | Runtime | Notes |
|---|---|---|
| MatterGen dataset build | 2–4 hours | Depends on Materials Project API rate limits; caching reduces repeat runs |
| MatterGen cleaning | < 5 minutes | — |
| CSLLM dataset generation | 10–30 minutes | — |
| FabAgent feature processing | 5–15 minutes | — |

### 7.6 Verifying Outputs

After running all steps, verify with:

```bash
python prepare_datasets.py --verify
```

Expected output summary:

```
[OK] dataset/Perovskite_database_content_all_data.csv  (43412 rows)
[OK] dataset/mattergen/mattergen_dataset/train.csv     (90% split)
[OK] dataset/mattergen/mattergen_dataset/val.csv       (10% split)
[OK] dataset/CSLLM/finetune_datasets/dataset_synthesis.json
[OK] dataset/CSLLM/finetune_datasets/dataset_method.json
[OK] dataset/CSLLM/finetune_datasets/dataset_precursor.json
[OK] mcp/fab_agent/Perovskite_PI_Multi/data/raw/full_dataset.csv
[OK] mcp/fab_agent/Perovskite_PI_Multi/data/csr/cif_only_sp1_oliynyk_zero_csr.npz
[OK] mcp/fab_agent/Perovskite_PI_Multi/data/csr/comp_only_sp1_oliynyk_zero_csr.npz
```

---

## 8. Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

Key packages for dataset processing:

| Package | Version | Purpose |
|---|---|---|
| `pandas` | >=2.0.0 | Data manipulation |
| `numpy` | >=1.24.0 | Numerical computing |
| `pymatgen` | >=2024.1.1 | CIF parsing, crystal structure analysis |
| `mp-api` | >=0.41.0 | Materials Project API client |
| `scikit-learn` | >=1.3.0 | Feature processing, ML models |
| `scipy` | >=1.9.0 | Sparse matrix (CSR) operations |
| `tqdm` | any | Progress bars |
| `joblib` | >=1.3.0 | Model serialization |

---

## Citation

If you use these datasets in your research, please cite the PeroMAS paper and the original Perovskite Database:

```bibtex
@misc{wang2026peromasmultiagentperovskitematerial,
      title     = {PeroMAS: A Multi-agent System of Perovskite Material Discovery},
      author    = {Yishu Wang and Wei Liu and Yifan Li and Shengxiang Xu and
                   Xujie Yuan and Ran Li and Yuyu Luo and Jia Zhu and
                   Shimin Di and Min-Ling Zhang and Guixiang Li},
      year      = {2026},
      eprint    = {2602.13312},
      archivePrefix = {arXiv},
      primaryClass  = {cs.MA},
      url       = {https://arxiv.org/abs/2602.13312},
}

@article{perovskitedatabase,
  title   = {An open-access database and analysis tool for perovskite solar cells
             based on the FAIR data principles},
  author  = {Jacobsson, T.J. and others},
  journal = {Nature Energy},
  volume  = {7},
  pages   = {107--115},
  year    = {2022},
  doi     = {10.1038/s41560-021-00941-3}
}
```

---

## License

The Perovskite Database is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Processed datasets and model weights in this repository follow the same license. See `LICENSE` for details.
