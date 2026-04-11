# smarts-evolution

Evolutionary algorithm system that discovers optimal SMARTS patterns for each node in hierarchical chemical taxonomies. Fitness is evaluated using MCC with k-fold stratified cross-validation.

## Prerequisites

- Rust (nightly, edition 2024)
- Conda environment with matching RDKit runtime and C++ headers from conda-forge.
  Older RDKit builds fail in `rdkit-sys` with missing `PeriodicTable::getMaxAtomicNumber()` /
  `PeriodicTable::getRow()` symbols. Pin `rdkit` and `rdkit-dev` to the same conda-forge
  release line. As of March 31, 2026, the conda-forge package pages list `2025.09.6` as
  the latest version; if your channel already has `2026.03.*`, use that same line for both.
  Install `libboost-headers` explicitly so the C++ build can find `boost/config.hpp`.
  Do not add `boost-cpp` explicitly here: with current conda-forge packaging it can make the
  solve fail.
  ```bash
  conda create -n smarts-evolution -c conda-forge \
    "rdkit=2025.09.*" \
    "rdkit-dev=2025.09.*" \
    libboost-headers \
    cxx-compiler
  conda activate smarts-evolution
  ```

If you already have an environment, update it in place instead:

```bash
conda install -c conda-forge \
  "rdkit=2025.09.*" \
  "rdkit-dev=2025.09.*" \
  libboost-headers \
  cxx-compiler
```

## Build

```bash
# Debug (fast compile, slow runtime)
cargo build

# Release (recommended for actual runs)
cargo build --release
```

## Usage

If you omit `--path`, `smarts-evolution` now resolves the latest weekly Zenodo snapshot automatically:

- `npc` downloads the latest `completed.jsonl.zst` from `npc-labeler` and rebuilds `npc.fully_labeled.jsonl.zst`
- `classyfire` downloads the latest `classyfire-labels.jsonl.zst` from `classyfire`, fetches PubChem `CID-SMILES.gz` if needed, and rebuilds `classyfire.jsonl.zst`

Prepared datasets are cached in the repository root, and raw downloads/state live under `.tmp/datasets/`.

### Inspect a dataset

```bash
# NPC classifier (managed weekly snapshot)
cargo run --release -- load --dataset npc

# ClassyFire taxonomy (managed weekly snapshot)
cargo run --release -- load --dataset classyfire

# Optional explicit path override
cargo run --release -- load --dataset npc --path /path/to/custom.jsonl.zst
```

### Run evolution

```bash
# NPC dataset — full run with defaults (200 pop, 500 gens, 5-fold CV)
RUST_LOG=info cargo run --release -- evolve \
  --dataset npc \
  --checkpoint-dir checkpoints/npc

# ClassyFire dataset
RUST_LOG=info cargo run --release -- evolve \
  --dataset classyfire \
  --population-size 2048 \
  --generation-limit 1000 \
  --stagnation-limit 25 \
  --folds 5 \
  --checkpoint-dir checkpoints/classyfire

# Recommended NPC run: moderate population, bounded stagnation
RUST_LOG=info cargo run --release -- evolve \
  --dataset npc \
  --population-size 2048 \
  --generation-limit 1000 \
  --stagnation-limit 25 \
  --folds 5 \
  --checkpoint-dir checkpoints/npc-reasonable

# Short test run
RUST_LOG=info cargo run --release -- evolve \
  --dataset npc \
  --population-size 50 \
  --generation-limit 10 \
  --stagnation-limit 100 \
  --checkpoint-dir /tmp/smarts-test
```

### CLI options for `evolve`

| Flag | Default | Description |
|---|---|---|
| `--dataset` | required | `npc` or `classyfire` |
| `--path` | latest managed snapshot | Optional explicit path to a data file |
| `--population-size` | 200 | Genomes per generation |
| `--generation-limit` | 500 | Max generations per node |
| `--stagnation-limit` | 50 | Stop if no improvement for N gens |
| `--folds` | 5 | Number of cross-validation folds |
| `--checkpoint-dir` | `checkpoints` | Directory for checkpoint JSON |
| `--resume` | false | Resume from existing checkpoint |

### Resume with new data

As new weekly Zenodo releases arrive, rerun `evolve --resume` and the managed dataset will refresh before loading:

```bash
RUST_LOG=info cargo run --release -- evolve \
  --dataset npc \
  --checkpoint-dir checkpoints/npc \
  --resume
```

Already-evolved nodes are skipped; new nodes above the 20-compound threshold are evolved.

## Output

### Logs

With `RUST_LOG=info`, each node logs per-generation progress:

```
Gen 1: lead_score=43612, lead_mcc=0.152, global_mcc=0.152, unique=182/200, duplicates=18, avg_len=7.4, best_len=6, cache_hits=0/182, smarts=[#6][#6]=[#6]
```

### Checkpoints

Evolved SMARTS are saved to `checkpoint.json`:

```json
{
  "nodes": {
    "42": {
      "node_name": "Penicillins",
      "level": 2,
      "best_smarts": "[#6][#6]([#6])[#16][C@@H][C@H]",
      "best_mcc": 0.856
    }
  }
}
```

## Architecture

- **Data**: Loads ClassyFire and NPC from `.jsonl` or `.jsonl.zst`
- **Taxonomy DAG**: Auto-constructed from data, grows with new classes
- **Genome**: SMARTS token sequences with parser/display roundtrip
- **Operators**: bounded mutation and crossover with structural validation
- **Fitness**: MCC-only, 5-fold CV
- **Evolution**: Per-node in topological DAG order; parent SMARTS pre-filters candidates

## Tests

```bash
cargo test
```

The test suite covers parser roundtrip, compressed dataset loading, fitness metrics, operator validity rates, worker-process evaluation, and stratified splitting.
