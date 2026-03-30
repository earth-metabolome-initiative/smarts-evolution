<!-- cSpell:ignore librdkit classyfire SMARTS smarts jsonl -->
# smarts-evolution

Evolutionary algorithm system that discovers optimal SMARTS patterns for each node in hierarchical chemical taxonomies. Fitness is evaluated using MCC, Information Gain, and Lin semantic similarity with k-fold stratified cross-validation.

## Prerequisites

- Rust (nightly, edition 2024)
- Conda with RDKit C++ libraries:
  ```bash
  conda install -c conda-forge librdkit-dev
  ```

## Build

```bash
# Debug (fast compile, slow runtime)
cargo build

# Release (recommended for actual runs)
cargo build --release
```

## Usage

### Inspect a dataset

```bash
# NPC classifier (multi-label, .jsonl)
cargo run --release -- load --dataset npc --path part-000002.jsonl

# ClassyFire taxonomy (single-label, .jsonl.zst)
cargo run --release -- load --dataset classyfire --path success-000001.jsonl.zst
```

### Run evolution

```bash
# NPC dataset — full run with defaults (200 pop, 500 gens, 5-fold CV)
RUST_LOG=info cargo run --release -- evolve \
  --dataset npc \
  --path part-000002.jsonl \
  --checkpoint-dir checkpoints/npc

# ClassyFire dataset
RUST_LOG=info cargo run --release -- evolve \
  --dataset classyfire \
  --path success-000001.jsonl.zst \
  --checkpoint-dir checkpoints/classyfire

# Short test run
RUST_LOG=info cargo run --release -- evolve \
  --dataset npc \
  --path part-000003.jsonl \
  --population-size 50 \
  --generation-limit 10 \
  --stagnation-limit 100 \
  --checkpoint-dir /tmp/smarts-test
```

### CLI options for `evolve`

| Flag | Default | Description |
|---|---|---|
| `--dataset` | required | `npc` or `classyfire` |
| `--path` | required | Path to data file |
| `--population-size` | 200 | Genomes per generation |
| `--generation-limit` | 500 | Max generations per node |
| `--stagnation-limit` | 50 | Stop if no improvement for N gens |
| `--folds` | 5 | Number of cross-validation folds |
| `--checkpoint-dir` | `checkpoints` | Directory for checkpoint JSON |
| `--resume` | false | Resume from existing checkpoint |

### Resume with new data

As new labels arrive (~10k/day), append to the JSONL file and resume:

```bash
RUST_LOG=info cargo run --release -- evolve \
  --dataset npc \
  --path part-000002.jsonl \
  --checkpoint-dir checkpoints/npc \
  --resume
```

Already-evolved nodes are skipped; new nodes above the 20-compound threshold are evolved.

## Output

### Logs

With `RUST_LOG=info`, each node logs per-generation progress:

```
Gen 0: fitness=43612, MCC=0.152, IG=0.087, Lin=0.634, smarts=[#6][#6]=[#6]
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
      "best_fitness": 79248,
      "best_mcc": 0.856,
      "best_ig": 0.234,
      "best_lin": 0.912
    }
  }
}
```

## Architecture

- **Data**: Loads ClassyFire (.jsonl.zst) and NPC (.jsonl) with multi-label support
- **Taxonomy DAG**: Auto-constructed from data, grows with new classes
- **Genome**: SMARTS token sequences with parser/display roundtrip
- **Operators**: 6 mutation types, fragment-exchange crossover, all RDKit-validated
- **Fitness**: MCC + Information Gain + Lin similarity (weighted composite, 5-fold CV)
- **Evolution**: Per-node in topological DAG order; parent SMARTS pre-filters candidates

## Tests

```bash
cargo test
```

20 tests covering parser roundtrip, fitness metrics, operator validity rates, and stratified splitting.
