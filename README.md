# smarts-evolution

`smarts-evolution` is a Rust library for evolving SMARTS queries against a binary task.

It is only the search engine. Data loading, label systems, taxonomy traversal, fold construction, experiment orchestration, and reporting belong in downstream code.

## What You Use It For

Use this crate when you already have:

- a binary problem such as `class A` vs `everything else`
- molecules prepared as `PreparedTarget`
- one or more evaluation folds
- a seed corpus, positive SMILES, or both

The crate then searches for a SMARTS query that scores well on those folds.

## What The Library Does

- evolves SMARTS with aggressive mutation and crossover
- seeds the population from a real corpus instead of only random generation
- scores candidates with fold-averaged MCC
- penalizes slow SMARTS so cheap queries are preferred over equally accurate expensive ones
- returns the best SMARTS found for one task

## What It Does Not Do

- read datasets
- build folds for you
- know about NPClassifier, ClassyFire, or any other label system
- traverse DAGs or taxonomies
- run experiment batches
- generate reports

## Main Entry Point

```rust
pub fn evolve_task(
    task: &EvolutionTask,
    config: &EvolutionConfig,
    seed_corpus: &SeedCorpus,
) -> Result<TaskResult, Box<dyn std::error::Error>>
```

You pass in:

- `EvolutionTask`
- `EvolutionConfig`
- `SeedCorpus`

`EvolutionTask` contains:

- `task_id`
- `positive_smiles`
- `folds: Vec<FoldData>`

Each `FoldData` contains:

- `targets: Vec<PreparedTarget>`
- `is_positive: Vec<bool>`

The result contains:

- `best_smarts`
- `best_mcc`
- `best_eval_time`
- `best_score`
- `generations`

## Scoring

The objective is:

- higher fold-averaged MCC is better
- slower compile and match time is worse

This keeps the search from drifting toward bloated SMARTS that only win by being expensive.

## Seeding

The initial population is mixed from several sources:

- a curated SMARTS seed corpus
- fragments derived from positive-example SMILES
- small hand-written primitive patterns
- random genomes

`SeedCorpus` supports:

- `SeedCorpus::builtin()`
- `SeedCorpus::from_smarts(Vec<String>)`
- `insert_smarts(...)`
- `extend_from_smarts(...)`

When the `std-io` feature is enabled, it also supports:

- `SeedCorpus::from_file(...)`

File-backed corpora are plain text:

- one SMARTS per line
- blank lines ignored
- lines starting with `#` ignored

In-memory corpora can be built directly from `Vec<String>`.

## Minimal Example

```rust
use std::str::FromStr;

use smiles_parser::Smiles;
use smarts_evolution::evolution::config::EvolutionConfig;
use smarts_evolution::evolution::runner::{EvolutionTask, evolve_task};
use smarts_evolution::fitness::evaluator::FoldData;
use smarts_evolution::genome::seed::SeedCorpus;
use smarts_validator::PreparedTarget;

fn prepared(smiles: &str) -> PreparedTarget {
    PreparedTarget::new(Smiles::from_str(smiles).unwrap())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let task = EvolutionTask {
        task_id: "amide-vs-rest".to_string(),
        positive_smiles: vec!["CC(=O)N".to_string(), "NC(=O)C".to_string()],
        folds: vec![FoldData {
            targets: vec![
                prepared("CC(=O)N"),
                prepared("NC(=O)C"),
                prepared("CCO"),
                prepared("c1ccccc1"),
            ],
            is_positive: vec![true, true, false, false],
        }],
    };

    let seed_corpus = SeedCorpus::from_smarts(vec![
        "[#6](=[#8])[#7]".to_string(),
        "[#6]~[#7]".to_string(),
    ])?;

    let result = evolve_task(&task, &EvolutionConfig::default(), &seed_corpus)?;

    println!("{}", result.best_smarts);
    println!("{}", result.best_mcc);
    Ok(())
}
```

## Add To `Cargo.toml`

During local development, depend on the crate by path:

```toml
[dependencies]
smarts-evolution = { path = "../smarts-evolution" }
smiles-parser = { git = "https://github.com/earth-metabolome-initiative/smiles-parser", branch = "main" }
smarts-validator = { git = "https://github.com/earth-metabolome-initiative/smarts-rs", branch = "main", package = "smarts-validator" }
```

## Features

- `std-io`
  Enabled by default. Adds `SeedCorpus::from_file(...)`.
- `wasm`
  Enables `genevo`'s `wasm-bindgen` support for `wasm32` targets.

For wasm builds, use:

```bash
cargo check --target wasm32-unknown-unknown --no-default-features --features wasm
```

## Build

```bash
cargo build
```

## Test

```bash
cargo test
```
