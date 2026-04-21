# smarts-evolution

`smarts-evolution` is a Rust library for evolving one SMARTS query against one
binary task.

This crate is only the search engine. Dataset loading, label systems, fold
construction, experiment orchestration, and reporting belong in downstream
code.

## Scope

Use this crate when you already have:

- prepared molecules as `PreparedTarget`
- one or more labeled evaluation folds
- a binary objective such as `class A` vs `everything else`
- a SMARTS seed corpus, or willingness to start from built-in seeds

What the crate does:

- evolves SMARTS with mutation and crossover
- seeds the population from a corpus plus built-in fragments
- scores candidates with fold-averaged MCC
- penalizes slower SMARTS when MCC is otherwise similar

What the crate does not do:

- read datasets
- build folds for you
- know about any taxonomy or label hierarchy
- run experiment batches
- generate reports

## Quick Start

```rust
use std::str::FromStr;

use smiles_parser::Smiles;
use smarts_evolution::{
    EvolutionConfig, EvolutionTask, FoldData, FoldSample, SeedCorpus, evolve_task,
};
use smarts_validator::PreparedTarget;

fn prepared(smiles: &str) -> PreparedTarget {
    PreparedTarget::new(Smiles::from_str(smiles).unwrap())
}

let task = EvolutionTask::new(
    "amide-vs-rest",
    vec![FoldData::new(vec![
        FoldSample::positive(prepared("CC(=O)N")),
        FoldSample::positive(prepared("NC(=O)C")),
        FoldSample::negative(prepared("CCO")),
        FoldSample::negative(prepared("c1ccccc1")),
    ])],
);

let config = EvolutionConfig::builder()
    .population_size(8)
    .generation_limit(2)
    .stagnation_limit(2)
    .build()?;

let seed_corpus = SeedCorpus::from_smarts(vec![
    "[#6](=[#8])[#7]".to_string(),
    "[#6]~[#7]".to_string(),
])?;

let result = evolve_task(&task, &config, &seed_corpus)?;
assert!(!result.best_smarts().is_empty());
assert!(result.best_mcc().is_finite());
# Ok::<(), Box<dyn std::error::Error>>(())
```

The main entry point is:

- `evolve_task(task, config, seed_corpus)`

You provide prepared labeled folds and a seed corpus. The result contains the
best SMARTS found, its MCC, its evaluation time, and the number of generations
run.

## Seeding

The initial population is mixed from:

- a curated `SeedCorpus`
- built-in SMARTS fragments
- random genomes

`SeedCorpus` supports:

- `SeedCorpus::builtin()`
- `SeedCorpus::from_smarts(Vec<String>)`
- `insert_smarts(...)`
- `extend_from_smarts(...)`

With the default `std-io` feature, it also supports:

- `SeedCorpus::from_file(...)`

File-backed corpora are plain text:

- one SMARTS per line
- blank lines ignored
- lines starting with `#` ignored

## Objective

The search objective is:

- maximize fold-averaged MCC
- prefer lower compile-and-match time as a tiebreaking pressure

That keeps the search from drifting toward bloated SMARTS that are only
competitive because they are expensive.

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

For wasm builds:

```bash
cargo check --target wasm32-unknown-unknown --no-default-features
```

## Build And Test

```bash
cargo build
cargo test
```
