# smarts-evolution

[![CI](https://github.com/LucaCappelletti94/smarts-evolution/actions/workflows/ci.yml/badge.svg)](https://github.com/LucaCappelletti94/smarts-evolution/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/LucaCappelletti94/smarts-evolution/graph/badge.svg)](https://codecov.io/gh/LucaCappelletti94/smarts-evolution)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`smarts-evolution` is a Rust library for evolving one SMARTS pattern against one binary classification task.

## Quick Start

```rust
use core::str::FromStr;

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
    .build()
    .unwrap();

let seed_corpus = SeedCorpus::from_smarts(vec![
    "[#6](=[#8])[#7]".to_string(),
    "[#6]~[#7]".to_string(),
])
.unwrap();

let result = evolve_task(&task, &config, &seed_corpus).unwrap();
assert!(!result.best_smarts().is_empty());
assert!(result.best_mcc().is_finite());
```

## Search Objective

1. maximize fold-averaged MCC
2. if MCC ties, prefer shorter SMARTS
3. if both still tie, use lexical order for determinism

## Seed Corpus

`SeedCorpus` supports:

- `SeedCorpus::builtin()`
- `SeedCorpus::from_smarts(Vec<String>)`
- `insert_smarts(...)`
- `extend_from_smarts(...)`

## no_std

```bash
cargo check --target wasm32-unknown-unknown
```

## Development

```bash
cargo fmt
cargo clippy --all-targets --all-features
cargo test
cargo bench --bench evolution -- --noplot
```

## License

[MIT](LICENSE)
