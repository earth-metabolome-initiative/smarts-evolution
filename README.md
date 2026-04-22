# smarts-evolution

[![CI](https://github.com/LucaCappelletti94/smarts-evolution/actions/workflows/ci.yml/badge.svg)](https://github.com/LucaCappelletti94/smarts-evolution/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/LucaCappelletti94/smarts-evolution/graph/badge.svg)](https://codecov.io/gh/LucaCappelletti94/smarts-evolution)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Evolving SMARTS patterns against a binary classification task.

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

let seed_corpus = SeedCorpus::try_from([
    "[#6](=[#8])[#7]",
    "[#6]~[#7]",
])
.unwrap();

let result = evolve_task(&task, &config, &seed_corpus).unwrap();
assert!(!result.best_smarts().is_empty());
assert!(result.best_mcc().is_finite());
```
