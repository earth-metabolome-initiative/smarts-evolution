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
    EvolutionConfig, EvolutionTask, FoldData, FoldSample, SeedCorpus,
};
use smarts_rs::PreparedTarget;

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

let result = task.evolve(&config, &seed_corpus).unwrap();
assert!(!result.best_smarts().is_empty());
assert!(result.best_mcc().is_finite());
```

## Terminal Progress Bars

Enable the `indicatif` feature and call `task.evolve_with_indicatif(&config, &seed_corpus)` to get a generation progress bar plus a per-generation SMARTS evaluation bar. The evaluation bar reports completion, current SMARTS and MCC, generation-best SMARTS and MCC, and incumbent-best SMARTS and MCC. For large prepared datasets, use `task.evolve_owned_with_indicatif_progress(...)` to move the task folds into the session; use `IndicatifEvolutionProgress::attach_to(&multi)` or `from_bars(...)` to embed the bars in an existing `MultiProgress`. Non-terminal callers can implement `EvolutionProgressObserver` and pass it to `task.evolve_with_observer(...)`.

## Pathological SMARTS Evaluation

The GA logs each SMARTS evaluation at `debug` level, applies `EvolutionConfig::match_time_limit` as a cooperative per-match safety fuse, and emits a `warn` log when matching exceeds that limit. `MatchLimitResult::Exceeded` is treated as unknown, so the affected genome receives invalid fitness instead of counting the sample as a non-match. SMARTS length is used only for deterministic tie-breaking and optional `max_evaluation_smarts_len` filtering.

For a standalone `.log` file, initialize the built-in file logger before starting evolution:

```rust,no_run
smarts_evolution::FileLogConfig::new("smarts-evolution.log")
    .level(smarts_evolution::LevelFilter::Debug)
    .init()
    .expect("initialize smarts-evolution file logger");
```

The file logger is file-only by default so indicatif bars keep control of stderr. Use `.mirror_to_stderr(true)` only when terminal log lines are useful.
