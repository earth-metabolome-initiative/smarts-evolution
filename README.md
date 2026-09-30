# smarts-evolution

[![CI](https://github.com/LucaCappelletti94/smarts-evolution/actions/workflows/ci.yml/badge.svg)](https://github.com/LucaCappelletti94/smarts-evolution/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/LucaCappelletti94/smarts-evolution/graph/badge.svg)](https://codecov.io/gh/LucaCappelletti94/smarts-evolution)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Evolving SMARTS patterns against a binary classification task.

## Quick Start

```rust
use smiles_rs::Smiles;
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

## TUI

Enable the `tui` feature and call `task.evolve_with_tui(&config, &seed_corpus)` for a native terminal dashboard with `[copy]` buttons for the best and change-point SMARTS.

![The smarts-evolution TUI dashboard plotting best-so-far MCC per generation](docs/tui.gif)

The `examples/tui.rs` example runs a sample "amide vs the rest" classification on the dashboard. Try it with:

```sh
cargo run --release --example tui --features tui
```

Copy uses the OS clipboard on a local desktop. Over SSH or tmux it falls back to an OSC 52 escape so the SMARTS reaches the clipboard on the machine you are viewing. Two things are needed for that path:

- An outer terminal that supports OSC 52. Most modern terminals do (kitty, Alacritty, WezTerm, foot, iTerm2, recent xterm). GNOME Terminal and other VTE-based terminals do not.
- Passthrough enabled inside tmux, which defaults off on tmux 3.3 and later. Add this to your `tmux.conf`:

  ```tmux
  set -g allow-passthrough on
  ```

When passthrough is off, the status line says so and shows the command to fix it instead of falsely reporting success.

## Progress Bars

Enable the `indicatif` feature and call `task.evolve_with_indicatif(&config, &seed_corpus)` for generation, evaluation, and offspring progress bars. Use `evolve_owned_with_indicatif_progress(...)` for large prepared datasets, or implement `EvolutionProgressObserver` and pass it to `evolve_with_observer(...)` for non-terminal callers.

## Logging

The GA logs each SMARTS evaluation at `debug` level and applies `EvolutionConfig::match_time_limit` as a per-SMARTS safety fuse. Patterns that exceed the limit get invalid fitness rather than counting as a non-match.

For a standalone log file, initialize the file logger before starting evolution:

```rust,no_run
smarts_evolution::FileLogConfig::new("smarts-evolution.log")
    .level(smarts_evolution::LevelFilter::Debug)
    .init()
    .expect("initialize smarts-evolution file logger");
```
