#![no_std]
#![doc = include_str!("../README.md")]

extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod evolution;
pub mod fitness;
pub mod genome;
pub mod operators;

pub use evolution::config::{EvolutionConfig, EvolutionConfigBuilder};
pub use evolution::runner::{
    EvolutionError, EvolutionProgress, EvolutionStatus, EvolutionTask, RankedSmarts, TaskResult,
    evolve_task, evolve_task_with_progress,
};
pub use fitness::evaluator::{FoldData, FoldSample, SmartsEvaluator};
pub use genome::SmartsGenome;
pub use genome::seed::SeedCorpus;
