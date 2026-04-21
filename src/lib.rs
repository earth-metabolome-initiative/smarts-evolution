#![doc = include_str!("../README.md")]

pub mod evolution;
pub mod fitness;
pub mod genome;
pub mod operators;

pub use evolution::config::{EvolutionConfig, EvolutionConfigBuilder};
pub use evolution::runner::{EvolutionTask, TaskResult, evolve_task};
pub use fitness::evaluator::{FoldData, FoldSample, SmartsEvaluator};
pub use genome::SmartsGenome;
pub use genome::seed::SeedCorpus;
