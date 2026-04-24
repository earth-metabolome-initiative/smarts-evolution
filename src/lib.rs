#![no_std]
#![doc = include_str!("../README.md")]

extern crate alloc;
#[cfg(any(test, feature = "std"))]
extern crate std;

pub mod evolution;
pub mod fitness;
pub mod genome;
pub mod operators;

pub use evolution::config::{EvolutionConfig, EvolutionConfigBuilder};
#[cfg(feature = "indicatif")]
pub use evolution::indicatif::IndicatifEvolutionProgress;
pub use evolution::runner::{
    EvolutionError, EvolutionEvaluationProgress, EvolutionProgress, EvolutionProgressObserver,
    EvolutionSession, EvolutionStatus, EvolutionTask, RankedSmarts, TaskResult,
};
pub use fitness::evaluator::{
    EvaluatedSmarts, EvaluationProgress as SmartsEvaluationProgress, EvaluationSet, FoldData,
    FoldSample, LabeledCorpus, SmartsEvaluator,
};
pub use genome::SmartsGenome;
pub use genome::seed::SeedCorpus;
