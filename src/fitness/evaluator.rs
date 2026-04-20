use std::str::FromStr;
use std::time::Instant;

use genevo::genetic::{Fitness, FitnessFunction};
use smarts_parser::QueryMol;
use smarts_validator::{CompiledQuery, PreparedTarget, matches_compiled};

use super::mcc::{ConfusionCounts, compute_fold_averaged_mcc};
use super::objective::ObjectiveFitness;
use crate::genome::genome::SmartsGenome;
use crate::genome::parser::parse_and_validate_smarts;

/// A single fold's data for MCC evaluation.
#[derive(Clone, Debug)]
pub struct FoldData {
    /// Test-set prepared targets in this fold.
    pub targets: Vec<PreparedTarget>,
    /// Whether each test molecule is a positive example.
    pub is_positive: Vec<bool>,
}

/// Evaluates SMARTS genomes using fold-averaged MCC with a runtime penalty.
#[derive(Clone)]
pub struct SmartsEvaluator {
    folds: std::rc::Rc<Vec<FoldData>>,
}

impl std::fmt::Debug for SmartsEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SmartsEvaluator")
            .field("num_folds", &self.folds.len())
            .finish()
    }
}

impl SmartsEvaluator {
    pub fn new(folds: Vec<FoldData>) -> Result<Self, Box<dyn std::error::Error>> {
        for fold in &folds {
            if fold.targets.len() != fold.is_positive.len() {
                return Err("local evaluator received inconsistent fold lengths".into());
            }
        }

        Ok(Self {
            folds: std::rc::Rc::new(folds),
        })
    }

    pub fn num_folds(&self) -> usize {
        self.folds.len()
    }

    pub fn fold_counts_of(&self, genome: &SmartsGenome) -> Option<Vec<ConfusionCounts>> {
        self.fold_counts_for_smarts(&genome.smarts_string)
    }

    pub fn fold_counts_for_smarts(&self, smarts: &str) -> Option<Vec<ConfusionCounts>> {
        if parse_and_validate_smarts(smarts).is_err() {
            return None;
        }
        let query = QueryMol::from_str(smarts).ok()?;
        let compiled = CompiledQuery::new(query).ok()?;

        let mut fold_counts = Vec::with_capacity(self.folds.len());
        for fold in self.folds.iter() {
            fold_counts.push(count_matches_in_fold(&compiled, fold));
        }
        Some(fold_counts)
    }

    /// Evaluate the fold-averaged MCC for a genome.
    pub fn mcc_of(&self, genome: &SmartsGenome) -> f64 {
        self.fold_counts_of(genome)
            .map(|fold_counts| compute_fold_averaged_mcc(&fold_counts))
            .unwrap_or(-1.0)
    }

    /// Evaluate the combined MCC + runtime objective for a genome.
    pub fn objective_of(&self, genome: &SmartsGenome) -> ObjectiveFitness {
        self.objective_for_smarts(&genome.smarts_string)
    }

    pub fn objective_for_smarts(&self, smarts: &str) -> ObjectiveFitness {
        let started_at = Instant::now();
        let Some(fold_counts) = self.fold_counts_for_smarts(smarts) else {
            return ObjectiveFitness::invalid(started_at.elapsed());
        };

        let mcc = compute_fold_averaged_mcc(&fold_counts);
        ObjectiveFitness::from_metrics(mcc, started_at.elapsed())
    }
}

impl FitnessFunction<SmartsGenome, ObjectiveFitness> for SmartsEvaluator {
    fn fitness_of(&self, genome: &SmartsGenome) -> ObjectiveFitness {
        self.objective_of(genome)
    }

    fn average(&self, values: &[ObjectiveFitness]) -> ObjectiveFitness {
        if values.is_empty() {
            return ObjectiveFitness::zero();
        }
        let average_mcc = values.iter().map(|value| value.mcc()).sum::<f64>() / values.len() as f64;
        let average_elapsed_micros = values
            .iter()
            .map(|value| value.elapsed_micros as u128)
            .sum::<u128>()
            / values.len() as u128;

        ObjectiveFitness::from_metrics(
            average_mcc,
            std::time::Duration::from_micros(average_elapsed_micros as u64),
        )
    }

    fn highest_possible_fitness(&self) -> ObjectiveFitness {
        ObjectiveFitness::from_metrics(1.0, std::time::Duration::ZERO)
    }

    fn lowest_possible_fitness(&self) -> ObjectiveFitness {
        ObjectiveFitness::zero()
    }
}

fn count_matches_in_fold(query: &CompiledQuery, fold: &FoldData) -> ConfusionCounts {
    let mut counts = ConfusionCounts::default();

    for (target, is_positive) in fold.targets.iter().zip(&fold.is_positive) {
        let matched = matches_compiled(query, target);
        match (matched, *is_positive) {
            (true, true) => counts.tp += 1,
            (true, false) => counts.fp += 1,
            (false, true) => counts.fn_ += 1,
            (false, false) => counts.tn += 1,
        }
    }

    counts
}
