use std::time::Instant;

use genevo::genetic::{Fitness, FitnessFunction};
use smarts_validator::{CompiledQuery, PreparedTarget, matches_compiled};

use super::mcc::{ConfusionCounts, compute_fold_averaged_mcc};
use super::objective::ObjectiveFitness;
use crate::genome::SmartsGenome;

/// One labeled evaluation target.
#[derive(Clone, Debug)]
pub struct FoldSample {
    target: PreparedTarget,
    is_positive: bool,
}

impl FoldSample {
    /// Create one labeled evaluation sample.
    pub fn new(target: PreparedTarget, is_positive: bool) -> Self {
        Self {
            target,
            is_positive,
        }
    }

    pub fn positive(target: PreparedTarget) -> Self {
        Self::new(target, true)
    }

    pub fn negative(target: PreparedTarget) -> Self {
        Self::new(target, false)
    }

    pub fn target(&self) -> &PreparedTarget {
        &self.target
    }

    pub fn is_positive(&self) -> bool {
        self.is_positive
    }
}

/// A single fold's data for MCC evaluation.
#[derive(Clone, Debug)]
pub struct FoldData {
    /// Test-set prepared targets in this fold.
    samples: Vec<FoldSample>,
}

impl FoldData {
    /// Create one evaluation fold from prepared labeled samples.
    pub fn new(samples: Vec<FoldSample>) -> Self {
        Self { samples }
    }

    pub fn samples(&self) -> &[FoldSample] {
        &self.samples
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Evaluates SMARTS genomes using fold-averaged MCC with a runtime penalty.
#[derive(Clone)]
pub struct SmartsEvaluator {
    folds: Vec<FoldData>,
}

impl std::fmt::Debug for SmartsEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SmartsEvaluator")
            .field("num_folds", &self.folds.len())
            .finish()
    }
}

impl SmartsEvaluator {
    /// Create a new evaluator over one or more prepared folds.
    pub fn new(folds: Vec<FoldData>) -> Self {
        Self { folds }
    }

    fn fold_counts_for_query(
        &self,
        query: &smarts_parser::QueryMol,
    ) -> Option<Vec<ConfusionCounts>> {
        let compiled = CompiledQuery::new(query.clone()).ok()?;
        let mut fold_counts = Vec::with_capacity(self.folds.len());
        for fold in &self.folds {
            fold_counts.push(count_matches_in_fold(&compiled, fold));
        }
        Some(fold_counts)
    }

    /// Evaluate the combined MCC + runtime objective for a genome.
    pub fn objective_of(&self, genome: &SmartsGenome) -> ObjectiveFitness {
        let started_at = Instant::now();
        let Some(fold_counts) = self.fold_counts_for_query(genome.query()) else {
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
            .map(|value| value.elapsed_micros() as u128)
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

    for sample in fold.samples() {
        let matched = matches_compiled(query, sample.target());
        match (matched, sample.is_positive()) {
            (true, true) => counts.tp += 1,
            (true, false) => counts.fp += 1,
            (false, true) => counts.fn_ += 1,
            (false, false) => counts.tn += 1,
        }
    }

    counts
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use genevo::genetic::FitnessFunction;
    use smiles_parser::Smiles;

    use super::*;

    fn prepared(smiles: &str) -> PreparedTarget {
        PreparedTarget::new(Smiles::from_str(smiles).unwrap())
    }

    fn sample(smiles: &str, is_positive: bool) -> FoldSample {
        FoldSample::new(prepared(smiles), is_positive)
    }

    #[test]
    fn fold_data_wraps_samples() {
        let fold = FoldData::new(vec![sample("CN", true), sample("CC", false)]);
        assert_eq!(fold.len(), 2);
        assert!(!fold.is_empty());
        assert!(FoldData::new(Vec::new()).is_empty());
    }

    #[test]
    fn count_matches_in_fold_tracks_all_confusion_buckets() {
        let fold = FoldData::new(vec![
            sample("CN", true),
            sample("CN", false),
            sample("CC", true),
            sample("CC", false),
        ]);
        let genome = SmartsGenome::from_smarts("[#6]~[#7]").unwrap();
        let compiled = CompiledQuery::new(genome.query().clone()).unwrap();

        let counts = count_matches_in_fold(&compiled, &fold);

        assert_eq!(
            counts,
            ConfusionCounts {
                tp: 1,
                fp: 1,
                tn: 1,
                fn_: 1,
            }
        );
    }

    #[test]
    fn evaluator_debug_and_objective_work_for_simple_fold() {
        let evaluator = SmartsEvaluator::new(vec![FoldData::new(vec![
            sample("CN", true),
            sample("CC", false),
        ])]);
        let genome = SmartsGenome::from_smarts("[#6]~[#7]").unwrap();
        let fitness = evaluator.objective_of(&genome);

        assert!(format!("{evaluator:?}").contains("num_folds"));
        assert!(fitness.mcc() > 0.9);
    }

    #[test]
    fn evaluator_average_and_bounds_cover_empty_and_non_empty_inputs() {
        let evaluator = SmartsEvaluator::new(vec![FoldData::new(vec![sample("CN", true)])]);
        let low = ObjectiveFitness::from_metrics(0.25, std::time::Duration::from_micros(100));
        let high = ObjectiveFitness::from_metrics(0.75, std::time::Duration::from_micros(300));

        assert_eq!(evaluator.average(&[]), ObjectiveFitness::zero());

        let average = evaluator.average(&[low, high]);
        assert!((average.mcc() - 0.5).abs() < 0.01);
        assert_eq!(average.elapsed(), std::time::Duration::from_micros(200));
        assert_eq!(
            evaluator.highest_possible_fitness(),
            ObjectiveFitness::from_metrics(1.0, std::time::Duration::ZERO)
        );
        assert_eq!(
            evaluator.lowest_possible_fitness(),
            ObjectiveFitness::zero()
        );
        let genome = SmartsGenome::from_smarts("[#6]~[#7]").unwrap();
        let via_trait = evaluator.fitness_of(&genome);
        let direct = evaluator.objective_of(&genome);
        assert!((via_trait.mcc() - direct.mcc()).abs() < 1e-12);
        assert!(via_trait.elapsed() > std::time::Duration::ZERO);
        assert!(direct.elapsed() > std::time::Duration::ZERO);
    }
}
