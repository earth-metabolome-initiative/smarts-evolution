use alloc::vec::Vec;
use core::fmt;

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

/// Evaluates SMARTS genomes using fold-averaged MCC.
#[derive(Clone)]
pub struct SmartsEvaluator {
    folds: Vec<FoldData>,
}

impl fmt::Debug for SmartsEvaluator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
        Some(self.fold_counts_for_compiled(&compiled))
    }

    // Keep fold aggregation localized so future evaluator changes stay local.
    fn fold_counts_for_compiled(&self, compiled: &CompiledQuery) -> Vec<ConfusionCounts> {
        self.folds
            .iter()
            .map(|fold| count_matches_in_fold(compiled, fold))
            .collect()
    }

    /// Evaluate the MCC objective for a genome.
    pub fn objective_of(&self, genome: &SmartsGenome) -> ObjectiveFitness {
        let Some(fold_counts) = self.fold_counts_for_query(genome.query()) else {
            return ObjectiveFitness::invalid();
        };

        let mcc = compute_fold_averaged_mcc(&fold_counts);
        ObjectiveFitness::from_mcc(mcc)
    }

    pub fn fitness_of(&self, genome: &SmartsGenome) -> ObjectiveFitness {
        self.objective_of(genome)
    }
}

fn count_matches_in_fold(query: &CompiledQuery, fold: &FoldData) -> ConfusionCounts {
    let mut counts = ConfusionCounts::default();

    for sample in fold.samples() {
        counts += count_match_for_sample(query, sample);
    }

    counts
}

fn count_match_for_sample(query: &CompiledQuery, sample: &FoldSample) -> ConfusionCounts {
    let mut counts = ConfusionCounts::default();
    counts.record_match(
        matches_compiled(query, sample.target()),
        sample.is_positive(),
    );
    counts
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use std::{format, vec};

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

        assert_eq!(counts, ConfusionCounts::new(1, 1, 1, 1));
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
    fn fitness_of_matches_objective_of() {
        let evaluator = SmartsEvaluator::new(vec![FoldData::new(vec![sample("CN", true)])]);
        let genome = SmartsGenome::from_smarts("[#6]~[#7]").unwrap();
        let via_trait = evaluator.fitness_of(&genome);
        let direct = evaluator.objective_of(&genome);
        assert!((via_trait.mcc() - direct.mcc()).abs() < 1e-12);
    }
}
