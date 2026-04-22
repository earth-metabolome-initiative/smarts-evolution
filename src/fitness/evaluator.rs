use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
use rayon::prelude::*;
use smarts_validator::{CompiledQuery, PreparedTarget, matches_compiled};

use super::mcc::{ConfusionCounts, compute_fold_averaged_mcc};
use super::objective::ObjectiveFitness;
use crate::genome::SmartsGenome;

#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
const MIN_PARALLEL_GENOMES: usize = 8;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenomeEvaluation {
    fitness: ObjectiveFitness,
    phenotype: Arc<[u64]>,
}

impl GenomeEvaluation {
    pub fn fitness(&self) -> ObjectiveFitness {
        self.fitness
    }

    pub fn phenotype(&self) -> &Arc<[u64]> {
        &self.phenotype
    }
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

    fn fold_counts_and_behavior_for_query(
        &self,
        query: &smarts_parser::QueryMol,
    ) -> Option<(Vec<ConfusionCounts>, Arc<[u64]>)> {
        let compiled = CompiledQuery::new(query.clone()).ok()?;
        Some(self.fold_counts_and_behavior_for_compiled(&compiled))
    }

    // Keep fold aggregation localized so future evaluator changes stay local.
    fn fold_counts_and_behavior_for_compiled(
        &self,
        compiled: &CompiledQuery,
    ) -> (Vec<ConfusionCounts>, Arc<[u64]>) {
        let total_samples: usize = self.folds.iter().map(FoldData::len).sum();
        let mut behavior_words = vec![0u64; total_samples.div_ceil(64)];
        let mut bit_index = 0usize;
        let fold_counts = self
            .folds
            .iter()
            .map(|fold| count_matches_in_fold(compiled, fold, &mut behavior_words, &mut bit_index))
            .collect();
        (fold_counts, Arc::from(behavior_words))
    }

    /// Evaluate the MCC objective for a genome.
    pub fn objective_of(&self, genome: &SmartsGenome) -> ObjectiveFitness {
        self.evaluate(genome).fitness()
    }

    pub fn fitness_of(&self, genome: &SmartsGenome) -> ObjectiveFitness {
        self.objective_of(genome)
    }

    pub fn evaluate(&self, genome: &SmartsGenome) -> GenomeEvaluation {
        let Some((fold_counts, phenotype)) =
            self.fold_counts_and_behavior_for_query(genome.query())
        else {
            return GenomeEvaluation {
                fitness: ObjectiveFitness::invalid(),
                phenotype: Arc::from([]),
            };
        };

        let mcc = compute_fold_averaged_mcc(&fold_counts);
        GenomeEvaluation {
            fitness: ObjectiveFitness::from_mcc(mcc),
            phenotype,
        }
    }

    /// Evaluate a batch of genomes, using Rayon on std targets when the batch
    /// is large enough to amortize scheduling overhead.
    pub fn evaluate_many(
        &self,
        genomes: Vec<SmartsGenome>,
    ) -> Vec<(SmartsGenome, GenomeEvaluation)> {
        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        if genomes.len() >= MIN_PARALLEL_GENOMES {
            return genomes
                .into_par_iter()
                .map(|genome| {
                    let evaluation = self.evaluate(&genome);
                    (genome, evaluation)
                })
                .collect();
        }

        genomes
            .into_iter()
            .map(|genome| {
                let evaluation = self.evaluate(&genome);
                (genome, evaluation)
            })
            .collect()
    }

    pub fn fitness_of_many(
        &self,
        genomes: Vec<SmartsGenome>,
    ) -> Vec<(SmartsGenome, ObjectiveFitness)> {
        self.evaluate_many(genomes)
            .into_iter()
            .map(|(genome, evaluation)| (genome, evaluation.fitness()))
            .collect()
    }
}

fn count_matches_in_fold(
    query: &CompiledQuery,
    fold: &FoldData,
    behavior_words: &mut [u64],
    bit_index: &mut usize,
) -> ConfusionCounts {
    let mut counts = ConfusionCounts::default();

    for sample in fold.samples() {
        let matched = matches_compiled(query, sample.target());
        record_behavior_match(behavior_words, *bit_index, matched);
        *bit_index += 1;
        counts.record_match(matched, sample.is_positive());
    }

    counts
}

fn record_behavior_match(behavior_words: &mut [u64], bit_index: usize, matched: bool) {
    if !matched {
        return;
    }

    let word_index = bit_index / 64;
    let bit_offset = bit_index % 64;
    behavior_words[word_index] |= 1u64 << bit_offset;
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
        let mut behavior = vec![0u64; 1];
        let mut bit_index = 0usize;

        let counts = count_matches_in_fold(&compiled, &fold, &mut behavior, &mut bit_index);

        assert_eq!(counts, ConfusionCounts::new(1, 1, 1, 1));
        assert_eq!(behavior[0], 0b0011);
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

    #[test]
    fn fold_sample_constructors_and_batch_helpers_work() {
        let positive = FoldSample::positive(prepared("CN"));
        let negative = FoldSample::negative(prepared("CC"));
        assert!(positive.is_positive());
        assert!(!negative.is_positive());
        let _ = positive.target();
        let _ = negative.target();

        let evaluator = SmartsEvaluator::new(vec![FoldData::new(vec![
            positive.clone(),
            negative.clone(),
        ])]);
        let genomes = vec![
            SmartsGenome::from_smarts("[#6]").unwrap(),
            SmartsGenome::from_smarts("[#7]").unwrap(),
            SmartsGenome::from_smarts("[#8]").unwrap(),
            SmartsGenome::from_smarts("[#6]~[#7]").unwrap(),
            SmartsGenome::from_smarts("[#6]~[#8]").unwrap(),
            SmartsGenome::from_smarts("[#6]=[#8]").unwrap(),
            SmartsGenome::from_smarts("[#6]-[#6]").unwrap(),
            SmartsGenome::from_smarts("[#6]-[#7]").unwrap(),
        ];

        let evaluated = evaluator.evaluate_many(genomes.clone());
        let fitness_only = evaluator.fitness_of_many(genomes);
        assert_eq!(evaluated.len(), 8);
        assert_eq!(fitness_only.len(), 8);
        assert!(evaluated.iter().zip(fitness_only.iter()).all(
            |((left_genome, left_eval), (right_genome, right_fitness))| {
                left_genome == right_genome && left_eval.fitness() == *right_fitness
            }
        ));
    }

    #[test]
    fn evaluate_tracks_exact_training_behavior() {
        let evaluator = SmartsEvaluator::new(vec![FoldData::new(vec![
            sample("CN", true),
            sample("CC", false),
            sample("CN", false),
        ])]);
        let genome = SmartsGenome::from_smarts("[#6]~[#7]").unwrap();

        let evaluation = evaluator.evaluate(&genome);

        assert_eq!(evaluation.phenotype().as_ref(), &[0b0101]);
        assert!(evaluation.fitness().mcc() > 0.49);
    }
}
