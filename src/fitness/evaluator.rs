use genevo::genetic::{Fitness, FitnessFunction};

use super::mcc::{ConfusionCounts, MccFitness, compute_fold_averaged_mcc};
use crate::genome::genome::SmartsGenome;
use crate::genome::parser::parse_and_validate_smarts;
use crate::rdkit_substruct_library::{CompiledSmartsQuery, SubstructLibraryIndex};

/// A single fold's data for MCC evaluation.
pub struct FoldData {
    /// Test-set SMILES in this fold.
    pub smiles: Vec<String>,
    /// Whether each test molecule is a positive example.
    pub is_positive: Vec<bool>,
}

/// Evaluates SMARTS genomes using fold-averaged MCC only.
#[derive(Clone)]
pub struct SmartsEvaluator {
    folds: std::rc::Rc<Vec<SubstructLibraryIndex>>,
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
        let mut libraries = Vec::with_capacity(folds.len());
        for fold in folds {
            if fold.smiles.len() != fold.is_positive.len() {
                return Err("local evaluator received inconsistent fold lengths".into());
            }

            let mut library = SubstructLibraryIndex::new()?;
            for (smiles, is_positive) in fold.smiles.into_iter().zip(fold.is_positive) {
                library.add_smiles(&smiles, is_positive)?;
            }
            libraries.push(library);
        }

        Ok(Self {
            folds: std::rc::Rc::new(libraries),
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
        let query = CompiledSmartsQuery::new(smarts)?;

        let mut fold_counts = Vec::with_capacity(self.folds.len());
        for library in self.folds.iter() {
            fold_counts.push(library.count_matches_compiled(&query, 1)?);
        }
        Some(fold_counts)
    }

    /// Evaluate the fold-averaged MCC for a genome.
    pub fn mcc_of(&self, genome: &SmartsGenome) -> f64 {
        self.fold_counts_of(genome)
            .map(|fold_counts| compute_fold_averaged_mcc(&fold_counts))
            .unwrap_or(-1.0)
    }
}

impl FitnessFunction<SmartsGenome, MccFitness> for SmartsEvaluator {
    fn fitness_of(&self, genome: &SmartsGenome) -> MccFitness {
        MccFitness::from_mcc(self.mcc_of(genome))
    }

    fn average(&self, values: &[MccFitness]) -> MccFitness {
        if values.is_empty() {
            return MccFitness::zero();
        }
        let sum: i64 = values.iter().map(|v| v.score).sum();
        MccFitness {
            score: sum / values.len() as i64,
        }
    }

    fn highest_possible_fitness(&self) -> MccFitness {
        MccFitness {
            score: MccFitness::max_score(),
        }
    }

    fn lowest_possible_fitness(&self) -> MccFitness {
        MccFitness::zero()
    }
}
