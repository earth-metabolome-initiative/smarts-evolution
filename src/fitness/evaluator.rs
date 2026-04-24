use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::time::Duration;

#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
use rayon::prelude::*;
use smarts_rs::{
    CompiledQuery, MatchScratch, PreparedTarget, QueryMol, QueryScreen, TargetCorpusIndex,
    TargetCorpusScratch,
};
#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
use std::sync::Mutex;
#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
use std::time::Instant;

use super::mcc::{ConfusionCounts, compute_fold_averaged_mcc};
use super::objective::ObjectiveFitness;
use crate::genome::SmartsGenome;
use log::debug;
#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
use log::warn;

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

/// A single labeled evaluation set.
///
/// The name comes from fold-averaged MCC support. A `FoldData` value does not
/// imply that the caller is doing cross-validation.
#[derive(Clone, Debug)]
pub struct FoldData {
    /// Test-set prepared targets in this fold.
    samples: Vec<FoldSample>,
    positive_count: usize,
    negative_count: usize,
    index: TargetCorpusIndex,
}

impl FoldData {
    /// Create one evaluation fold from prepared labeled samples.
    pub fn new(samples: Vec<FoldSample>) -> Self {
        let positive_count = samples.iter().filter(|sample| sample.is_positive()).count();
        let negative_count = samples.len().saturating_sub(positive_count);
        let targets = samples
            .iter()
            .map(FoldSample::target)
            .cloned()
            .collect::<Vec<_>>();
        let index = TargetCorpusIndex::new(&targets);
        Self {
            samples,
            positive_count,
            negative_count,
            index,
        }
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

    pub fn positive_count(&self) -> usize {
        self.positive_count
    }

    pub fn negative_count(&self) -> usize {
        self.negative_count
    }

    fn index(&self) -> &TargetCorpusIndex {
        &self.index
    }
}

/// Alias for callers that are not modeling cross-validation folds.
pub type EvaluationSet = FoldData;

/// Alias for callers that use one labeled corpus per task.
pub type LabeledCorpus = FoldData;

/// Evaluates SMARTS genomes using fold-averaged MCC.
#[derive(Clone)]
pub struct SmartsEvaluator {
    folds: Vec<FoldData>,
}

#[derive(Clone, Debug)]
pub(crate) struct EvaluationLogSettings {
    task_id: String,
    generation: u64,
    fold_count: usize,
    target_count: usize,
    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    slow_log_threshold: Option<Duration>,
}

impl EvaluationLogSettings {
    pub(crate) fn new(
        task_id: String,
        generation: u64,
        fold_count: usize,
        target_count: usize,
        slow_log_threshold: Option<Duration>,
    ) -> Self {
        #[cfg(not(all(feature = "std", not(target_arch = "wasm32"))))]
        let _ = slow_log_threshold;

        Self {
            task_id,
            generation,
            fold_count,
            target_count,
            #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
            slow_log_threshold,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatedSmarts {
    smarts: String,
    mcc: f64,
    complexity: usize,
}

impl EvaluatedSmarts {
    fn new(genome: &SmartsGenome, fitness: ObjectiveFitness) -> Self {
        Self {
            smarts: genome.smarts().to_string(),
            mcc: fitness.mcc(),
            complexity: genome.complexity(),
        }
    }

    pub fn smarts(&self) -> &str {
        &self.smarts
    }

    pub fn mcc(&self) -> f64 {
        self.mcc
    }

    pub fn complexity(&self) -> usize {
        self.complexity
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvaluationProgress {
    completed: usize,
    total: usize,
    last: Option<EvaluatedSmarts>,
    best: Option<EvaluatedSmarts>,
}

impl EvaluationProgress {
    fn new(
        completed: usize,
        total: usize,
        last: Option<EvaluatedSmarts>,
        best: Option<EvaluatedSmarts>,
    ) -> Self {
        Self {
            completed,
            total: total.max(1),
            last,
            best,
        }
    }

    pub fn completed(&self) -> usize {
        self.completed
    }

    pub fn total(&self) -> usize {
        self.total
    }

    /// Last evaluated SMARTS. With parallel evaluation this follows completion
    /// order, not input order.
    pub fn last(&self) -> Option<&EvaluatedSmarts> {
        self.last.as_ref()
    }

    pub fn last_smarts(&self) -> Option<&str> {
        self.last().map(EvaluatedSmarts::smarts)
    }

    pub fn last_mcc(&self) -> Option<f64> {
        self.last().map(EvaluatedSmarts::mcc)
    }

    pub fn best(&self) -> Option<&EvaluatedSmarts> {
        self.best.as_ref()
    }

    pub fn best_smarts(&self) -> Option<&str> {
        self.best().map(EvaluatedSmarts::smarts)
    }

    pub fn best_mcc(&self) -> Option<f64> {
        self.best().map(EvaluatedSmarts::mcc)
    }
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

    pub(crate) fn fold_count(&self) -> usize {
        self.folds.len()
    }

    pub(crate) fn target_count(&self) -> usize {
        self.folds.iter().map(FoldData::len).sum()
    }

    fn fold_counts_and_behavior_for_query(
        &self,
        query: &QueryMol,
    ) -> Option<(Vec<ConfusionCounts>, Arc<[u64]>)> {
        let compiled = CompiledQuery::new(query.clone()).ok()?;
        let screen = QueryScreen::new(compiled.query());
        Some(self.fold_counts_and_behavior_for_compiled(&compiled, &screen))
    }

    // Keep fold aggregation localized so future evaluator changes stay local.
    fn fold_counts_and_behavior_for_compiled(
        &self,
        compiled: &CompiledQuery,
        screen: &QueryScreen,
    ) -> (Vec<ConfusionCounts>, Arc<[u64]>) {
        let total_samples: usize = self.folds.iter().map(FoldData::len).sum();
        let mut behavior_words = vec![0u64; total_samples.div_ceil(64)];
        let mut bit_index = 0usize;
        let fold_counts = self
            .folds
            .iter()
            .map(|fold| {
                count_matches_in_fold(compiled, screen, fold, &mut behavior_words, &mut bit_index)
            })
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

    fn evaluate_with_logging(
        &self,
        genome: &SmartsGenome,
        settings: Option<&EvaluationLogSettings>,
    ) -> GenomeEvaluation {
        if let Some(settings) = settings {
            log_evaluation_start(genome, settings);
        }

        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        let started = Instant::now();

        let evaluation = self.evaluate(genome);

        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        if let Some(settings) = settings {
            log_slow_evaluation_if_needed(
                genome,
                evaluation.fitness(),
                settings,
                started.elapsed(),
            );
        }

        evaluation
    }

    /// Evaluate a batch of genomes, using Rayon on std targets when the batch
    /// is large enough to amortize scheduling overhead.
    pub fn evaluate_many(
        &self,
        genomes: Vec<SmartsGenome>,
    ) -> Vec<(SmartsGenome, GenomeEvaluation)> {
        self.evaluate_many_with_optional_logging(genomes, None)
    }

    pub(crate) fn evaluate_many_with_logging(
        &self,
        genomes: Vec<SmartsGenome>,
        settings: &EvaluationLogSettings,
    ) -> Vec<(SmartsGenome, GenomeEvaluation)> {
        self.evaluate_many_with_optional_logging(genomes, Some(settings))
    }

    fn evaluate_many_with_optional_logging(
        &self,
        genomes: Vec<SmartsGenome>,
        settings: Option<&EvaluationLogSettings>,
    ) -> Vec<(SmartsGenome, GenomeEvaluation)> {
        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        if genomes.len() >= MIN_PARALLEL_GENOMES {
            return genomes
                .into_par_iter()
                .map(|genome| {
                    let evaluation = self.evaluate_with_logging(&genome, settings);
                    (genome, evaluation)
                })
                .collect();
        }

        genomes
            .into_iter()
            .map(|genome| {
                let evaluation = self.evaluate_with_logging(&genome, settings);
                (genome, evaluation)
            })
            .collect()
    }

    /// Evaluate a batch of genomes and report completion progress.
    ///
    /// On std non-wasm targets this keeps the same Rayon path as
    /// [`Self::evaluate_many`] for large batches. Progress callbacks can arrive
    /// in completion order rather than input order.
    pub fn evaluate_many_with_progress(
        &self,
        genomes: Vec<SmartsGenome>,
        on_progress: impl FnMut(EvaluationProgress) + Send,
    ) -> Vec<(SmartsGenome, GenomeEvaluation)> {
        self.evaluate_many_with_progress_and_optional_logging(genomes, None, on_progress)
    }

    pub(crate) fn evaluate_many_with_progress_and_logging(
        &self,
        genomes: Vec<SmartsGenome>,
        settings: &EvaluationLogSettings,
        on_progress: impl FnMut(EvaluationProgress) + Send,
    ) -> Vec<(SmartsGenome, GenomeEvaluation)> {
        self.evaluate_many_with_progress_and_optional_logging(genomes, Some(settings), on_progress)
    }

    fn evaluate_many_with_progress_and_optional_logging(
        &self,
        genomes: Vec<SmartsGenome>,
        settings: Option<&EvaluationLogSettings>,
        mut on_progress: impl FnMut(EvaluationProgress) + Send,
    ) -> Vec<(SmartsGenome, GenomeEvaluation)> {
        let total = genomes.len().max(1);
        if genomes.is_empty() {
            on_progress(EvaluationProgress::new(0, total, None, None));
            return Vec::new();
        }

        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        if genomes.len() >= MIN_PARALLEL_GENOMES {
            let progress_state = Mutex::new((0usize, None::<EvaluatedSmarts>, on_progress));
            return genomes
                .into_par_iter()
                .map(|genome| {
                    let evaluation = self.evaluate_with_logging(&genome, settings);
                    let last = EvaluatedSmarts::new(&genome, evaluation.fitness());
                    let mut progress_state = progress_state.lock().unwrap();
                    progress_state.0 += 1;
                    update_best_evaluated(&mut progress_state.1, &last);
                    let event = EvaluationProgress::new(
                        progress_state.0,
                        total,
                        Some(last),
                        progress_state.1.clone(),
                    );
                    (progress_state.2)(event);
                    (genome, evaluation)
                })
                .collect();
        }

        let mut completed = 0usize;
        let mut best = None::<EvaluatedSmarts>;
        genomes
            .into_iter()
            .map(|genome| {
                let evaluation = self.evaluate_with_logging(&genome, settings);
                let last = EvaluatedSmarts::new(&genome, evaluation.fitness());
                update_best_evaluated(&mut best, &last);
                completed += 1;
                on_progress(EvaluationProgress::new(
                    completed,
                    total,
                    Some(last),
                    best.clone(),
                ));
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

fn log_evaluation_start(genome: &SmartsGenome, settings: &EvaluationLogSettings) {
    debug!(
        target: "smarts_evolution::fitness::evaluator",
        "starting SMARTS evaluation task_id={} generation={} folds={} targets={} complexity={} smarts_len={} smarts={}",
        settings.task_id,
        settings.generation,
        settings.fold_count,
        settings.target_count,
        genome.complexity(),
        genome.smarts().len(),
        genome.smarts(),
    );
}

#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
fn log_slow_evaluation_if_needed(
    genome: &SmartsGenome,
    fitness: ObjectiveFitness,
    settings: &EvaluationLogSettings,
    elapsed: Duration,
) {
    let Some(threshold) = settings.slow_log_threshold else {
        return;
    };
    if elapsed < threshold {
        return;
    }

    warn!(
        target: "smarts_evolution::fitness::evaluator",
        "slow SMARTS evaluation task_id={} generation={} elapsed_ms={} threshold_ms={} folds={} targets={} complexity={} smarts_len={} mcc={:.3} smarts={}",
        settings.task_id,
        settings.generation,
        elapsed.as_millis(),
        threshold.as_millis(),
        settings.fold_count,
        settings.target_count,
        genome.complexity(),
        genome.smarts().len(),
        fitness.mcc(),
        genome.smarts(),
    );
}

fn update_best_evaluated(best: &mut Option<EvaluatedSmarts>, candidate: &EvaluatedSmarts) {
    let should_update = best
        .as_ref()
        .is_none_or(|current| evaluated_is_better(candidate, current));
    if should_update {
        *best = Some(candidate.clone());
    }
}

fn evaluated_is_better(candidate: &EvaluatedSmarts, current: &EvaluatedSmarts) -> bool {
    candidate.mcc() > current.mcc()
        || (candidate.mcc() == current.mcc()
            && (candidate.smarts().len() < current.smarts().len()
                || (candidate.smarts().len() == current.smarts().len()
                    && candidate.smarts() < current.smarts())))
}

fn count_matches_in_fold(
    query: &CompiledQuery,
    screen: &QueryScreen,
    fold: &FoldData,
    behavior_words: &mut [u64],
    bit_index: &mut usize,
) -> ConfusionCounts {
    let fold_start_bit_index = *bit_index;
    *bit_index += fold.len();

    let mut candidates = Vec::new();
    let mut corpus_scratch = TargetCorpusScratch::new();
    fold.index()
        .candidate_ids_with_scratch_into(screen, &mut corpus_scratch, &mut candidates);

    let mut match_scratch = MatchScratch::new();
    let mut true_positives = 0u64;
    let mut false_positives = 0u64;

    for target_id in candidates {
        let sample = &fold.samples()[target_id];
        let matched = query.matches_with_scratch(sample.target(), &mut match_scratch);
        if !matched {
            continue;
        }

        record_behavior_match(behavior_words, fold_start_bit_index + target_id, true);
        if sample.is_positive() {
            true_positives += 1;
        } else {
            false_positives += 1;
        }
    }

    ConfusionCounts::new(
        true_positives,
        false_positives,
        fold.negative_count() as u64 - false_positives,
        fold.positive_count() as u64 - true_positives,
    )
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

        let _: EvaluationSet = FoldData::new(Vec::new());
        let _: LabeledCorpus = FoldData::new(Vec::new());
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
        let screen = QueryScreen::new(compiled.query());
        let mut behavior = vec![0u64; 1];
        let mut bit_index = 0usize;

        let counts =
            count_matches_in_fold(&compiled, &screen, &fold, &mut behavior, &mut bit_index);

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
    fn evaluate_many_with_progress_matches_batch_results_and_reports_scores() {
        let evaluator = SmartsEvaluator::new(vec![FoldData::new(vec![
            FoldSample::positive(prepared("CN")),
            FoldSample::positive(prepared("CCN")),
            FoldSample::negative(prepared("CC")),
            FoldSample::negative(prepared("CO")),
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
        let expected = evaluator.evaluate_many(genomes.clone());
        let mut progress_events = Vec::new();

        let actual = evaluator.evaluate_many_with_progress(genomes, |progress| {
            progress_events.push(progress);
        });

        assert_eq!(actual, expected);
        assert_eq!(progress_events.len(), expected.len());
        let mut completed = progress_events
            .iter()
            .map(EvaluationProgress::completed)
            .collect::<Vec<_>>();
        completed.sort_unstable();
        assert_eq!(completed, (1..=expected.len()).collect::<Vec<_>>());
        assert!(
            progress_events
                .iter()
                .all(|progress| progress.total() == expected.len()
                    && progress.last().is_some()
                    && progress.last_smarts().is_some()
                    && progress.last_mcc().is_some_and(f64::is_finite)
                    && progress.best().is_some()
                    && progress.best_smarts().is_some()
                    && progress.best_mcc().is_some_and(f64::is_finite))
        );
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
