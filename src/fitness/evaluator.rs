use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::time::Duration;

#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
use rayon::prelude::*;
use smarts_rs::{
    CompiledQuery, MatchOutcome, MatchOutcomeLimitResult, MatchScratch, PreparedTarget, QueryMol,
    QueryScreen, TargetCorpusIndex, TargetCorpusScratch,
};
#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
use std::sync::Mutex;
#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
use std::time::Instant;

use super::mcc::{ConfusionCounts, compute_fold_averaged_mcc};
use super::objective::ObjectiveFitness;
use crate::genome::SmartsGenome;
use log::{debug, warn};

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
    match_time_limit: Option<Duration>,
    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    slow_log_threshold: Option<Duration>,
}

impl EvaluationLogSettings {
    pub(crate) fn new(
        task_id: String,
        generation: u64,
        fold_count: usize,
        target_count: usize,
        match_time_limit: Option<Duration>,
        slow_log_threshold: Option<Duration>,
    ) -> Self {
        #[cfg(not(all(feature = "std", not(target_arch = "wasm32"))))]
        let _ = slow_log_threshold;

        Self {
            task_id,
            generation,
            fold_count,
            target_count,
            match_time_limit,
            #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
            slow_log_threshold,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum EvaluationMatchLimit<'a> {
    None,
    Configured {
        max_elapsed: Duration,
        #[cfg(target_arch = "wasm32")]
        now_ms: Option<&'a dyn Fn() -> f64>,
        #[cfg(not(target_arch = "wasm32"))]
        _marker: core::marker::PhantomData<&'a ()>,
    },
}

impl<'a> EvaluationMatchLimit<'a> {
    fn from_settings(
        settings: Option<&EvaluationLogSettings>,
        #[cfg(target_arch = "wasm32")] now_ms: Option<&'a dyn Fn() -> f64>,
    ) -> Self {
        match settings.and_then(|settings| settings.match_time_limit) {
            Some(max_elapsed) => Self::Configured {
                max_elapsed,
                #[cfg(target_arch = "wasm32")]
                now_ms,
                #[cfg(not(target_arch = "wasm32"))]
                _marker: core::marker::PhantomData,
            },
            None => Self::None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvaluationFailure {
    InvalidQuery,
    LimitExceeded,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatedSmarts {
    smarts: String,
    mcc: f64,
    smarts_len: usize,
    coverage_score: f64,
}

impl EvaluatedSmarts {
    fn new(genome: &SmartsGenome, evaluation: &GenomeEvaluation) -> Self {
        Self {
            smarts: genome.smarts().to_string(),
            mcc: evaluation.fitness().mcc(),
            smarts_len: genome.smarts_len(),
            coverage_score: evaluation.coverage_score(),
        }
    }

    pub fn smarts(&self) -> &str {
        &self.smarts
    }

    pub fn mcc(&self) -> f64 {
        self.mcc
    }

    pub fn smarts_len(&self) -> usize {
        self.smarts_len
    }

    pub fn coverage_score(&self) -> f64 {
        self.coverage_score
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

    pub fn last_coverage_score(&self) -> Option<f64> {
        self.last().map(EvaluatedSmarts::coverage_score)
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

    pub fn best_coverage_score(&self) -> Option<f64> {
        self.best().map(EvaluatedSmarts::coverage_score)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ScreeningProxyFitness {
    mcc: f64,
    positive_candidates: u64,
    negative_candidates: u64,
}

impl ScreeningProxyFitness {
    fn new(mcc: f64, positive_candidates: u64, negative_candidates: u64) -> Self {
        Self {
            mcc,
            positive_candidates,
            negative_candidates,
        }
    }

    pub(crate) fn mcc(self) -> f64 {
        self.mcc
    }

    pub(crate) fn positive_candidates(self) -> u64 {
        self.positive_candidates
    }

    pub(crate) fn negative_candidates(self) -> u64 {
        self.negative_candidates
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenomeEvaluation {
    fitness: ObjectiveFitness,
    phenotype: Arc<[u64]>,
    coverage_score: f64,
    limit_exceeded: bool,
}

impl GenomeEvaluation {
    pub fn fitness(&self) -> ObjectiveFitness {
        self.fitness
    }

    pub fn phenotype(&self) -> &Arc<[u64]> {
        &self.phenotype
    }

    pub fn coverage_score(&self) -> f64 {
        self.coverage_score
    }

    pub fn limit_exceeded(&self) -> bool {
        self.limit_exceeded
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FoldMatchSummary {
    counts: ConfusionCounts,
    positive_coverage_sum: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct EvaluationSummary {
    fold_counts: Vec<ConfusionCounts>,
    phenotype: Arc<[u64]>,
    coverage_score: f64,
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
        limit: EvaluationMatchLimit<'_>,
    ) -> Result<EvaluationSummary, EvaluationFailure> {
        let compiled =
            CompiledQuery::new(query.clone()).map_err(|_| EvaluationFailure::InvalidQuery)?;
        let screen = QueryScreen::new(compiled.query());
        self.fold_counts_and_behavior_for_compiled(&compiled, &screen, limit)
    }

    // Keep fold aggregation localized so future evaluator changes stay local.
    fn fold_counts_and_behavior_for_compiled(
        &self,
        compiled: &CompiledQuery,
        screen: &QueryScreen,
        limit: EvaluationMatchLimit<'_>,
    ) -> Result<EvaluationSummary, EvaluationFailure> {
        let total_samples: usize = self.folds.iter().map(FoldData::len).sum();
        let positive_count: usize = self.folds.iter().map(FoldData::positive_count).sum();
        let mut behavior_words = vec![0u64; total_samples.div_ceil(64)];
        let mut bit_index = 0usize;
        let mut fold_counts = Vec::with_capacity(self.folds.len());
        let mut positive_coverage_sum = 0.0;
        for fold in &self.folds {
            let summary = count_matches_in_fold(
                compiled,
                screen,
                fold,
                &mut behavior_words,
                &mut bit_index,
                limit,
            )?;
            fold_counts.push(summary.counts);
            positive_coverage_sum += summary.positive_coverage_sum;
        }
        let coverage_score = if positive_count == 0 {
            0.0
        } else {
            positive_coverage_sum / positive_count as f64
        };
        Ok(EvaluationSummary {
            fold_counts,
            phenotype: Arc::from(behavior_words),
            coverage_score,
        })
    }

    /// Evaluate the MCC objective for a genome.
    pub fn objective_of(&self, genome: &SmartsGenome) -> ObjectiveFitness {
        self.evaluate(genome).fitness()
    }

    pub fn fitness_of(&self, genome: &SmartsGenome) -> ObjectiveFitness {
        self.objective_of(genome)
    }

    pub(crate) fn screening_proxy_of(&self, genome: &SmartsGenome) -> ScreeningProxyFitness {
        let screen = QueryScreen::new(genome.query());
        let mut fold_counts = Vec::with_capacity(self.folds.len());
        let mut positive_candidates = 0u64;
        let mut negative_candidates = 0u64;

        for fold in &self.folds {
            let mut candidates = Vec::new();
            let mut scratch = TargetCorpusScratch::new();
            fold.index()
                .candidate_ids_with_scratch_into(&screen, &mut scratch, &mut candidates);

            let mut true_positives = 0u64;
            let mut false_positives = 0u64;
            for target_id in candidates {
                if fold.samples()[target_id].is_positive() {
                    true_positives += 1;
                } else {
                    false_positives += 1;
                }
            }

            positive_candidates += true_positives;
            negative_candidates += false_positives;
            fold_counts.push(ConfusionCounts::new(
                true_positives,
                false_positives,
                fold.negative_count() as u64 - false_positives,
                fold.positive_count() as u64 - true_positives,
            ));
        }

        ScreeningProxyFitness::new(
            compute_fold_averaged_mcc(&fold_counts),
            positive_candidates,
            negative_candidates,
        )
    }

    pub(crate) fn confusion_for_phenotype(&self, phenotype: &[u64]) -> ConfusionCounts {
        let mut counts = ConfusionCounts::default();
        let mut bit_index = 0usize;
        for fold in &self.folds {
            for sample in fold.samples() {
                counts.record_match(phenotype_bit(phenotype, bit_index), sample.is_positive());
                bit_index += 1;
            }
        }
        counts
    }

    pub fn evaluate(&self, genome: &SmartsGenome) -> GenomeEvaluation {
        self.evaluate_with_limit(genome, EvaluationMatchLimit::None)
    }

    fn evaluate_with_limit(
        &self,
        genome: &SmartsGenome,
        limit: EvaluationMatchLimit<'_>,
    ) -> GenomeEvaluation {
        let summary = match self.fold_counts_and_behavior_for_query(genome.query(), limit) {
            Ok(result) => result,
            Err(failure) => {
                return GenomeEvaluation {
                    fitness: ObjectiveFitness::invalid(),
                    phenotype: Arc::from([]),
                    coverage_score: 0.0,
                    limit_exceeded: failure == EvaluationFailure::LimitExceeded,
                };
            }
        };

        let mcc = compute_fold_averaged_mcc(&summary.fold_counts);
        GenomeEvaluation {
            fitness: ObjectiveFitness::from_mcc(mcc),
            phenotype: summary.phenotype,
            coverage_score: summary.coverage_score,
            limit_exceeded: false,
        }
    }

    fn evaluate_with_logging(
        &self,
        genome: &SmartsGenome,
        settings: Option<&EvaluationLogSettings>,
        #[cfg(target_arch = "wasm32")] now_ms: Option<&dyn Fn() -> f64>,
    ) -> GenomeEvaluation {
        if let Some(settings) = settings {
            log_evaluation_start(genome, settings);
        }

        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        let started = Instant::now();

        let evaluation = self.evaluate_with_limit(
            genome,
            EvaluationMatchLimit::from_settings(
                settings,
                #[cfg(target_arch = "wasm32")]
                now_ms,
            ),
        );

        if evaluation.limit_exceeded()
            && let Some(settings) = settings
        {
            log_limit_exceeded(genome, settings);
        }

        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        if let Some(settings) = settings {
            log_slow_evaluation_if_needed(
                genome,
                evaluation.fitness(),
                evaluation.limit_exceeded(),
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
        self.evaluate_many_with_optional_logging(
            genomes,
            None,
            #[cfg(target_arch = "wasm32")]
            None,
        )
    }

    pub(crate) fn evaluate_many_with_logging(
        &self,
        genomes: Vec<SmartsGenome>,
        settings: &EvaluationLogSettings,
    ) -> Vec<(SmartsGenome, GenomeEvaluation)> {
        self.evaluate_many_with_optional_logging(
            genomes,
            Some(settings),
            #[cfg(target_arch = "wasm32")]
            None,
        )
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn evaluate_many_with_logging_and_clock(
        &self,
        genomes: Vec<SmartsGenome>,
        settings: &EvaluationLogSettings,
        now_ms: &dyn Fn() -> f64,
    ) -> Vec<(SmartsGenome, GenomeEvaluation)> {
        self.evaluate_many_with_optional_logging(genomes, Some(settings), Some(now_ms))
    }

    fn evaluate_many_with_optional_logging(
        &self,
        genomes: Vec<SmartsGenome>,
        settings: Option<&EvaluationLogSettings>,
        #[cfg(target_arch = "wasm32")] now_ms: Option<&dyn Fn() -> f64>,
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
                let evaluation = self.evaluate_with_logging(
                    &genome,
                    settings,
                    #[cfg(target_arch = "wasm32")]
                    now_ms,
                );
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
        self.evaluate_many_with_progress_and_optional_logging(
            genomes,
            None,
            #[cfg(target_arch = "wasm32")]
            None,
            on_progress,
        )
    }

    pub(crate) fn evaluate_many_with_progress_and_logging(
        &self,
        genomes: Vec<SmartsGenome>,
        settings: &EvaluationLogSettings,
        on_progress: impl FnMut(EvaluationProgress) + Send,
    ) -> Vec<(SmartsGenome, GenomeEvaluation)> {
        self.evaluate_many_with_progress_and_optional_logging(
            genomes,
            Some(settings),
            #[cfg(target_arch = "wasm32")]
            None,
            on_progress,
        )
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn evaluate_many_with_progress_logging_and_clock(
        &self,
        genomes: Vec<SmartsGenome>,
        settings: &EvaluationLogSettings,
        now_ms: &dyn Fn() -> f64,
        on_progress: impl FnMut(EvaluationProgress) + Send,
    ) -> Vec<(SmartsGenome, GenomeEvaluation)> {
        self.evaluate_many_with_progress_and_optional_logging(
            genomes,
            Some(settings),
            Some(now_ms),
            on_progress,
        )
    }

    fn evaluate_many_with_progress_and_optional_logging(
        &self,
        genomes: Vec<SmartsGenome>,
        settings: Option<&EvaluationLogSettings>,
        #[cfg(target_arch = "wasm32")] now_ms: Option<&dyn Fn() -> f64>,
        on_progress: impl FnMut(EvaluationProgress) + Send,
    ) -> Vec<(SmartsGenome, GenomeEvaluation)> {
        let total = genomes.len().max(1);
        let mut progress = EvaluationProgressState::new(total, on_progress);
        if genomes.is_empty() {
            progress.emit_empty();
            return Vec::new();
        }

        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        if genomes.len() >= MIN_PARALLEL_GENOMES {
            let progress_state = Mutex::new(progress);
            return genomes
                .into_par_iter()
                .map(|genome| {
                    let evaluation = self.evaluate_with_logging(&genome, settings);
                    match progress_state.lock() {
                        Ok(mut progress) => progress.record(&genome, &evaluation),
                        Err(poisoned) => poisoned.into_inner().record(&genome, &evaluation),
                    }
                    (genome, evaluation)
                })
                .collect();
        }

        genomes
            .into_iter()
            .map(|genome| {
                let evaluation = self.evaluate_with_logging(
                    &genome,
                    settings,
                    #[cfg(target_arch = "wasm32")]
                    now_ms,
                );
                progress.record(&genome, &evaluation);
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

struct EvaluationProgressState<C> {
    completed: usize,
    total: usize,
    best: Option<EvaluatedSmarts>,
    on_progress: C,
}

impl<C> EvaluationProgressState<C>
where
    C: FnMut(EvaluationProgress),
{
    fn new(total: usize, on_progress: C) -> Self {
        Self {
            completed: 0,
            total: total.max(1),
            best: None,
            on_progress,
        }
    }

    fn emit_empty(&mut self) {
        (self.on_progress)(EvaluationProgress::new(0, self.total, None, None));
    }

    fn record(&mut self, genome: &SmartsGenome, evaluation: &GenomeEvaluation) {
        let last = EvaluatedSmarts::new(genome, evaluation);
        update_best_evaluated(&mut self.best, &last);
        self.completed += 1;
        (self.on_progress)(EvaluationProgress::new(
            self.completed,
            self.total,
            Some(last),
            self.best.clone(),
        ));
    }
}

fn log_evaluation_start(genome: &SmartsGenome, settings: &EvaluationLogSettings) {
    debug!(
        target: "smarts_evolution::fitness::evaluator",
        "starting SMARTS evaluation task_id={} generation={} folds={} targets={} smarts_len={} match_time_limit_ms={} smarts={}",
        settings.task_id,
        settings.generation,
        settings.fold_count,
        settings.target_count,
        genome.smarts().len(),
        settings
            .match_time_limit
            .map_or(0, |limit| limit.as_millis()),
        genome.smarts(),
    );
}

fn log_limit_exceeded(genome: &SmartsGenome, settings: &EvaluationLogSettings) {
    warn!(
        target: "smarts_evolution::fitness::evaluator",
        "SMARTS evaluation exceeded match limit task_id={} generation={} folds={} targets={} smarts_len={} match_time_limit_ms={} smarts={}",
        settings.task_id,
        settings.generation,
        settings.fold_count,
        settings.target_count,
        genome.smarts().len(),
        settings
            .match_time_limit
            .map_or(0, |limit| limit.as_millis()),
        genome.smarts(),
    );
}

#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
fn log_slow_evaluation_if_needed(
    genome: &SmartsGenome,
    fitness: ObjectiveFitness,
    limit_exceeded: bool,
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
        "slow SMARTS evaluation task_id={} generation={} elapsed_ms={} threshold_ms={} folds={} targets={} smarts_len={} limit_exceeded={} mcc={:.3} smarts={}",
        settings.task_id,
        settings.generation,
        elapsed.as_millis(),
        threshold.as_millis(),
        settings.fold_count,
        settings.target_count,
        genome.smarts().len(),
        limit_exceeded,
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
            && match candidate
                .coverage_score()
                .total_cmp(&current.coverage_score())
            {
                core::cmp::Ordering::Greater => true,
                core::cmp::Ordering::Equal => candidate.smarts() < current.smarts(),
                core::cmp::Ordering::Less => false,
            })
}

fn count_matches_in_fold(
    query: &CompiledQuery,
    screen: &QueryScreen,
    fold: &FoldData,
    behavior_words: &mut [u64],
    bit_index: &mut usize,
    limit: EvaluationMatchLimit<'_>,
) -> Result<FoldMatchSummary, EvaluationFailure> {
    let fold_start_bit_index = *bit_index;
    *bit_index += fold.len();

    let mut candidates = Vec::new();
    let mut corpus_scratch = TargetCorpusScratch::new();
    fold.index()
        .candidate_ids_with_scratch_into(screen, &mut corpus_scratch, &mut candidates);

    let mut match_scratch = MatchScratch::new();
    let mut true_positives = 0u64;
    let mut false_positives = 0u64;
    let mut positive_coverage_sum = 0.0;

    for target_id in candidates {
        let sample = &fold.samples()[target_id];
        let outcome = match_outcome_with_limit(query, sample.target(), &mut match_scratch, limit)?;
        if !outcome.matched {
            continue;
        }

        record_behavior_match(behavior_words, fold_start_bit_index + target_id, true);
        if sample.is_positive() {
            true_positives += 1;
            positive_coverage_sum += outcome.coverage;
        } else {
            false_positives += 1;
        }
    }

    Ok(FoldMatchSummary {
        counts: ConfusionCounts::new(
            true_positives,
            false_positives,
            fold.negative_count() as u64 - false_positives,
            fold.positive_count() as u64 - true_positives,
        ),
        positive_coverage_sum,
    })
}

fn match_outcome_with_limit(
    query: &CompiledQuery,
    target: &PreparedTarget,
    scratch: &mut MatchScratch,
    limit: EvaluationMatchLimit<'_>,
) -> Result<MatchOutcome, EvaluationFailure> {
    match limit {
        EvaluationMatchLimit::None => Ok(query.match_outcome_with_scratch(target, scratch)),
        EvaluationMatchLimit::Configured {
            max_elapsed,
            #[cfg(target_arch = "wasm32")]
            now_ms,
            #[cfg(not(target_arch = "wasm32"))]
                _marker: _,
        } => match match_outcome_with_configured_limit(
            query,
            target,
            scratch,
            max_elapsed,
            #[cfg(target_arch = "wasm32")]
            now_ms,
        ) {
            MatchOutcomeLimitResult::Complete(outcome) => Ok(outcome),
            MatchOutcomeLimitResult::Exceeded => Err(EvaluationFailure::LimitExceeded),
        },
    }
}

fn match_outcome_with_configured_limit(
    query: &CompiledQuery,
    target: &PreparedTarget,
    scratch: &mut MatchScratch,
    max_elapsed: Duration,
    #[cfg(target_arch = "wasm32")] now_ms: Option<&dyn Fn() -> f64>,
) -> MatchOutcomeLimitResult {
    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    {
        query.match_outcome_with_scratch_and_time_limit(target, scratch, max_elapsed)
    }

    #[cfg(target_arch = "wasm32")]
    {
        if let Some(now_ms) = now_ms {
            let deadline_ms = now_ms() + max_elapsed.as_secs_f64() * 1_000.0;
            return query.match_outcome_with_scratch_and_interrupt(target, scratch, || {
                now_ms() >= deadline_ms
            });
        }
    }

    #[cfg(not(all(feature = "std", not(target_arch = "wasm32"))))]
    {
        let _ = max_elapsed;
        MatchOutcomeLimitResult::Complete(query.match_outcome_with_scratch(target, scratch))
    }
}

fn record_behavior_match(behavior_words: &mut [u64], bit_index: usize, matched: bool) {
    if !matched {
        return;
    }

    let word_index = bit_index / 64;
    let bit_offset = bit_index % 64;
    behavior_words[word_index] |= 1u64 << bit_offset;
}

fn phenotype_bit(phenotype: &[u64], bit_index: usize) -> bool {
    let word_index = bit_index / 64;
    let bit_offset = bit_index % 64;
    phenotype
        .get(word_index)
        .is_some_and(|word| word & (1u64 << bit_offset) != 0)
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

        let summary = count_matches_in_fold(
            &compiled,
            &screen,
            &fold,
            &mut behavior,
            &mut bit_index,
            EvaluationMatchLimit::None,
        );

        assert_eq!(
            summary,
            Ok(FoldMatchSummary {
                counts: ConfusionCounts::new(1, 1, 1, 1),
                positive_coverage_sum: 1.0,
            })
        );
        assert_eq!(behavior[0], 0b0011);
    }

    #[test]
    fn evaluator_scores_positive_match_coverage() {
        let evaluator = SmartsEvaluator::new(vec![FoldData::new(vec![
            sample("CN", true),
            sample("CCN", true),
            sample("CC", false),
        ])]);
        let atom = SmartsGenome::from_smarts("[#7]").unwrap();
        let bond = SmartsGenome::from_smarts("[#6]~[#7]").unwrap();

        let atom_eval = evaluator.evaluate(&atom);
        let bond_eval = evaluator.evaluate(&bond);

        assert_eq!(atom_eval.fitness(), bond_eval.fitness());
        assert!((atom_eval.coverage_score() - (4.0 / 15.0)).abs() < 1e-12);
        assert!((bond_eval.coverage_score() - 0.8).abs() < 1e-12);
        assert!(bond_eval.coverage_score() > atom_eval.coverage_score());
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
    fn screening_proxy_counts_indexed_positive_and_negative_candidates() {
        let evaluator = SmartsEvaluator::new(vec![FoldData::new(vec![
            sample("CN", true),
            sample("CO", true),
            sample("CC", false),
            sample("OO", false),
        ])]);
        let nitrogen = SmartsGenome::from_smarts("[#7]").unwrap();
        let sulfur = SmartsGenome::from_smarts("[#16]").unwrap();

        let nitrogen_proxy = evaluator.screening_proxy_of(&nitrogen);
        let sulfur_proxy = evaluator.screening_proxy_of(&sulfur);

        assert_eq!(nitrogen_proxy.positive_candidates(), 1);
        assert_eq!(nitrogen_proxy.negative_candidates(), 0);
        assert!(nitrogen_proxy.mcc() > 0.0);
        assert_eq!(sulfur_proxy.positive_candidates(), 0);
        assert_eq!(sulfur_proxy.negative_candidates(), 0);
    }

    #[test]
    fn confusion_for_phenotype_reconstructs_label_counts() {
        let evaluator = SmartsEvaluator::new(vec![FoldData::new(vec![
            sample("CN", true),
            sample("CO", true),
            sample("CC", false),
            sample("OO", false),
        ])]);

        assert_eq!(
            evaluator.confusion_for_phenotype(&[0b0101]),
            ConfusionCounts::new(1, 1, 1, 1)
        );
        assert_eq!(
            evaluator.confusion_for_phenotype(&[]),
            ConfusionCounts::new(0, 0, 2, 2)
        );
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
                    && progress.last_coverage_score().is_some_and(f64::is_finite)
                    && progress.best().is_some()
                    && progress.best_smarts().is_some()
                    && progress.best_mcc().is_some_and(f64::is_finite)
                    && progress.best_coverage_score().is_some_and(f64::is_finite))
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
