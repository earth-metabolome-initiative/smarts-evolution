use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;

use hashbrown::{HashMap, HashSet};
use log::{debug, info, warn};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use smarts_rs::BondLabel;

use super::config::EvolutionConfig;
use crate::fitness::evaluator::{
    EvaluatedSmarts, EvaluationLogSettings, FoldData, GenomeEvaluation, ScreeningProxyFitness,
    SmartsEvaluator,
};
use crate::fitness::mcc::{ConfusionCounts, compute_mcc_from_counts};
use crate::fitness::objective::ObjectiveFitness;
use crate::genome::SmartsGenome;
use crate::genome::seed::{SeedCorpus, SmartsGenomeBuilder};
use crate::operators::crossover::SmartsCrossover;
use crate::operators::mutation::{MutationDirection, SmartsMutation};
use elements_rs::AtomicNumber;

const RESET_POOL_SIZE: usize = 32;
const DIVERSE_ELITE_POOL_FACTOR: usize = 4;
const GUIDED_MUTATION_PROPOSAL_COUNT: usize = 4;
const DATASET_SEED_LIMIT: usize = 24;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvolutionError {
    InvalidConfig(String),
    EmptyFolds,
    NoEvaluationTargets(String),
}

impl fmt::Display for EvolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "{message}"),
            Self::EmptyFolds => write!(f, "evolution task requires at least one fold"),
            Self::NoEvaluationTargets(task_id) => {
                write!(f, "task '{task_id}' has no evaluation targets")
            }
        }
    }
}

impl core::error::Error for EvolutionError {}

struct GenerationStats {
    unique_count: usize,
    total_count: usize,
    duplicate_count: usize,
    cache_hits: usize,
    rejection_counts: EvaluationRejectionCounts,
    match_timeout_count: usize,
    lead_smarts_len: usize,
    average_smarts_len: f64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct EvaluationRejectionCounts {
    smarts_length: usize,
}

impl EvaluationRejectionCounts {
    fn record(&mut self, rejection: GenomeEvaluationRejection) {
        match rejection {
            GenomeEvaluationRejection::SmartsLength { .. } => self.smarts_length += 1,
        }
    }

    fn total(self) -> usize {
        self.smarts_length
    }
}

#[derive(Clone)]
struct CachedEvaluation {
    fitness: ObjectiveFitness,
    phenotype: Arc<[u64]>,
    stamp: u64,
}

struct FitnessCache {
    entries: HashMap<Arc<str>, CachedEvaluation>,
    access_order: VecDeque<(Arc<str>, u64)>,
    capacity: usize,
    next_stamp: u64,
}

impl FitnessCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            capacity,
            next_stamp: 0,
        }
    }

    fn get(&mut self, key: &Arc<str>) -> Option<(ObjectiveFitness, Arc<[u64]>)> {
        if self.capacity == 0 {
            return None;
        }

        let stamp = self.bump_stamp();
        let evaluation = {
            let cached = self.entries.get_mut(key)?;
            cached.stamp = stamp;
            (cached.fitness, cached.phenotype.clone())
        };
        self.access_order.push_back((key.clone(), stamp));
        self.compact_if_needed();
        Some(evaluation)
    }

    fn insert(&mut self, key: Arc<str>, fitness: ObjectiveFitness, phenotype: Arc<[u64]>) {
        if self.capacity == 0 {
            return;
        }

        let stamp = self.bump_stamp();
        self.entries.insert(
            key.clone(),
            CachedEvaluation {
                fitness,
                phenotype,
                stamp,
            },
        );
        self.access_order.push_back((key, stamp));
        self.evict_to_capacity();
        self.compact_if_needed();
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn bump_stamp(&mut self) -> u64 {
        self.next_stamp = self.next_stamp.wrapping_add(1).max(1);
        self.next_stamp
    }

    fn evict_to_capacity(&mut self) {
        if self.capacity == 0 {
            self.entries.clear();
            self.access_order.clear();
            return;
        }

        while self.entries.len() > self.capacity {
            let Some((key, stamp)) = self.access_order.pop_front() else {
                break;
            };
            let remove = self
                .entries
                .get(&key)
                .is_some_and(|cached| cached.stamp == stamp);
            if remove {
                self.entries.remove(&key);
            }
        }
    }

    fn compact_if_needed(&mut self) {
        let compact_threshold = self.capacity.max(1).saturating_mul(4).max(1024);
        if self.access_order.len() <= compact_threshold {
            return;
        }

        let mut entries = self
            .entries
            .iter()
            .map(|(key, cached)| (key.clone(), cached.stamp))
            .collect::<Vec<_>>();
        entries.sort_by_key(|(_, stamp)| *stamp);
        self.access_order = entries.into();
    }
}

#[derive(Clone)]
struct ScoredGenome {
    genome: SmartsGenome,
    fitness: ObjectiveFitness,
    phenotype: Arc<[u64]>,
}

struct PreparedPopulationScores {
    scored: Vec<ScoredGenome>,
    uncached: Vec<SmartsGenome>,
    cache_hits: usize,
    rejection_counts: EvaluationRejectionCounts,
}

struct FreshPopulationScores {
    scored: Vec<ScoredGenome>,
    match_timeout_count: usize,
}

/// One generic evolution task.
///
/// Experiment-specific code is expected to prepare folds, labels, and task
/// iteration elsewhere, then call into this library per task.
#[derive(Clone, Debug)]
pub struct EvolutionTask {
    task_id: String,
    folds: Vec<FoldData>,
}

impl EvolutionTask {
    /// Create one evolution task from a task identifier and prepared folds.
    pub fn new(task_id: impl Into<String>, folds: Vec<FoldData>) -> Self {
        Self {
            task_id: task_id.into(),
            folds,
        }
    }

    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    pub fn folds(&self) -> &[FoldData] {
        &self.folds
    }

    /// Evolve this binary task from already-prepared folds.
    ///
    /// This is the main library entry point.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::str::FromStr;
    ///
    /// use smiles_parser::Smiles;
    /// use smarts_evolution::{
    ///     EvolutionConfig, EvolutionTask, FoldData, FoldSample, SeedCorpus,
    /// };
    /// use smarts_rs::PreparedTarget;
    ///
    /// fn prepared(smiles: &str) -> PreparedTarget {
    ///     PreparedTarget::new(Smiles::from_str(smiles).unwrap())
    /// }
    ///
    /// let task = EvolutionTask::new(
    ///     "amide-vs-rest",
    ///     vec![FoldData::new(vec![
    ///         FoldSample::positive(prepared("CC(=O)N")),
    ///         FoldSample::positive(prepared("NC(=O)C")),
    ///         FoldSample::negative(prepared("CCO")),
    ///         FoldSample::negative(prepared("c1ccccc1")),
    ///     ])],
    /// );
    ///
    /// let config = EvolutionConfig::builder()
    ///     .population_size(8)
    ///     .generation_limit(2)
    ///     .stagnation_limit(2)
    ///     .build()
    ///     .unwrap();
    /// let seed_corpus = SeedCorpus::try_from([
    ///     "[#6](=[#8])[#7]",
    ///     "[#6]~[#7]",
    /// ])
    /// .unwrap();
    ///
    /// let result = task.evolve(&config, &seed_corpus).unwrap();
    /// assert!(!result.best_smarts().is_empty());
    /// assert!(result.best_mcc().is_finite());
    /// ```
    pub fn evolve(
        &self,
        config: &EvolutionConfig,
        seed_corpus: &SeedCorpus,
    ) -> Result<TaskResult, EvolutionError> {
        EvolutionSession::new(self, config, seed_corpus, 1)?.evolve()
    }

    /// Evolve this task while emitting one snapshot after each scored generation.
    pub fn evolve_with_progress(
        &self,
        config: &EvolutionConfig,
        seed_corpus: &SeedCorpus,
        leaderboard_size: usize,
        on_progress: impl FnMut(EvolutionProgress),
    ) -> Result<TaskResult, EvolutionError> {
        EvolutionSession::new(self, config, seed_corpus, leaderboard_size)?
            .evolve_with_progress(on_progress)
    }

    /// Evolve this task while reporting full lifecycle and progress events.
    pub fn evolve_with_observer(
        &self,
        config: &EvolutionConfig,
        seed_corpus: &SeedCorpus,
        leaderboard_size: usize,
        mut observer: impl EvolutionProgressObserver + Send,
    ) -> Result<TaskResult, EvolutionError> {
        match EvolutionSession::new(self, config, seed_corpus, leaderboard_size) {
            Ok(session) => session.evolve_with_observer(observer),
            Err(error) => {
                observer.on_start(self.task_id(), config.generation_limit());
                observer.on_error(&error);
                Err(error)
            }
        }
    }

    /// Evolve this task by moving its folds directly into the session.
    pub fn evolve_owned(
        self,
        config: &EvolutionConfig,
        seed_corpus: &SeedCorpus,
    ) -> Result<TaskResult, EvolutionError> {
        EvolutionSession::from_owned_task(self, config, seed_corpus, 1)?.evolve()
    }

    /// Evolve this task by moving its folds into the session while reporting
    /// generation snapshots.
    pub fn evolve_owned_with_progress(
        self,
        config: &EvolutionConfig,
        seed_corpus: &SeedCorpus,
        leaderboard_size: usize,
        on_progress: impl FnMut(EvolutionProgress),
    ) -> Result<TaskResult, EvolutionError> {
        EvolutionSession::from_owned_task(self, config, seed_corpus, leaderboard_size)?
            .evolve_with_progress(on_progress)
    }

    /// Evolve this task by moving its folds into the session while reporting
    /// full lifecycle and progress events.
    pub fn evolve_owned_with_observer(
        self,
        config: &EvolutionConfig,
        seed_corpus: &SeedCorpus,
        leaderboard_size: usize,
        mut observer: impl EvolutionProgressObserver + Send,
    ) -> Result<TaskResult, EvolutionError> {
        let task_id = self.task_id().to_string();
        match EvolutionSession::from_owned_task(self, config, seed_corpus, leaderboard_size) {
            Ok(session) => session.evolve_with_observer(observer),
            Err(error) => {
                observer.on_start(&task_id, config.generation_limit());
                observer.on_error(&error);
                Err(error)
            }
        }
    }
}

/// Result of evolving a single task.
#[derive(Clone, Debug)]
pub struct TaskResult {
    task_id: String,
    best_smarts: String,
    best_mcc: f64,
    best_smarts_len: usize,
    generations: u64,
    leaders: Vec<RankedSmarts>,
}

impl TaskResult {
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    pub fn best_smarts(&self) -> &str {
        &self.best_smarts
    }

    pub fn best_mcc(&self) -> f64 {
        self.best_mcc
    }

    pub fn best_smarts_len(&self) -> usize {
        self.best_smarts_len
    }

    pub fn generations(&self) -> u64 {
        self.generations
    }

    pub fn leaders(&self) -> &[RankedSmarts] {
        &self.leaders
    }
}

/// One ranked SMARTS candidate and its current performance.
#[derive(Clone, Debug, PartialEq)]
pub struct RankedSmarts {
    smarts: String,
    mcc: f64,
    smarts_len: usize,
}

impl RankedSmarts {
    fn new(genome: &SmartsGenome, fitness: ObjectiveFitness) -> Self {
        Self::from_parts(
            genome.smarts().to_string(),
            fitness.mcc(),
            genome.smarts_len(),
        )
    }

    fn from_parts(smarts: String, mcc: f64, smarts_len: usize) -> Self {
        Self {
            smarts,
            mcc,
            smarts_len,
        }
    }

    fn from_evaluated(evaluated: &EvaluatedSmarts) -> Self {
        Self::from_parts(
            evaluated.smarts().to_string(),
            evaluated.mcc(),
            evaluated.smarts_len(),
        )
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
}

/// High-level status for one evolution snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvolutionStatus {
    Running,
    Stagnated,
    Completed,
}

/// One per-generation snapshot of the search state.
#[derive(Clone, Debug, PartialEq)]
pub struct EvolutionProgress {
    task_id: String,
    generation: u64,
    generation_limit: u64,
    status: EvolutionStatus,
    best: RankedSmarts,
    best_so_far: RankedSmarts,
    leaders: Vec<RankedSmarts>,
    unique_count: usize,
    total_count: usize,
    duplicate_count: usize,
    cache_hits: usize,
    smarts_length_rejection_count: usize,
    match_timeout_count: usize,
    lead_smarts_len: usize,
    average_smarts_len: f64,
    stagnation: u64,
}

/// Progress while the current generation is being scored.
#[derive(Clone, Debug, PartialEq)]
pub struct EvolutionEvaluationProgress {
    task_id: String,
    generation: u64,
    generation_limit: u64,
    completed: usize,
    total: usize,
    last: Option<RankedSmarts>,
    generation_best: Option<RankedSmarts>,
    incumbent_best: RankedSmarts,
}

/// Observer for evolution progress without imposing a rendering backend.
pub trait EvolutionProgressObserver {
    fn on_start(&mut self, _task_id: &str, _generation_limit: u64) {}

    fn on_evaluation(&mut self, _progress: &EvolutionEvaluationProgress) {}

    fn on_generation(&mut self, _progress: &EvolutionProgress) {}

    fn on_finish(&mut self, _result: &TaskResult) {}

    fn on_error(&mut self, _error: &EvolutionError) {}
}

impl EvolutionProgressObserver for () {}

struct ProgressSnapshot<'a> {
    task_id: &'a str,
    generation: u64,
    generation_limit: u64,
    status: EvolutionStatus,
    best: RankedSmarts,
    best_so_far: RankedSmarts,
    leaders: Vec<RankedSmarts>,
    stats: &'a GenerationStats,
    stagnation: u64,
}

struct EvaluationProgressSnapshot<'a> {
    task_id: &'a str,
    generation: u64,
    generation_limit: u64,
    completed: usize,
    total: usize,
    last: Option<RankedSmarts>,
    generation_best: Option<RankedSmarts>,
    incumbent_best: RankedSmarts,
}

impl EvolutionEvaluationProgress {
    fn from_snapshot(snapshot: EvaluationProgressSnapshot<'_>) -> Self {
        Self {
            task_id: snapshot.task_id.to_string(),
            generation: snapshot.generation,
            generation_limit: snapshot.generation_limit,
            completed: snapshot.completed,
            total: snapshot.total.max(1),
            last: snapshot.last,
            generation_best: snapshot.generation_best,
            incumbent_best: snapshot.incumbent_best,
        }
    }

    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn generation_limit(&self) -> u64 {
        self.generation_limit
    }

    pub fn completed(&self) -> usize {
        self.completed
    }

    pub fn total(&self) -> usize {
        self.total
    }

    /// Last evaluated SMARTS. With parallel scoring this follows completion
    /// order, not input order.
    pub fn last(&self) -> Option<&RankedSmarts> {
        self.last.as_ref()
    }

    pub fn last_smarts(&self) -> Option<&str> {
        self.last().map(RankedSmarts::smarts)
    }

    pub fn last_mcc(&self) -> Option<f64> {
        self.last().map(RankedSmarts::mcc)
    }

    pub fn generation_best(&self) -> Option<&RankedSmarts> {
        self.generation_best.as_ref()
    }

    pub fn generation_best_smarts(&self) -> Option<&str> {
        self.generation_best().map(RankedSmarts::smarts)
    }

    pub fn generation_best_mcc(&self) -> Option<f64> {
        self.generation_best().map(RankedSmarts::mcc)
    }

    pub fn incumbent_best(&self) -> &RankedSmarts {
        &self.incumbent_best
    }

    pub fn incumbent_best_smarts(&self) -> &str {
        self.incumbent_best.smarts()
    }

    pub fn incumbent_best_mcc(&self) -> f64 {
        self.incumbent_best.mcc()
    }
}

impl EvolutionProgress {
    fn from_snapshot(snapshot: ProgressSnapshot<'_>) -> Self {
        Self {
            task_id: snapshot.task_id.to_string(),
            generation: snapshot.generation,
            generation_limit: snapshot.generation_limit,
            status: snapshot.status,
            best: snapshot.best,
            best_so_far: snapshot.best_so_far,
            leaders: snapshot.leaders,
            unique_count: snapshot.stats.unique_count,
            total_count: snapshot.stats.total_count,
            duplicate_count: snapshot.stats.duplicate_count,
            cache_hits: snapshot.stats.cache_hits,
            smarts_length_rejection_count: snapshot.stats.rejection_counts.smarts_length,
            match_timeout_count: snapshot.stats.match_timeout_count,
            lead_smarts_len: snapshot.stats.lead_smarts_len,
            average_smarts_len: snapshot.stats.average_smarts_len,
            stagnation: snapshot.stagnation,
        }
    }

    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn generation_limit(&self) -> u64 {
        self.generation_limit
    }

    pub fn status(&self) -> EvolutionStatus {
        self.status
    }

    pub fn best(&self) -> &RankedSmarts {
        &self.best
    }

    pub fn best_so_far(&self) -> &RankedSmarts {
        &self.best_so_far
    }

    pub fn leaders(&self) -> &[RankedSmarts] {
        &self.leaders
    }

    pub fn unique_count(&self) -> usize {
        self.unique_count
    }

    pub fn total_count(&self) -> usize {
        self.total_count
    }

    pub fn duplicate_count(&self) -> usize {
        self.duplicate_count
    }

    pub fn cache_hits(&self) -> usize {
        self.cache_hits
    }

    pub fn smarts_length_rejection_count(&self) -> usize {
        self.smarts_length_rejection_count
    }

    pub fn evaluation_rejection_count(&self) -> usize {
        self.smarts_length_rejection_count
    }

    pub fn match_timeout_count(&self) -> usize {
        self.match_timeout_count
    }

    pub fn lead_smarts_len(&self) -> usize {
        self.lead_smarts_len
    }

    pub fn average_smarts_len(&self) -> f64 {
        self.average_smarts_len
    }

    pub fn stagnation(&self) -> u64 {
        self.stagnation
    }
}

/// One incremental evolution session that can be paused and resumed between
/// generations.
pub struct EvolutionSession {
    task_id: String,
    config: EvolutionConfig,
    evaluator: SmartsEvaluator,
    genome_builder: SmartsGenomeBuilder,
    population: Vec<SmartsGenome>,
    crossover: SmartsCrossover,
    mutator: SmartsMutation,
    rng: SmallRng,
    best_fitness: ObjectiveFitness,
    best_smarts: String,
    best_smarts_len: usize,
    stagnation: u64,
    fitness_cache: FitnessCache,
    last_leaders: Vec<RankedSmarts>,
    generation: u64,
    leaderboard_size: usize,
    finished_result: Option<TaskResult>,
}

impl EvolutionSession {
    /// Build one resumable evolution session for a task.
    pub fn new(
        task: &EvolutionTask,
        config: &EvolutionConfig,
        seed_corpus: &SeedCorpus,
        leaderboard_size: usize,
    ) -> Result<Self, EvolutionError> {
        Self::from_task_parts(
            task.task_id().to_string(),
            task.folds().to_vec(),
            config,
            seed_corpus,
            leaderboard_size,
        )
    }

    /// Build one resumable evolution session by moving a task's folds into the evaluator.
    pub fn from_owned_task(
        task: EvolutionTask,
        config: &EvolutionConfig,
        seed_corpus: &SeedCorpus,
        leaderboard_size: usize,
    ) -> Result<Self, EvolutionError> {
        let EvolutionTask { task_id, folds } = task;
        Self::from_task_parts(task_id, folds, config, seed_corpus, leaderboard_size)
    }

    fn from_task_parts(
        task_id: String,
        folds: Vec<FoldData>,
        config: &EvolutionConfig,
        seed_corpus: &SeedCorpus,
        leaderboard_size: usize,
    ) -> Result<Self, EvolutionError> {
        config.validate().map_err(EvolutionError::InvalidConfig)?;
        if folds.is_empty() {
            return Err(EvolutionError::EmptyFolds);
        }
        let total_targets: usize = folds.iter().map(FoldData::len).sum();
        if total_targets == 0 {
            return Err(EvolutionError::NoEvaluationTargets(task_id));
        }
        let fold_count = folds.len();

        info!(
            "Evolving task '{}' over {} folds ({} evaluation targets)",
            task_id, fold_count, total_targets
        );

        let task_seed_corpus = task_seed_corpus(seed_corpus, &folds);
        let evaluator = SmartsEvaluator::new(folds);
        let genome_builder = SmartsGenomeBuilder::new(task_seed_corpus.clone());
        let mut rng = build_rng(config);
        let population: Vec<SmartsGenome> = (0..config.population_size())
            .map(|i| genome_builder.build_genome(i, &mut rng))
            .collect();
        info!(
            "Built initial population for task '{}' ({} genomes)",
            task_id,
            population.len(),
        );

        let crossover = SmartsCrossover::new(config.crossover_rate());
        let reset_pool = build_reset_pool(&task_seed_corpus, &population, RESET_POOL_SIZE);
        let mutator = SmartsMutation::with_reset_pool(config.mutation_rate(), reset_pool);

        Ok(Self {
            task_id,
            config: config.clone(),
            evaluator,
            genome_builder,
            population,
            crossover,
            mutator,
            rng,
            best_fitness: ObjectiveFitness::invalid(),
            best_smarts: String::from("[#6]"),
            best_smarts_len: 0,
            stagnation: 0,
            fitness_cache: FitnessCache::new(config.fitness_cache_capacity()),
            last_leaders: Vec::new(),
            generation: 0,
            leaderboard_size: leaderboard_size.max(1),
            finished_result: None,
        })
    }

    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    pub fn generation_limit(&self) -> u64 {
        self.config.generation_limit()
    }

    /// Drive this session to completion without reporting generation snapshots.
    pub fn evolve(self) -> Result<TaskResult, EvolutionError> {
        self.evolve_with_progress(|_| {})
    }

    /// Drive this session to completion while reporting generation snapshots.
    pub fn evolve_with_progress(
        mut self,
        mut on_progress: impl FnMut(EvolutionProgress),
    ) -> Result<TaskResult, EvolutionError> {
        while let Some(progress) = self.step() {
            on_progress(progress);
            if self.is_finished() {
                if let Some(result) = self.take_result() {
                    return Ok(result);
                }
                return Err(EvolutionError::InvalidConfig(
                    "finished evolution session did not expose a terminal result".into(),
                ));
            }
        }

        self.take_result().ok_or_else(|| {
            EvolutionError::InvalidConfig("evolution session ended unexpectedly".into())
        })
    }

    /// Drive this session to completion while reporting lifecycle and progress events.
    pub fn evolve_with_observer(
        mut self,
        mut observer: impl EvolutionProgressObserver + Send,
    ) -> Result<TaskResult, EvolutionError> {
        observer.on_start(self.task_id(), self.generation_limit());
        while let Some(progress) =
            self.step_with_evaluation_progress(|evaluation| observer.on_evaluation(&evaluation))
        {
            observer.on_generation(&progress);
            if self.is_finished() {
                if let Some(result) = self.take_result() {
                    observer.on_finish(&result);
                    return Ok(result);
                }
                let error = EvolutionError::InvalidConfig(
                    "finished evolution session did not expose a terminal result".into(),
                );
                observer.on_error(&error);
                return Err(error);
            }
        }

        match self.take_result() {
            Some(result) => {
                observer.on_finish(&result);
                Ok(result)
            }
            None => {
                let error =
                    EvolutionError::InvalidConfig("evolution session ended unexpectedly".into());
                observer.on_error(&error);
                Err(error)
            }
        }
    }

    /// Advance the search by one scored generation.
    pub fn step(&mut self) -> Option<EvolutionProgress> {
        self.step_internal(
            None,
            #[cfg(target_arch = "wasm32")]
            None,
        )
    }

    /// Advance the search by one generation and report scoring progress.
    pub fn step_with_evaluation_progress(
        &mut self,
        mut on_evaluation_progress: impl FnMut(EvolutionEvaluationProgress) + Send,
    ) -> Option<EvolutionProgress> {
        self.step_internal(
            Some(&mut on_evaluation_progress),
            #[cfg(target_arch = "wasm32")]
            None,
        )
    }

    #[cfg(target_arch = "wasm32")]
    pub fn step_with_evaluation_progress_and_clock(
        &mut self,
        now_ms: &dyn Fn() -> f64,
        mut on_evaluation_progress: impl FnMut(EvolutionEvaluationProgress) + Send,
    ) -> Option<EvolutionProgress> {
        self.step_internal(Some(&mut on_evaluation_progress), Some(now_ms))
    }

    fn step_internal(
        &mut self,
        mut on_evaluation_progress: Option<&mut (dyn FnMut(EvolutionEvaluationProgress) + Send)>,
        #[cfg(target_arch = "wasm32")] now_ms: Option<&dyn Fn() -> f64>,
    ) -> Option<EvolutionProgress> {
        if self.finished_result.is_some() || self.generation >= self.config.generation_limit() {
            return None;
        }

        let generation = self.generation;
        let generation_number = generation + 1;
        let generation_limit = self.config.generation_limit();
        let total_count = self.population.len();
        let evaluation_total = total_count.max(1);
        let average_smarts_len = average_smarts_len(&self.population);
        let mut evaluation_progress = EvaluationProgressTracker::new(
            self.task_id.clone(),
            generation_number,
            generation_limit,
            evaluation_total,
            self.incumbent_best(),
        );

        evaluation_progress.emit(None, &mut on_evaluation_progress);

        let (unique_population, duplicate_count) =
            self.drain_unique_population(&mut evaluation_progress, &mut on_evaluation_progress);
        let unique_count = unique_population.len();
        let prepared_scores = self.score_cached_or_rejected_population(
            unique_population,
            generation_number,
            &mut evaluation_progress,
            &mut on_evaluation_progress,
        );
        let fresh_scores = self.score_uncached_population(
            prepared_scores.uncached,
            generation_number,
            &mut evaluation_progress,
            &mut on_evaluation_progress,
            #[cfg(target_arch = "wasm32")]
            now_ms,
        );
        let mut unique_scored = prepared_scores.scored;
        unique_scored.extend(fresh_scores.scored);

        unique_scored = phenotypically_deduplicate(unique_scored);
        unique_scored.sort_by(compare_scored_genomes);
        let leaders = leaderboard_from_scored(&unique_scored, self.leaderboard_size);

        let lead = &unique_scored[0];
        let lead_smarts_len = lead.genome.smarts_len();
        self.update_incumbent_from_lead(generation, lead);

        let stats = GenerationStats {
            unique_count,
            total_count,
            duplicate_count,
            cache_hits: prepared_scores.cache_hits,
            rejection_counts: prepared_scores.rejection_counts,
            match_timeout_count: fresh_scores.match_timeout_count,
            lead_smarts_len,
            average_smarts_len,
        };

        debug!(
            "Task '{}' gen {}: {}",
            self.task_id,
            generation + 1,
            generation_progress_message(&stats, lead.fitness, self.best_fitness,),
        );

        let generations = generation_number;
        let status = if self.stagnation >= self.config.stagnation_limit() {
            EvolutionStatus::Stagnated
        } else if generations == generation_limit {
            EvolutionStatus::Completed
        } else {
            EvolutionStatus::Running
        };

        self.last_leaders = leaders.clone();
        let progress = EvolutionProgress::from_snapshot(ProgressSnapshot {
            task_id: &self.task_id,
            generation: generations,
            generation_limit,
            status,
            best: RankedSmarts::new(&lead.genome, lead.fitness),
            best_so_far: RankedSmarts {
                smarts: self.best_smarts.clone(),
                mcc: self.best_fitness.mcc(),
                smarts_len: self.best_smarts_len,
            },
            leaders,
            stats: &stats,
            stagnation: self.stagnation,
        });

        self.generation = generations;

        if status == EvolutionStatus::Stagnated {
            info!(
                "Task '{}' stagnated after {} generations; best MCC={:.3}",
                self.task_id,
                generations,
                self.best_fitness.mcc(),
            );
            self.finished_result = Some(build_task_result(
                &self.task_id,
                self.best_smarts.clone(),
                self.best_fitness,
                self.best_smarts_len,
                generations,
                self.last_leaders.clone(),
            ));
            return Some(progress);
        }

        if status == EvolutionStatus::Completed {
            info!(
                "Task '{}' completed after {} generations; best MCC={:.3}",
                self.task_id,
                generations,
                self.best_fitness.mcc(),
            );
            self.finished_result = Some(build_task_result(
                &self.task_id,
                self.best_smarts.clone(),
                self.best_fitness,
                self.best_smarts_len,
                generations,
                self.last_leaders.clone(),
            ));
            return Some(progress);
        }

        self.repopulate_from_scored(&unique_scored, generation);

        Some(progress)
    }

    fn drain_unique_population(
        &mut self,
        evaluation_progress: &mut EvaluationProgressTracker,
        on_evaluation_progress: &mut Option<&mut (dyn FnMut(EvolutionEvaluationProgress) + Send)>,
    ) -> (Vec<SmartsGenome>, usize) {
        let mut seen_population = HashSet::with_capacity(self.population.len());
        let mut unique_population = Vec::with_capacity(self.population.len());
        let mut duplicate_count = 0usize;

        for genome in self.population.drain(..) {
            let smarts = genome.smarts_shared().clone();
            if seen_population.insert(smarts) {
                unique_population.push(genome);
            } else {
                duplicate_count += 1;
                evaluation_progress.record_completed(None, on_evaluation_progress);
            }
        }

        (unique_population, duplicate_count)
    }

    fn score_cached_or_rejected_population(
        &mut self,
        unique_population: Vec<SmartsGenome>,
        generation: u64,
        evaluation_progress: &mut EvaluationProgressTracker,
        on_evaluation_progress: &mut Option<&mut (dyn FnMut(EvolutionEvaluationProgress) + Send)>,
    ) -> PreparedPopulationScores {
        let mut unique_scored = Vec::with_capacity(unique_population.len());
        let mut uncached_genomes = Vec::new();
        let mut cache_hits = 0usize;
        let mut rejection_counts = EvaluationRejectionCounts::default();

        for genome in unique_population {
            let smarts = genome.smarts_shared().clone();
            if let Some(rejection) = genome_evaluation_rejection(&self.config, &genome) {
                rejection_counts.record(rejection);
                log_rejected_genome(&self.task_id, generation, &genome, rejection);
                let ranked = RankedSmarts::new(&genome, ObjectiveFitness::invalid());
                unique_scored.push(build_rejected_scored_genome(genome));
                evaluation_progress.record_completed(Some(ranked), on_evaluation_progress);
            } else if let Some((fitness, phenotype)) = self.fitness_cache.get(&smarts) {
                cache_hits += 1;
                let ranked = RankedSmarts::new(&genome, fitness);
                unique_scored.push(ScoredGenome {
                    genome,
                    fitness,
                    phenotype,
                });
                evaluation_progress.record_completed(Some(ranked), on_evaluation_progress);
            } else {
                uncached_genomes.push(genome);
            }
        }

        PreparedPopulationScores {
            scored: unique_scored,
            uncached: uncached_genomes,
            cache_hits,
            rejection_counts,
        }
    }

    fn score_uncached_population(
        &mut self,
        uncached_genomes: Vec<SmartsGenome>,
        generation: u64,
        evaluation_progress: &mut EvaluationProgressTracker,
        on_evaluation_progress: &mut Option<&mut (dyn FnMut(EvolutionEvaluationProgress) + Send)>,
        #[cfg(target_arch = "wasm32")] now_ms: Option<&dyn Fn() -> f64>,
    ) -> FreshPopulationScores {
        let log_settings = EvaluationLogSettings::new(
            self.task_id.clone(),
            generation,
            self.evaluator.fold_count(),
            self.evaluator.target_count(),
            self.config.match_time_limit(),
            self.config.slow_evaluation_log_threshold(),
        );

        let freshly_scored = if on_evaluation_progress.is_some() {
            #[cfg(target_arch = "wasm32")]
            let freshly_scored = if let Some(now_ms) = now_ms {
                self.evaluator
                    .evaluate_many_with_progress_logging_and_clock(
                        uncached_genomes,
                        &log_settings,
                        now_ms,
                        |progress| {
                            let last = progress.last().map(RankedSmarts::from_evaluated);
                            evaluation_progress.record_completed(last, on_evaluation_progress);
                        },
                    )
            } else {
                self.evaluator.evaluate_many_with_progress_and_logging(
                    uncached_genomes,
                    &log_settings,
                    |progress| {
                        let last = progress.last().map(RankedSmarts::from_evaluated);
                        evaluation_progress.record_completed(last, on_evaluation_progress);
                    },
                )
            };

            #[cfg(target_arch = "wasm32")]
            {
                freshly_scored
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                self.evaluator.evaluate_many_with_progress_and_logging(
                    uncached_genomes,
                    &log_settings,
                    |progress| {
                        let last = progress.last().map(RankedSmarts::from_evaluated);
                        evaluation_progress.record_completed(last, on_evaluation_progress);
                    },
                )
            }
        } else {
            #[cfg(target_arch = "wasm32")]
            let freshly_scored = if let Some(now_ms) = now_ms {
                self.evaluator.evaluate_many_with_logging_and_clock(
                    uncached_genomes,
                    &log_settings,
                    now_ms,
                )
            } else {
                self.evaluator
                    .evaluate_many_with_logging(uncached_genomes, &log_settings)
            };

            #[cfg(target_arch = "wasm32")]
            {
                freshly_scored
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                self.evaluator
                    .evaluate_many_with_logging(uncached_genomes, &log_settings)
            }
        };

        let match_timeout_count = freshly_scored
            .iter()
            .filter(|(_, evaluation)| evaluation.limit_exceeded())
            .count();
        let scored = freshly_scored
            .into_iter()
            .map(|(genome, evaluation)| {
                build_scored_genome(&mut self.fitness_cache, genome, evaluation)
            })
            .collect();

        FreshPopulationScores {
            scored,
            match_timeout_count,
        }
    }

    fn update_incumbent_from_lead(&mut self, generation: u64, lead: &ScoredGenome) {
        if generation == 0
            || is_better_candidate(
                lead.fitness,
                &lead.genome,
                self.best_fitness,
                self.best_smarts_len,
                &self.best_smarts,
            )
        {
            self.best_fitness = lead.fitness;
            self.best_smarts = lead.genome.smarts().to_string();
            self.best_smarts_len = lead.genome.smarts_len();
            self.stagnation = 0;
        } else {
            self.stagnation += 1;
        }
    }

    fn repopulate_from_scored(&mut self, unique_scored: &[ScoredGenome], generation: u64) {
        let num_parents = ((self.config.population_size() as f64 * self.config.selection_ratio())
            .round() as usize)
            .max(2)
            .min(self.config.population_size().max(2));
        let parents = tournament_select(
            unique_scored,
            num_parents,
            self.config.tournament_size(),
            &mut self.rng,
        );

        let mut offspring = Vec::with_capacity(self.config.population_size());
        let mut pi = 0;
        while offspring.len() < self.config.population_size() {
            let p1 = &parents[pi % parents.len()];
            let p2 = &parents[(pi + 1) % parents.len()];
            pi += 2;

            let p1_direction = self.mutation_direction_for_parent(p1);
            let p2_direction = self.mutation_direction_for_parent(p2);
            let (child_a, child_b) =
                self.crossover
                    .crossover_pair(&p1.genome, &p2.genome, &mut self.rng);
            let child_a = self.mutate_offspring(child_a, p1_direction);
            offspring.push(child_a);
            if offspring.len() >= self.config.population_size() {
                break;
            }
            let child_b = self.mutate_offspring(child_b, p2_direction);
            offspring.push(child_b);
        }

        let elite_count = self
            .config
            .elite_count()
            .max(1)
            .min(self.config.population_size());
        let immigrant_count = ((self.config.random_immigrant_ratio()
            * self.config.population_size() as f64)
            .round() as usize)
            .min(self.config.population_size().saturating_sub(elite_count));
        let immigrant_base = generation as usize * self.config.population_size();
        let mut immigrant_idx = 0usize;
        self.population = reinsert_population(
            unique_scored,
            offspring,
            self.config.population_size(),
            elite_count,
            immigrant_count,
            || {
                let genome = self
                    .genome_builder
                    .build_genome(immigrant_base + immigrant_idx, &mut self.rng);
                immigrant_idx += 1;
                genome
            },
        );
    }

    fn mutate_offspring(
        &mut self,
        genome: SmartsGenome,
        direction: MutationDirection,
    ) -> SmartsGenome {
        let evaluator = &self.evaluator;
        self.mutator.mutate_guided(
            genome,
            GUIDED_MUTATION_PROPOSAL_COUNT,
            direction,
            &mut self.rng,
            |_, candidates| select_screened_mutation_candidate(evaluator, candidates),
        )
    }

    fn mutation_direction_for_parent(&self, parent: &ScoredGenome) -> MutationDirection {
        mutation_direction_from_counts(
            self.evaluator
                .confusion_for_phenotype(parent.phenotype.as_ref()),
        )
    }

    /// Returns true once the session has reached a terminal state.
    pub fn is_finished(&self) -> bool {
        self.finished_result.is_some()
    }

    /// Takes the terminal task result once the session has finished.
    pub fn take_result(&mut self) -> Option<TaskResult> {
        self.finished_result.take()
    }

    fn incumbent_best(&self) -> RankedSmarts {
        RankedSmarts::from_parts(
            self.best_smarts.clone(),
            self.best_fitness.mcc(),
            self.best_smarts_len,
        )
    }
}

struct EvaluationProgressTracker {
    task_id: String,
    generation: u64,
    generation_limit: u64,
    completed: usize,
    total: usize,
    generation_best: Option<RankedSmarts>,
    incumbent_best: RankedSmarts,
}

impl EvaluationProgressTracker {
    fn new(
        task_id: String,
        generation: u64,
        generation_limit: u64,
        total: usize,
        incumbent_best: RankedSmarts,
    ) -> Self {
        Self {
            task_id,
            generation,
            generation_limit,
            completed: 0,
            total: total.max(1),
            generation_best: None,
            incumbent_best,
        }
    }

    fn record_completed(
        &mut self,
        last: Option<RankedSmarts>,
        on_evaluation_progress: &mut Option<&mut (dyn FnMut(EvolutionEvaluationProgress) + Send)>,
    ) {
        self.completed += 1;
        if let Some(candidate) = last.as_ref() {
            update_ranked_best(&mut self.generation_best, candidate);
        }
        self.emit(last, on_evaluation_progress);
    }

    fn emit(
        &self,
        last: Option<RankedSmarts>,
        on_evaluation_progress: &mut Option<&mut (dyn FnMut(EvolutionEvaluationProgress) + Send)>,
    ) {
        if let Some(callback) = on_evaluation_progress.as_deref_mut() {
            callback(EvolutionEvaluationProgress::from_snapshot(
                EvaluationProgressSnapshot {
                    task_id: &self.task_id,
                    generation: self.generation,
                    generation_limit: self.generation_limit,
                    completed: self.completed,
                    total: self.total,
                    last,
                    generation_best: self.generation_best.clone(),
                    incumbent_best: self.incumbent_best.clone(),
                },
            ));
        }
    }
}

fn update_ranked_best(best: &mut Option<RankedSmarts>, candidate: &RankedSmarts) {
    let should_update = best
        .as_ref()
        .is_none_or(|current| ranked_is_better(candidate, current));
    if should_update {
        *best = Some(candidate.clone());
    }
}

fn ranked_is_better(candidate: &RankedSmarts, current: &RankedSmarts) -> bool {
    candidate.mcc() > current.mcc()
        || (candidate.mcc() == current.mcc()
            && (candidate.smarts_len() < current.smarts_len()
                || (candidate.smarts_len() == current.smarts_len()
                    && candidate.smarts() < current.smarts())))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GenomeEvaluationRejection {
    SmartsLength { actual: usize, max: usize },
}

fn genome_evaluation_rejection(
    config: &EvolutionConfig,
    genome: &SmartsGenome,
) -> Option<GenomeEvaluationRejection> {
    let smarts_len = genome.smarts_len();
    config
        .max_evaluation_smarts_len()
        .filter(|&max_len| smarts_len > max_len)
        .map(|max_len| GenomeEvaluationRejection::SmartsLength {
            actual: smarts_len,
            max: max_len,
        })
}

fn log_rejected_genome(
    task_id: &str,
    generation: u64,
    genome: &SmartsGenome,
    rejection: GenomeEvaluationRejection,
) {
    match rejection {
        GenomeEvaluationRejection::SmartsLength { actual, max } => {
            warn!(
                target: "smarts_evolution::evolution::runner",
                "rejecting SMARTS before evaluation task_id={} generation={} reason=smarts_len actual={} max={} smarts={}",
                task_id,
                generation,
                actual,
                max,
                genome.smarts(),
            );
        }
    }
}

fn build_task_result(
    task_id: &str,
    best_smarts: String,
    best_fitness: ObjectiveFitness,
    best_smarts_len: usize,
    generations: u64,
    leaders: Vec<RankedSmarts>,
) -> TaskResult {
    TaskResult {
        task_id: task_id.to_string(),
        best_smarts,
        best_mcc: best_fitness.mcc(),
        best_smarts_len,
        generations,
        leaders,
    }
}

fn build_scored_genome(
    fitness_cache: &mut FitnessCache,
    genome: SmartsGenome,
    evaluation: GenomeEvaluation,
) -> ScoredGenome {
    let phenotype = evaluation.phenotype().clone();
    let fitness = evaluation.fitness();
    fitness_cache.insert(genome.smarts_shared().clone(), fitness, phenotype.clone());
    ScoredGenome {
        genome,
        fitness,
        phenotype,
    }
}

fn build_rejected_scored_genome(genome: SmartsGenome) -> ScoredGenome {
    ScoredGenome {
        genome,
        fitness: ObjectiveFitness::invalid(),
        phenotype: Arc::from([]),
    }
}

#[cfg(test)]
fn deduplicate_population(population: &[SmartsGenome]) -> (Vec<SmartsGenome>, usize) {
    let mut unique = Vec::with_capacity(population.len());
    let mut seen: HashSet<Arc<str>> = HashSet::with_capacity(population.len());

    for genome in population {
        if seen.insert(genome.smarts_shared().clone()) {
            unique.push(genome.clone());
        }
    }

    let duplicate_count = population.len().saturating_sub(unique.len());
    (unique, duplicate_count)
}

fn insert_unique_genome(
    next_gen: &mut Vec<SmartsGenome>,
    seen: &mut HashSet<Arc<str>>,
    genome: SmartsGenome,
) -> bool {
    if seen.insert(genome.smarts_shared().clone()) {
        next_gen.push(genome);
        true
    } else {
        false
    }
}

fn reinsert_population(
    scored_unique: &[ScoredGenome],
    offspring: Vec<SmartsGenome>,
    population_size: usize,
    elite_count: usize,
    immigrant_count: usize,
    mut next_immigrant: impl FnMut() -> SmartsGenome,
) -> Vec<SmartsGenome> {
    let pre_immigrant_target = population_size.saturating_sub(immigrant_count);
    let mut next_gen = Vec::with_capacity(population_size);
    let mut seen: HashSet<Arc<str>> = HashSet::with_capacity(population_size);

    for genome in select_diverse_elites(scored_unique, elite_count) {
        insert_unique_genome(&mut next_gen, &mut seen, genome);
    }

    for genome in offspring {
        if next_gen.len() >= pre_immigrant_target {
            break;
        }
        insert_unique_genome(&mut next_gen, &mut seen, genome);
    }

    let unique_attempt_limit = population_size.saturating_mul(8).max(32);
    let mut attempts = 0usize;
    while next_gen.len() < population_size && attempts < unique_attempt_limit {
        attempts += 1;
        insert_unique_genome(&mut next_gen, &mut seen, next_immigrant());
    }

    while next_gen.len() < population_size {
        next_gen.push(next_immigrant());
    }

    next_gen
}

fn build_reset_pool(
    seed_corpus: &SeedCorpus,
    population: &[SmartsGenome],
    limit: usize,
) -> Vec<SmartsGenome> {
    let mut reset_pool = Vec::with_capacity(limit);
    let mut seen: HashSet<Arc<str>> = HashSet::with_capacity(limit);

    for genome in seed_corpus.entries() {
        if reset_pool.len() >= limit {
            return reset_pool;
        }
        if seen.insert(genome.smarts_shared().clone()) {
            reset_pool.push(genome.clone());
        }
    }

    for genome in population {
        if reset_pool.len() >= limit {
            break;
        }
        if seen.insert(genome.smarts_shared().clone()) {
            reset_pool.push(genome.clone());
        }
    }

    reset_pool
}

fn task_seed_corpus(seed_corpus: &SeedCorpus, folds: &[FoldData]) -> SeedCorpus {
    let mut corpus = dataset_seed_corpus(folds, DATASET_SEED_LIMIT);
    corpus.extend(seed_corpus.clone());
    corpus
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum DatasetSeedFeature {
    Atom(u16),
    RingAtom(u16),
    TotalHydrogen {
        atomic_number: u16,
        hydrogens: u8,
    },
    Bond {
        left_atomic_number: u16,
        bond: DatasetSeedBond,
        right_atomic_number: u16,
    },
}

impl DatasetSeedFeature {
    fn smarts(self) -> String {
        match self {
            Self::Atom(atomic_number) => format!("[#{atomic_number}]"),
            Self::RingAtom(atomic_number) => format!("[#{atomic_number};R]"),
            Self::TotalHydrogen {
                atomic_number,
                hydrogens,
            } => {
                format!("[#{atomic_number};H{hydrogens}]")
            }
            Self::Bond {
                left_atomic_number,
                bond,
                right_atomic_number,
            } => {
                format!(
                    "[#{left_atomic_number}]{}[#{right_atomic_number}]",
                    bond.smarts()
                )
            }
        }
    }

    fn smarts_len(self) -> usize {
        self.smarts().len()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum DatasetSeedBond {
    Any,
    Single,
    Double,
    Triple,
    Aromatic,
}

impl DatasetSeedBond {
    fn from_label(label: BondLabel) -> Self {
        match label {
            BondLabel::Single | BondLabel::Up | BondLabel::Down => Self::Single,
            BondLabel::Double => Self::Double,
            BondLabel::Triple => Self::Triple,
            BondLabel::Aromatic => Self::Aromatic,
            BondLabel::Any => Self::Any,
        }
    }

    fn smarts(self) -> &'static str {
        match self {
            Self::Any => "~",
            Self::Single => "-",
            Self::Double => "=",
            Self::Triple => "#",
            Self::Aromatic => ":",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct DatasetSeedFeatureCounts {
    positive_targets: u64,
    negative_targets: u64,
}

#[derive(Clone, Copy, Debug)]
struct DatasetSeedFeatureScore {
    mcc: f64,
    positive_targets: u64,
    negative_targets: u64,
}

fn dataset_seed_corpus(folds: &[FoldData], limit: usize) -> SeedCorpus {
    if limit == 0 {
        return SeedCorpus::default();
    }

    let positive_total = folds.iter().map(FoldData::positive_count).sum::<usize>() as u64;
    let negative_total = folds.iter().map(FoldData::negative_count).sum::<usize>() as u64;
    if positive_total == 0 || negative_total == 0 {
        return SeedCorpus::default();
    }

    let mut counts: BTreeMap<DatasetSeedFeature, DatasetSeedFeatureCounts> = BTreeMap::new();
    for fold in folds {
        for sample in fold.samples() {
            for feature in target_dataset_seed_features(sample.target()) {
                let entry = counts.entry(feature).or_default();
                if sample.is_positive() {
                    entry.positive_targets += 1;
                } else {
                    entry.negative_targets += 1;
                }
            }
        }
    }

    let mut ranked = counts
        .into_iter()
        .filter_map(|(feature, counts)| {
            let score = dataset_seed_feature_score(counts, positive_total, negative_total)?;
            Some((feature, score))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(compare_dataset_seed_features);

    let mut corpus = SeedCorpus::default();
    for (feature, _) in ranked.into_iter().take(limit) {
        let _ = corpus.insert_smarts(&feature.smarts());
    }
    corpus
}

fn target_dataset_seed_features(target: &smarts_rs::PreparedTarget) -> Vec<DatasetSeedFeature> {
    let mut features = BTreeSet::new();
    for atom_id in 0..target.atom_count() {
        let Some(atomic_number) = target_atomic_number(target, atom_id) else {
            continue;
        };
        features.insert(DatasetSeedFeature::Atom(atomic_number));
        if target.is_ring_atom(atom_id) {
            features.insert(DatasetSeedFeature::RingAtom(atomic_number));
        }
        if let Some(hydrogens @ 1..=4) = target.total_hydrogen_count(atom_id) {
            features.insert(DatasetSeedFeature::TotalHydrogen {
                atomic_number,
                hydrogens,
            });
        }

        for (left, right, _, _) in target.target().edges_for_node(atom_id) {
            let other_atom_id = if left == atom_id { right } else { left };
            if other_atom_id < atom_id {
                continue;
            }
            let Some(other_atomic_number) = target_atomic_number(target, other_atom_id) else {
                continue;
            };
            let Some(label) = target.bond(atom_id, other_atom_id) else {
                continue;
            };
            let (left_atomic_number, right_atomic_number) =
                ordered_pair(atomic_number, other_atomic_number);
            features.insert(DatasetSeedFeature::Bond {
                left_atomic_number,
                bond: DatasetSeedBond::Any,
                right_atomic_number,
            });
            let exact_bond = DatasetSeedBond::from_label(label);
            if exact_bond == DatasetSeedBond::Any {
                continue;
            }
            features.insert(DatasetSeedFeature::Bond {
                left_atomic_number,
                bond: exact_bond,
                right_atomic_number,
            });
        }
    }
    features.into_iter().collect()
}

fn target_atomic_number(target: &smarts_rs::PreparedTarget, atom_id: usize) -> Option<u16> {
    target
        .atom(atom_id)?
        .element()
        .map(|element| u16::from(element.atomic_number()))
}

fn ordered_pair(left: u16, right: u16) -> (u16, u16) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn dataset_seed_feature_score(
    counts: DatasetSeedFeatureCounts,
    positive_total: u64,
    negative_total: u64,
) -> Option<DatasetSeedFeatureScore> {
    let mcc = compute_mcc_from_counts(
        counts.positive_targets,
        counts.negative_targets,
        negative_total.saturating_sub(counts.negative_targets),
        positive_total.saturating_sub(counts.positive_targets),
    );
    (mcc > 0.0).then_some(DatasetSeedFeatureScore {
        mcc,
        positive_targets: counts.positive_targets,
        negative_targets: counts.negative_targets,
    })
}

fn compare_dataset_seed_features(
    left: &(DatasetSeedFeature, DatasetSeedFeatureScore),
    right: &(DatasetSeedFeature, DatasetSeedFeatureScore),
) -> Ordering {
    right
        .1
        .mcc
        .total_cmp(&left.1.mcc)
        .then_with(|| right.1.positive_targets.cmp(&left.1.positive_targets))
        .then_with(|| left.1.negative_targets.cmp(&right.1.negative_targets))
        .then_with(|| left.0.smarts_len().cmp(&right.0.smarts_len()))
        .then_with(|| left.0.cmp(&right.0))
}

fn average_smarts_len(population: &[SmartsGenome]) -> f64 {
    if population.is_empty() {
        return 0.0;
    }

    let total_smarts_len: usize = population.iter().map(SmartsGenome::smarts_len).sum();
    total_smarts_len as f64 / population.len() as f64
}

fn build_rng(config: &EvolutionConfig) -> SmallRng {
    match config.rng_seed() {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => {
            let mut seed = <SmallRng as SeedableRng>::Seed::default();
            if getrandom::fill(seed.as_mut()).is_ok() {
                SmallRng::from_seed(seed)
            } else {
                SmallRng::seed_from_u64(0x5EED_5EED_5EED_5EED)
            }
        }
    }
}

fn generation_progress_message(
    stats: &GenerationStats,
    lead: ObjectiveFitness,
    best: ObjectiveFitness,
) -> String {
    format!(
        "lead={:.3} best={:.3} uniq={}/{} dup={} cache={}/{} rejected={} smarts_len_rejected={} match_timeouts={} smarts_len={} avg_smarts_len={:.1}",
        lead.mcc(),
        best.mcc(),
        stats.unique_count,
        stats.total_count,
        stats.duplicate_count,
        stats.cache_hits,
        stats.unique_count,
        stats.rejection_counts.total(),
        stats.rejection_counts.smarts_length,
        stats.match_timeout_count,
        stats.lead_smarts_len,
        stats.average_smarts_len,
    )
}

fn compare_scored_genomes(left: &ScoredGenome, right: &ScoredGenome) -> Ordering {
    right
        .fitness
        .cmp(&left.fitness)
        .then_with(|| left.genome.smarts_len().cmp(&right.genome.smarts_len()))
        .then_with(|| left.genome.smarts().cmp(right.genome.smarts()))
}

fn is_better_candidate(
    candidate_fitness: ObjectiveFitness,
    candidate: &SmartsGenome,
    best_fitness: ObjectiveFitness,
    best_smarts_len: usize,
    best_smarts: &str,
) -> bool {
    candidate_fitness > best_fitness
        || (candidate_fitness == best_fitness
            && (candidate.smarts_len() < best_smarts_len
                || (candidate.smarts_len() == best_smarts_len && candidate.smarts() < best_smarts)))
}

fn select_screened_mutation_candidate(
    evaluator: &SmartsEvaluator,
    candidates: &[SmartsGenome],
) -> usize {
    let mut best_idx = 0usize;
    let mut best_score = evaluator.screening_proxy_of(&candidates[0]);
    for (idx, candidate) in candidates.iter().enumerate().skip(1) {
        let score = evaluator.screening_proxy_of(candidate);
        if screened_mutation_candidate_is_better(
            candidate,
            score,
            &candidates[best_idx],
            best_score,
        ) {
            best_idx = idx;
            best_score = score;
        }
    }
    best_idx
}

fn screened_mutation_candidate_is_better(
    candidate: &SmartsGenome,
    candidate_score: ScreeningProxyFitness,
    current: &SmartsGenome,
    current_score: ScreeningProxyFitness,
) -> bool {
    let candidate_hits_positive = candidate_score.positive_candidates() > 0;
    let current_hits_positive = current_score.positive_candidates() > 0;
    if candidate_hits_positive != current_hits_positive {
        return candidate_hits_positive;
    }

    candidate_score.mcc() > current_score.mcc()
        || (candidate_score.mcc() == current_score.mcc()
            && (candidate_score.negative_candidates() < current_score.negative_candidates()
                || (candidate_score.negative_candidates() == current_score.negative_candidates()
                    && (candidate.smarts_len() < current.smarts_len()
                        || (candidate.smarts_len() == current.smarts_len()
                            && candidate.smarts() < current.smarts())))))
}

fn mutation_direction_from_counts(counts: ConfusionCounts) -> MutationDirection {
    let false_positive_rate = rate(
        counts.false_positives(),
        counts.false_positives() + counts.true_negatives(),
    );
    let false_negative_rate = rate(
        counts.false_negatives(),
        counts.false_negatives() + counts.true_positives(),
    );
    const DIRECTION_MARGIN: f64 = 1.20;

    if false_positive_rate > 0.0 && false_positive_rate > false_negative_rate * DIRECTION_MARGIN {
        MutationDirection::Specialize
    } else if false_negative_rate > 0.0
        && false_negative_rate > false_positive_rate * DIRECTION_MARGIN
    {
        MutationDirection::Generalize
    } else {
        MutationDirection::Balanced
    }
}

fn rate(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn tournament_select(
    scored: &[ScoredGenome],
    count: usize,
    tournament_size: usize,
    rng: &mut impl rand::Rng,
) -> Vec<ScoredGenome> {
    let mut parents = Vec::with_capacity(count);
    for _ in 0..count {
        let mut best_idx = rng.random_range(0..scored.len());
        for _ in 1..tournament_size {
            let idx = rng.random_range(0..scored.len());
            if compare_scored_genomes(&scored[idx], &scored[best_idx]).is_lt() {
                best_idx = idx;
            }
        }
        parents.push(scored[best_idx].clone());
    }
    parents
}

fn leaderboard_from_scored(scored: &[ScoredGenome], limit: usize) -> Vec<RankedSmarts> {
    scored
        .iter()
        .take(limit.max(1))
        .map(|candidate| RankedSmarts::new(&candidate.genome, candidate.fitness))
        .collect()
}

fn phenotypically_deduplicate(scored: Vec<ScoredGenome>) -> Vec<ScoredGenome> {
    let mut best_by_phenotype: HashMap<Arc<[u64]>, ScoredGenome> =
        HashMap::with_capacity(scored.len());

    for candidate in scored {
        let phenotype = candidate.phenotype.clone();
        match best_by_phenotype.entry(phenotype) {
            hashbrown::hash_map::Entry::Occupied(mut entry) => {
                if compare_scored_genomes(&candidate, entry.get()).is_lt() {
                    entry.insert(candidate);
                }
            }
            hashbrown::hash_map::Entry::Vacant(entry) => {
                entry.insert(candidate);
            }
        }
    }

    best_by_phenotype.into_values().collect()
}

fn select_diverse_elites(scored: &[ScoredGenome], elite_count: usize) -> Vec<SmartsGenome> {
    if scored.is_empty() || elite_count == 0 {
        return Vec::new();
    }

    let pool_len = scored
        .len()
        .min(elite_count.max(1).saturating_mul(DIVERSE_ELITE_POOL_FACTOR));
    let pool = &scored[..pool_len];
    let target = elite_count.min(pool.len());
    let mut selected = Vec::with_capacity(target);
    let mut selected_indices = Vec::with_capacity(target);

    selected.push(pool[0].genome.clone());
    selected_indices.push(0usize);

    while selected.len() < target {
        let mut best_idx = None;
        let mut best_distance = 0u32;

        for candidate_idx in 0..pool.len() {
            if selected_indices.contains(&candidate_idx) {
                continue;
            }

            let distance = selected_indices
                .iter()
                .map(|&selected_idx| {
                    phenotype_distance(
                        pool[candidate_idx].phenotype.as_ref(),
                        pool[selected_idx].phenotype.as_ref(),
                    )
                })
                .min()
                .unwrap_or(0);

            let is_better = distance > best_distance
                || (distance == best_distance
                    && best_idx.is_some_and(|current_best| {
                        compare_scored_genomes(&pool[candidate_idx], &pool[current_best]).is_lt()
                    }));
            if best_idx.is_none() || is_better {
                best_idx = Some(candidate_idx);
                best_distance = distance;
            }
        }

        let Some(next_idx) = best_idx else {
            break;
        };
        selected.push(pool[next_idx].genome.clone());
        selected_indices.push(next_idx);
    }

    selected
}

fn phenotype_distance(left: &[u64], right: &[u64]) -> u32 {
    left.iter()
        .zip(right.iter())
        .map(|(left_word, right_word)| (left_word ^ right_word).count_ones())
        .sum()
}

#[cfg(test)]
mod regression_tests {
    use std::str::FromStr;
    use std::sync::{Arc as StdArc, Mutex};
    use std::vec;

    use crate::fitness::evaluator::FoldSample;
    use smarts_rs::PreparedTarget;
    use smiles_parser::Smiles;

    use super::*;

    fn fitness(mcc: f64) -> ObjectiveFitness {
        ObjectiveFitness::from_mcc(mcc)
    }

    fn scored(smarts: &str, mcc: f64, phenotype: &[u64]) -> ScoredGenome {
        ScoredGenome {
            genome: SmartsGenome::from_smarts(smarts).unwrap(),
            fitness: fitness(mcc),
            phenotype: Arc::from(phenotype.to_vec()),
        }
    }

    fn target(smiles: &str) -> PreparedTarget {
        PreparedTarget::new(Smiles::from_str(smiles).unwrap())
    }

    fn sample(smiles: &str, is_positive: bool) -> FoldSample {
        FoldSample::new(target(smiles), is_positive)
    }

    #[derive(Default)]
    struct RecordedObserverEvents {
        starts: Vec<(String, u64)>,
        evaluations: usize,
        generations: usize,
        finishes: Vec<String>,
        errors: Vec<String>,
    }

    struct RecordingObserver {
        events: StdArc<Mutex<RecordedObserverEvents>>,
    }

    impl RecordingObserver {
        fn new(events: StdArc<Mutex<RecordedObserverEvents>>) -> Self {
            Self { events }
        }
    }

    impl EvolutionProgressObserver for RecordingObserver {
        fn on_start(&mut self, task_id: &str, generation_limit: u64) {
            self.events
                .lock()
                .unwrap()
                .starts
                .push((task_id.to_string(), generation_limit));
        }

        fn on_evaluation(&mut self, _progress: &EvolutionEvaluationProgress) {
            self.events.lock().unwrap().evaluations += 1;
        }

        fn on_generation(&mut self, _progress: &EvolutionProgress) {
            self.events.lock().unwrap().generations += 1;
        }

        fn on_finish(&mut self, result: &TaskResult) {
            self.events
                .lock()
                .unwrap()
                .finishes
                .push(result.task_id().to_string());
        }

        fn on_error(&mut self, error: &EvolutionError) {
            self.events.lock().unwrap().errors.push(error.to_string());
        }
    }

    #[test]
    fn task_evolve_preserves_task_id() {
        let task = EvolutionTask {
            task_id: "leaf-1".to_string(),
            folds: vec![FoldData::new(vec![sample("CC", true), sample("CN", false)])],
        };
        let config = EvolutionConfig::builder()
            .population_size(4)
            .generation_limit(1)
            .stagnation_limit(1)
            .build()
            .unwrap();

        let result = task.evolve(&config, &SeedCorpus::builtin()).unwrap();
        assert_eq!(result.task_id(), "leaf-1");
    }

    #[test]
    fn task_evolve_rejects_missing_folds_and_targets() {
        let config = EvolutionConfig::default();
        let empty_task = EvolutionTask {
            task_id: "empty".to_string(),
            folds: Vec::new(),
        };
        let no_target_task = EvolutionTask {
            task_id: "no-targets".to_string(),
            folds: vec![FoldData::new(Vec::new())],
        };

        assert_eq!(
            empty_task
                .evolve(&config, &SeedCorpus::builtin())
                .unwrap_err()
                .to_string(),
            "evolution task requires at least one fold"
        );
        assert_eq!(
            no_target_task
                .evolve(&config, &SeedCorpus::builtin())
                .unwrap_err()
                .to_string(),
            "task 'no-targets' has no evaluation targets"
        );
    }

    #[test]
    fn task_evolve_runs_end_to_end_and_returns_valid_smarts() {
        let task = EvolutionTask {
            task_id: "amide-vs-rest".to_string(),
            folds: vec![FoldData::new(vec![
                sample("CC(=O)N", true),
                sample("NC(=O)C", true),
                sample("CC(=O)NC", true),
                sample("CCO", false),
                sample("c1ccccc1", false),
                sample("CCCl", false),
            ])],
        };
        let config = EvolutionConfig::builder()
            .population_size(12)
            .generation_limit(3)
            .stagnation_limit(3)
            .build()
            .unwrap();
        let seed_corpus = SeedCorpus::try_from(["[#6](=[#8])[#7]", "[#6]~[#7]", "[#7]"]).unwrap();

        let result = task.evolve(&config, &seed_corpus).unwrap();

        assert_eq!(result.task_id(), "amide-vs-rest");
        assert!(result.generations() >= 1);
        assert!(result.generations() <= config.generation_limit());
        assert!(SmartsGenome::from_smarts(result.best_smarts()).is_ok());
        assert!(result.best_mcc().is_finite());
        assert!(!result.leaders().is_empty());
    }

    #[test]
    fn task_evolve_owned_and_owned_session_run_end_to_end() {
        let folds = vec![FoldData::new(vec![
            sample("CC(=O)N", true),
            sample("NC(=O)C", true),
            sample("CCO", false),
            sample("CCCl", false),
        ])];
        let config = EvolutionConfig::builder()
            .population_size(8)
            .generation_limit(2)
            .stagnation_limit(2)
            .rng_seed(17)
            .build()
            .unwrap();
        let seed_corpus = SeedCorpus::try_from(["[#6](=[#8])[#7]", "[#6]~[#7]", "[#7]"]).unwrap();

        let task_result = EvolutionTask::new("owned-task", folds.clone())
            .evolve_owned(&config, &seed_corpus)
            .unwrap();
        let session_result = EvolutionSession::from_owned_task(
            EvolutionTask::new("owned-session", folds),
            &config,
            &seed_corpus,
            2,
        )
        .unwrap()
        .evolve()
        .unwrap();

        assert_eq!(task_result.task_id(), "owned-task");
        assert_eq!(session_result.task_id(), "owned-session");
        assert!(task_result.best_mcc().is_finite());
        assert!(session_result.best_mcc().is_finite());
        assert!(!session_result.leaders().is_empty());
    }

    #[test]
    fn task_evolve_handles_static_population() {
        let task = EvolutionTask {
            task_id: "stagnation".to_string(),
            folds: vec![FoldData::new(vec![sample("CC", true), sample("CN", false)])],
        };
        let config = EvolutionConfig::builder()
            .population_size(2)
            .generation_limit(5)
            .stagnation_limit(1)
            .mutation_rate(0.0)
            .crossover_rate(0.0)
            .random_immigrant_ratio(0.0)
            .elite_count(1)
            .build()
            .unwrap();
        let seed_corpus = SeedCorpus::try_from(["[#6]"]).unwrap();

        let result = task.evolve(&config, &seed_corpus).unwrap();

        assert!(result.generations() >= 2);
        assert!(result.generations() <= config.generation_limit());
        assert!(SmartsGenome::from_smarts(result.best_smarts()).is_ok());
    }

    #[test]
    fn task_evolve_with_progress_emits_generation_snapshots_and_leaderboard() {
        let task = EvolutionTask::new(
            "amide-progress",
            vec![FoldData::new(vec![
                sample("CC(=O)N", true),
                sample("NC(=O)C", true),
                sample("CCO", false),
                sample("CCCl", false),
            ])],
        );
        let config = EvolutionConfig::builder()
            .population_size(8)
            .generation_limit(2)
            .stagnation_limit(2)
            .build()
            .unwrap();
        let seed_corpus = SeedCorpus::try_from(["[#6](=[#8])[#7]", "[#6]~[#7]", "[#7]"]).unwrap();
        let mut snapshots = Vec::new();

        let result = task
            .evolve_with_progress(&config, &seed_corpus, 3, |snapshot| {
                snapshots.push(snapshot);
            })
            .unwrap();

        assert_eq!(snapshots.len(), result.generations() as usize);
        assert_eq!(snapshots[0].task_id(), "amide-progress");
        assert_eq!(
            snapshots.last().unwrap().status(),
            EvolutionStatus::Completed
        );
        assert!(snapshots.iter().all(|snapshot| snapshot.generation() >= 1));
        assert!(
            snapshots
                .iter()
                .all(|snapshot| snapshot.generation_limit() == 2)
        );
        assert!(
            snapshots
                .iter()
                .all(|snapshot| snapshot.leaders().len() <= 3 && !snapshot.leaders().is_empty())
        );
        assert_eq!(
            snapshots.last().unwrap().leaders()[0].smarts(),
            result.leaders()[0].smarts()
        );
    }

    #[test]
    fn task_evolve_with_observer_reports_lifecycle_events() {
        let task = EvolutionTask::new(
            "amide-observer",
            vec![FoldData::new(vec![
                sample("CC(=O)N", true),
                sample("NC(=O)C", true),
                sample("CCO", false),
                sample("CCCl", false),
            ])],
        );
        let config = EvolutionConfig::builder()
            .population_size(8)
            .generation_limit(2)
            .stagnation_limit(2)
            .build()
            .unwrap();
        let seed_corpus = SeedCorpus::try_from(["[#6](=[#8])[#7]", "[#6]~[#7]", "[#7]"]).unwrap();
        let events = StdArc::new(Mutex::new(RecordedObserverEvents::default()));

        let result = task
            .evolve_with_observer(
                &config,
                &seed_corpus,
                2,
                RecordingObserver::new(events.clone()),
            )
            .unwrap();

        let events = events.lock().unwrap();
        assert_eq!(events.starts, vec![("amide-observer".to_string(), 2)]);
        assert!(events.evaluations > 0);
        assert_eq!(events.generations, result.generations() as usize);
        assert_eq!(events.finishes, vec!["amide-observer".to_string()]);
        assert!(events.errors.is_empty());
    }

    #[test]
    fn task_evolve_with_observer_reports_setup_errors() {
        let task = EvolutionTask::new("empty-observer", Vec::new());
        let config = EvolutionConfig::default();
        let events = StdArc::new(Mutex::new(RecordedObserverEvents::default()));

        let error = task
            .evolve_with_observer(
                &config,
                &SeedCorpus::builtin(),
                1,
                RecordingObserver::new(events.clone()),
            )
            .unwrap_err();

        let events = events.lock().unwrap();
        assert_eq!(error, EvolutionError::EmptyFolds);
        assert_eq!(
            events.starts,
            vec![("empty-observer".to_string(), config.generation_limit())]
        );
        assert!(events.finishes.is_empty());
        assert_eq!(
            events.errors,
            vec!["evolution task requires at least one fold".to_string()]
        );
    }

    #[test]
    fn evolution_session_reports_within_generation_evaluation_progress() {
        let task = EvolutionTask::new(
            "amide-evaluation-progress",
            vec![FoldData::new(vec![
                sample("CC(=O)N", true),
                sample("NC(=O)C", true),
                sample("CCO", false),
                sample("CCCl", false),
            ])],
        );
        let config = EvolutionConfig::builder()
            .population_size(8)
            .generation_limit(1)
            .stagnation_limit(1)
            .build()
            .unwrap();
        let seed_corpus = SeedCorpus::try_from(["[#6](=[#8])[#7]", "[#6]~[#7]", "[#7]"]).unwrap();
        let mut session = EvolutionSession::new(&task, &config, &seed_corpus, 3).unwrap();
        let mut evaluations = Vec::new();

        let progress = session
            .step_with_evaluation_progress(|progress| evaluations.push(progress))
            .unwrap();

        assert_eq!(progress.generation(), 1);
        assert!(!evaluations.is_empty());
        let first = evaluations.first().unwrap();
        assert_eq!(first.completed(), 0);
        assert!(first.last().is_none());
        assert!(first.generation_best().is_none());
        assert_eq!(first.incumbent_best_smarts(), "[#6]");
        assert_eq!(first.incumbent_best_mcc(), -1.0);
        let mut completed = evaluations
            .iter()
            .map(EvolutionEvaluationProgress::completed)
            .collect::<Vec<_>>();
        completed.sort_unstable();
        assert_eq!(
            completed,
            (0..=config.population_size()).collect::<Vec<_>>()
        );
        assert!(evaluations.iter().all(|progress| progress.task_id()
            == "amide-evaluation-progress"
            && progress.generation() == 1
            && progress.generation_limit() == 1
            && progress.total() == config.population_size()));
        assert!(
            evaluations
                .iter()
                .all(|progress| progress.incumbent_best_smarts() == "[#6]"
                    && progress.incumbent_best_mcc() == -1.0)
        );
        let scored_events = evaluations
            .iter()
            .filter(|progress| progress.last().is_some())
            .collect::<Vec<_>>();
        assert!(!scored_events.is_empty());
        assert!(
            scored_events
                .iter()
                .all(|progress| progress.last_smarts().is_some()
                    && progress.last_mcc().is_some_and(f64::is_finite)
                    && progress.generation_best_smarts().is_some()
                    && progress.generation_best_mcc().is_some_and(f64::is_finite))
        );
    }

    #[test]
    fn deduplicate_population_removes_repeated_smarts() {
        let population = vec![
            SmartsGenome::from_smarts("[#6]").unwrap(),
            SmartsGenome::from_smarts("[#7]").unwrap(),
            SmartsGenome::from_smarts("[#6]").unwrap(),
            SmartsGenome::from_smarts("[#8]").unwrap(),
        ];

        let (unique, duplicate_count) = deduplicate_population(&population);

        assert_eq!(duplicate_count, 1);
        assert_eq!(unique.len(), 3);
        assert_eq!(unique[0].smarts(), "[#6]");
        assert_eq!(unique[1].smarts(), "[#7]");
        assert_eq!(unique[2].smarts(), "[#8]");
    }

    #[test]
    fn reinsert_population_prefers_unique_offspring_and_immigrants() {
        let scored = vec![
            scored("[#6]", 0.90, &[0b0001]),
            scored("[#7]", 0.80, &[0b0010]),
        ];
        let offspring = vec![
            SmartsGenome::from_smarts("[#6]").unwrap(),
            SmartsGenome::from_smarts("[#7]").unwrap(),
            SmartsGenome::from_smarts("[#8]").unwrap(),
            SmartsGenome::from_smarts("[#16]").unwrap(),
        ];
        let immigrant_smarts = ["[#9]", "[#15]", "[#17]"];
        let mut immigrant_idx = 0usize;

        let next_gen = reinsert_population(&scored, offspring, 5, 1, 1, || {
            let genome =
                SmartsGenome::from_smarts(immigrant_smarts[immigrant_idx % immigrant_smarts.len()])
                    .unwrap();
            immigrant_idx += 1;
            genome
        });

        let smarts: Vec<&str> = next_gen.iter().map(|genome| genome.smarts()).collect();
        assert_eq!(smarts, vec!["[#6]", "[#7]", "[#8]", "[#16]", "[#9]"]);
    }

    #[test]
    fn reinsert_population_falls_back_to_duplicate_immigrants_after_unique_budget() {
        let scored = vec![scored("[#6]", 0.90, &[0b0001])];
        let offspring = vec![SmartsGenome::from_smarts("[#6]").unwrap()];

        let next_gen = reinsert_population(&scored, offspring, 3, 1, 0, || {
            SmartsGenome::from_smarts("[#6]").unwrap()
        });

        assert_eq!(next_gen.len(), 3);
        assert_eq!(next_gen[0].smarts(), "[#6]");
        assert_eq!(next_gen[1].smarts(), "[#6]");
        assert_eq!(next_gen[2].smarts(), "[#6]");
    }

    #[test]
    fn build_reset_pool_prefers_corpus_and_respects_limit() {
        let seed_corpus = SeedCorpus::try_from(["[#6]", "[#7]", "[#8]"]).unwrap();
        let population = vec![
            SmartsGenome::from_smarts("[#9]").unwrap(),
            SmartsGenome::from_smarts("[#15]").unwrap(),
        ];

        let limited = build_reset_pool(&seed_corpus, &population, 2);
        let from_population = build_reset_pool(&SeedCorpus::default(), &population, 1);

        assert_eq!(
            limited
                .iter()
                .map(|genome| genome.smarts())
                .collect::<Vec<_>>(),
            vec!["[#6]", "[#7]"]
        );
        assert_eq!(
            from_population
                .iter()
                .map(|genome| genome.smarts())
                .collect::<Vec<_>>(),
            vec!["[#9]"]
        );
    }

    #[test]
    fn dataset_seed_corpus_prefers_positive_enriched_features() {
        let folds = vec![FoldData::new(vec![
            sample("CN", true),
            sample("CCN", true),
            sample("CC", false),
            sample("CO", false),
        ])];
        let corpus = dataset_seed_corpus(&folds, 8);
        let smarts = corpus
            .entries()
            .iter()
            .map(|genome| genome.smarts())
            .collect::<Vec<_>>();

        assert!(smarts.contains(&"[#7]"));
        assert!(smarts.contains(&"[#6]~[#7]"));
        assert!(!smarts.contains(&"[#8]"));
    }

    #[test]
    fn task_seed_corpus_preserves_user_seeds_and_adds_dataset_seeds() {
        let user = SeedCorpus::try_from(["[#16]"]).unwrap();
        let folds = vec![FoldData::new(vec![sample("CN", true), sample("CC", false)])];
        let corpus = task_seed_corpus(&user, &folds);
        let smarts = corpus
            .entries()
            .iter()
            .map(|genome| genome.smarts())
            .collect::<Vec<_>>();

        assert!(smarts.contains(&"[#16]"));
        assert!(smarts.contains(&"[#7]"));
        assert!(
            smarts.iter().position(|smarts| *smarts == "[#7]")
                < smarts.iter().position(|smarts| *smarts == "[#16]")
        );
    }

    #[test]
    fn lower_smarts_len_smarts_win_when_scores_tie() {
        let simple = scored("[#6]", 0.75, &[0b0001]);
        let longer = scored("[#6]~[#7]", 0.75, &[0b0010]);

        assert!(simple.genome.smarts_len() < longer.genome.smarts_len());

        assert!(compare_scored_genomes(&simple, &longer).is_lt());
        assert!(compare_scored_genomes(&longer, &simple).is_gt());
    }

    #[test]
    fn phenotypic_dedup_keeps_best_representative_for_same_behavior() {
        let deduped = phenotypically_deduplicate(vec![
            scored("[#6]~[#7]", 0.80, &[0b0101]),
            scored("[#6]", 0.80, &[0b0101]),
            scored("[#8]", 0.70, &[0b0011]),
        ]);
        let mut smarts = deduped
            .iter()
            .map(|candidate| candidate.genome.smarts())
            .collect::<Vec<_>>();
        smarts.sort_unstable();

        assert_eq!(smarts, vec!["[#6]", "[#8]"]);
    }

    #[test]
    fn select_diverse_elites_prefers_distinct_behaviors_within_front() {
        let elites = select_diverse_elites(
            &[
                scored("[#6]", 0.90, &[0b0000]),
                scored("[#7]", 0.89, &[0b0000]),
                scored("[#8]", 0.88, &[0b1111]),
                scored("[#16]", 0.87, &[0b1100]),
            ],
            2,
        );

        assert_eq!(
            elites
                .iter()
                .map(|genome| genome.smarts())
                .collect::<Vec<_>>(),
            vec!["[#6]", "[#8]"]
        );
    }

    #[test]
    fn mutation_direction_tracks_parent_error_profile() {
        assert_eq!(
            mutation_direction_from_counts(ConfusionCounts::new(8, 4, 0, 0)),
            MutationDirection::Specialize
        );
        assert_eq!(
            mutation_direction_from_counts(ConfusionCounts::new(0, 0, 8, 4)),
            MutationDirection::Generalize
        );
        assert_eq!(
            mutation_direction_from_counts(ConfusionCounts::new(9, 1, 9, 1)),
            MutationDirection::Balanced
        );
    }

    #[test]
    fn generation_progress_message_surfaces_diversity_stats() {
        let message = generation_progress_message(
            &GenerationStats {
                unique_count: 382,
                total_count: 1024,
                duplicate_count: 642,
                cache_hits: 208,
                rejection_counts: EvaluationRejectionCounts { smarts_length: 5 },
                match_timeout_count: 12,
                lead_smarts_len: 16,
                average_smarts_len: 12.4,
            },
            fitness(0.6621),
            fitness(0.7314),
        );

        assert_eq!(
            message,
            "lead=0.662 best=0.731 uniq=382/1024 dup=642 cache=208/382 rejected=5 smarts_len_rejected=5 match_timeouts=12 smarts_len=16 avg_smarts_len=12.4"
        );
    }

    #[test]
    fn average_smarts_len_handles_empty_population() {
        assert_eq!(average_smarts_len(&[]), 0.0);
    }

    #[test]
    fn fitness_cache_evicts_least_recently_used_entries() {
        let mut cache = FitnessCache::new(2);
        let a: Arc<str> = "[#6]".into();
        let b: Arc<str> = "[#7]".into();
        let c: Arc<str> = "[#8]".into();

        cache.insert(a.clone(), fitness(0.8), Arc::from([0b0001u64]));
        cache.insert(b.clone(), fitness(0.7), Arc::from([0b0010u64]));
        assert!(cache.get(&a).is_some());
        cache.insert(c.clone(), fitness(0.6), Arc::from([0b0100u64]));

        assert_eq!(cache.len(), 2);
        assert!(cache.get(&a).is_some());
        assert!(cache.get(&c).is_some());
        assert!(cache.get(&b).is_none());
    }

    #[test]
    fn fitness_cache_zero_capacity_and_compaction_paths_are_safe() {
        let mut cache = FitnessCache::new(0);
        let key: Arc<str> = "[#6]".into();
        cache.insert(key.clone(), fitness(0.8), Arc::from([0b1u64]));
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.len(), 0);

        let mut cache = FitnessCache::new(1);
        let key: Arc<str> = "[#7]".into();
        cache.insert(key.clone(), fitness(0.7), Arc::from([0b10u64]));
        for _ in 0..1100 {
            assert!(cache.get(&key).is_some());
        }
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn public_accessors_and_display_helpers_round_trip() {
        let genome = SmartsGenome::from_smarts("[#6]").unwrap();
        let ranked = RankedSmarts::new(&genome, fitness(0.5));
        assert_eq!(ranked.smarts(), "[#6]");
        assert_eq!(ranked.mcc(), 0.5);
        assert_eq!(ranked.smarts_len(), genome.smarts_len());

        let stats = GenerationStats {
            unique_count: 2,
            total_count: 3,
            duplicate_count: 1,
            cache_hits: 1,
            rejection_counts: EvaluationRejectionCounts { smarts_length: 2 },
            match_timeout_count: 3,
            lead_smarts_len: 4,
            average_smarts_len: 2.5,
        };
        let progress = EvolutionProgress::from_snapshot(ProgressSnapshot {
            task_id: "task-1",
            generation: 3,
            generation_limit: 7,
            status: EvolutionStatus::Running,
            best: ranked.clone(),
            best_so_far: ranked.clone(),
            leaders: vec![ranked.clone()],
            stats: &stats,
            stagnation: 2,
        });

        assert_eq!(progress.task_id(), "task-1");
        assert_eq!(progress.generation(), 3);
        assert_eq!(progress.generation_limit(), 7);
        assert_eq!(progress.status(), EvolutionStatus::Running);
        assert_eq!(progress.best().smarts(), "[#6]");
        assert_eq!(progress.best_so_far().smarts(), "[#6]");
        assert_eq!(progress.leaders().len(), 1);
        assert_eq!(progress.unique_count(), 2);
        assert_eq!(progress.total_count(), 3);
        assert_eq!(progress.duplicate_count(), 1);
        assert_eq!(progress.cache_hits(), 1);
        assert_eq!(progress.smarts_length_rejection_count(), 2);
        assert_eq!(progress.evaluation_rejection_count(), 2);
        assert_eq!(progress.match_timeout_count(), 3);
        assert_eq!(progress.lead_smarts_len(), 4);
        assert_eq!(progress.average_smarts_len(), 2.5);
        assert_eq!(progress.stagnation(), 2);

        let evaluation_progress =
            EvolutionEvaluationProgress::from_snapshot(EvaluationProgressSnapshot {
                task_id: "task-1",
                generation: 4,
                generation_limit: 7,
                completed: 2,
                total: 5,
                last: Some(ranked.clone()),
                generation_best: Some(ranked.clone()),
                incumbent_best: ranked.clone(),
            });
        assert_eq!(evaluation_progress.task_id(), "task-1");
        assert_eq!(evaluation_progress.generation(), 4);
        assert_eq!(evaluation_progress.generation_limit(), 7);
        assert_eq!(evaluation_progress.completed(), 2);
        assert_eq!(evaluation_progress.total(), 5);
        assert_eq!(evaluation_progress.last().unwrap().smarts(), "[#6]");
        assert_eq!(evaluation_progress.last_smarts(), Some("[#6]"));
        assert_eq!(evaluation_progress.last_mcc(), Some(0.5));
        assert_eq!(
            evaluation_progress.generation_best().unwrap().smarts(),
            "[#6]"
        );
        assert_eq!(evaluation_progress.generation_best_smarts(), Some("[#6]"));
        assert_eq!(evaluation_progress.generation_best_mcc(), Some(0.5));
        assert_eq!(evaluation_progress.incumbent_best().smarts(), "[#6]");
        assert_eq!(evaluation_progress.incumbent_best_smarts(), "[#6]");
        assert_eq!(evaluation_progress.incumbent_best_mcc(), 0.5);

        let result = build_task_result(
            "task-1",
            "[#7]".to_string(),
            fitness(0.75),
            11,
            4,
            vec![ranked],
        );
        assert_eq!(result.task_id(), "task-1");
        assert_eq!(result.best_smarts(), "[#7]");
        assert_eq!(result.best_mcc(), 0.75);
        assert_eq!(result.best_smarts_len(), 11);
        assert_eq!(result.generations(), 4);
        assert_eq!(result.leaders().len(), 1);

        assert_eq!(
            EvolutionError::InvalidConfig("bad".to_string()).to_string(),
            "bad"
        );
    }

    #[test]
    fn helper_branches_cover_tie_breaks_and_empty_inputs() {
        let seeded = EvolutionConfig::builder().rng_seed(17).build().unwrap();
        let mut rng_a = build_rng(&seeded);
        let mut rng_b = build_rng(&seeded);
        assert_eq!(rng_a.random::<u64>(), rng_b.random::<u64>());

        let simple = SmartsGenome::from_smarts("[#6]").unwrap();
        let longer = SmartsGenome::from_smarts("[#6]~[#7]").unwrap();
        assert!(simple.smarts_len() < longer.smarts_len());
        assert!(is_better_candidate(
            fitness(0.5),
            &simple,
            fitness(0.5),
            longer.smarts_len(),
            longer.smarts(),
        ));
        assert!(!is_better_candidate(
            fitness(0.5),
            &longer,
            fitness(0.5),
            simple.smarts_len(),
            simple.smarts(),
        ));

        let alpha = SmartsGenome::from_smarts("[#6]").unwrap();
        let beta = SmartsGenome::from_smarts("[#7]").unwrap();
        assert_eq!(alpha.smarts_len(), beta.smarts_len());
        assert!(is_better_candidate(
            fitness(0.5),
            &alpha,
            fitness(0.5),
            beta.smarts_len(),
            beta.smarts(),
        ));

        let leaders = leaderboard_from_scored(&[scored("[#6]", 0.9, &[0b1])], 0);
        assert_eq!(leaders.len(), 1);
        assert_eq!(leaders[0].smarts(), "[#6]");

        let deduped = phenotypically_deduplicate(vec![
            scored("[#6]", 0.8, &[0b1]),
            scored("[#7]", 0.7, &[0b1]),
        ]);
        assert_eq!(deduped.len(), 1);
        assert_eq!(deduped[0].genome.smarts(), "[#6]");

        assert!(select_diverse_elites(&[], 2).is_empty());
        assert!(select_diverse_elites(&[scored("[#6]", 0.8, &[0b1])], 0).is_empty());
        assert_eq!(phenotype_distance(&[0b1010], &[0b0011]), 2);
    }

    #[test]
    fn screened_mutation_selector_prefers_positive_enriched_candidates() {
        let evaluator = SmartsEvaluator::new(vec![FoldData::new(vec![
            sample("CN", true),
            sample("CCN", true),
            sample("CC", false),
            sample("CO", false),
        ])]);
        let candidates = vec![
            SmartsGenome::from_smarts("[#16]").unwrap(),
            SmartsGenome::from_smarts("[#6]").unwrap(),
            SmartsGenome::from_smarts("[#7]").unwrap(),
        ];

        let selected = select_screened_mutation_candidate(&evaluator, &candidates);

        assert_eq!(candidates[selected].smarts(), "[#7]");
    }

    #[test]
    fn optional_length_limit_rejects_over_budget_genomes_before_matching() {
        let by_length = EvolutionConfig::builder()
            .max_evaluation_smarts_len(4)
            .build()
            .unwrap();
        let genome = SmartsGenome::from_smarts("[#6]~[#7]").unwrap();

        assert_eq!(
            genome_evaluation_rejection(&by_length, &genome),
            Some(GenomeEvaluationRejection::SmartsLength {
                actual: genome.smarts_len(),
                max: 4
            })
        );

        let rejected = build_rejected_scored_genome(genome);
        assert_eq!(rejected.fitness, ObjectiveFitness::invalid());
        assert!(rejected.phenotype.is_empty());
    }
}
