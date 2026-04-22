use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;

use hashbrown::{HashMap, HashSet};
use log::{debug, info};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use super::config::EvolutionConfig;
use crate::fitness::evaluator::{FoldData, SmartsEvaluator};
use crate::fitness::objective::ObjectiveFitness;
use crate::genome::SmartsGenome;
use crate::genome::seed::{SeedCorpus, SmartsGenomeBuilder};
use crate::operators::crossover::SmartsCrossover;
use crate::operators::mutation::SmartsMutation;

const RESET_POOL_SIZE: usize = 32;

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
    lead_complexity: usize,
    average_complexity: f64,
}

#[derive(Clone, Copy)]
struct CachedFitness {
    fitness: ObjectiveFitness,
    stamp: u64,
}

struct FitnessCache {
    entries: HashMap<Arc<str>, CachedFitness>,
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

    fn get(&mut self, key: &Arc<str>) -> Option<ObjectiveFitness> {
        if self.capacity == 0 {
            return None;
        }

        let stamp = self.bump_stamp();
        let fitness = {
            let cached = self.entries.get_mut(key)?;
            cached.stamp = stamp;
            cached.fitness
        };
        self.access_order.push_back((key.clone(), stamp));
        self.compact_if_needed();
        Some(fitness)
    }

    fn insert(&mut self, key: Arc<str>, fitness: ObjectiveFitness) {
        if self.capacity == 0 {
            return;
        }

        let stamp = self.bump_stamp();
        self.entries
            .insert(key.clone(), CachedFitness { fitness, stamp });
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
}

/// Result of evolving a single task.
#[derive(Clone, Debug)]
pub struct TaskResult {
    task_id: String,
    best_smarts: String,
    best_mcc: f64,
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
    complexity: usize,
}

impl RankedSmarts {
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
    leaders: Vec<RankedSmarts>,
    unique_count: usize,
    total_count: usize,
    duplicate_count: usize,
    cache_hits: usize,
    lead_complexity: usize,
    average_complexity: f64,
    stagnation: u64,
}

struct ProgressSnapshot<'a> {
    task_id: &'a str,
    generation: u64,
    generation_limit: u64,
    status: EvolutionStatus,
    best: RankedSmarts,
    leaders: Vec<RankedSmarts>,
    stats: &'a GenerationStats,
    stagnation: u64,
}

impl EvolutionProgress {
    fn from_snapshot(snapshot: ProgressSnapshot<'_>) -> Self {
        Self {
            task_id: snapshot.task_id.to_string(),
            generation: snapshot.generation,
            generation_limit: snapshot.generation_limit,
            status: snapshot.status,
            best: snapshot.best,
            leaders: snapshot.leaders,
            unique_count: snapshot.stats.unique_count,
            total_count: snapshot.stats.total_count,
            duplicate_count: snapshot.stats.duplicate_count,
            cache_hits: snapshot.stats.cache_hits,
            lead_complexity: snapshot.stats.lead_complexity,
            average_complexity: snapshot.stats.average_complexity,
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

    pub fn lead_complexity(&self) -> usize {
        self.lead_complexity
    }

    pub fn average_complexity(&self) -> f64 {
        self.average_complexity
    }

    pub fn stagnation(&self) -> u64 {
        self.stagnation
    }
}

/// Evolve one binary task from already-prepared folds.
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
///     EvolutionConfig, EvolutionTask, FoldData, FoldSample, SeedCorpus, evolve_task,
/// };
/// use smarts_validator::PreparedTarget;
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
/// let seed_corpus = SeedCorpus::from_smarts(vec![
///     "[#6](=[#8])[#7]".to_string(),
///     "[#6]~[#7]".to_string(),
/// ])
/// .unwrap();
///
/// let result = evolve_task(&task, &config, &seed_corpus).unwrap();
/// assert!(!result.best_smarts().is_empty());
/// assert!(result.best_mcc().is_finite());
/// ```
pub fn evolve_task(
    task: &EvolutionTask,
    config: &EvolutionConfig,
    seed_corpus: &SeedCorpus,
) -> Result<TaskResult, EvolutionError> {
    evolve_task_with_progress(task, config, seed_corpus, 1, |_| {})
}

/// Evolve one binary task while emitting one snapshot after each scored generation.
pub fn evolve_task_with_progress(
    task: &EvolutionTask,
    config: &EvolutionConfig,
    seed_corpus: &SeedCorpus,
    leaderboard_size: usize,
    mut on_progress: impl FnMut(EvolutionProgress),
) -> Result<TaskResult, EvolutionError> {
    config.validate().map_err(EvolutionError::InvalidConfig)?;
    if task.folds().is_empty() {
        return Err(EvolutionError::EmptyFolds);
    }
    let total_targets: usize = task.folds().iter().map(FoldData::len).sum();
    if total_targets == 0 {
        return Err(EvolutionError::NoEvaluationTargets(
            task.task_id().to_string(),
        ));
    }

    info!(
        "Evolving task '{}' over {} folds ({} evaluation targets)",
        task.task_id(),
        task.folds().len(),
        total_targets
    );

    let evaluator = SmartsEvaluator::new(task.folds().to_vec());
    evolve_task_inner(
        task.task_id(),
        config,
        &evaluator,
        seed_corpus,
        leaderboard_size.max(1),
        &mut on_progress,
    )
}

fn evolve_task_inner(
    task_id: &str,
    config: &EvolutionConfig,
    evaluator: &SmartsEvaluator,
    seed_corpus: &SeedCorpus,
    leaderboard_size: usize,
    on_progress: &mut impl FnMut(EvolutionProgress),
) -> Result<TaskResult, EvolutionError> {
    let genome_builder = SmartsGenomeBuilder::new(seed_corpus.clone());
    let mut rng = build_rng(config);

    let mut population: Vec<SmartsGenome> = (0..config.population_size())
        .map(|i| genome_builder.build_genome(i, &mut rng))
        .collect();
    info!(
        "Built initial population for task '{}' ({} genomes)",
        task_id,
        population.len(),
    );

    let crossover = SmartsCrossover::new(config.crossover_rate());
    let reset_pool = build_reset_pool(seed_corpus, &population, RESET_POOL_SIZE);
    let mutator = SmartsMutation::with_reset_pool(config.mutation_rate(), reset_pool);

    let mut best_fitness = ObjectiveFitness::invalid();
    let mut best_smarts = String::from("[#6]");
    let mut best_text_len = usize::MAX;
    let mut stagnation = 0u64;
    let mut fitness_cache = FitnessCache::new(config.fitness_cache_capacity());
    let mut last_leaders = Vec::new();

    for generation in 0..config.generation_limit() {
        let total_count = population.len();
        let average_complexity = average_complexity(&population);
        let mut seen_population = HashSet::with_capacity(total_count);
        let mut duplicate_count = 0usize;
        let mut cache_hits = 0usize;
        let mut unique_population = Vec::with_capacity(total_count);

        for genome in population.drain(..) {
            let smarts = genome.smarts_shared().clone();
            if !seen_population.insert(smarts.clone()) {
                duplicate_count += 1;
                continue;
            }

            unique_population.push(genome);
        }
        let unique_count = seen_population.len();
        let mut unique_scored = Vec::with_capacity(unique_count);
        let mut uncached_genomes = Vec::new();

        for genome in unique_population {
            let smarts = genome.smarts_shared().clone();
            if let Some(fitness) = fitness_cache.get(&smarts) {
                cache_hits += 1;
                unique_scored.push((genome, fitness));
            } else {
                uncached_genomes.push(genome);
            }
        }

        let freshly_scored = evaluator.fitness_of_many(uncached_genomes);
        for (genome, fitness) in freshly_scored {
            fitness_cache.insert(genome.smarts_shared().clone(), fitness);
            unique_scored.push((genome, fitness));
        }

        unique_scored.sort_by(compare_scored_genomes);
        let leaders = leaderboard_from_scored(&unique_scored, leaderboard_size);

        let lead = &unique_scored[0];
        let lead_complexity = lead.0.complexity();
        if generation == 0
            || is_better_candidate(lead.1, &lead.0, best_fitness, best_text_len, &best_smarts)
        {
            best_fitness = lead.1;
            best_smarts = lead.0.smarts().to_string();
            best_text_len = lead.0.smarts().len();
            stagnation = 0;
        } else {
            stagnation += 1;
        }

        let stats = GenerationStats {
            unique_count,
            total_count,
            duplicate_count,
            cache_hits,
            lead_complexity,
            average_complexity,
        };

        debug!(
            "Task '{}' gen {}: {}",
            task_id,
            generation + 1,
            generation_progress_message(&stats, lead.1, best_fitness,),
        );

        let generations = generation + 1;
        let status = if stagnation >= config.stagnation_limit() {
            EvolutionStatus::Stagnated
        } else if generations == config.generation_limit() {
            EvolutionStatus::Completed
        } else {
            EvolutionStatus::Running
        };

        last_leaders = leaders.clone();
        on_progress(EvolutionProgress::from_snapshot(ProgressSnapshot {
            task_id,
            generation: generations,
            generation_limit: config.generation_limit(),
            status,
            best: RankedSmarts::new(&lead.0, lead.1),
            leaders,
            stats: &stats,
            stagnation,
        }));

        if status == EvolutionStatus::Stagnated {
            info!(
                "Task '{}' stagnated after {} generations; best MCC={:.3}",
                task_id,
                generations,
                best_fitness.mcc(),
            );
            return Ok(build_task_result(
                task_id,
                best_smarts,
                best_fitness,
                generations,
                last_leaders,
            ));
        }

        if status == EvolutionStatus::Completed {
            info!(
                "Task '{}' completed after {} generations; best MCC={:.3}",
                task_id,
                generations,
                best_fitness.mcc(),
            );
            return Ok(build_task_result(
                task_id,
                best_smarts,
                best_fitness,
                generations,
                last_leaders,
            ));
        }

        let num_parents = ((config.population_size() as f64 * config.selection_ratio()).round()
            as usize)
            .max(2)
            .min(config.population_size().max(2));
        let parents = tournament_select(
            &unique_scored,
            num_parents,
            config.tournament_size(),
            &mut rng,
        );

        let mut offspring = Vec::with_capacity(config.population_size());
        let mut pi = 0;
        while offspring.len() < config.population_size() {
            let p1 = &parents[pi % parents.len()];
            let p2 = &parents[(pi + 1) % parents.len()];
            pi += 2;

            let (child_a, child_b) = crossover.crossover_pair(p1, p2, &mut rng);
            offspring.push(mutator.mutate(child_a, &mut rng));
            if offspring.len() >= config.population_size() {
                break;
            }
            offspring.push(mutator.mutate(child_b, &mut rng));
        }

        let elite_count = config.elite_count().max(1).min(config.population_size());
        let immigrant_count = ((config.random_immigrant_ratio() * config.population_size() as f64)
            .round() as usize)
            .min(config.population_size().saturating_sub(elite_count));
        let immigrant_base = generation as usize * config.population_size();
        let mut immigrant_idx = 0usize;
        population = reinsert_population(
            &unique_scored,
            offspring,
            config.population_size(),
            elite_count,
            immigrant_count,
            || {
                let genome = genome_builder.build_genome(immigrant_base + immigrant_idx, &mut rng);
                immigrant_idx += 1;
                genome
            },
        );
    }

    Ok(build_task_result(
        task_id,
        best_smarts,
        best_fitness,
        config.generation_limit(),
        last_leaders,
    ))
}

fn build_task_result(
    task_id: &str,
    best_smarts: String,
    best_fitness: ObjectiveFitness,
    generations: u64,
    leaders: Vec<RankedSmarts>,
) -> TaskResult {
    TaskResult {
        task_id: task_id.to_string(),
        best_smarts,
        best_mcc: best_fitness.mcc(),
        generations,
        leaders,
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
    scored_unique: &[(SmartsGenome, ObjectiveFitness)],
    offspring: Vec<SmartsGenome>,
    population_size: usize,
    elite_count: usize,
    immigrant_count: usize,
    mut next_immigrant: impl FnMut() -> SmartsGenome,
) -> Vec<SmartsGenome> {
    let pre_immigrant_target = population_size.saturating_sub(immigrant_count);
    let mut next_gen = Vec::with_capacity(population_size);
    let mut seen: HashSet<Arc<str>> = HashSet::with_capacity(population_size);

    for (genome, _) in scored_unique.iter().take(elite_count) {
        insert_unique_genome(&mut next_gen, &mut seen, genome.clone());
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

fn average_complexity(population: &[SmartsGenome]) -> f64 {
    if population.is_empty() {
        return 0.0;
    }

    let total_complexity: usize = population.iter().map(SmartsGenome::complexity).sum();
    total_complexity as f64 / population.len() as f64
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
        "lead={:.3} best={:.3} uniq={}/{} dup={} cache={}/{} size={} avg_size={:.1}",
        lead.mcc(),
        best.mcc(),
        stats.unique_count,
        stats.total_count,
        stats.duplicate_count,
        stats.cache_hits,
        stats.unique_count,
        stats.lead_complexity,
        stats.average_complexity,
    )
}

fn compare_scored_genomes(
    left: &(SmartsGenome, ObjectiveFitness),
    right: &(SmartsGenome, ObjectiveFitness),
) -> Ordering {
    right
        .1
        .cmp(&left.1)
        .then_with(|| left.0.smarts().len().cmp(&right.0.smarts().len()))
        .then_with(|| left.0.smarts().cmp(right.0.smarts()))
}

fn is_better_candidate(
    candidate_fitness: ObjectiveFitness,
    candidate: &SmartsGenome,
    best_fitness: ObjectiveFitness,
    best_text_len: usize,
    best_smarts: &str,
) -> bool {
    candidate_fitness > best_fitness
        || (candidate_fitness == best_fitness
            && (candidate.smarts().len() < best_text_len
                || (candidate.smarts().len() == best_text_len && candidate.smarts() < best_smarts)))
}

fn tournament_select(
    scored: &[(SmartsGenome, ObjectiveFitness)],
    count: usize,
    tournament_size: usize,
    rng: &mut impl rand::Rng,
) -> Vec<SmartsGenome> {
    let mut parents = Vec::with_capacity(count);
    for _ in 0..count {
        let mut best_idx = rng.random_range(0..scored.len());
        for _ in 1..tournament_size {
            let idx = rng.random_range(0..scored.len());
            if compare_scored_genomes(&scored[idx], &scored[best_idx]).is_lt() {
                best_idx = idx;
            }
        }
        parents.push(scored[best_idx].0.clone());
    }
    parents
}

fn leaderboard_from_scored(
    scored: &[(SmartsGenome, ObjectiveFitness)],
    limit: usize,
) -> Vec<RankedSmarts> {
    scored
        .iter()
        .take(limit.max(1))
        .map(|(genome, fitness)| RankedSmarts::new(genome, *fitness))
        .collect()
}

#[cfg(test)]
mod regression_tests {
    use std::str::FromStr;
    use std::vec;

    use crate::fitness::evaluator::FoldSample;
    use smarts_validator::PreparedTarget;
    use smiles_parser::Smiles;

    use super::*;

    fn fitness(mcc: f64) -> ObjectiveFitness {
        ObjectiveFitness::from_mcc(mcc)
    }

    fn target(smiles: &str) -> PreparedTarget {
        PreparedTarget::new(Smiles::from_str(smiles).unwrap())
    }

    fn sample(smiles: &str, is_positive: bool) -> FoldSample {
        FoldSample::new(target(smiles), is_positive)
    }

    #[test]
    fn evolve_task_preserves_task_id() {
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

        let result = evolve_task(&task, &config, &SeedCorpus::builtin()).unwrap();
        assert_eq!(result.task_id(), "leaf-1");
    }

    #[test]
    fn evolve_task_rejects_missing_folds_and_targets() {
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
            evolve_task(&empty_task, &config, &SeedCorpus::builtin())
                .unwrap_err()
                .to_string(),
            "evolution task requires at least one fold"
        );
        assert_eq!(
            evolve_task(&no_target_task, &config, &SeedCorpus::builtin())
                .unwrap_err()
                .to_string(),
            "task 'no-targets' has no evaluation targets"
        );
    }

    #[test]
    fn evolve_task_runs_end_to_end_and_returns_valid_smarts() {
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
        let seed_corpus = SeedCorpus::from_smarts(vec![
            "[#6](=[#8])[#7]".to_string(),
            "[#6]~[#7]".to_string(),
            "[#7]".to_string(),
        ])
        .unwrap();

        let result = evolve_task(&task, &config, &seed_corpus).unwrap();

        assert_eq!(result.task_id(), "amide-vs-rest");
        assert!(result.generations() >= 1);
        assert!(result.generations() <= config.generation_limit());
        assert!(SmartsGenome::from_smarts(result.best_smarts()).is_ok());
        assert!(result.best_mcc().is_finite());
        assert!(!result.leaders().is_empty());
    }

    #[test]
    fn evolve_task_handles_static_population() {
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
        let seed_corpus = SeedCorpus::from_smarts(vec!["[#6]".to_string()]).unwrap();

        let result = evolve_task(&task, &config, &seed_corpus).unwrap();

        assert!(result.generations() >= 2);
        assert!(result.generations() <= config.generation_limit());
        assert!(SmartsGenome::from_smarts(result.best_smarts()).is_ok());
    }

    #[test]
    fn evolve_task_with_progress_emits_generation_snapshots_and_leaderboard() {
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
        let seed_corpus = SeedCorpus::from_smarts(vec![
            "[#6](=[#8])[#7]".to_string(),
            "[#6]~[#7]".to_string(),
            "[#7]".to_string(),
        ])
        .unwrap();
        let mut snapshots = Vec::new();

        let result = evolve_task_with_progress(&task, &config, &seed_corpus, 3, |snapshot| {
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
            (SmartsGenome::from_smarts("[#6]").unwrap(), fitness(0.90)),
            (SmartsGenome::from_smarts("[#7]").unwrap(), fitness(0.80)),
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
        let scored = vec![(SmartsGenome::from_smarts("[#6]").unwrap(), fitness(0.90))];
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
        let seed_corpus = SeedCorpus::from_smarts(vec![
            "[#6]".to_string(),
            "[#7]".to_string(),
            "[#8]".to_string(),
        ])
        .unwrap();
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
    fn shorter_textual_smarts_win_when_scores_tie() {
        let short = (SmartsGenome::from_smarts("[N]").unwrap(), fitness(0.75));
        let long = (SmartsGenome::from_smarts("[#7]").unwrap(), fitness(0.75));

        assert_eq!(short.0.complexity(), long.0.complexity());
        assert!(short.0.smarts().len() < long.0.smarts().len());

        assert!(compare_scored_genomes(&short, &long).is_lt());
        assert!(compare_scored_genomes(&long, &short).is_gt());
    }

    #[test]
    fn generation_progress_message_surfaces_diversity_stats() {
        let message = generation_progress_message(
            &GenerationStats {
                unique_count: 382,
                total_count: 1024,
                duplicate_count: 642,
                cache_hits: 208,
                lead_complexity: 16,
                average_complexity: 12.4,
            },
            fitness(0.6621),
            fitness(0.7314),
        );

        assert_eq!(
            message,
            "lead=0.662 best=0.731 uniq=382/1024 dup=642 cache=208/382 size=16 avg_size=12.4"
        );
    }

    #[test]
    fn average_complexity_handles_empty_population() {
        assert_eq!(average_complexity(&[]), 0.0);
    }

    #[test]
    fn fitness_cache_evicts_least_recently_used_entries() {
        let mut cache = FitnessCache::new(2);
        let a: Arc<str> = "[#6]".into();
        let b: Arc<str> = "[#7]".into();
        let c: Arc<str> = "[#8]".into();

        cache.insert(a.clone(), fitness(0.8));
        cache.insert(b.clone(), fitness(0.7));
        assert!(cache.get(&a).is_some());
        cache.insert(c.clone(), fitness(0.6));

        assert_eq!(cache.len(), 2);
        assert!(cache.get(&a).is_some());
        assert!(cache.get(&c).is_some());
        assert!(cache.get(&b).is_none());
    }
}
