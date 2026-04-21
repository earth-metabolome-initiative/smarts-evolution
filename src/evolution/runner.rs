use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::time::Duration;

use genevo::genetic::FitnessFunction;
use genevo::operator::{CrossoverOp, MutationOp};
use genevo::population::GenomeBuilder;
use log::{debug, info};

use super::config::EvolutionConfig;
use crate::fitness::evaluator::{FoldData, SmartsEvaluator};
use crate::fitness::objective::ObjectiveFitness;
use crate::genome::SmartsGenome;
use crate::genome::seed::{SeedCorpus, SmartsGenomeBuilder};
use crate::operators::crossover::SmartsCrossover;
use crate::operators::mutation::SmartsMutation;

const RESET_POOL_SIZE: usize = 32;

struct GenerationStats {
    unique_count: usize,
    total_count: usize,
    duplicate_count: usize,
    cache_hits: usize,
    lead_complexity: usize,
    average_complexity: f64,
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
    best_eval_time: Duration,
    best_score: i64,
    generations: u64,
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

    pub fn best_eval_time(&self) -> Duration {
        self.best_eval_time
    }

    pub fn best_score(&self) -> i64 {
        self.best_score
    }

    pub fn generations(&self) -> u64 {
        self.generations
    }
}

/// Evolve one binary task from already-prepared folds.
///
/// This is the main library entry point.
///
/// # Examples
///
/// ```
/// use std::str::FromStr;
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
///     .build()?;
/// let seed_corpus = SeedCorpus::from_smarts(vec![
///     "[#6](=[#8])[#7]".to_string(),
///     "[#6]~[#7]".to_string(),
/// ])?;
///
/// let result = evolve_task(&task, &config, &seed_corpus)?;
/// assert!(!result.best_smarts().is_empty());
/// assert!(result.best_mcc().is_finite());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn evolve_task(
    task: &EvolutionTask,
    config: &EvolutionConfig,
    seed_corpus: &SeedCorpus,
) -> Result<TaskResult, Box<dyn std::error::Error>> {
    config.validate()?;
    if task.folds().is_empty() {
        return Err("evolution task requires at least one fold".into());
    }
    let total_targets: usize = task.folds().iter().map(FoldData::len).sum();
    if total_targets == 0 {
        return Err(format!("task '{}' has no evaluation targets", task.task_id()).into());
    }

    info!(
        "Evolving task '{}' over {} folds ({} evaluation targets)",
        task.task_id(),
        task.folds().len(),
        total_targets
    );

    let evaluator = SmartsEvaluator::new(task.folds().to_vec());
    evolve_task_inner(task.task_id(), config, &evaluator, seed_corpus)
}

fn evolve_task_inner(
    task_id: &str,
    config: &EvolutionConfig,
    evaluator: &SmartsEvaluator,
    seed_corpus: &SeedCorpus,
) -> Result<TaskResult, Box<dyn std::error::Error>> {
    let genome_builder = SmartsGenomeBuilder::new(seed_corpus.clone());
    let mut rng = rand::thread_rng();

    let mut population: Vec<SmartsGenome> = (0..config.population_size())
        .map(|i| genome_builder.build_genome(i, &mut rng))
        .collect();
    info!(
        "Built initial population for task '{}' ({} genomes)",
        task_id,
        population.len()
    );

    let crossover = SmartsCrossover::new(config.crossover_rate());
    let reset_pool = build_reset_pool(seed_corpus, &population, RESET_POOL_SIZE);
    let mutator = SmartsMutation::with_reset_pool(config.mutation_rate(), reset_pool);

    let mut best_fitness = ObjectiveFitness::from_metrics(-1.0, Duration::ZERO);
    let mut best_smarts = String::from("[#6]");
    let mut best_len = usize::MAX;
    let mut stagnation = 0u64;
    let mut fitness_cache: HashMap<String, ObjectiveFitness> = HashMap::new();

    for generation in 0..config.generation_limit() {
        let (unique_population, duplicate_count) = deduplicate_population(&population);
        let unique_count = unique_population.len();
        let total_count = population.len();
        let average_complexity = average_complexity(&population);
        let mut cache_hits = 0usize;
        let mut fitness_by_smarts: HashMap<String, ObjectiveFitness> =
            HashMap::with_capacity(unique_count);
        let mut uncached_population = Vec::with_capacity(unique_count);

        for genome in unique_population {
            if let Some(&fitness) = fitness_cache.get(genome.smarts()) {
                fitness_by_smarts.insert(genome.smarts().to_string(), fitness);
                cache_hits += 1;
            } else {
                uncached_population.push(genome);
            }
        }

        if !uncached_population.is_empty() {
            for genome in uncached_population {
                let fitness = evaluator.fitness_of(&genome);
                fitness_cache.insert(genome.smarts().to_string(), fitness);
                fitness_by_smarts.insert(genome.smarts().to_string(), fitness);
            }
        }

        let mut scored = Vec::with_capacity(population.len());
        for genome in population {
            let fitness = *fitness_by_smarts
                .get(genome.smarts())
                .ok_or_else(|| format!("missing fitness for SMARTS '{}'", genome.smarts()))?;
            scored.push((genome, fitness));
        }

        scored.sort_by(compare_scored_genomes);
        let unique_scored = deduplicate_scored_population(&scored);

        let lead = &unique_scored[0];
        let lead_complexity = lead.0.complexity();
        if generation == 0
            || is_better_candidate(lead.1, &lead.0, best_fitness, best_len, &best_smarts)
        {
            best_fitness = lead.1;
            best_smarts = lead.0.smarts().to_string();
            best_len = lead_complexity;
            stagnation = 0;
        } else {
            stagnation += 1;
        }

        debug!(
            "Task '{}' gen {}: {}",
            task_id,
            generation + 1,
            generation_progress_message(
                &GenerationStats {
                    unique_count,
                    total_count,
                    duplicate_count,
                    cache_hits,
                    lead_complexity,
                    average_complexity,
                },
                lead.1,
                best_fitness,
            ),
        );

        if stagnation >= config.stagnation_limit() {
            info!(
                "Task '{}' stagnated after {} generations; best MCC={:.3}, eval_ms={:.3}",
                task_id,
                generation + 1,
                best_fitness.mcc(),
                duration_ms(best_fitness.elapsed())
            );
            return Ok(TaskResult {
                task_id: task_id.to_string(),
                best_smarts,
                best_mcc: best_fitness.mcc(),
                best_eval_time: best_fitness.elapsed(),
                best_score: best_fitness.score(),
                generations: generation + 1,
            });
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

            let children = crossover.crossover(vec![p1.clone(), p2.clone()], &mut rng);
            for child in children {
                let mutated = mutator.mutate(child, &mut rng);
                offspring.push(mutated);
                if offspring.len() >= config.population_size() {
                    break;
                }
            }
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

    Ok(TaskResult {
        task_id: task_id.to_string(),
        best_smarts,
        best_mcc: best_fitness.mcc(),
        best_eval_time: best_fitness.elapsed(),
        best_score: best_fitness.score(),
        generations: config.generation_limit(),
    })
}

fn deduplicate_population(population: &[SmartsGenome]) -> (Vec<SmartsGenome>, usize) {
    let mut unique = Vec::with_capacity(population.len());
    let mut seen = HashSet::with_capacity(population.len());

    for genome in population {
        if seen.insert(genome.smarts().to_string()) {
            unique.push(genome.clone());
        }
    }

    let duplicate_count = population.len().saturating_sub(unique.len());
    (unique, duplicate_count)
}

fn deduplicate_scored_population(
    scored: &[(SmartsGenome, ObjectiveFitness)],
) -> Vec<(SmartsGenome, ObjectiveFitness)> {
    let mut unique = Vec::with_capacity(scored.len());
    let mut seen = HashSet::with_capacity(scored.len());

    for (genome, score) in scored {
        if seen.insert(genome.smarts().to_string()) {
            unique.push((genome.clone(), *score));
        }
    }

    unique
}

fn insert_unique_genome(
    next_gen: &mut Vec<SmartsGenome>,
    seen: &mut HashSet<String>,
    genome: SmartsGenome,
) -> bool {
    if seen.insert(genome.smarts().to_string()) {
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
    let mut seen = HashSet::with_capacity(population_size);

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
    let mut seen = HashSet::with_capacity(limit);

    for genome in seed_corpus.entries() {
        if reset_pool.len() >= limit {
            return reset_pool;
        }
        if seen.insert(genome.smarts().to_string()) {
            reset_pool.push(genome.clone());
        }
    }

    let (unique_population, _) = deduplicate_population(population);
    for genome in unique_population {
        if reset_pool.len() >= limit {
            break;
        }
        if seen.insert(genome.smarts().to_string()) {
            reset_pool.push(genome);
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

fn generation_progress_message(
    stats: &GenerationStats,
    lead: ObjectiveFitness,
    best: ObjectiveFitness,
) -> String {
    format!(
        "lead={:.3}/{:.2}ms best={:.3}/{:.2}ms uniq={}/{} dup={} cache={}/{} size={} avg_size={:.1}",
        lead.mcc(),
        duration_ms(lead.elapsed()),
        best.mcc(),
        duration_ms(best.elapsed()),
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
        .then_with(|| left.0.complexity().cmp(&right.0.complexity()))
        .then_with(|| left.0.smarts().cmp(right.0.smarts()))
}

fn is_better_candidate(
    candidate_fitness: ObjectiveFitness,
    candidate: &SmartsGenome,
    best_fitness: ObjectiveFitness,
    best_len: usize,
    best_smarts: &str,
) -> bool {
    candidate_fitness > best_fitness
        || (candidate_fitness == best_fitness
            && (candidate.complexity() < best_len
                || (candidate.complexity() == best_len && candidate.smarts() < best_smarts)))
}

fn tournament_select(
    scored: &[(SmartsGenome, ObjectiveFitness)],
    count: usize,
    tournament_size: usize,
    rng: &mut impl rand::Rng,
) -> Vec<SmartsGenome> {
    let mut parents = Vec::with_capacity(count);
    for _ in 0..count {
        let mut best_idx = rng.gen_range(0..scored.len());
        for _ in 1..tournament_size {
            let idx = rng.gen_range(0..scored.len());
            if compare_scored_genomes(&scored[idx], &scored[best_idx]).is_lt() {
                best_idx = idx;
            }
        }
        parents.push(scored[best_idx].0.clone());
    }
    parents
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod regression_tests {
    use std::str::FromStr;

    use crate::fitness::evaluator::FoldSample;
    use smarts_validator::PreparedTarget;
    use smiles_parser::Smiles;

    use super::*;

    fn fitness(mcc: f64, elapsed_micros: u64) -> ObjectiveFitness {
        ObjectiveFitness::from_metrics(mcc, Duration::from_micros(elapsed_micros))
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
    }

    #[test]
    fn evolve_task_returns_early_when_search_stagnates() {
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
        assert!(result.generations() < config.generation_limit());
        assert!(SmartsGenome::from_smarts(result.best_smarts()).is_ok());
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
    fn deduplicate_scored_population_keeps_first_best_entry_per_smarts() {
        let scored = vec![
            (
                SmartsGenome::from_smarts("[#6]").unwrap(),
                fitness(0.80, 400),
            ),
            (
                SmartsGenome::from_smarts("[#7]").unwrap(),
                fitness(0.60, 400),
            ),
            (
                SmartsGenome::from_smarts("[#6]").unwrap(),
                fitness(0.20, 400),
            ),
            (
                SmartsGenome::from_smarts("[#8]").unwrap(),
                fitness(0.10, 400),
            ),
        ];

        let unique = deduplicate_scored_population(&scored);

        assert_eq!(unique.len(), 3);
        assert_eq!(unique[0].0.smarts(), "[#6]");
        assert_eq!(unique[0].1, fitness(0.80, 400));
        assert_eq!(unique[1].0.smarts(), "[#7]");
        assert_eq!(unique[2].0.smarts(), "[#8]");
    }

    #[test]
    fn reinsert_population_prefers_unique_offspring_and_immigrants() {
        let scored = vec![
            (
                SmartsGenome::from_smarts("[#6]").unwrap(),
                fitness(0.90, 500),
            ),
            (
                SmartsGenome::from_smarts("[#7]").unwrap(),
                fitness(0.80, 500),
            ),
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
        let scored = vec![(
            SmartsGenome::from_smarts("[#6]").unwrap(),
            fitness(0.90, 500),
        )];
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
    fn shorter_smarts_win_when_scores_tie() {
        let short = (
            SmartsGenome::from_smarts("[#6]").unwrap(),
            fitness(0.75, 250),
        );
        let long = (
            SmartsGenome::from_smarts("[#6]~[#7]").unwrap(),
            fitness(0.75, 250),
        );

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
            fitness(0.6621, 320),
            fitness(0.7314, 480),
        );

        assert_eq!(
            message,
            "lead=0.662/0.32ms best=0.731/0.48ms uniq=382/1024 dup=642 cache=208/382 size=16 avg_size=12.4"
        );
    }

    #[test]
    fn average_complexity_handles_empty_population() {
        assert_eq!(average_complexity(&[]), 0.0);
    }
}
