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
use crate::genome::genome::SmartsGenome;
use crate::genome::seed::{SeedCorpus, SmartsGenomeBuilder};
use crate::operators::crossover::SmartsCrossover;
use crate::operators::mutation::SmartsMutation;

const RESET_POOL_SIZE: usize = 32;

/// One generic evolution task.
///
/// Experiment-specific code is expected to prepare folds, labels, and task
/// iteration elsewhere, then call into this library per task.
#[derive(Clone, Debug)]
pub struct EvolutionTask {
    pub task_id: String,
    pub positive_smiles: Vec<String>,
    pub folds: Vec<FoldData>,
}

/// Result of evolving a single task.
#[derive(Clone, Debug)]
pub struct TaskResult {
    pub task_id: String,
    pub best_smarts: String,
    pub best_mcc: f64,
    pub best_eval_time: Duration,
    pub best_score: i64,
    pub generations: u64,
}

trait EvaluationBackend {
    fn score_population(
        &mut self,
        population: Vec<SmartsGenome>,
    ) -> Result<Vec<(SmartsGenome, ObjectiveFitness)>, Box<dyn std::error::Error>>;
}

struct LocalEvaluationBackend {
    evaluator: SmartsEvaluator,
}

impl LocalEvaluationBackend {
    fn new(evaluator: SmartsEvaluator) -> Self {
        Self { evaluator }
    }
}

impl EvaluationBackend for LocalEvaluationBackend {
    fn score_population(
        &mut self,
        population: Vec<SmartsGenome>,
    ) -> Result<Vec<(SmartsGenome, ObjectiveFitness)>, Box<dyn std::error::Error>> {
        let mut scored = Vec::with_capacity(population.len());
        for genome in population {
            let fitness = self.evaluator.fitness_of(&genome);
            scored.push((genome, fitness));
        }
        Ok(scored)
    }
}

/// Evolve one generic task from already-prepared folds.
pub fn evolve_task(
    task: &EvolutionTask,
    config: &EvolutionConfig,
    seed_corpus: &SeedCorpus,
) -> Result<TaskResult, Box<dyn std::error::Error>> {
    if config.population_size == 0 {
        return Err("population_size must be greater than zero".into());
    }
    if task.folds.is_empty() {
        return Err("evolution task requires at least one fold".into());
    }
    let total_targets: usize = task.folds.iter().map(|fold| fold.targets.len()).sum();
    if total_targets == 0 {
        return Err(format!("task '{}' has no evaluation targets", task.task_id).into());
    }

    info!(
        "Evolving task '{}' over {} folds ({} evaluation targets)",
        task.task_id,
        task.folds.len(),
        total_targets
    );

    let evaluator = SmartsEvaluator::new(task.folds.clone())?;
    let mut evaluator_backend = LocalEvaluationBackend::new(evaluator);
    evolve_task_inner(
        &task.task_id,
        config,
        &mut evaluator_backend,
        &task.positive_smiles,
        seed_corpus,
    )
}

fn evolve_task_inner(
    task_id: &str,
    config: &EvolutionConfig,
    evaluator_backend: &mut impl EvaluationBackend,
    positive_smiles: &[String],
    seed_corpus: &SeedCorpus,
) -> Result<TaskResult, Box<dyn std::error::Error>> {
    let genome_builder =
        SmartsGenomeBuilder::with_seed_corpus(positive_smiles.to_vec(), seed_corpus.clone());
    let mut rng = rand::thread_rng();

    let mut population: Vec<SmartsGenome> = (0..config.population_size)
        .map(|i| genome_builder.build_genome(i, &mut rng))
        .collect();
    info!(
        "Built initial population for task '{}' ({} genomes)",
        task_id,
        population.len()
    );

    let crossover = SmartsCrossover::new(config.crossover_rate);
    let reset_pool = build_reset_pool(seed_corpus, &population, RESET_POOL_SIZE);
    let mutator = SmartsMutation::with_reset_pool(config.mutation_rate, reset_pool);

    let mut best_fitness = ObjectiveFitness::from_metrics(-1.0, Duration::ZERO);
    let mut best_smarts = String::from("[#6]");
    let mut best_mcc = 0.0f64;
    let mut best_eval_time = Duration::ZERO;
    let mut best_len = usize::MAX;
    let mut stagnation = 0u64;
    let mut fitness_cache: HashMap<String, ObjectiveFitness> = HashMap::new();

    for generation in 0..config.generation_limit {
        let (unique_population, duplicate_count) = deduplicate_population(&population);
        let unique_count = unique_population.len();
        let total_count = population.len();
        let average_token_len = average_token_len(&population);
        let mut cache_hits = 0usize;
        let mut fitness_by_smarts: HashMap<String, ObjectiveFitness> =
            HashMap::with_capacity(unique_count);
        let mut uncached_population = Vec::with_capacity(unique_count);

        for genome in unique_population {
            if let Some(&fitness) = fitness_cache.get(&genome.smarts_string) {
                fitness_by_smarts.insert(genome.smarts_string.clone(), fitness);
                cache_hits += 1;
            } else {
                uncached_population.push(genome);
            }
        }

        if !uncached_population.is_empty() {
            let unique_scored = evaluator_backend.score_population(uncached_population)?;
            for (genome, fitness) in unique_scored {
                fitness_cache.insert(genome.smarts_string.clone(), fitness);
                fitness_by_smarts.insert(genome.smarts_string, fitness);
            }
        }

        let mut scored = Vec::with_capacity(population.len());
        for genome in population {
            let fitness = *fitness_by_smarts
                .get(&genome.smarts_string)
                .ok_or_else(|| format!("missing fitness for SMARTS '{}'", genome.smarts_string))?;
            scored.push((genome, fitness));
        }

        scored.sort_by(compare_scored_genomes);
        let unique_scored = deduplicate_scored_population(&scored);

        let lead = &unique_scored[0];
        let lead_len = lead.0.tokens.len();
        let lead_mcc = lead.1.mcc();
        let lead_eval_time = lead.1.elapsed();
        if generation == 0
            || is_better_candidate(lead.1, &lead.0, best_fitness, best_len, &best_smarts)
        {
            best_fitness = lead.1;
            best_smarts = lead.0.smarts_string.clone();
            best_mcc = best_fitness.mcc();
            best_eval_time = best_fitness.elapsed();
            best_len = lead_len;
            stagnation = 0;
        } else {
            stagnation += 1;
        }

        debug!(
            "Task '{}' gen {}: {}",
            task_id,
            generation + 1,
            generation_progress_message(
                lead.1,
                best_fitness,
                unique_count,
                total_count,
                duplicate_count,
                cache_hits,
                lead_len,
                average_token_len,
            ),
        );

        if stagnation >= config.stagnation_limit {
            info!(
                "Task '{}' stagnated after {} generations; best MCC={:.3}, eval_ms={:.3}",
                task_id,
                generation + 1,
                best_mcc,
                duration_ms(best_eval_time)
            );
            return Ok(TaskResult {
                task_id: task_id.to_string(),
                best_smarts,
                best_mcc,
                best_eval_time,
                best_score: best_fitness.score,
                generations: generation + 1,
            });
        }

        let num_parents = ((config.population_size as f64 * config.selection_ratio).round()
            as usize)
            .max(2)
            .min(config.population_size.max(2));
        let parents = tournament_select(
            &unique_scored,
            num_parents,
            config.tournament_size,
            &mut rng,
        );

        let mut offspring = Vec::with_capacity(config.population_size);
        let mut pi = 0;
        while offspring.len() < config.population_size {
            let p1 = &parents[pi % parents.len()];
            let p2 = &parents[(pi + 1) % parents.len()];
            pi += 2;

            let children = crossover.crossover(vec![p1.clone(), p2.clone()], &mut rng);
            for child in children {
                let mutated = mutator.mutate(child, &mut rng);
                offspring.push(mutated);
                if offspring.len() >= config.population_size {
                    break;
                }
            }
        }

        let elite_count = config.elite_count.max(1).min(config.population_size);
        let immigrant_count = ((config.random_immigrant_ratio * config.population_size as f64)
            .round() as usize)
            .min(config.population_size.saturating_sub(elite_count));
        let immigrant_base = generation as usize * config.population_size;
        let mut immigrant_idx = 0usize;
        population = reinsert_population(
            &unique_scored,
            offspring,
            config.population_size,
            elite_count,
            immigrant_count,
            || {
                let genome = genome_builder.build_genome(immigrant_base + immigrant_idx, &mut rng);
                immigrant_idx += 1;
                genome
            },
        );

        let _ = lead_mcc;
        let _ = lead_eval_time;
    }

    Ok(TaskResult {
        task_id: task_id.to_string(),
        best_smarts,
        best_mcc,
        best_eval_time,
        best_score: best_fitness.score,
        generations: config.generation_limit,
    })
}

fn deduplicate_population(population: &[SmartsGenome]) -> (Vec<SmartsGenome>, usize) {
    let mut unique = Vec::with_capacity(population.len());
    let mut seen = HashSet::with_capacity(population.len());

    for genome in population {
        if seen.insert(genome.smarts_string.clone()) {
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
        if seen.insert(genome.smarts_string.clone()) {
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
    if seen.insert(genome.smarts_string.clone()) {
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
        if seen.insert(genome.smarts_string.clone()) {
            reset_pool.push(genome.clone());
        }
    }

    let (unique_population, _) = deduplicate_population(population);
    for genome in unique_population {
        if reset_pool.len() >= limit {
            break;
        }
        if seen.insert(genome.smarts_string.clone()) {
            reset_pool.push(genome);
        }
    }

    reset_pool
}

fn average_token_len(population: &[SmartsGenome]) -> f64 {
    if population.is_empty() {
        return 0.0;
    }

    let total_tokens: usize = population.iter().map(|genome| genome.tokens.len()).sum();
    total_tokens as f64 / population.len() as f64
}

fn generation_progress_message(
    lead: ObjectiveFitness,
    best: ObjectiveFitness,
    unique_count: usize,
    total_count: usize,
    duplicate_count: usize,
    cache_hits: usize,
    lead_len: usize,
    average_token_len: f64,
) -> String {
    format!(
        "lead={:.3}/{:.2}ms best={:.3}/{:.2}ms uniq={unique_count}/{total_count} dup={duplicate_count} cache={cache_hits}/{unique_count} len={lead_len} avg_len={average_token_len:.1}",
        lead.mcc(),
        duration_ms(lead.elapsed()),
        best.mcc(),
        duration_ms(best.elapsed()),
    )
}

fn compare_scored_genomes(
    left: &(SmartsGenome, ObjectiveFitness),
    right: &(SmartsGenome, ObjectiveFitness),
) -> Ordering {
    right
        .1
        .cmp(&left.1)
        .then_with(|| left.0.tokens.len().cmp(&right.0.tokens.len()))
        .then_with(|| left.0.smarts_string.cmp(&right.0.smarts_string))
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
            && (candidate.tokens.len() < best_len
                || (candidate.tokens.len() == best_len
                    && candidate.smarts_string.as_str() < best_smarts)))
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

    use smarts_validator::PreparedTarget;
    use smiles_parser::Smiles;

    use super::*;

    fn fitness(mcc: f64, elapsed_micros: u64) -> ObjectiveFitness {
        ObjectiveFitness::from_metrics(mcc, Duration::from_micros(elapsed_micros))
    }

    fn target(smiles: &str) -> PreparedTarget {
        PreparedTarget::new(Smiles::from_str(smiles).unwrap())
    }

    #[test]
    fn evolve_task_preserves_task_id() {
        let task = EvolutionTask {
            task_id: "leaf-1".to_string(),
            positive_smiles: vec!["CC".to_string()],
            folds: vec![FoldData {
                targets: vec![target("CC"), target("CN")],
                is_positive: vec![true, false],
            }],
        };
        let config = EvolutionConfig {
            population_size: 4,
            generation_limit: 1,
            stagnation_limit: 1,
            ..EvolutionConfig::default()
        };

        let result = evolve_task(&task, &config, &SeedCorpus::builtin()).unwrap();
        assert_eq!(result.task_id, "leaf-1");
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
        assert_eq!(unique[0].smarts_string, "[#6]");
        assert_eq!(unique[1].smarts_string, "[#7]");
        assert_eq!(unique[2].smarts_string, "[#8]");
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
        assert_eq!(unique[0].0.smarts_string, "[#6]");
        assert_eq!(unique[0].1, fitness(0.80, 400));
        assert_eq!(unique[1].0.smarts_string, "[#7]");
        assert_eq!(unique[2].0.smarts_string, "[#8]");
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

        let smarts: Vec<&str> = next_gen
            .iter()
            .map(|genome| genome.smarts_string.as_str())
            .collect();
        assert_eq!(smarts, vec!["[#6]", "[#7]", "[#8]", "[#16]", "[#9]"]);
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
            fitness(0.6621, 320),
            fitness(0.7314, 480),
            382,
            1024,
            642,
            208,
            16,
            12.4,
        );

        assert_eq!(
            message,
            "lead=0.662/0.32ms best=0.731/0.48ms uniq=382/1024 dup=642 cache=208/382 len=16 avg_len=12.4"
        );
    }
}
