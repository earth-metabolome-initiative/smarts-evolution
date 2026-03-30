use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use genevo::operator::{CrossoverOp, MutationOp};
use genevo::population::GenomeBuilder;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::info;

use crate::genome::genome::SmartsGenome;

use super::checkpoint::{FullCheckpoint, NodeCheckpoint};
use super::config::EvolutionConfig;
use crate::data::compound::Compound;
use crate::evolution::process_pool::{
    ProcessEvaluationBackend, WorkerFoldPayload, WorkerNodeContext,
};
#[cfg(test)]
use crate::fitness::evaluator::FoldData;
#[cfg(test)]
use crate::fitness::evaluator::SmartsEvaluator;
use crate::fitness::mcc::MccFitness;
use crate::genome::seed::SmartsGenomeBuilder;
use crate::operators::crossover::SmartsCrossover;
use crate::operators::mutation::SmartsMutation;
use crate::rdkit_substruct_library::SubstructLibraryIndex;
use crate::taxonomy::dag::TaxonomyDag;
use crate::validation::splits::FoldSplits;
#[cfg(test)]
use genevo::genetic::FitnessFunction;

const GENERATION_STEPS: u64 = 4;
const RESET_POOL_SIZE: usize = 32;

/// Result of evolving a single node.
pub struct NodeResult {
    pub node_id: usize,
    pub best_smarts: String,
    pub best_mcc: f64,
    pub generations: u64,
}

trait EvaluationBackend {
    fn score_population(
        &mut self,
        population: Vec<SmartsGenome>,
        step_pb: &ProgressBar,
    ) -> Result<Vec<(SmartsGenome, i64)>, Box<dyn std::error::Error>>;
}

#[cfg(test)]
struct LocalEvaluationBackend {
    evaluator: SmartsEvaluator,
}

#[cfg(test)]
impl LocalEvaluationBackend {
    fn new(evaluator: SmartsEvaluator) -> Self {
        Self { evaluator }
    }
}

#[cfg(test)]
impl EvaluationBackend for LocalEvaluationBackend {
    fn score_population(
        &mut self,
        population: Vec<SmartsGenome>,
        step_pb: &ProgressBar,
    ) -> Result<Vec<(SmartsGenome, i64)>, Box<dyn std::error::Error>> {
        let mut scored = Vec::with_capacity(population.len());
        for genome in population {
            let score = self.evaluator.fitness_of(&genome).score;
            scored.push((genome, score));
            step_pb.inc(1);
        }
        Ok(scored)
    }
}

impl EvaluationBackend for ProcessEvaluationBackend {
    fn score_population(
        &mut self,
        population: Vec<SmartsGenome>,
        step_pb: &ProgressBar,
    ) -> Result<Vec<(SmartsGenome, i64)>, Box<dyn std::error::Error>> {
        ProcessEvaluationBackend::score_population(self, population, step_pb)
    }
}

/// Run evolution for all eligible nodes in the taxonomy DAG.
pub fn run_evolution(
    compounds: &[Compound],
    dag: &TaxonomyDag,
    splits: &FoldSplits,
    config: &EvolutionConfig,
    checkpoint: &mut FullCheckpoint,
) -> Result<Vec<NodeResult>, Box<dyn std::error::Error>> {
    let valid_compound_count = compounds.iter().filter(|compound| compound.parsed).count();
    let startup_index_pb = startup_progress_bar(
        "startup: building global candidate index",
        valid_compound_count as u64,
    );
    let mut candidate_filter = SubstructLibraryIndex::new()?;
    let mut valid_compound_indices = Vec::new();
    for (compound_idx, compound) in compounds.iter().enumerate() {
        if compound.parsed {
            candidate_filter.add_smiles(&compound.smiles, true)?;
            valid_compound_indices.push(compound_idx);
            startup_index_pb.inc(1);
        }
    }
    startup_index_pb.finish_with_message(format!(
        "startup: global candidate index ready ({valid_compound_count} compounds)"
    ));
    let all_valid_compounds: HashSet<usize> = valid_compound_indices.iter().copied().collect();

    // Create checkpoint dir
    std::fs::create_dir_all(&config.checkpoint_dir)?;

    let mut results = Vec::new();
    let mut evolved_smarts: HashMap<usize, String> = HashMap::new();
    let worker_processes = config
        .worker_processes
        .max(1)
        .min(config.population_size.max(1));
    info!("Starting {worker_processes} evaluator worker processes...");
    let startup_workers_pb = startup_progress_bar(
        "startup: spawning worker processes",
        worker_processes as u64,
    );
    let mut evaluator_backend =
        ProcessEvaluationBackend::spawn_with_progress(worker_processes, Some(&startup_workers_pb))?;
    startup_workers_pb.finish_with_message(format!(
        "startup: worker pool ready ({worker_processes} workers)"
    ));

    // Load already-evolved SMARTS from checkpoint
    for (nid, nc) in &checkpoint.nodes {
        evolved_smarts.insert(*nid, nc.best_smarts.clone());
    }

    // Count eligible nodes for the progress bar
    let eligible_count = dag
        .nodes
        .iter()
        .filter(|n| n.compound_indices.len() >= 20)
        .count();

    let mp = MultiProgress::new();
    let node_style = ProgressStyle::with_template(
        "{prefix} [{bar:30.cyan/dim}] {pos}/{len} nodes ({eta} remaining)",
    )
    .unwrap()
    .progress_chars("=> ");
    let node_pb = mp.add(ProgressBar::new(eligible_count as u64));
    node_pb.set_style(node_style);
    node_pb.set_prefix("DAG");

    let gen_pb = mp.add(ProgressBar::new(0));
    gen_pb.set_style(
        ProgressStyle::with_template(
            "  {prefix} [{bar:25.green/dim}] gen {pos}/{len} | best MCC={msg}",
        )
        .unwrap()
        .progress_chars("=> "),
    );

    let step_pb = mp.add(ProgressBar::new(GENERATION_STEPS));
    step_pb.set_style(
        ProgressStyle::with_template("  {prefix} [{bar:25.yellow/dim}] {msg} {pos}/{len}")
            .unwrap()
            .progress_chars("=> "),
    );

    // Process nodes in topological order
    let total_nodes = dag.topological_order.len();
    for (order_idx, &node_id) in dag.topological_order.iter().enumerate() {
        let node = &dag.nodes[node_id];

        // Skip nodes below threshold
        if node.compound_indices.len() < 20 {
            continue;
        }

        // Skip already-evolved nodes (from checkpoint)
        if evolved_smarts.contains_key(&node_id) {
            node_pb.inc(1);
            continue;
        }

        node_pb.set_message(format!("{}", node.name));
        gen_pb.set_prefix(format!("{}", node.name));
        gen_pb.set_length(config.generation_limit);
        gen_pb.set_position(0);
        gen_pb.set_message("--");
        step_pb.set_prefix(format!("{}", node.name));
        step_pb.set_length(GENERATION_STEPS);
        step_pb.set_position(0);
        step_pb.set_message("waiting");

        info!(
            "[{}/{}] Evolving node: {} (level={}, compounds={})",
            order_idx + 1,
            total_nodes,
            node.name,
            node.level,
            node.compound_indices.len()
        );

        // Determine candidate set: if parent has evolved SMARTS, filter to matching molecules
        let candidate_set = determine_candidate_set(
            node_id,
            dag,
            &evolved_smarts,
            &all_valid_compounds,
            &candidate_filter,
            &valid_compound_indices,
            Some(&step_pb),
        );

        let positive_set: HashSet<usize> = node.compound_indices.iter().copied().collect();

        // Build fold data for the evaluator
        begin_node_setup_step(
            &step_pb,
            "building fold payloads",
            candidate_set.len() as u64,
        );
        let fold_payloads =
            build_fold_payloads(compounds, splits, &positive_set, &candidate_set, &step_pb);
        complete_generation_step(&step_pb);

        // Check we have enough data in folds
        let total_test: usize = fold_payloads.iter().map(|fold| fold.smiles.len()).sum();
        if total_test < 10 {
            info!("  Skipping: insufficient test data ({total_test} molecules across folds)");
            continue;
        }

        // Collect positive SMILES for seeding
        let positive_smiles: Vec<String> = node
            .compound_indices
            .iter()
            .take(200) // Don't need all of them for seeding
            .map(|&ci| compounds[ci].smiles.clone())
            .collect();

        evaluator_backend.set_node_context_with_progress(
            WorkerNodeContext {
                folds: fold_payloads,
            },
            Some(&step_pb),
        )?;
        complete_generation_step(&step_pb);

        let result = evolve_node(
            node_id,
            config,
            &mut evaluator_backend,
            positive_smiles,
            &gen_pb,
            &step_pb,
        )?;

        gen_pb.set_position(result.generations);
        gen_pb.set_message(format!("{:.3}", result.best_mcc));
        step_pb.set_position(GENERATION_STEPS);
        step_pb.set_message("generation phases complete");
        node_pb.inc(1);

        info!(
            "  Best: {} (MCC={:.3}), {} gens",
            result.best_smarts, result.best_mcc, result.generations
        );

        evolved_smarts.insert(node_id, result.best_smarts.clone());

        // Save to checkpoint
        checkpoint.nodes.insert(
            node_id,
            NodeCheckpoint {
                node_id,
                node_name: node.name.clone(),
                level: node.level,
                generation: result.generations,
                best_smarts: result.best_smarts.clone(),
                best_mcc: result.best_mcc,
                population: Vec::new(),
            },
        );

        // Periodic checkpoint save
        if (order_idx + 1) % 5 == 0 || order_idx + 1 == total_nodes {
            let path = config.checkpoint_dir.join("checkpoint.json");
            checkpoint.save(&path)?;
            info!("  Checkpoint saved to {}", path.display());
        }

        results.push(result);
    }

    // Final checkpoint
    let path = config.checkpoint_dir.join("checkpoint.json");
    checkpoint.save(&path)?;

    gen_pb.finish_and_clear();
    step_pb.finish_and_clear();
    node_pb.finish_with_message("done");

    info!("Final checkpoint saved to {}", path.display());

    Ok(results)
}

/// Determine the set of candidate compound indices for a node.
///
/// If the node has a parent with an evolved SMARTS, only include compounds
/// that match the parent's pattern. Otherwise, use all compounds.
fn determine_candidate_set(
    node_id: usize,
    dag: &TaxonomyDag,
    evolved_smarts: &HashMap<usize, String>,
    all_valid_compounds: &HashSet<usize>,
    candidate_filter: &SubstructLibraryIndex,
    valid_compound_indices: &[usize],
    progress_pb: Option<&ProgressBar>,
) -> HashSet<usize> {
    let node = &dag.nodes[node_id];

    // Find parent nodes (nodes at level-1 that have an edge to this node)
    if node.level == 0 {
        // Root level: all compounds are candidates
        return all_valid_compounds.clone();
    }

    // Check if any parent has an evolved SMARTS
    let parent_nodes: Vec<usize> = dag
        .parents_of(node_id)
        .iter()
        .copied()
        .filter(|parent_id| {
            dag.nodes[*parent_id].level == node.level - 1 && evolved_smarts.contains_key(parent_id)
        })
        .collect();

    if parent_nodes.is_empty() {
        return all_valid_compounds.clone();
    }

    // Filter: compound must match at least one actual parent's SMARTS
    if let Some(pb) = progress_pb {
        begin_node_setup_step(pb, "filtering parent matches", parent_nodes.len() as u64);
    }
    let mut candidates = HashSet::new();

    for &parent_id in &parent_nodes {
        if let Some(smarts) = evolved_smarts.get(&parent_id) {
            if let Some(match_indices) = candidate_filter.positive_matches(smarts, 1) {
                for match_idx in match_indices {
                    if let Some(&compound_idx) = valid_compound_indices.get(match_idx) {
                        candidates.insert(compound_idx);
                    }
                }
            }
        }
        if let Some(pb) = progress_pb {
            pb.inc(1);
        }
    }
    candidates
}

/// Build per-fold compound payloads for worker evaluation.
fn build_fold_payloads(
    compounds: &[Compound],
    splits: &FoldSplits,
    positive_set: &HashSet<usize>,
    candidate_set: &HashSet<usize>,
    progress_pb: &ProgressBar,
) -> Vec<WorkerFoldPayload> {
    (0..splits.k)
        .map(|fold_idx| {
            let (_, _, test_indices, test_labels) =
                splits.train_test_for_node(fold_idx, positive_set, candidate_set);

            let mut smiles = Vec::new();
            let mut is_positive = Vec::new();

            for (i, &ci) in test_indices.iter().enumerate() {
                smiles.push(compounds[ci].smiles.clone());
                is_positive.push(test_labels[i]);
                progress_pb.inc(1);
            }

            WorkerFoldPayload {
                smiles,
                is_positive,
            }
        })
        .collect()
}

/// Run a custom GA loop for a single node.
///
/// We don't use genevo's Simulator because it uses rayon internally for
/// fitness evaluation and breeding, which causes segfaults with RDKit's C++ FFI.
/// Instead, we run a simple sequential GA: evaluate → select → breed → mutate → reinsert.
fn evolve_node(
    node_id: usize,
    config: &EvolutionConfig,
    evaluator_backend: &mut impl EvaluationBackend,
    positive_smiles: Vec<String>,
    gen_pb: &ProgressBar,
    step_pb: &ProgressBar,
) -> Result<NodeResult, Box<dyn std::error::Error>> {
    let genome_builder = SmartsGenomeBuilder::new(positive_smiles);
    let mut rng = rand::thread_rng();

    // Build initial population
    info!(
        "  Building initial population ({})...",
        config.population_size
    );
    let mut population: Vec<SmartsGenome> = (0..config.population_size)
        .map(|i| genome_builder.build_genome(i, &mut rng))
        .collect();
    info!(
        "  Population built. First SMARTS: {}",
        population[0].smarts_string
    );

    let crossover = SmartsCrossover::new(config.crossover_rate);
    let reset_pool = build_reset_pool(&population, RESET_POOL_SIZE);
    let mutator = SmartsMutation::with_reset_pool(config.mutation_rate, reset_pool);

    let mut best_score: i64 = 0;
    let mut best_smarts = String::from("[#6]");
    let mut best_mcc = 0.0f64;
    let mut best_len = usize::MAX;
    let mut stagnation = 0u64;
    let mut fitness_cache: HashMap<String, i64> = HashMap::new();

    for generation in 0..config.generation_limit {
        let (unique_population, duplicate_count) = deduplicate_population(&population);
        let unique_count = unique_population.len();
        let total_count = population.len();
        let average_token_len = average_token_len(&population);
        let mut cache_hits = 0usize;
        let mut score_by_smarts: HashMap<String, i64> = HashMap::with_capacity(unique_count);
        let mut uncached_population = Vec::with_capacity(unique_count);
        for genome in unique_population {
            if let Some(&score) = fitness_cache.get(&genome.smarts_string) {
                score_by_smarts.insert(genome.smarts_string.clone(), score);
                cache_hits += 1;
            } else {
                uncached_population.push(genome);
            }
        }
        begin_generation_step(
            step_pb,
            generation,
            "evaluate",
            uncached_population.len() as u64,
        );

        // 1. Evaluate fitness in worker processes, reusing per-node cached scores.
        if !uncached_population.is_empty() {
            let unique_scored = evaluator_backend.score_population(uncached_population, step_pb)?;
            for (genome, score) in unique_scored {
                fitness_cache.insert(genome.smarts_string.clone(), score);
                score_by_smarts.insert(genome.smarts_string, score);
            }
        }
        let mut scored = Vec::with_capacity(population.len());
        for genome in population {
            let score = *score_by_smarts
                .get(&genome.smarts_string)
                .ok_or_else(|| format!("missing score for SMARTS '{}'", genome.smarts_string))?;
            scored.push((genome, score));
        }
        complete_generation_step(step_pb);

        // 2. Sort by fitness descending, preferring shorter SMARTS on ties.
        scored.sort_by(compare_scored_genomes);

        // Track best MCC for the top-scoring genome
        let lead = &scored[0];
        let lead_len = lead.0.tokens.len();
        let lead_mcc = MccFitness::to_mcc(lead.1);
        if generation == 0
            || is_better_candidate(lead.1, &lead.0, best_score, best_len, &best_smarts)
        {
            best_score = lead.1;
            best_smarts = lead.0.smarts_string.clone();
            best_mcc = MccFitness::to_mcc(best_score);
            best_len = lead_len;
            stagnation = 0;
        } else {
            stagnation += 1;
        }

        gen_pb.set_position(generation);
        gen_pb.set_message(format!("{best_mcc:.3}"));

        info!(
            "    Gen {}: lead_score={}, lead_mcc={lead_mcc:.3}, global_mcc={best_mcc:.3}, unique={unique_count}/{total_count}, duplicates={duplicate_count}, avg_len={average_token_len:.1}, best_len={lead_len}, cache_hits={cache_hits}/{unique_count}, smarts={}",
            generation + 1,
            lead.1,
            lead.0.smarts_string,
        );

        if stagnation >= config.stagnation_limit {
            info!("    Stagnation after {generation} generations");
            return Ok(NodeResult {
                node_id,
                best_smarts,
                best_mcc,
                generations: generation,
            });
        }

        // 3. Selection: tournament selection
        let num_parents = ((config.population_size as f64 * config.selection_ratio).round()
            as usize)
            .max(2)
            .min(config.population_size.max(2));
        begin_generation_step(step_pb, generation, "select", num_parents as u64);
        let parents = tournament_select(
            &scored,
            num_parents,
            config.tournament_size,
            &mut rng,
            step_pb,
        );
        complete_generation_step(step_pb);

        // 4. Crossover + Mutation → offspring
        begin_generation_step(step_pb, generation, "breed", config.population_size as u64);
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
                step_pb.inc(1);
                if offspring.len() >= config.population_size {
                    break;
                }
            }
        }
        complete_generation_step(step_pb);

        // 5. Elitist reinsertion: keep a small fixed elite set, then offspring, then immigrants.
        begin_generation_step(
            step_pb,
            generation,
            "reinsert",
            config.population_size as u64,
        );
        let elite_count = config.elite_count.max(1).min(config.population_size);
        let immigrant_count = ((config.random_immigrant_ratio * config.population_size as f64)
            .round() as usize)
            .min(config.population_size.saturating_sub(elite_count));
        let offspring_target = config.population_size - elite_count - immigrant_count;
        let mut next_gen: Vec<SmartsGenome> = Vec::with_capacity(config.population_size);
        for (g, _) in scored.iter().take(elite_count) {
            next_gen.push(g.clone());
            step_pb.inc(1);
        }
        for genome in offspring.into_iter().take(offspring_target) {
            next_gen.push(genome);
            step_pb.inc(1);
        }
        let immigrant_base = generation as usize * config.population_size;
        for immigrant_idx in 0..immigrant_count {
            next_gen.push(genome_builder.build_genome(immigrant_base + immigrant_idx, &mut rng));
            step_pb.inc(1);
        }
        complete_generation_step(step_pb);

        population = next_gen;
    }

    Ok(NodeResult {
        node_id,
        best_smarts,
        best_mcc,
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

fn build_reset_pool(population: &[SmartsGenome], limit: usize) -> Vec<SmartsGenome> {
    let (unique, _) = deduplicate_population(population);
    unique.into_iter().take(limit).collect()
}

fn average_token_len(population: &[SmartsGenome]) -> f64 {
    if population.is_empty() {
        return 0.0;
    }

    let total_tokens: usize = population.iter().map(|genome| genome.tokens.len()).sum();
    total_tokens as f64 / population.len() as f64
}

fn compare_scored_genomes(left: &(SmartsGenome, i64), right: &(SmartsGenome, i64)) -> Ordering {
    right
        .1
        .cmp(&left.1)
        .then_with(|| left.0.tokens.len().cmp(&right.0.tokens.len()))
        .then_with(|| left.0.smarts_string.cmp(&right.0.smarts_string))
}

fn is_better_candidate(
    candidate_score: i64,
    candidate: &SmartsGenome,
    best_score: i64,
    best_len: usize,
    best_smarts: &str,
) -> bool {
    candidate_score > best_score
        || (candidate_score == best_score
            && (candidate.tokens.len() < best_len
                || (candidate.tokens.len() == best_len
                    && candidate.smarts_string.as_str() < best_smarts)))
}

/// Tournament selection: pick `count` parents via tournaments of `size`.
fn tournament_select(
    scored: &[(SmartsGenome, i64)],
    count: usize,
    tournament_size: usize,
    rng: &mut impl rand::Rng,
    step_pb: &ProgressBar,
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
        step_pb.inc(1);
    }
    parents
}

fn begin_generation_step(step_pb: &ProgressBar, generation: u64, step_name: &str, length: u64) {
    step_pb.set_length(length.max(1));
    step_pb.set_position(0);
    step_pb.set_message(format!("gen {}: {step_name}", generation + 1));
}

fn startup_progress_bar(message: &str, length: u64) -> ProgressBar {
    let pb = ProgressBar::new(length);
    pb.set_style(
        ProgressStyle::with_template(
            "{msg} [{bar:30.blue/dim}] {pos}/{len} items ({eta} remaining)",
        )
        .unwrap()
        .progress_chars("=> "),
    );
    pb.set_message(message.to_string());
    pb
}

fn begin_node_setup_step(step_pb: &ProgressBar, step_name: &str, length: u64) {
    step_pb.set_length(length.max(1));
    step_pb.set_position(0);
    step_pb.set_message(format!("setup: {step_name}"));
}

fn complete_generation_step(step_pb: &ProgressBar) {
    if let Some(length) = step_pb.length() {
        step_pb.set_position(length);
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use indicatif::ProgressBar;

    use crate::data::rdkit_lock::with_rdkit_lock;
    use crate::taxonomy::builder;

    fn make_compound(cid: u64, smiles: &str, labels: &[&[&str]]) -> Compound {
        let parsed = with_rdkit_lock(|| rdkit::ROMol::from_smiles(smiles).is_ok());
        let labels = labels
            .iter()
            .map(|level| level.iter().map(|label| (*label).to_string()).collect())
            .collect();
        Compound {
            cid,
            smiles: smiles.to_string(),
            parsed,
            labels,
        }
    }

    fn sample_dataset() -> (Vec<Compound>, TaxonomyDag) {
        let compounds = vec![
            make_compound(1, "CC", &[&["A"], &["A1"]]),
            make_compound(2, "CN", &[&["A"], &["A2"]]),
            make_compound(3, "O", &[&["B"], &["B1"]]),
            make_compound(4, "N", &[&["B"], &["B2"]]),
        ];
        let dag = builder::build_dag(&compounds, &["coarse", "fine"]).unwrap();
        (compounds, dag)
    }

    #[test]
    fn candidate_set_uses_actual_parents_only() {
        let (compounds, dag) = sample_dataset();
        let mut candidate_filter = SubstructLibraryIndex::new().unwrap();
        let mut valid_compound_indices = Vec::new();
        for (compound_idx, compound) in compounds.iter().enumerate() {
            if compound.parsed {
                candidate_filter.add_smiles(&compound.smiles, true).unwrap();
                valid_compound_indices.push(compound_idx);
            }
        }
        let all_valid_compounds: HashSet<usize> = valid_compound_indices.iter().copied().collect();

        let mut evolved_smarts = HashMap::new();
        let a_id = *dag.name_to_id.get(&(0, "A".to_string())).unwrap();
        let b_id = *dag.name_to_id.get(&(0, "B".to_string())).unwrap();
        let a1_id = *dag.name_to_id.get(&(1, "A1".to_string())).unwrap();
        evolved_smarts.insert(a_id, "[#6]".to_string());
        evolved_smarts.insert(b_id, "[#8]".to_string());

        let candidates = determine_candidate_set(
            a1_id,
            &dag,
            &evolved_smarts,
            &all_valid_compounds,
            &candidate_filter,
            &valid_compound_indices,
            None,
        );

        assert!(candidates.contains(&0));
        assert!(candidates.contains(&1));
        assert!(
            !candidates.contains(&2),
            "candidate set for A1 should not include compounds matched only by unrelated parent B"
        );
    }

    #[test]
    fn evolve_node_preserves_node_id() {
        let (compounds, dag) = sample_dataset();
        let predicted_node = *dag.name_to_id.get(&(1, "A1".to_string())).unwrap();
        let folds = vec![FoldData {
            smiles: vec![compounds[0].smiles.clone(), compounds[1].smiles.clone()],
            is_positive: vec![true, false],
        }];
        let evaluator = SmartsEvaluator::new(folds).unwrap();
        let config = EvolutionConfig {
            population_size: 4,
            generation_limit: 1,
            stagnation_limit: 1,
            ..EvolutionConfig::default()
        };
        let mut evaluator_backend = LocalEvaluationBackend::new(evaluator);

        let result = evolve_node(
            predicted_node,
            &config,
            &mut evaluator_backend,
            vec!["CC".to_string()],
            &ProgressBar::hidden(),
            &ProgressBar::hidden(),
        )
        .unwrap();

        assert_eq!(result.node_id, predicted_node);
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
    fn shorter_smarts_win_when_scores_tie() {
        let short = (SmartsGenome::from_smarts("[#6]").unwrap(), 123);
        let long = (SmartsGenome::from_smarts("[#6]~[#7]").unwrap(), 123);

        assert!(compare_scored_genomes(&short, &long).is_lt());
        assert!(compare_scored_genomes(&long, &short).is_gt());
    }
}
