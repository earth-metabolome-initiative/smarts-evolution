use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::str::FromStr;

use smarts_evolution::fitness::mcc::compute_mcc_from_counts;
use smarts_evolution::{
    EvolutionConfig, EvolutionTask, FoldData, FoldSample, SeedCorpus, SmartsGenome, TaskResult,
    evolve_task,
};
use smarts_validator::{CompiledQuery, PreparedTarget, matches_compiled};
use smiles_parser::{Smiles, bond::Bond};

pub struct ExampleDataset {
    pub name: &'static str,
    pub positive_smiles: &'static str,
    pub negative_smiles: &'static str,
}

pub const EXAMPLE_DATASETS: &[ExampleDataset] = &[
    ExampleDataset {
        name: "amphetamines",
        positive_smiles: include_str!(
            "../apps/web/examples/amphetamines_and_derivatives_positive.smiles"
        ),
        negative_smiles: include_str!(
            "../apps/web/examples/amphetamines_and_derivatives_negative.smiles"
        ),
    },
    ExampleDataset {
        name: "flavonoids",
        positive_smiles: include_str!("../apps/web/examples/flavonoids_positive.smiles"),
        negative_smiles: include_str!("../apps/web/examples/flavonoids_negative.smiles"),
    },
    ExampleDataset {
        name: "fatty-acids",
        positive_smiles: include_str!(
            "../apps/web/examples/fatty_acids_and_conjugates_positive.smiles"
        ),
        negative_smiles: include_str!(
            "../apps/web/examples/fatty_acids_and_conjugates_negative.smiles"
        ),
    },
    ExampleDataset {
        name: "penicillins",
        positive_smiles: include_str!("../apps/web/examples/penicillins_positive.smiles"),
        negative_smiles: include_str!("../apps/web/examples/penicillins_negative.smiles"),
    },
    ExampleDataset {
        name: "steroids",
        positive_smiles: include_str!(
            "../apps/web/examples/steroids_and_steroid_derivatives_positive.smiles"
        ),
        negative_smiles: include_str!(
            "../apps/web/examples/steroids_and_steroid_derivatives_negative.smiles"
        ),
    },
];

pub const EXAMPLE_BENCH_POPULATION_SIZE: usize = 64;
pub const EXAMPLE_BENCH_GENERATION_LIMIT: u64 = 20;
pub const EXAMPLE_BENCH_STAGNATION_LIMIT: u64 = 10;
#[allow(dead_code)]
pub const EXAMPLE_BENCH_SEED: u64 = 17;
#[allow(dead_code)]
pub const EXAMPLE_EVALUATOR_QUERY: &str = "[#6](~[#6])~[#6]";
#[allow(dead_code)]
pub const QUALITY_SEEDS: &[u64] = &[17, 23, 41];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SeedStrategy {
    Builtin,
    DiscriminativePaths,
}

impl SeedStrategy {
    pub fn name(self) -> &'static str {
        match self {
            Self::Builtin => "builtin",
            Self::DiscriminativePaths => "discriminative-paths",
        }
    }
}

pub struct ExampleContext {
    positive_graphs: Vec<Smiles>,
    negative_graphs: Vec<Smiles>,
    positive_targets: Vec<PreparedTarget>,
    negative_targets: Vec<PreparedTarget>,
    task: EvolutionTask,
}

pub fn load_example_context(dataset: &ExampleDataset) -> ExampleContext {
    let positive_graphs = parse_smiles_block(dataset.positive_smiles);
    let negative_graphs = parse_smiles_block(dataset.negative_smiles);
    let positive_targets = positive_graphs
        .iter()
        .cloned()
        .map(PreparedTarget::new)
        .collect::<Vec<_>>();
    let negative_targets = negative_graphs
        .iter()
        .cloned()
        .map(PreparedTarget::new)
        .collect::<Vec<_>>();

    let mut samples = positive_targets
        .iter()
        .cloned()
        .map(FoldSample::positive)
        .collect::<Vec<_>>();
    samples.extend(negative_targets.iter().cloned().map(FoldSample::negative));

    ExampleContext {
        positive_graphs,
        negative_graphs,
        positive_targets,
        negative_targets,
        task: EvolutionTask::new(
            format!("example-{}", dataset.name),
            vec![FoldData::new(samples)],
        ),
    }
}

#[allow(dead_code)]
pub fn build_example_task(dataset: &ExampleDataset) -> EvolutionTask {
    load_example_context(dataset).task
}

pub fn build_example_config(seed: u64) -> EvolutionConfig {
    EvolutionConfig::builder()
        .population_size(EXAMPLE_BENCH_POPULATION_SIZE)
        .generation_limit(EXAMPLE_BENCH_GENERATION_LIMIT)
        .stagnation_limit(EXAMPLE_BENCH_STAGNATION_LIMIT)
        .rng_seed(seed)
        .build()
        .unwrap()
}

pub fn run_example_strategy(
    dataset: &ExampleDataset,
    strategy: SeedStrategy,
    seed: u64,
) -> TaskResult {
    let context = load_example_context(dataset);
    let config = build_example_config(seed);
    let seed_corpus = build_seed_corpus(&context, strategy);
    evolve_task(&context.task, &config, &seed_corpus).unwrap()
}

pub fn build_seed_corpus(context: &ExampleContext, strategy: SeedStrategy) -> SeedCorpus {
    match strategy {
        SeedStrategy::Builtin => SeedCorpus::builtin(),
        SeedStrategy::DiscriminativePaths => build_discriminative_seed_corpus(context),
    }
}

#[derive(Clone)]
struct CandidateStats {
    smarts: String,
    positive_support: usize,
    negative_support: usize,
    mcc: f64,
}

#[derive(Clone, Copy)]
struct MiningConfig {
    max_path_atoms: usize,
    shortlist_size: usize,
    top_k: usize,
    min_positive_support: usize,
    max_negative_support: usize,
}

impl MiningConfig {
    fn for_context(context: &ExampleContext) -> Self {
        Self {
            max_path_atoms: 4,
            shortlist_size: 96,
            min_positive_support: (context.positive_graphs.len() / 50).max(10),
            max_negative_support: (context.negative_graphs.len() / 40).max(25),
            top_k: 8,
        }
    }
}

fn build_discriminative_seed_corpus(context: &ExampleContext) -> SeedCorpus {
    let config = MiningConfig::for_context(context);
    let positive_support = mine_support_counts(&context.positive_graphs, config.max_path_atoms);
    let negative_support = mine_support_counts(&context.negative_graphs, config.max_path_atoms);

    let mut shortlist = positive_support
        .iter()
        .filter_map(|(smarts, &pos)| {
            let neg = negative_support.get(smarts).copied().unwrap_or(0);
            if pos < config.min_positive_support || neg > config.max_negative_support {
                return None;
            }
            Some((smarts.clone(), pos, neg))
        })
        .collect::<Vec<_>>();

    shortlist.sort_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| left.0.len().cmp(&right.0.len()))
            .then_with(|| left.0.cmp(&right.0))
    });
    shortlist.truncate(config.shortlist_size);

    let mut candidates = shortlist
        .into_iter()
        .filter_map(|(smarts, _, _)| evaluate_candidate(context, &smarts, config))
        .collect::<Vec<_>>();

    candidates.sort_by(compare_candidate_stats);

    let mut seed_corpus = SeedCorpus::builtin();
    let _ = seed_corpus.extend_from_smarts(
        candidates
            .iter()
            .take(config.top_k)
            .map(|candidate| candidate.smarts.as_str()),
    );
    seed_corpus
}

fn evaluate_candidate(
    context: &ExampleContext,
    smarts: &str,
    config: MiningConfig,
) -> Option<CandidateStats> {
    let genome = SmartsGenome::from_smarts(smarts).ok()?;
    let compiled = CompiledQuery::new(genome.query().clone()).ok()?;

    let positive_support = context
        .positive_targets
        .iter()
        .filter(|target| matches_compiled(&compiled, target))
        .count();
    let negative_support = context
        .negative_targets
        .iter()
        .filter(|target| matches_compiled(&compiled, target))
        .count();

    if positive_support < config.min_positive_support
        || negative_support > config.max_negative_support
    {
        return None;
    }

    let tp = positive_support as u64;
    let fp = negative_support as u64;
    let fn_ = (context.positive_targets.len() - positive_support) as u64;
    let tn = (context.negative_targets.len() - negative_support) as u64;

    Some(CandidateStats {
        smarts: genome.smarts().to_string(),
        positive_support,
        negative_support,
        mcc: compute_mcc_from_counts(tp, fp, tn, fn_),
    })
}

fn compare_candidate_stats(left: &CandidateStats, right: &CandidateStats) -> Ordering {
    right
        .mcc
        .partial_cmp(&left.mcc)
        .unwrap_or(Ordering::Equal)
        .then_with(|| right.positive_support.cmp(&left.positive_support))
        .then_with(|| left.negative_support.cmp(&right.negative_support))
        .then_with(|| left.smarts.len().cmp(&right.smarts.len()))
        .then_with(|| left.smarts.cmp(&right.smarts))
}

fn parse_smiles_block(smiles_block: &str) -> Vec<Smiles> {
    smiles_block
        .lines()
        .map(str::trim)
        .filter(|smiles| !smiles.is_empty())
        .map(|smiles| Smiles::from_str(smiles).unwrap())
        .collect()
}

fn mine_support_counts(smiles_set: &[Smiles], max_path_atoms: usize) -> HashMap<String, usize> {
    let mut support = HashMap::new();
    for smiles in smiles_set {
        for candidate in molecule_candidates(smiles, max_path_atoms) {
            *support.entry(candidate).or_insert(0) += 1;
        }
    }
    support
}

fn molecule_candidates(smiles: &Smiles, max_path_atoms: usize) -> HashSet<String> {
    let ring_membership = smiles.ring_membership();
    let mut candidates = HashSet::new();
    let mut visited = vec![false; smiles.nodes().len()];
    let mut atom_ids = Vec::new();
    let mut bond_tokens = Vec::new();

    for atom_id in 0..smiles.nodes().len() {
        collect_paths(
            smiles,
            &ring_membership,
            atom_id,
            max_path_atoms,
            &mut visited,
            &mut atom_ids,
            &mut bond_tokens,
            &mut candidates,
        );
        collect_branch_pairs(smiles, &ring_membership, atom_id, &mut candidates);
    }

    candidates
}

#[allow(clippy::too_many_arguments)]
fn collect_paths(
    smiles: &Smiles,
    ring_membership: &smiles_parser::smiles::RingMembership,
    atom_id: usize,
    max_path_atoms: usize,
    visited: &mut [bool],
    atom_ids: &mut Vec<usize>,
    bond_tokens: &mut Vec<&'static str>,
    candidates: &mut HashSet<String>,
) {
    visited[atom_id] = true;
    atom_ids.push(atom_id);
    candidates.insert(canonical_path_smarts(
        smiles,
        ring_membership,
        atom_ids,
        bond_tokens,
    ));

    if atom_ids.len() < max_path_atoms {
        for edge in smiles.edges_for_node(atom_id) {
            let next = if edge.0 == atom_id { edge.1 } else { edge.0 };
            if visited[next] {
                continue;
            }
            bond_tokens.push(bond_smarts(edge.2));
            collect_paths(
                smiles,
                ring_membership,
                next,
                max_path_atoms,
                visited,
                atom_ids,
                bond_tokens,
                candidates,
            );
            bond_tokens.pop();
        }
    }

    atom_ids.pop();
    visited[atom_id] = false;
}

fn collect_branch_pairs(
    smiles: &Smiles,
    ring_membership: &smiles_parser::smiles::RingMembership,
    center: usize,
    candidates: &mut HashSet<String>,
) {
    let neighbors = smiles.edges_for_node(center).collect::<Vec<_>>();
    for left_index in 0..neighbors.len() {
        for right_index in (left_index + 1)..neighbors.len() {
            let left = neighbors[left_index];
            let right = neighbors[right_index];
            let left_id = if left.0 == center { left.1 } else { left.0 };
            let right_id = if right.0 == center { right.1 } else { right.0 };
            if left_id == right_id {
                continue;
            }
            let forward = branch_pair_smarts(
                smiles,
                ring_membership,
                center,
                left_id,
                bond_smarts(left.2),
                right_id,
                bond_smarts(right.2),
            );
            let reverse = branch_pair_smarts(
                smiles,
                ring_membership,
                center,
                right_id,
                bond_smarts(right.2),
                left_id,
                bond_smarts(left.2),
            );
            candidates.insert(forward.min(reverse));
        }
    }
}

fn branch_pair_smarts(
    smiles: &Smiles,
    ring_membership: &smiles_parser::smiles::RingMembership,
    center: usize,
    branch_atom: usize,
    branch_bond: &str,
    tail_atom: usize,
    tail_bond: &str,
) -> String {
    format!(
        "{}({}{}){}{}",
        atom_smarts(smiles, ring_membership, center),
        branch_bond,
        atom_smarts(smiles, ring_membership, branch_atom),
        tail_bond,
        atom_smarts(smiles, ring_membership, tail_atom)
    )
}

fn canonical_path_smarts(
    smiles: &Smiles,
    ring_membership: &smiles_parser::smiles::RingMembership,
    atom_ids: &[usize],
    bond_tokens: &[&str],
) -> String {
    let forward = path_smarts(smiles, ring_membership, atom_ids, bond_tokens);
    if atom_ids.len() <= 1 {
        return forward;
    }

    let mut reverse_atoms = atom_ids.to_vec();
    reverse_atoms.reverse();
    let mut reverse_bonds = bond_tokens.to_vec();
    reverse_bonds.reverse();
    let reverse = path_smarts(smiles, ring_membership, &reverse_atoms, &reverse_bonds);
    forward.min(reverse)
}

fn path_smarts(
    smiles: &Smiles,
    ring_membership: &smiles_parser::smiles::RingMembership,
    atom_ids: &[usize],
    bond_tokens: &[&str],
) -> String {
    let mut smarts = atom_smarts(smiles, ring_membership, atom_ids[0]);
    for (index, &atom_id) in atom_ids.iter().enumerate().skip(1) {
        smarts.push_str(bond_tokens[index - 1]);
        smarts.push_str(&atom_smarts(smiles, ring_membership, atom_id));
    }
    smarts
}

fn atom_smarts(
    smiles: &Smiles,
    ring_membership: &smiles_parser::smiles::RingMembership,
    atom_id: usize,
) -> String {
    let atom = smiles.node_by_id(atom_id).unwrap();
    let atomic_number = atom.element().map(u8::from).unwrap_or(0);
    let mut smarts = format!("[#{atomic_number}");
    if atom.aromatic() {
        smarts.push_str(";a");
    } else if ring_membership.contains_atom(atom_id) {
        smarts.push_str(";R");
    }
    smarts.push(']');
    smarts
}

fn bond_smarts(bond: Bond) -> &'static str {
    match bond {
        Bond::Single | Bond::Up | Bond::Down => "-",
        Bond::Double => "=",
        Bond::Triple => "#",
        Bond::Quadruple => "$",
        Bond::Aromatic => ":",
    }
}
