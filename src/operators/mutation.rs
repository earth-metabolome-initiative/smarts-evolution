use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::str::FromStr;

use elements_rs::isotopes::{CarbonIsotope, NitrogenIsotope, OxygenIsotope};
use elements_rs::{AtomicNumber, Element, ElementVariant, Isotope};
use log::debug;
use rand::Rng;
use rand::RngExt;
use rand::prelude::IndexedRandom;
use rand::seq::IteratorRandom;
use smarts_rs::{
    AtomExpr, AtomPrimitive, BondExpr, BondExprTree, BondPrimitive, BracketExpr, BracketExprTree,
    ComponentGroupId, EditableQueryMol, ExprPath, HydrogenKind, NumericQuery, NumericRange,
    QueryMol, add_atom_primitive, add_bond_primitive, normalize_bond_tree, remove_atom_primitive,
    remove_bond_primitive, replace_atom_primitive, replace_bond_primitive,
};
use smiles_parser::atom::bracketed::chirality::Chirality;
use smiles_parser::bond::Bond;

use crate::genome::SmartsGenome;
use crate::genome::limits::{MAX_PRIMITIVES_PER_BRACKET, MAX_SMARTS_LEN};

const ALL_ELEMENT_SYMBOLS: &[&str] = &[
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
    "Fl", "Mc", "Lv", "Ts", "Og",
];
const AROMATIC_ELEMENT_SYMBOLS: &[&str] = &["B", "C", "N", "O", "P", "S", "As", "Se"];
const MAX_ATOMIC_NUMBER: u16 = 118;
const MAX_ISOTOPE_WILDCARD: u16 = u16::MAX;
const MAX_DEGREE: u16 = 16;
const MAX_CONNECTIVITY: u16 = 16;
const MAX_VALENCE: u16 = 16;
const MAX_HYDROGEN_COUNT: u16 = 16;
const MAX_RING_MEMBERSHIP: u16 = 16;
const MAX_RING_SIZE: u16 = 32;
const MAX_RING_CONNECTIVITY: u16 = 16;
const MAX_HYBRIDIZATION: u16 = 16;
const MAX_HETERO_NEIGHBOR: u16 = 16;
const MAX_ALIPHATIC_HETERO_NEIGHBOR: u16 = 16;
const MAX_ABS_CHARGE: i8 = 8;
const MAX_EXPR_TREE_DEPTH: usize = 4;
const MAX_RECURSIVE_QUERY_DEPTH: usize = 4;
const MAX_COMPONENTS: usize = 16;
const MAX_ATOM_MAP: u32 = 65_535;
const MAX_BOOLEAN_CHILDREN: usize = 4;
const RING_CLOSE_ATTEMPTS: usize = 32;
const RESET_RESTART_ATTEMPTS: usize = 4;
const DEFAULT_RESET_PROBABILITY: f64 = 0.20;
const DEFAULT_MAX_MUTATION_STEPS: usize = 6;
const DEFAULT_ATTEMPT_BUDGET: usize = 24;
const NEAR_SMARTS_LEN_LIMIT_PERCENT: usize = 85;
const ONE_STEP_MUTATION_PROBABILITY: f64 = 0.62;
const TWO_STEP_MUTATION_PROBABILITY: f64 = 0.84;
const THREE_STEP_MUTATION_PROBABILITY: f64 = 0.95;
const REMOVE_ATOM_MAP_PROBABILITY: f64 = 0.80;
const COMMON_ALIPHATIC_ELEMENT_SYMBOLS: &[&str] = &[
    "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I",
];
const COMMON_ATOMIC_NUMBERS: &[u16] = &[5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53];
const RECURSIVE_FRAGMENT_SMARTS: &[&str] = &[
    "[#6]",
    "[#7]",
    "[#8]",
    "[#6]=[#8]",
    "[#6]~[#7]",
    "[#6]~[#8]",
    "[#6,#7]",
    "[!#1]",
    "[#6]1~[#6]~[#6]1",
    "[#6]-[#8].[#7]",
];

const MUTATION_LOG_TARGET: &str = "smarts_evolution::operators::mutation";

#[derive(Clone, Copy, Debug)]
enum MutationSource {
    Parent,
    Reset,
}

impl MutationSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Parent => "parent",
            Self::Reset => "reset",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum MutationDirection {
    #[default]
    Balanced,
    Generalize,
    Specialize,
}

#[repr(usize)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MutationOperator {
    AtomConstraints,
    AtomRefinement,
    AtomMap,
    RecursiveQuery,
    AttachLeafAtom,
    InsertAtomOnBond,
    RemoveLeafAtom,
    BondConstraints,
    RingEdit,
    ComponentEdit,
    ToggleAtomNot,
    GraftSeedFragment,
}

impl MutationOperator {
    const ALL: [Self; 12] = [
        Self::AtomConstraints,
        Self::AtomRefinement,
        Self::AtomMap,
        Self::RecursiveQuery,
        Self::AttachLeafAtom,
        Self::InsertAtomOnBond,
        Self::RemoveLeafAtom,
        Self::BondConstraints,
        Self::RingEdit,
        Self::ComponentEdit,
        Self::ToggleAtomNot,
        Self::GraftSeedFragment,
    ];
    const COUNT: usize = Self::ALL.len();

    fn as_index(self) -> usize {
        self as usize
    }

    fn weight(self, near_smarts_len_limit: bool, direction: MutationDirection) -> u32 {
        let (near_limit, normal) = match (direction, self) {
            (_, Self::AtomMap) => return 1,
            (MutationDirection::Balanced, Self::AtomConstraints) => (10, 12),
            (MutationDirection::Balanced, Self::AtomRefinement) => (18, 11),
            (MutationDirection::Balanced, Self::RecursiveQuery) => (2, 10),
            (MutationDirection::Balanced, Self::AttachLeafAtom) => (2, 13),
            (MutationDirection::Balanced, Self::InsertAtomOnBond) => (1, 9),
            (MutationDirection::Balanced, Self::RemoveLeafAtom) => (22, 10),
            (MutationDirection::Balanced, Self::BondConstraints) => (12, 12),
            (MutationDirection::Balanced, Self::RingEdit) => (8, 8),
            (MutationDirection::Balanced, Self::ComponentEdit) => (3, 7),
            (MutationDirection::Balanced, Self::ToggleAtomNot) => (5, 5),
            (MutationDirection::Balanced, Self::GraftSeedFragment) => (0, 2),

            (MutationDirection::Generalize, Self::AtomConstraints) => (12, 8),
            (MutationDirection::Generalize, Self::AtomRefinement) => (24, 22),
            (MutationDirection::Generalize, Self::RecursiveQuery) => (1, 2),
            (MutationDirection::Generalize, Self::AttachLeafAtom) => (1, 2),
            (MutationDirection::Generalize, Self::InsertAtomOnBond) => (1, 2),
            (MutationDirection::Generalize, Self::RemoveLeafAtom) => (24, 20),
            (MutationDirection::Generalize, Self::BondConstraints) => (8, 8),
            (MutationDirection::Generalize, Self::RingEdit) => (10, 10),
            (MutationDirection::Generalize, Self::ComponentEdit) => (4, 4),
            (MutationDirection::Generalize, Self::ToggleAtomNot) => (2, 2),
            (MutationDirection::Generalize, Self::GraftSeedFragment) => (0, 1),

            (MutationDirection::Specialize, Self::AtomConstraints) => (10, 18),
            (MutationDirection::Specialize, Self::AtomRefinement) => (18, 16),
            (MutationDirection::Specialize, Self::RecursiveQuery) => (2, 8),
            (MutationDirection::Specialize, Self::AttachLeafAtom) => (2, 8),
            (MutationDirection::Specialize, Self::InsertAtomOnBond) => (1, 5),
            (MutationDirection::Specialize, Self::RemoveLeafAtom) => (22, 4),
            (MutationDirection::Specialize, Self::BondConstraints) => (12, 18),
            (MutationDirection::Specialize, Self::RingEdit) => (8, 8),
            (MutationDirection::Specialize, Self::ComponentEdit) => (3, 4),
            (MutationDirection::Specialize, Self::ToggleAtomNot) => (5, 5),
            (MutationDirection::Specialize, Self::GraftSeedFragment) => (0, 6),
        };
        length_sensitive_weight(near_smarts_len_limit, near_limit, normal)
    }

    fn is_eligible(
        self,
        query: &QueryMol,
        reset_pool: &[SmartsGenome],
        recursive_depth: usize,
    ) -> bool {
        match self {
            Self::AtomConstraints | Self::AtomRefinement | Self::AttachLeafAtom => {
                query.atom_count() > 0
            }
            Self::AtomMap => has_mapped_atom(query),
            Self::RecursiveQuery => {
                recursive_depth < MAX_RECURSIVE_QUERY_DEPTH && query.atom_count() > 0
            }
            Self::InsertAtomOnBond | Self::BondConstraints => !query.bonds().is_empty(),
            Self::RemoveLeafAtom => query.atom_count() > 1 && !query.leaf_atoms().is_empty(),
            Self::RingEdit => can_edit_ring(query),
            Self::ComponentEdit => true,
            Self::ToggleAtomNot => has_bracket_atom(query),
            Self::GraftSeedFragment => !reset_pool.is_empty() && query.atom_count() > 0,
        }
    }

    fn apply<R: Rng>(
        self,
        editable: &mut EditableQueryMol,
        reset_pool: &[SmartsGenome],
        rng: &mut R,
        recursive_depth: usize,
        direction: MutationDirection,
    ) -> bool {
        match self {
            Self::AtomConstraints => mutate_atom_constraints(editable, rng, recursive_depth),
            Self::AtomRefinement => {
                generalize_specialize_atom(editable, rng, recursive_depth, direction)
            }
            Self::AtomMap => mutate_atom_map(editable, rng),
            Self::RecursiveQuery => {
                mutate_recursive_query(editable, reset_pool, rng, recursive_depth, direction)
            }
            Self::AttachLeafAtom => attach_leaf_atom(editable, rng),
            Self::InsertAtomOnBond => insert_atom_on_bond(editable, rng),
            Self::RemoveLeafAtom => remove_leaf_atom(editable, rng),
            Self::BondConstraints => mutate_bond_constraints(editable, rng),
            Self::RingEdit => ring_edit(editable, rng),
            Self::ComponentEdit => component_edit(editable, rng, recursive_depth),
            Self::ToggleAtomNot => toggle_atom_not(editable, rng),
            Self::GraftSeedFragment => graft_seed_fragment(editable, reset_pool, rng),
        }
    }
}

fn length_sensitive_weight(near_smarts_len_limit: bool, near_limit: u32, normal: u32) -> u32 {
    if near_smarts_len_limit {
        near_limit
    } else {
        normal
    }
}

#[derive(Clone, Copy, Debug)]
struct MutationEdit {
    operator: MutationOperator,
    changed: bool,
}

#[derive(Clone, Debug, Default)]
struct MutationAttemptStats {
    attempts: usize,
    steps: usize,
    changed_steps: usize,
    query_errors: usize,
    rejected_genomes: usize,
    canonical_noops: usize,
    operator_attempts: [usize; MutationOperator::COUNT],
    operator_changes: [usize; MutationOperator::COUNT],
}

impl MutationAttemptStats {
    fn record_edit(&mut self, edit: MutationEdit) {
        let index = edit.operator.as_index();
        self.steps += 1;
        self.operator_attempts[index] += 1;
        if edit.changed {
            self.changed_steps += 1;
            self.operator_changes[index] += 1;
        }
    }

    fn log_accepted(
        &self,
        source: MutationSource,
        parent: &SmartsGenome,
        candidate: &SmartsGenome,
    ) {
        debug!(
            target: MUTATION_LOG_TARGET,
            "mutation accepted source={} attempts={} steps={} changed_steps={} query_errors={} rejected_genomes={} canonical_noops={} before_smarts_len={} after_smarts_len={} operator_attempts={:?} operator_changes={:?}",
            source.as_str(),
            self.attempts,
            self.steps,
            self.changed_steps,
            self.query_errors,
            self.rejected_genomes,
            self.canonical_noops,
            parent.smarts_len(),
            candidate.smarts_len(),
            self.operator_attempts,
            self.operator_changes,
        );
    }

    fn log_rejected(&self, source: MutationSource, parent: &SmartsGenome) {
        debug!(
            target: MUTATION_LOG_TARGET,
            "mutation rejected source={} attempts={} steps={} changed_steps={} query_errors={} rejected_genomes={} canonical_noops={} before_smarts_len={} operator_attempts={:?} operator_changes={:?}",
            source.as_str(),
            self.attempts,
            self.steps,
            self.changed_steps,
            self.query_errors,
            self.rejected_genomes,
            self.canonical_noops,
            parent.smarts_len(),
            self.operator_attempts,
            self.operator_changes,
        );
    }
}

/// SMARTS-aware mutation operator.
///
/// Mutation uses `smarts-parser`'s editable query graph as the mutation
/// surface and returns a fresh parsed genome after each successful edit.
#[derive(Clone, Debug)]
pub struct SmartsMutation {
    mutation_rate: f64,
    reset_pool: Vec<SmartsGenome>,
    reset_probability: f64,
    max_mutation_steps: usize,
    attempt_budget: usize,
}

impl SmartsMutation {
    pub fn new(mutation_rate: f64) -> Self {
        Self::with_reset_pool(mutation_rate, Vec::new())
    }

    pub fn with_reset_pool(mutation_rate: f64, reset_pool: Vec<SmartsGenome>) -> Self {
        Self {
            mutation_rate,
            reset_pool,
            reset_probability: DEFAULT_RESET_PROBABILITY,
            max_mutation_steps: DEFAULT_MAX_MUTATION_STEPS,
            attempt_budget: DEFAULT_ATTEMPT_BUDGET,
        }
    }

    pub fn mutate<R>(&self, genome: SmartsGenome, rng: &mut R) -> SmartsGenome
    where
        R: Rng + Sized,
    {
        if !rng.random_bool(self.mutation_rate) {
            return genome;
        }

        if !self.reset_pool.is_empty()
            && rng.random_bool(self.reset_probability)
            && let Some(candidate) = self.mutated_reset_candidate(rng)
        {
            return candidate;
        }

        self.mutated_candidate(
            &genome,
            MutationSource::Parent,
            MutationDirection::Balanced,
            rng,
        )
        .unwrap_or(genome)
    }

    pub(crate) fn mutate_guided<R, F>(
        &self,
        genome: SmartsGenome,
        candidate_count: usize,
        direction: MutationDirection,
        rng: &mut R,
        mut select_candidate: F,
    ) -> SmartsGenome
    where
        R: Rng + Sized,
        F: FnMut(&SmartsGenome, &[SmartsGenome]) -> usize,
    {
        if !rng.random_bool(self.mutation_rate) {
            return genome;
        }

        let candidate_count = candidate_count.max(1);
        let mut candidates = Vec::with_capacity(candidate_count);

        if !self.reset_pool.is_empty() && rng.random_bool(self.reset_probability) {
            candidates.extend(self.mutated_reset_candidates(candidate_count, rng));
        }
        if candidates.is_empty() {
            candidates.extend(self.mutated_candidates(
                &genome,
                MutationSource::Parent,
                direction,
                candidate_count,
                rng,
            ));
        }
        if candidates.is_empty() {
            return genome;
        }

        let selected = select_candidate(&genome, &candidates).min(candidates.len() - 1);
        candidates.swap_remove(selected)
    }

    fn mutated_reset_candidate<R>(&self, rng: &mut R) -> Option<SmartsGenome>
    where
        R: Rng + Sized,
    {
        self.mutated_reset_candidates(1, rng).into_iter().next()
    }

    fn mutated_reset_candidates<R>(&self, max_candidates: usize, rng: &mut R) -> Vec<SmartsGenome>
    where
        R: Rng + Sized,
    {
        let mut candidates: Vec<SmartsGenome> = Vec::with_capacity(max_candidates.max(1));
        let mut remaining_budget = self.attempt_budget;
        for restart in 0..RESET_RESTART_ATTEMPTS {
            if remaining_budget == 0 || candidates.len() >= max_candidates {
                break;
            }
            let restarts_left = RESET_RESTART_ATTEMPTS - restart;
            let restart_budget = remaining_budget.div_ceil(restarts_left);
            let reset_idx = rng.random_range(0..self.reset_pool.len());
            let reset = &self.reset_pool[reset_idx];
            candidates.extend(self.mutated_candidates_with_budget(
                reset,
                MutationSource::Reset,
                MutationDirection::Balanced,
                restart_budget,
                max_candidates - candidates.len(),
                rng,
            ));
            remaining_budget = remaining_budget.saturating_sub(restart_budget);
        }

        candidates
    }

    fn mutated_candidate<R>(
        &self,
        genome: &SmartsGenome,
        source: MutationSource,
        direction: MutationDirection,
        rng: &mut R,
    ) -> Option<SmartsGenome>
    where
        R: Rng + Sized,
    {
        self.mutated_candidates(genome, source, direction, 1, rng)
            .into_iter()
            .next()
    }

    fn mutated_candidates<R>(
        &self,
        genome: &SmartsGenome,
        source: MutationSource,
        direction: MutationDirection,
        max_candidates: usize,
        rng: &mut R,
    ) -> Vec<SmartsGenome>
    where
        R: Rng + Sized,
    {
        self.mutated_candidates_with_budget(
            genome,
            source,
            direction,
            self.attempt_budget,
            max_candidates,
            rng,
        )
    }

    fn mutated_candidates_with_budget<R>(
        &self,
        genome: &SmartsGenome,
        source: MutationSource,
        direction: MutationDirection,
        attempt_budget: usize,
        max_candidates: usize,
        rng: &mut R,
    ) -> Vec<SmartsGenome>
    where
        R: Rng + Sized,
    {
        let mut stats = MutationAttemptStats::default();
        let mut candidates: Vec<SmartsGenome> = Vec::with_capacity(max_candidates.max(1));

        for _ in 0..attempt_budget {
            if candidates.len() >= max_candidates {
                break;
            }
            stats.attempts += 1;
            let mut editable = genome.query().edit();
            let mutation_steps = sample_mutation_steps(self.max_mutation_steps, rng);
            let mut mutated = false;

            for _ in 0..mutation_steps {
                let edit =
                    apply_random_mutation(&mut editable, &self.reset_pool, rng, 0, direction);
                stats.record_edit(edit);
                mutated |= edit.changed;
            }

            if !mutated {
                continue;
            }

            let Ok(query) = editable.into_query_mol() else {
                stats.query_errors += 1;
                continue;
            };
            let Some(candidate) = genome_from_query(&query) else {
                stats.rejected_genomes += 1;
                continue;
            };
            if candidate.smarts() == genome.smarts() {
                stats.canonical_noops += 1;
                continue;
            }
            if candidates
                .iter()
                .any(|existing| existing.smarts() == candidate.smarts())
            {
                stats.canonical_noops += 1;
                continue;
            }
            stats.log_accepted(source, genome, &candidate);
            candidates.push(candidate);
        }

        if candidates.is_empty() {
            stats.log_rejected(source, genome);
        }
        candidates
    }
}

fn sample_mutation_steps<R: Rng>(max_mutation_steps: usize, rng: &mut R) -> usize {
    let max_steps = max_mutation_steps.max(1);
    let roll: f64 = rng.random();
    if roll < ONE_STEP_MUTATION_PROBABILITY || max_steps == 1 {
        1
    } else if roll < TWO_STEP_MUTATION_PROBABILITY || max_steps == 2 {
        2
    } else if roll < THREE_STEP_MUTATION_PROBABILITY || max_steps == 3 {
        3
    } else {
        rng.random_range(4..=max_steps)
    }
}

fn genome_from_query(query: &QueryMol) -> Option<SmartsGenome> {
    let genome = SmartsGenome::from_query_mol(query);
    (genome.smarts_len() <= MAX_SMARTS_LEN && genome.is_valid()).then_some(genome)
}

fn apply_random_mutation<R: Rng>(
    editable: &mut EditableQueryMol,
    reset_pool: &[SmartsGenome],
    rng: &mut R,
    recursive_depth: usize,
    direction: MutationDirection,
) -> MutationEdit {
    let operator = sample_mutation_operator(
        editable.as_query_mol(),
        reset_pool,
        rng,
        recursive_depth,
        direction,
    );
    let changed = operator.apply(editable, reset_pool, rng, recursive_depth, direction);

    MutationEdit { operator, changed }
}

fn sample_mutation_operator<R: Rng>(
    query: &QueryMol,
    reset_pool: &[SmartsGenome],
    rng: &mut R,
    recursive_depth: usize,
    direction: MutationDirection,
) -> MutationOperator {
    let near_smarts_len_limit = is_near_smarts_len_limit(query);
    let total_weight = MutationOperator::ALL
        .into_iter()
        .map(|operator| {
            operator_weight(
                operator,
                query,
                reset_pool,
                recursive_depth,
                near_smarts_len_limit,
                direction,
            )
        })
        .sum();

    if total_weight == 0 {
        return MutationOperator::AtomConstraints;
    }

    let mut roll = rng.random_range(0..total_weight);
    for operator in MutationOperator::ALL {
        let weight = operator_weight(
            operator,
            query,
            reset_pool,
            recursive_depth,
            near_smarts_len_limit,
            direction,
        );
        if roll < weight {
            return operator;
        }
        roll -= weight;
    }

    MutationOperator::AtomConstraints
}

fn is_near_smarts_len_limit(query: &QueryMol) -> bool {
    query.to_string().len() * 100 >= MAX_SMARTS_LEN * NEAR_SMARTS_LEN_LIMIT_PERCENT
}

fn operator_weight(
    operator: MutationOperator,
    query: &QueryMol,
    reset_pool: &[SmartsGenome],
    recursive_depth: usize,
    near_smarts_len_limit: bool,
    direction: MutationDirection,
) -> u32 {
    if operator.is_eligible(query, reset_pool, recursive_depth) {
        operator.weight(near_smarts_len_limit, direction)
    } else {
        0
    }
}

fn random_atom<R: Rng>(editable: &EditableQueryMol, rng: &mut R) -> (usize, AtomExpr) {
    let atoms = editable.as_query_mol().atoms();
    debug_assert!(!atoms.is_empty());
    let atom = &atoms[rng.random_range(0..atoms.len())];
    (atom.id, atom.expr.clone())
}

fn random_atom_id<R: Rng>(editable: &EditableQueryMol, rng: &mut R) -> usize {
    let atoms = editable.as_query_mol().atoms();
    debug_assert!(!atoms.is_empty());
    atoms[rng.random_range(0..atoms.len())].id
}

fn random_bond_id<R: Rng>(editable: &EditableQueryMol, rng: &mut R) -> Option<usize> {
    editable
        .as_query_mol()
        .bonds()
        .iter()
        .choose(rng)
        .map(|bond| bond.id)
}

fn random_bracket_atom<R: Rng>(
    editable: &EditableQueryMol,
    rng: &mut R,
) -> Option<(usize, BracketExpr)> {
    editable
        .as_query_mol()
        .atoms()
        .iter()
        .filter_map(|atom| match &atom.expr {
            AtomExpr::Bracket(expr) => Some((atom.id, expr.clone())),
            _ => None,
        })
        .choose(rng)
}

fn random_mapped_bracket_atom<R: Rng>(
    editable: &EditableQueryMol,
    rng: &mut R,
) -> Option<(usize, BracketExpr)> {
    editable
        .as_query_mol()
        .atoms()
        .iter()
        .filter_map(|atom| match &atom.expr {
            AtomExpr::Bracket(expr) if expr.atom_map.is_some() => Some((atom.id, expr.clone())),
            _ => None,
        })
        .choose(rng)
}

fn has_bracket_atom(query: &QueryMol) -> bool {
    query
        .atoms()
        .iter()
        .any(|atom| matches!(&atom.expr, AtomExpr::Bracket(_)))
}

fn has_mapped_atom(query: &QueryMol) -> bool {
    query.atoms().iter().any(|atom| {
        matches!(
            &atom.expr,
            AtomExpr::Bracket(BracketExpr {
                atom_map: Some(_),
                ..
            })
        )
    })
}

fn can_edit_ring(query: &QueryMol) -> bool {
    !query.ring_bonds().is_empty() || can_close_ring(query)
}

fn can_close_ring(query: &QueryMol) -> bool {
    for component in 0..query.component_count() {
        let atoms = query.component_atoms(component);
        for (idx, &left) in atoms.iter().enumerate() {
            for &right in &atoms[idx + 1..] {
                if query.bond_between(left, right).is_none() {
                    return true;
                }
            }
        }
    }

    false
}

fn mutate_atom_constraints<R: Rng>(
    editable: &mut EditableQueryMol,
    rng: &mut R,
    recursive_depth: usize,
) -> bool {
    let (atom_id, atom_expr) = random_atom(editable, rng);

    match atom_expr {
        AtomExpr::Bracket(mut expr) => {
            let primitive_paths = bracket_primitive_paths(&expr);
            let mutated = match rng.random_range(0..5) {
                0 if primitive_paths.len() < MAX_PRIMITIVES_PER_BRACKET => {
                    add_atom_primitive(&mut expr, random_atom_primitive(rng, recursive_depth))
                        .is_ok()
                }
                1 if !primitive_paths.is_empty() => {
                    let Some(path) = primitive_paths.choose(rng) else {
                        return false;
                    };
                    replace_atom_primitive(
                        &mut expr,
                        path,
                        random_atom_primitive(rng, recursive_depth),
                    )
                    .is_ok()
                }
                2 if primitive_paths.len() > 1 => {
                    let Some(path) = primitive_paths.choose(rng) else {
                        return false;
                    };
                    remove_atom_primitive(&mut expr, path).is_ok()
                }
                3 => mutate_bracket_tree_shape(&mut expr, rng, recursive_depth),
                _ => {
                    expr = random_bracket_expr(rng, recursive_depth);
                    true
                }
            };

            mutated
                && editable
                    .replace_atom_expr(atom_id, AtomExpr::Bracket(expr))
                    .is_ok()
        }
        other => editable
            .replace_atom_expr(atom_id, local_atom_expr_mutation(other, rng))
            .is_ok(),
    }
}

fn generalize_specialize_atom<R: Rng>(
    editable: &mut EditableQueryMol,
    rng: &mut R,
    recursive_depth: usize,
    direction: MutationDirection,
) -> bool {
    let (atom_id, atom_expr) = random_atom(editable, rng);

    let AtomExpr::Bracket(mut expr) = atom_expr else {
        return editable
            .replace_atom_expr(atom_id, local_atom_expr_mutation(atom_expr, rng))
            .is_ok();
    };
    let primitive_paths = bracket_primitive_paths(&expr);
    let Some(path) = primitive_paths.choose(rng) else {
        return false;
    };
    let Some(BracketExprTree::Primitive(current)) = expr.tree.get(path) else {
        return false;
    };
    let should_generalize = match direction {
        MutationDirection::Balanced => rng.random_bool(0.5),
        MutationDirection::Generalize => rng.random_bool(0.80),
        MutationDirection::Specialize => rng.random_bool(0.20),
    };
    let replacement = if should_generalize {
        generalize_atom_primitive(current, rng, recursive_depth)
    } else {
        specialize_atom_primitive(current, rng, recursive_depth)
    };

    replace_atom_primitive(&mut expr, path, replacement).is_ok()
        && editable
            .replace_atom_expr(atom_id, AtomExpr::Bracket(expr))
            .is_ok()
}

fn mutate_atom_map<R: Rng>(editable: &mut EditableQueryMol, rng: &mut R) -> bool {
    let Some((atom_id, mut expr)) = random_mapped_bracket_atom(editable, rng) else {
        return false;
    };

    expr.atom_map = if rng.random_bool(REMOVE_ATOM_MAP_PROBABILITY) {
        None
    } else {
        Some(rng.random_range(1..=MAX_ATOM_MAP))
    };

    editable
        .replace_atom_expr(atom_id, AtomExpr::Bracket(expr))
        .is_ok()
}

fn mutate_recursive_query<R: Rng>(
    editable: &mut EditableQueryMol,
    reset_pool: &[SmartsGenome],
    rng: &mut R,
    recursive_depth: usize,
    direction: MutationDirection,
) -> bool {
    if recursive_depth >= MAX_RECURSIVE_QUERY_DEPTH {
        return false;
    }

    let sites = recursive_sites(editable.as_query_mol());
    if sites.is_empty() {
        let (atom_id, atom_expr) = random_atom(editable, rng);
        let recursive = random_recursive_query(rng, recursive_depth + 1);
        let expr = match atom_expr {
            AtomExpr::Bracket(mut expr) => {
                if add_atom_primitive(
                    &mut expr,
                    AtomPrimitive::RecursiveQuery(Box::new(recursive.clone())),
                )
                .is_err()
                {
                    expr.tree = BracketExprTree::HighAnd(vec![
                        expr.tree,
                        BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(Box::new(
                            recursive,
                        ))),
                    ]);
                    let _ = expr.normalize();
                }
                expr
            }
            other => {
                let mut expr = atom_expr_to_bracket(other);
                expr.tree = BracketExprTree::HighAnd(vec![
                    expr.tree,
                    BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(Box::new(recursive))),
                ]);
                let _ = expr.normalize();
                expr
            }
        };
        return editable
            .replace_atom_expr(atom_id, AtomExpr::Bracket(expr))
            .is_ok();
    }

    let Some((atom_id, path)) = sites.choose(rng).cloned() else {
        return false;
    };
    let Some(atom) = editable.as_query_mol().atom(atom_id) else {
        return false;
    };
    let mut atom_expr = atom.expr.clone();
    let AtomExpr::Bracket(bracket) = &mut atom_expr else {
        return false;
    };
    let Some(BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(inner))) =
        bracket.tree.get_mut(&path)
    else {
        return false;
    };

    let mutated_inner = if rng.random_bool(0.30) {
        let candidate = random_recursive_query(rng, recursive_depth + 1);
        if genome_from_query(&candidate).is_some() {
            **inner = candidate;
            true
        } else {
            false
        }
    } else {
        let mut nested = inner.edit();
        if !apply_random_mutation(&mut nested, reset_pool, rng, recursive_depth + 1, direction)
            .changed
        {
            false
        } else {
            let Ok(query) = nested.into_query_mol() else {
                return false;
            };
            if genome_from_query(&query).is_none() {
                return false;
            }
            **inner = query;
            true
        }
    };

    mutated_inner && editable.replace_atom_expr(atom_id, atom_expr).is_ok()
}

fn attach_leaf_atom<R: Rng>(editable: &mut EditableQueryMol, rng: &mut R) -> bool {
    let parent = random_atom_id(editable, rng);

    editable
        .attach_leaf(parent, random_bond_expr(rng), random_atom_expr(rng, 0))
        .is_ok()
}

fn insert_atom_on_bond<R: Rng>(editable: &mut EditableQueryMol, rng: &mut R) -> bool {
    let Some(bond_id) = random_bond_id(editable, rng) else {
        return false;
    };

    editable
        .insert_atom_on_bond(
            bond_id,
            random_bond_expr(rng),
            random_atom_expr(rng, 0),
            random_bond_expr(rng),
        )
        .is_ok()
}

fn remove_leaf_atom<R: Rng>(editable: &mut EditableQueryMol, rng: &mut R) -> bool {
    if editable.as_query_mol().atom_count() <= 1 {
        return false;
    }

    let leaves = editable.as_query_mol().leaf_atoms();
    let Some(&leaf) = leaves.choose(rng) else {
        return false;
    };
    editable.remove_leaf_atom(leaf).is_ok()
}

fn mutate_bond_constraints<R: Rng>(editable: &mut EditableQueryMol, rng: &mut R) -> bool {
    let Some(bond_id) = random_bond_id(editable, rng) else {
        return false;
    };

    let Some(bond) = editable.as_query_mol().bond(bond_id) else {
        return false;
    };
    let current = bond.expr.clone();
    match current {
        BondExpr::Elided => editable
            .replace_bond_expr(bond_id, random_bond_expr(rng))
            .is_ok(),
        BondExpr::Query(mut tree) => {
            let primitive_paths = bond_primitive_paths(&tree);
            let mutated = match rng.random_range(0..5) {
                0 if primitive_paths.len() < MAX_PRIMITIVES_PER_BRACKET => {
                    add_bond_primitive(&mut tree, random_bond_primitive(rng)).is_ok()
                }
                1 if !primitive_paths.is_empty() => {
                    let Some(path) = primitive_paths.choose(rng) else {
                        return false;
                    };
                    replace_bond_primitive(&mut tree, path, random_bond_primitive(rng)).is_ok()
                }
                2 if primitive_paths.len() > 1 => {
                    let Some(path) = primitive_paths.choose(rng) else {
                        return false;
                    };
                    remove_bond_primitive(&mut tree, path).is_ok()
                }
                3 => mutate_bond_tree_shape(&mut tree, rng),
                _ => {
                    tree = random_bond_tree(rng, 0);
                    true
                }
            };

            mutated
                && normalize_bond_tree(&mut tree).is_ok()
                && editable
                    .replace_bond_expr(bond_id, BondExpr::Query(tree))
                    .is_ok()
        }
    }
}

fn toggle_atom_not<R: Rng>(editable: &mut EditableQueryMol, rng: &mut R) -> bool {
    let Some((atom_id, mut expr)) = random_bracket_atom(editable, rng) else {
        return false;
    };

    expr.tree = match expr.tree {
        BracketExprTree::Not(inner) => *inner,
        tree => BracketExprTree::Not(Box::new(tree)),
    };
    expr.normalize().is_ok()
        && editable
            .replace_atom_expr(atom_id, AtomExpr::Bracket(expr))
            .is_ok()
}

fn ring_edit<R: Rng>(editable: &mut EditableQueryMol, rng: &mut R) -> bool {
    let snapshot = editable.as_query_mol().clone();
    let ring_bonds = snapshot.ring_bonds();
    if !ring_bonds.is_empty() && rng.random_bool(0.5) {
        let Some(&bond_id) = ring_bonds.choose(rng) else {
            return false;
        };
        return editable.open_ring(bond_id).is_ok();
    }

    for _ in 0..RING_CLOSE_ATTEMPTS {
        let component = rng.random_range(0..snapshot.component_count());
        let atoms = snapshot.component_atoms(component);
        if atoms.len() < 2 {
            continue;
        }
        let left_idx = rng.random_range(0..atoms.len());
        let mut right_idx = rng.random_range(0..atoms.len() - 1);
        if right_idx >= left_idx {
            right_idx += 1;
        }
        let left = atoms[left_idx];
        let right = atoms[right_idx];
        if snapshot.bond_between(left, right).is_none() {
            return editable
                .close_ring(left, right, random_bond_expr(rng))
                .is_ok();
        }
    }

    false
}

fn component_edit<R: Rng>(
    editable: &mut EditableQueryMol,
    rng: &mut R,
    recursive_depth: usize,
) -> bool {
    let snapshot = editable.as_query_mol().clone();

    if snapshot.component_count() < MAX_COMPONENTS && rng.random_bool(0.55) {
        let new_component = snapshot.component_count();
        let Ok(root) = editable.add_atom(new_component, random_atom_expr(rng, recursive_depth))
        else {
            return false;
        };
        if rng.random_bool(0.5) {
            let _ = editable.attach_leaf(
                root,
                random_bond_expr(rng),
                random_atom_expr(rng, recursive_depth),
            );
        }
        return true;
    }

    if mutate_component_grouping(editable, rng) {
        return true;
    }

    if snapshot.component_count() <= 1 {
        return false;
    }

    let removable = snapshot
        .leaf_atoms()
        .into_iter()
        .filter_map(|atom_id| {
            let atom = snapshot.atom(atom_id)?;
            (snapshot.component_atoms(atom.component).len() == 1).then_some(atom_id)
        })
        .collect::<Vec<_>>();
    let Some(&atom_id) = removable.choose(rng) else {
        return false;
    };
    editable.remove_leaf_atom(atom_id).is_ok()
}

fn mutate_component_grouping<R: Rng>(editable: &mut EditableQueryMol, rng: &mut R) -> bool {
    let snapshot = editable.as_query_mol();
    if snapshot.component_count() < 2 {
        return false;
    }

    let mut groups = snapshot.component_groups().to_vec();
    if rng.random_bool(0.5) {
        let left = rng.random_range(0..groups.len());
        let mut right = rng.random_range(0..groups.len());
        if left == right {
            right = (right + 1) % groups.len();
        }
        let next_group = groups
            .iter()
            .flatten()
            .copied()
            .max()
            .map_or(0, |group| group + 1);
        let group = groups[left].or(groups[right]).unwrap_or(next_group);
        groups[left] = Some(group);
        groups[right] = Some(group);
    } else {
        let grouped = groups
            .iter()
            .enumerate()
            .filter_map(|(idx, group)| group.map(|_| idx))
            .collect::<Vec<_>>();
        let Some(&idx) = grouped.choose(rng) else {
            return false;
        };
        groups[idx] = None;
    }

    let groups = normalize_component_groups(groups);
    let query = QueryMol::from_parts(
        snapshot.atoms().to_vec(),
        snapshot.bonds().to_vec(),
        snapshot.component_count(),
        groups,
    );
    *editable = query.edit();
    true
}

fn graft_seed_fragment<R: Rng>(
    editable: &mut EditableQueryMol,
    reset_pool: &[SmartsGenome],
    rng: &mut R,
) -> bool {
    if reset_pool.is_empty() {
        return false;
    }

    let parent = random_atom_id(editable, rng);
    let fragment_genome = &reset_pool[rng.random_range(0..reset_pool.len())];

    let fragment_root = fragment_genome
        .query()
        .leaf_atoms()
        .into_iter()
        .next()
        .unwrap_or_else(|| fragment_genome.query().atoms()[0].id);

    editable
        .graft_subgraph(
            parent,
            random_bond_expr(rng),
            fragment_genome.query(),
            fragment_root,
        )
        .is_ok()
}

fn bracket_primitive_paths(expr: &BracketExpr) -> Vec<ExprPath> {
    expr.tree
        .enumerate_paths()
        .into_iter()
        .filter(|path| matches!(expr.tree.get(path), Some(BracketExprTree::Primitive(_))))
        .collect()
}

fn bond_primitive_paths(tree: &BondExprTree) -> Vec<ExprPath> {
    tree.enumerate_paths()
        .into_iter()
        .filter(|path| matches!(tree.get(path), Some(BondExprTree::Primitive(_))))
        .collect()
}

fn recursive_sites(query: &QueryMol) -> Vec<(usize, ExprPath)> {
    query
        .atoms()
        .iter()
        .filter_map(|atom| match &atom.expr {
            AtomExpr::Bracket(expr) => Some((atom.id, expr)),
            _ => None,
        })
        .flat_map(|(atom_id, expr)| {
            expr.tree
                .enumerate_paths()
                .into_iter()
                .filter(move |path| {
                    matches!(
                        expr.tree.get(path),
                        Some(BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(_)))
                    )
                })
                .map(move |path| (atom_id, path))
        })
        .collect()
}

fn atom_expr_to_bracket(atom_expr: AtomExpr) -> BracketExpr {
    match atom_expr {
        AtomExpr::Bracket(expr) => expr,
        AtomExpr::Wildcard => BracketExpr {
            tree: BracketExprTree::Primitive(AtomPrimitive::Wildcard),
            atom_map: None,
        },
        AtomExpr::Bare { element, aromatic } => BracketExpr {
            tree: BracketExprTree::Primitive(AtomPrimitive::Symbol { element, aromatic }),
            atom_map: None,
        },
    }
}

fn mutate_bracket_tree_shape<R: Rng>(
    expr: &mut BracketExpr,
    rng: &mut R,
    recursive_depth: usize,
) -> bool {
    expr.tree = match rng.random_range(0..4) {
        0 => BracketExprTree::Not(Box::new(expr.tree.clone())),
        1 => BracketExprTree::HighAnd(vec![
            expr.tree.clone(),
            BracketExprTree::Primitive(random_atom_primitive(rng, recursive_depth)),
        ]),
        2 => BracketExprTree::Or(vec![
            expr.tree.clone(),
            BracketExprTree::Primitive(random_atom_primitive(rng, recursive_depth)),
        ]),
        _ => BracketExprTree::LowAnd(vec![
            expr.tree.clone(),
            BracketExprTree::Primitive(random_atom_primitive(rng, recursive_depth)),
        ]),
    };
    expr.normalize().is_ok()
}

fn mutate_bond_tree_shape<R: Rng>(tree: &mut BondExprTree, rng: &mut R) -> bool {
    *tree = match rng.random_range(0..4) {
        0 => BondExprTree::Not(Box::new(tree.clone())),
        1 => BondExprTree::HighAnd(vec![
            tree.clone(),
            BondExprTree::Primitive(random_bond_primitive(rng)),
        ]),
        2 => BondExprTree::Or(vec![
            tree.clone(),
            BondExprTree::Primitive(random_bond_primitive(rng)),
        ]),
        _ => BondExprTree::LowAnd(vec![
            tree.clone(),
            BondExprTree::Primitive(random_bond_primitive(rng)),
        ]),
    };
    normalize_bond_tree(tree).is_ok()
}

fn random_atom_expr<R: Rng>(rng: &mut R, recursive_depth: usize) -> AtomExpr {
    match rng.random_range(0..4) {
        0 => AtomExpr::Wildcard,
        1 => random_bare_atom_expr(rng),
        _ => AtomExpr::Bracket(random_bracket_expr(rng, recursive_depth)),
    }
}

fn local_atom_expr_mutation<R: Rng>(atom_expr: AtomExpr, rng: &mut R) -> AtomExpr {
    match atom_expr {
        AtomExpr::Bracket(expr) => AtomExpr::Bracket(expr),
        AtomExpr::Wildcard => {
            if rng.random_bool(0.5) {
                random_bare_atom_expr(rng)
            } else {
                bracket_atom_expr(AtomPrimitive::Wildcard)
            }
        }
        AtomExpr::Bare { element, aromatic } => match rng.random_range(0..5) {
            0 => AtomExpr::Wildcard,
            1 => bracket_atom_expr(if aromatic {
                AtomPrimitive::AromaticAny
            } else {
                AtomPrimitive::AliphaticAny
            }),
            2 => bracket_atom_expr(AtomPrimitive::Symbol { element, aromatic }),
            3 => {
                let mut expr = BracketExpr {
                    tree: BracketExprTree::HighAnd(vec![
                        BracketExprTree::Primitive(AtomPrimitive::Symbol { element, aromatic }),
                        BracketExprTree::Primitive(random_local_atom_primitive(rng)),
                    ]),
                    atom_map: None,
                };
                let _ = expr.normalize();
                AtomExpr::Bracket(expr)
            }
            _ => random_bare_atom_expr(rng),
        },
    }
}

fn bracket_atom_expr(primitive: AtomPrimitive) -> AtomExpr {
    AtomExpr::Bracket(BracketExpr {
        tree: BracketExprTree::Primitive(primitive),
        atom_map: None,
    })
}

fn random_bare_atom_expr<R: Rng>(rng: &mut R) -> AtomExpr {
    if rng.random_bool(0.35) {
        AtomExpr::Bare {
            element: random_supported_element(rng, true),
            aromatic: true,
        }
    } else {
        AtomExpr::Bare {
            element: random_supported_element(rng, false),
            aromatic: false,
        }
    }
}

fn random_bracket_expr<R: Rng>(rng: &mut R, recursive_depth: usize) -> BracketExpr {
    let mut expr = BracketExpr {
        tree: random_bracket_tree(rng, recursive_depth, 0),
        atom_map: rng
            .random_bool(0.01)
            .then_some(rng.random_range(1..=MAX_ATOM_MAP)),
    };
    let _ = expr.normalize();
    expr
}

fn random_bracket_tree<R: Rng>(
    rng: &mut R,
    recursive_depth: usize,
    tree_depth: usize,
) -> BracketExprTree {
    if tree_depth >= MAX_EXPR_TREE_DEPTH {
        return BracketExprTree::Primitive(random_atom_primitive(rng, recursive_depth));
    }

    match rng.random_range(0..6) {
        0 | 1 => BracketExprTree::Primitive(random_atom_primitive(rng, recursive_depth)),
        2 => BracketExprTree::Not(Box::new(random_bracket_tree(
            rng,
            recursive_depth,
            tree_depth + 1,
        ))),
        3 => BracketExprTree::HighAnd(random_bracket_children(rng, recursive_depth, tree_depth)),
        4 => BracketExprTree::Or(random_bracket_children(rng, recursive_depth, tree_depth)),
        _ => BracketExprTree::LowAnd(random_bracket_children(rng, recursive_depth, tree_depth)),
    }
}

fn random_bracket_children<R: Rng>(
    rng: &mut R,
    recursive_depth: usize,
    tree_depth: usize,
) -> Vec<BracketExprTree> {
    (0..rng.random_range(2..=MAX_BOOLEAN_CHILDREN))
        .map(|_| random_bracket_tree(rng, recursive_depth, tree_depth + 1))
        .collect()
}

fn random_atom_primitive<R: Rng>(rng: &mut R, recursive_depth: usize) -> AtomPrimitive {
    match rng.random_range(0..100) {
        0..=5 => AtomPrimitive::Wildcard,
        6..=10 => AtomPrimitive::AliphaticAny,
        11..=15 => AtomPrimitive::AromaticAny,
        16..=38 => random_symbol_primitive(rng),
        39..=47 => AtomPrimitive::AtomicNumber(random_atomic_number(rng)),
        48..=52 => AtomPrimitive::Degree(random_optional_numeric_query(rng, 0, MAX_DEGREE)),
        53..=57 => {
            AtomPrimitive::Connectivity(random_optional_numeric_query(rng, 0, MAX_CONNECTIVITY))
        }
        58..=62 => AtomPrimitive::Valence(random_optional_numeric_query(rng, 0, MAX_VALENCE)),
        63..=69 => random_hydrogen_primitive(rng, MAX_HYDROGEN_COUNT),
        70..=74 => AtomPrimitive::RingMembership(random_optional_numeric_query(
            rng,
            0,
            MAX_RING_MEMBERSHIP,
        )),
        75..=79 => AtomPrimitive::RingSize(random_optional_numeric_query(rng, 3, MAX_RING_SIZE)),
        80..=83 => AtomPrimitive::RingConnectivity(random_optional_numeric_query(
            rng,
            0,
            MAX_RING_CONNECTIVITY,
        )),
        84..=86 if recursive_depth < MAX_RECURSIVE_QUERY_DEPTH => AtomPrimitive::RecursiveQuery(
            Box::new(random_recursive_query(rng, recursive_depth + 1)),
        ),
        84..=86 => random_hydrogen_primitive(rng, MAX_HYDROGEN_COUNT),
        87 => AtomPrimitive::Isotope {
            isotope: random_isotope(rng),
            aromatic: rng.random_bool(0.2),
        },
        88 => AtomPrimitive::IsotopeWildcard(random_isotope_wildcard(rng)),
        89 => AtomPrimitive::Hybridization(random_numeric_query(rng, 1, MAX_HYBRIDIZATION)),
        90 => AtomPrimitive::HeteroNeighbor(random_optional_numeric_query(
            rng,
            0,
            MAX_HETERO_NEIGHBOR,
        )),
        91 => AtomPrimitive::AliphaticHeteroNeighbor(random_optional_numeric_query(
            rng,
            0,
            MAX_ALIPHATIC_HETERO_NEIGHBOR,
        )),
        92 => AtomPrimitive::Chirality(random_chirality(rng)),
        _ => AtomPrimitive::Charge(random_charge(rng)),
    }
}

fn random_local_atom_primitive<R: Rng>(rng: &mut R) -> AtomPrimitive {
    match rng.random_range(0..100) {
        0..=18 => random_hydrogen_primitive(rng, 4),
        19..=34 => AtomPrimitive::Degree(random_optional_numeric_query(rng, 0, 6)),
        35..=50 => AtomPrimitive::Connectivity(random_optional_numeric_query(rng, 0, 6)),
        51..=63 => AtomPrimitive::Valence(random_optional_numeric_query(rng, 0, 6)),
        64..=73 => AtomPrimitive::RingMembership(random_optional_numeric_query(rng, 0, 6)),
        74..=82 => AtomPrimitive::RingSize(random_optional_numeric_query(rng, 3, 8)),
        83..=90 => AtomPrimitive::RingConnectivity(random_optional_numeric_query(rng, 0, 6)),
        91..=96 => AtomPrimitive::Charge(rng.random_range(-1..=1)),
        97..=98 => AtomPrimitive::HeteroNeighbor(random_optional_numeric_query(rng, 0, 4)),
        _ => AtomPrimitive::AliphaticHeteroNeighbor(random_optional_numeric_query(rng, 0, 4)),
    }
}

fn random_hydrogen_primitive<R: Rng>(rng: &mut R, max_count: u16) -> AtomPrimitive {
    let kind = if rng.random_bool(0.5) {
        HydrogenKind::Total
    } else {
        HydrogenKind::Implicit
    };
    AtomPrimitive::Hydrogen(kind, random_optional_numeric_query(rng, 0, max_count))
}

fn generalize_atom_primitive<R: Rng>(
    primitive: &AtomPrimitive,
    rng: &mut R,
    recursive_depth: usize,
) -> AtomPrimitive {
    match primitive {
        AtomPrimitive::Symbol { aromatic, .. } => {
            if *aromatic {
                AtomPrimitive::AromaticAny
            } else {
                AtomPrimitive::AliphaticAny
            }
        }
        AtomPrimitive::Isotope { isotope, aromatic } => AtomPrimitive::Symbol {
            element: isotope.element(),
            aromatic: *aromatic,
        },
        AtomPrimitive::IsotopeWildcard(_) | AtomPrimitive::AtomicNumber(_) => {
            AtomPrimitive::Wildcard
        }
        AtomPrimitive::Degree(Some(NumericQuery::Exact(value))) if *value > 1 => {
            AtomPrimitive::Degree(Some(NumericQuery::Exact(value - 1)))
        }
        AtomPrimitive::Connectivity(Some(NumericQuery::Exact(value))) if *value > 1 => {
            AtomPrimitive::Connectivity(Some(NumericQuery::Exact(value - 1)))
        }
        AtomPrimitive::Valence(Some(NumericQuery::Exact(value))) if *value > 1 => {
            AtomPrimitive::Valence(Some(NumericQuery::Exact(value - 1)))
        }
        AtomPrimitive::Hydrogen(kind, Some(NumericQuery::Exact(value)))
            if *value < MAX_HYDROGEN_COUNT =>
        {
            AtomPrimitive::Hydrogen(*kind, Some(NumericQuery::Exact(value + 1)))
        }
        AtomPrimitive::RingMembership(Some(_)) => AtomPrimitive::RingMembership(None),
        AtomPrimitive::RingSize(Some(_)) => AtomPrimitive::RingSize(None),
        AtomPrimitive::RingConnectivity(Some(_)) => AtomPrimitive::RingConnectivity(None),
        AtomPrimitive::HeteroNeighbor(Some(_)) => AtomPrimitive::HeteroNeighbor(None),
        AtomPrimitive::AliphaticHeteroNeighbor(Some(_)) => {
            AtomPrimitive::AliphaticHeteroNeighbor(None)
        }
        AtomPrimitive::Hybridization(_) | AtomPrimitive::Chirality(_) => AtomPrimitive::Wildcard,
        AtomPrimitive::RecursiveQuery(_) => AtomPrimitive::Wildcard,
        AtomPrimitive::Charge(charge) if *charge > 1 => AtomPrimitive::Charge(charge - 1),
        AtomPrimitive::Charge(charge) if *charge < -1 => AtomPrimitive::Charge(charge + 1),
        AtomPrimitive::Charge(_) => AtomPrimitive::Wildcard,
        _ => random_atom_primitive(rng, recursive_depth),
    }
}

fn specialize_atom_primitive<R: Rng>(
    primitive: &AtomPrimitive,
    rng: &mut R,
    recursive_depth: usize,
) -> AtomPrimitive {
    match primitive {
        AtomPrimitive::Wildcard | AtomPrimitive::AliphaticAny | AtomPrimitive::AromaticAny => {
            random_symbol_primitive(rng)
        }
        AtomPrimitive::RingMembership(None) => AtomPrimitive::RingMembership(Some(
            NumericQuery::Exact(rng.random_range(1..=MAX_RING_MEMBERSHIP)),
        )),
        AtomPrimitive::RingSize(None) => AtomPrimitive::RingSize(Some(NumericQuery::Exact(
            rng.random_range(3..=MAX_RING_SIZE),
        ))),
        AtomPrimitive::RingConnectivity(None) => AtomPrimitive::RingConnectivity(Some(
            NumericQuery::Exact(rng.random_range(1..=MAX_RING_CONNECTIVITY)),
        )),
        AtomPrimitive::HeteroNeighbor(None) => AtomPrimitive::HeteroNeighbor(Some(
            NumericQuery::Exact(rng.random_range(1..=MAX_HETERO_NEIGHBOR)),
        )),
        AtomPrimitive::AliphaticHeteroNeighbor(None) => {
            AtomPrimitive::AliphaticHeteroNeighbor(Some(NumericQuery::Exact(
                rng.random_range(1..=MAX_ALIPHATIC_HETERO_NEIGHBOR),
            )))
        }
        AtomPrimitive::Degree(Some(NumericQuery::Exact(value))) if *value < MAX_DEGREE => {
            AtomPrimitive::Degree(Some(NumericQuery::Exact(value + 1)))
        }
        AtomPrimitive::Connectivity(Some(NumericQuery::Exact(value)))
            if *value < MAX_CONNECTIVITY =>
        {
            AtomPrimitive::Connectivity(Some(NumericQuery::Exact(value + 1)))
        }
        AtomPrimitive::Valence(Some(NumericQuery::Exact(value))) if *value < MAX_VALENCE => {
            AtomPrimitive::Valence(Some(NumericQuery::Exact(value + 1)))
        }
        AtomPrimitive::Hydrogen(kind, Some(NumericQuery::Exact(value))) if *value > 0 => {
            AtomPrimitive::Hydrogen(*kind, Some(NumericQuery::Exact(value - 1)))
        }
        AtomPrimitive::Charge(charge) if *charge > 0 && *charge < MAX_ABS_CHARGE => {
            AtomPrimitive::Charge(charge + 1)
        }
        AtomPrimitive::Charge(charge) if *charge < 0 && *charge > -MAX_ABS_CHARGE => {
            AtomPrimitive::Charge(charge - 1)
        }
        AtomPrimitive::Charge(0) => {
            AtomPrimitive::Charge(if rng.random_bool(0.5) { 1 } else { -1 })
        }
        AtomPrimitive::Symbol { element, .. } => {
            AtomPrimitive::AtomicNumber(element.atomic_number().into())
        }
        AtomPrimitive::AtomicNumber(_) => {
            AtomPrimitive::IsotopeWildcard(random_isotope_wildcard(rng))
        }
        AtomPrimitive::RecursiveQuery(_) | AtomPrimitive::Chirality(_) => {
            random_atom_primitive(rng, recursive_depth)
        }
        _ => random_atom_primitive(rng, recursive_depth),
    }
}

fn random_bond_expr<R: Rng>(rng: &mut R) -> BondExpr {
    if rng.random_bool(0.15) {
        BondExpr::Elided
    } else {
        BondExpr::Query(random_bond_tree(rng, 0))
    }
}

fn random_bond_tree<R: Rng>(rng: &mut R, tree_depth: usize) -> BondExprTree {
    if tree_depth >= MAX_EXPR_TREE_DEPTH {
        return BondExprTree::Primitive(random_bond_primitive(rng));
    }

    match rng.random_range(0..6) {
        0 | 1 => BondExprTree::Primitive(random_bond_primitive(rng)),
        2 => BondExprTree::Not(Box::new(random_bond_tree(rng, tree_depth + 1))),
        3 => BondExprTree::HighAnd(random_bond_children(rng, tree_depth)),
        4 => BondExprTree::Or(random_bond_children(rng, tree_depth)),
        _ => BondExprTree::LowAnd(random_bond_children(rng, tree_depth)),
    }
}

fn random_bond_children<R: Rng>(rng: &mut R, tree_depth: usize) -> Vec<BondExprTree> {
    (0..rng.random_range(2..=MAX_BOOLEAN_CHILDREN))
        .map(|_| random_bond_tree(rng, tree_depth + 1))
        .collect()
}

fn random_bond_primitive<R: Rng>(rng: &mut R) -> BondPrimitive {
    match rng.random_range(0..100) {
        0..=34 => BondPrimitive::Bond(Bond::Single),
        35..=54 => BondPrimitive::Bond(Bond::Double),
        55..=66 => BondPrimitive::Bond(Bond::Aromatic),
        67..=78 => BondPrimitive::Any,
        79..=88 => BondPrimitive::Ring,
        89..=93 => BondPrimitive::Bond(Bond::Triple),
        94 => BondPrimitive::Bond(Bond::Quadruple),
        95..=97 => BondPrimitive::Bond(Bond::Up),
        _ => BondPrimitive::Bond(Bond::Down),
    }
}

fn random_atomic_number<R: Rng>(rng: &mut R) -> u16 {
    if rng.random_bool(0.90) {
        COMMON_ATOMIC_NUMBERS[rng.random_range(0..COMMON_ATOMIC_NUMBERS.len())]
    } else {
        rng.random_range(1..=MAX_ATOMIC_NUMBER)
    }
}

fn random_symbol_primitive<R: Rng>(rng: &mut R) -> AtomPrimitive {
    let aromatic = rng.random_bool(0.30);
    let element = random_supported_element(rng, aromatic);
    AtomPrimitive::Symbol { element, aromatic }
}

fn random_supported_element<R: Rng>(rng: &mut R, aromatic: bool) -> Element {
    let symbol = if aromatic {
        AROMATIC_ELEMENT_SYMBOLS[rng.random_range(0..AROMATIC_ELEMENT_SYMBOLS.len())]
    } else if rng.random_bool(0.90) {
        COMMON_ALIPHATIC_ELEMENT_SYMBOLS
            [rng.random_range(0..COMMON_ALIPHATIC_ELEMENT_SYMBOLS.len())]
    } else {
        ALL_ELEMENT_SYMBOLS[rng.random_range(0..ALL_ELEMENT_SYMBOLS.len())]
    };
    match Element::from_str(symbol) {
        Ok(element) => element,
        Err(error) => {
            debug_assert!(false, "invalid built-in element symbol {symbol}: {error}");
            Element::C
        }
    }
}

fn random_isotope<R: Rng>(rng: &mut R) -> Isotope {
    match rng.random_range(0..6) {
        0 => Isotope::C(CarbonIsotope::C12),
        1 => Isotope::C(CarbonIsotope::C13),
        2 => Isotope::N(NitrogenIsotope::N14),
        3 => Isotope::N(NitrogenIsotope::N15),
        4 => Isotope::O(OxygenIsotope::O16),
        _ => Isotope::O(OxygenIsotope::O18),
    }
}

fn random_isotope_wildcard<R: Rng>(rng: &mut R) -> u16 {
    if rng.random_bool(0.90) {
        rng.random_range(0..=255)
    } else {
        rng.random_range(0..=MAX_ISOTOPE_WILDCARD)
    }
}

fn random_charge<R: Rng>(rng: &mut R) -> i8 {
    if rng.random_bool(0.85) {
        rng.random_range(-2..=2)
    } else {
        rng.random_range(-MAX_ABS_CHARGE..=MAX_ABS_CHARGE)
    }
}

fn random_chirality<R: Rng>(rng: &mut R) -> Chirality {
    match rng.random_range(0..7) {
        0 => Chirality::At,
        1 => Chirality::AtAt,
        2 => Chirality::try_th(rng.random_range(1..=2)).unwrap_or(Chirality::At),
        3 => Chirality::try_al(rng.random_range(1..=2)).unwrap_or(Chirality::At),
        4 => Chirality::try_sp(rng.random_range(1..=3)).unwrap_or(Chirality::At),
        5 => Chirality::try_tb(rng.random_range(1..=20)).unwrap_or(Chirality::At),
        _ => Chirality::try_oh(rng.random_range(1..=30)).unwrap_or(Chirality::At),
    }
}

fn random_optional_numeric_query<R: Rng>(rng: &mut R, min: u16, max: u16) -> Option<NumericQuery> {
    if rng.random_bool(0.25) {
        None
    } else {
        Some(random_numeric_query(rng, min, max))
    }
}

fn random_numeric_query<R: Rng>(rng: &mut R, min: u16, max: u16) -> NumericQuery {
    if min == max || rng.random_bool(0.6) {
        NumericQuery::Exact(rng.random_range(min..=max))
    } else {
        let start = rng.random_range(min..=max);
        let end = rng.random_range(start..=max);
        let mut lower = rng.random_bool(0.2).then_some(start);
        let mut upper = rng.random_bool(0.2).then_some(end);
        if lower.is_none() && upper.is_none() {
            if rng.random_bool(0.5) {
                lower = Some(start);
            } else {
                upper = Some(end);
            }
        }
        NumericQuery::Range(NumericRange {
            min: lower,
            max: upper,
        })
    }
}

#[allow(clippy::unwrap_used)]
fn query_from_known_valid_smarts(smarts: &str) -> QueryMol {
    QueryMol::from_str(smarts).unwrap()
}

fn random_recursive_query<R: Rng>(rng: &mut R, recursive_depth: usize) -> QueryMol {
    let smarts = RECURSIVE_FRAGMENT_SMARTS[rng.random_range(0..RECURSIVE_FRAGMENT_SMARTS.len())];
    let mut query = query_from_known_valid_smarts(smarts);
    if recursive_depth < MAX_RECURSIVE_QUERY_DEPTH && rng.random_bool(0.35) {
        let mut editable = query.edit();
        let edit = apply_random_mutation(
            &mut editable,
            &[],
            rng,
            recursive_depth,
            MutationDirection::Balanced,
        );
        if edit.changed
            && let Ok(candidate) = editable.into_query_mol()
            && genome_from_query(&candidate).is_some()
        {
            query = candidate;
        }
    }
    query
}

fn normalize_component_groups(
    groups: Vec<Option<ComponentGroupId>>,
) -> Vec<Option<ComponentGroupId>> {
    let mut remap = BTreeMap::new();
    let mut next = 0;
    groups
        .into_iter()
        .map(|group| {
            group.map(|group_id| {
                *remap.entry(group_id).or_insert_with(|| {
                    let compact = next;
                    next += 1;
                    compact
                })
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use std::string::ToString;
    use std::vec;

    use super::*;
    use crate::genome::over_limit_smarts_fixture;
    use elements_rs::AtomicNumber;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn test_mutation_validity_rate() {
        let seeds = &[
            "[#6]",
            "[#7;H2]",
            "[#6](=[#8])[#7]",
            "[#6;R]~[#6;R]~[#6;R]",
            "[#6;X3](=[#8])[#8]",
        ];

        let mut rng = SmallRng::seed_from_u64(42);
        let mutator = SmartsMutation::new(1.0);

        let mut valid = 0;
        let total = 1000;

        for i in 0..total {
            let seed = seeds[i % seeds.len()];
            let genome = SmartsGenome::from_smarts(seed).unwrap();
            let mutated = mutator.mutate(genome, &mut rng);
            if mutated.is_valid() {
                valid += 1;
            }
        }

        let rate = valid as f64 / total as f64;
        assert!(
            rate >= 0.70,
            "Mutation validity rate {rate:.2} < 0.70 ({valid}/{total})"
        );
    }

    #[test]
    fn zero_rate_mutation_leaves_genome_unchanged() {
        let genome = SmartsGenome::from_smarts("[#6]~[#7]").unwrap();
        let mut rng = SmallRng::seed_from_u64(7);
        let mutator = SmartsMutation::new(0.0);

        assert_eq!(mutator.mutate(genome.clone(), &mut rng), genome);
    }

    #[test]
    fn mutation_step_sampler_prefers_local_edits_and_keeps_exploration_tail() {
        let mut rng = SmallRng::seed_from_u64(8);
        let mut counts = [0usize; DEFAULT_MAX_MUTATION_STEPS + 1];

        for _ in 0..4096 {
            let steps = sample_mutation_steps(DEFAULT_MAX_MUTATION_STEPS, &mut rng);
            assert!((1..=DEFAULT_MAX_MUTATION_STEPS).contains(&steps));
            counts[steps] += 1;
        }

        assert!(counts[1] > counts[2]);
        assert!(counts[2] > counts[3]);
        assert!(counts[4..].iter().sum::<usize>() > 0);
    }

    #[test]
    fn mutation_operator_metadata_is_consistent() {
        assert_eq!(MutationOperator::COUNT, MutationOperator::ALL.len());
        assert_eq!(length_sensitive_weight(true, 3, 7), 3);
        assert_eq!(length_sensitive_weight(false, 3, 7), 7);

        for (index, operator) in MutationOperator::ALL.iter().copied().enumerate() {
            assert_eq!(operator.as_index(), index);
            assert!(operator.weight(false, MutationDirection::Balanced) <= 22);
            assert!(operator.weight(true, MutationDirection::Balanced) <= 22);
        }

        assert!(
            MutationOperator::AtomRefinement.weight(true, MutationDirection::Balanced)
                > MutationOperator::AtomRefinement.weight(false, MutationDirection::Balanced)
        );
        assert!(
            MutationOperator::RemoveLeafAtom.weight(true, MutationDirection::Balanced)
                > MutationOperator::RemoveLeafAtom.weight(false, MutationDirection::Balanced)
        );
        assert_eq!(
            MutationOperator::GraftSeedFragment.weight(true, MutationDirection::Balanced),
            0
        );
        assert!(
            MutationOperator::AtomRefinement.weight(false, MutationDirection::Generalize)
                > MutationOperator::AtomRefinement.weight(false, MutationDirection::Balanced)
        );
        assert!(
            MutationOperator::BondConstraints.weight(false, MutationDirection::Specialize)
                > MutationOperator::BondConstraints.weight(false, MutationDirection::Balanced)
        );
        assert!(
            MutationOperator::RemoveLeafAtom.weight(false, MutationDirection::Generalize)
                > MutationOperator::RemoveLeafAtom.weight(false, MutationDirection::Specialize)
        );

        let single_atom = QueryMol::from_str("C").unwrap();
        let mapped_atom = QueryMol::from_str("[#6:1]").unwrap();
        let bonded = QueryMol::from_str("CC").unwrap();
        let ring_closable = QueryMol::from_str("CCC").unwrap();
        let reset_pool = vec![SmartsGenome::from_smarts("[#6]").unwrap()];

        assert!(MutationOperator::AtomConstraints.is_eligible(&single_atom, &[], 0));
        assert!(MutationOperator::AtomRefinement.is_eligible(&single_atom, &[], 0));
        assert!(MutationOperator::AttachLeafAtom.is_eligible(&single_atom, &[], 0));
        assert!(MutationOperator::RecursiveQuery.is_eligible(&single_atom, &[], 0));
        assert!(!MutationOperator::RecursiveQuery.is_eligible(
            &single_atom,
            &[],
            MAX_RECURSIVE_QUERY_DEPTH,
        ));

        assert!(!MutationOperator::AtomMap.is_eligible(&single_atom, &[], 0));
        assert!(MutationOperator::AtomMap.is_eligible(&mapped_atom, &[], 0));
        assert!(MutationOperator::InsertAtomOnBond.is_eligible(&bonded, &[], 0));
        assert!(MutationOperator::BondConstraints.is_eligible(&bonded, &[], 0));
        assert!(MutationOperator::RemoveLeafAtom.is_eligible(&bonded, &[], 0));
        assert!(MutationOperator::RingEdit.is_eligible(&ring_closable, &[], 0));
        assert!(!MutationOperator::ToggleAtomNot.is_eligible(&single_atom, &[], 0));
        assert!(MutationOperator::ToggleAtomNot.is_eligible(&mapped_atom, &[], 0));
        assert!(!MutationOperator::GraftSeedFragment.is_eligible(&single_atom, &[], 0));
        assert!(MutationOperator::GraftSeedFragment.is_eligible(&single_atom, &reset_pool, 0));

        assert_eq!(
            operator_weight(
                MutationOperator::AtomMap,
                &single_atom,
                &[],
                0,
                false,
                MutationDirection::Balanced,
            ),
            0
        );
        assert_eq!(
            operator_weight(
                MutationOperator::AtomMap,
                &mapped_atom,
                &[],
                0,
                false,
                MutationDirection::Balanced,
            ),
            MutationOperator::AtomMap.weight(false, MutationDirection::Balanced)
        );
        assert_eq!(
            operator_weight(
                MutationOperator::GraftSeedFragment,
                &single_atom,
                &reset_pool,
                0,
                true,
                MutationDirection::Balanced,
            ),
            0
        );
    }

    #[test]
    fn mutation_attempt_stats_track_sources_and_operator_counts() {
        assert_eq!(MutationSource::Parent.as_str(), "parent");
        assert_eq!(MutationSource::Reset.as_str(), "reset");

        let mut stats = MutationAttemptStats::default();
        stats.record_edit(MutationEdit {
            operator: MutationOperator::AtomConstraints,
            changed: false,
        });
        stats.record_edit(MutationEdit {
            operator: MutationOperator::AtomConstraints,
            changed: true,
        });
        stats.record_edit(MutationEdit {
            operator: MutationOperator::BondConstraints,
            changed: true,
        });

        assert_eq!(stats.steps, 3);
        assert_eq!(stats.changed_steps, 2);
        assert_eq!(
            stats.operator_attempts[MutationOperator::AtomConstraints.as_index()],
            2
        );
        assert_eq!(
            stats.operator_changes[MutationOperator::AtomConstraints.as_index()],
            1
        );
        assert_eq!(
            stats.operator_attempts[MutationOperator::BondConstraints.as_index()],
            1
        );
        assert_eq!(
            stats.operator_changes[MutationOperator::BondConstraints.as_index()],
            1
        );
    }

    #[test]
    fn mutation_operator_apply_dispatches_to_helpers() {
        let reset_pool = vec![SmartsGenome::from_smarts("[#6]").unwrap()];
        let cases = [
            (MutationOperator::AtomConstraints, "C"),
            (MutationOperator::AtomRefinement, "C"),
            (MutationOperator::AtomMap, "[#6:1]"),
            (MutationOperator::RecursiveQuery, "C"),
            (MutationOperator::AttachLeafAtom, "C"),
            (MutationOperator::InsertAtomOnBond, "CC"),
            (MutationOperator::RemoveLeafAtom, "CC"),
            (MutationOperator::BondConstraints, "CC"),
            (MutationOperator::RingEdit, "CCC"),
            (MutationOperator::ComponentEdit, "C"),
            (MutationOperator::ToggleAtomNot, "[#6]"),
            (MutationOperator::GraftSeedFragment, "C"),
        ];

        for (operator, smarts) in cases {
            let changed = (0..16).any(|seed| {
                let mut editable = QueryMol::from_str(smarts).unwrap().edit();
                let mut rng = SmallRng::seed_from_u64(seed);
                operator.apply(
                    &mut editable,
                    &reset_pool,
                    &mut rng,
                    0,
                    MutationDirection::Balanced,
                ) && editable
                    .into_query_mol()
                    .map(|query| SmartsGenome::from_query_mol(&query).is_valid())
                    .unwrap_or(false)
            });

            assert!(changed, "{operator:?} did not produce a valid change");
        }
    }

    #[test]
    fn operator_sampler_excludes_ineligible_single_atom_operations() {
        let query = QueryMol::from_str("C").unwrap();
        let mut rng = SmallRng::seed_from_u64(80);

        for _ in 0..512 {
            let operator =
                sample_mutation_operator(&query, &[], &mut rng, 0, MutationDirection::Balanced);
            assert!(!matches!(
                operator,
                MutationOperator::AtomMap
                    | MutationOperator::InsertAtomOnBond
                    | MutationOperator::RemoveLeafAtom
                    | MutationOperator::BondConstraints
                    | MutationOperator::RingEdit
                    | MutationOperator::ToggleAtomNot
                    | MutationOperator::GraftSeedFragment
            ));
        }
    }

    #[test]
    fn operator_sampler_can_use_existing_atom_maps_without_introducing_them() {
        let query = QueryMol::from_str("[#6:1]").unwrap();
        let mut rng = SmallRng::seed_from_u64(81);
        let mut saw_atom_map = false;

        for _ in 0..4096 {
            if matches!(
                sample_mutation_operator(&query, &[], &mut rng, 0, MutationDirection::Balanced),
                MutationOperator::AtomMap
            ) {
                saw_atom_map = true;
                break;
            }
        }

        assert!(saw_atom_map);
    }

    #[test]
    fn operator_sampler_disables_grafting_near_smarts_len_limit() {
        let over_limit = QueryMol::from_str(&over_limit_smarts_fixture()).unwrap();
        let reset_pool = vec![SmartsGenome::from_smarts("[#6]").unwrap()];
        let mut rng = SmallRng::seed_from_u64(82);

        for _ in 0..512 {
            assert!(!matches!(
                sample_mutation_operator(
                    &over_limit,
                    &reset_pool,
                    &mut rng,
                    0,
                    MutationDirection::Balanced,
                ),
                MutationOperator::GraftSeedFragment
            ));
        }
    }

    #[test]
    fn genome_from_query_rejects_overly_long_queries() {
        let over_limit = over_limit_smarts_fixture();
        let query = QueryMol::from_str(&over_limit).unwrap();

        assert!(genome_from_query(&query).is_none());
    }

    #[test]
    fn editable_remove_leaf_collapses_single_branch() {
        let query = QueryMol::from_str("[#6](~[#7])").unwrap();
        let leaf = query
            .leaf_atoms()
            .into_iter()
            .find(|&atom_id| atom_id != 0)
            .unwrap();
        let mut editable = query.edit();
        editable.remove_leaf_atom(leaf).unwrap();
        let collapsed = editable.into_query_mol().unwrap();

        assert_eq!(collapsed.to_string(), "[#6]");
    }

    #[test]
    fn mutation_helpers_handle_non_bracket_and_missing_structure_cases() {
        let query = QueryMol::from_str("C").unwrap();

        let mut editable = query.edit();
        assert!(mutate_atom_constraints(
            &mut editable,
            &mut SmallRng::seed_from_u64(3),
            0,
        ));

        let mut editable = query.edit();
        assert!(generalize_specialize_atom(
            &mut editable,
            &mut SmallRng::seed_from_u64(4),
            0,
            MutationDirection::Balanced,
        ));

        let mut editable = query.edit();
        assert!(!mutate_bond_constraints(
            &mut editable,
            &mut SmallRng::seed_from_u64(5)
        ));

        let mut editable = query.edit();
        assert!(!remove_leaf_atom(
            &mut editable,
            &mut SmallRng::seed_from_u64(6)
        ));

        let mut editable = query.edit();
        assert!(!toggle_atom_not(
            &mut editable,
            &mut SmallRng::seed_from_u64(7)
        ));

        let mut editable = query.edit();
        assert!(!graft_seed_fragment(
            &mut editable,
            &[],
            &mut SmallRng::seed_from_u64(8)
        ));
    }

    #[test]
    fn bare_atom_refinement_keeps_single_atom_shape() {
        let query = QueryMol::from_str("C").unwrap();

        for seed in 0..64 {
            let mut editable = query.clone().edit();
            assert!(generalize_specialize_atom(
                &mut editable,
                &mut SmallRng::seed_from_u64(seed),
                0,
                MutationDirection::Balanced,
            ));
            let refined = editable.into_query_mol().unwrap();

            assert_eq!(refined.atom_count(), 1);
            assert!(SmartsGenome::from_query_mol(&refined).is_valid());
        }
    }

    #[test]
    fn specialize_atom_primitive_covers_ring_specific_cases() {
        let mut rng = SmallRng::seed_from_u64(9);

        assert!(matches!(
            specialize_atom_primitive(&AtomPrimitive::Wildcard, &mut rng, 0),
            AtomPrimitive::Symbol { .. }
        ));
        assert!(matches!(
            specialize_atom_primitive(&AtomPrimitive::RingMembership(None), &mut rng, 0),
            AtomPrimitive::RingMembership(Some(_))
        ));
        assert!(matches!(
            specialize_atom_primitive(&AtomPrimitive::RingSize(None), &mut rng, 0),
            AtomPrimitive::RingSize(Some(_))
        ));
        assert!(matches!(
            specialize_atom_primitive(&AtomPrimitive::RingConnectivity(None), &mut rng, 0),
            AtomPrimitive::RingConnectivity(Some(_))
        ));
    }

    #[test]
    fn random_atom_primitive_reaches_recursive_and_rich_grammar_nodes() {
        let mut rng = SmallRng::seed_from_u64(123);
        let mut saw_recursive = false;
        let mut saw_isotope = false;
        let mut saw_chirality = false;
        let mut saw_charge = false;
        let mut saw_hybridization = false;
        let mut saw_hetero_neighbor = false;
        let mut saw_implicit_h = false;

        for _ in 0..4096 {
            match random_atom_primitive(&mut rng, 0) {
                AtomPrimitive::RecursiveQuery(_) => saw_recursive = true,
                AtomPrimitive::Isotope { .. } | AtomPrimitive::IsotopeWildcard(_) => {
                    saw_isotope = true
                }
                AtomPrimitive::Chirality(_) => saw_chirality = true,
                AtomPrimitive::Charge(_) => saw_charge = true,
                AtomPrimitive::Hybridization(_) => saw_hybridization = true,
                AtomPrimitive::HeteroNeighbor(_) | AtomPrimitive::AliphaticHeteroNeighbor(_) => {
                    saw_hetero_neighbor = true
                }
                AtomPrimitive::Hydrogen(HydrogenKind::Implicit, _) => saw_implicit_h = true,
                _ => {}
            }
        }

        assert!(saw_recursive);
        assert!(saw_isotope);
        assert!(saw_chirality);
        assert!(saw_charge);
        assert!(saw_hybridization);
        assert!(saw_hetero_neighbor);
        assert!(saw_implicit_h);
    }

    #[test]
    fn random_bond_expr_reaches_boolean_and_directional_grammar_nodes() {
        let mut rng = SmallRng::seed_from_u64(321);
        let mut saw_boolean = false;
        let mut saw_directional = false;
        let mut saw_quadruple = false;

        for _ in 0..1024 {
            match random_bond_expr(&mut rng) {
                BondExpr::Elided => {}
                BondExpr::Query(
                    BondExprTree::Not(_)
                    | BondExprTree::HighAnd(_)
                    | BondExprTree::Or(_)
                    | BondExprTree::LowAnd(_),
                ) => saw_boolean = true,
                BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Bond(
                    Bond::Up | Bond::Down,
                ))) => saw_directional = true,
                BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Bond(Bond::Quadruple))) => {
                    saw_quadruple = true
                }
                _ => {}
            }
        }

        assert!(saw_boolean);
        assert!(saw_directional);
        assert!(saw_quadruple);
    }

    #[test]
    fn recursive_query_mutation_round_trips() {
        let mut editable = QueryMol::from_str("[$(CC)]N").unwrap().edit();
        assert!(mutate_recursive_query(
            &mut editable,
            &[],
            &mut SmallRng::seed_from_u64(55),
            0,
            MutationDirection::Balanced,
        ));
        let query = editable.into_query_mol().unwrap();
        assert!(query.to_string().contains("$("));
        assert!(SmartsGenome::from_query_mol(&query).is_valid());
    }

    #[test]
    fn random_recursive_query_keeps_nested_mutations_valid() {
        for seed in 0..128 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let query = random_recursive_query(&mut rng, 0);
            assert!(genome_from_query(&query).is_some());
        }
    }

    #[test]
    fn ring_and_component_mutations_are_available() {
        let mut ring_query = QueryMol::from_str("CCC").unwrap().edit();
        assert!(ring_edit(&mut ring_query, &mut SmallRng::seed_from_u64(77)));
        let ring_query = ring_query.into_query_mol().unwrap();
        assert!(!ring_query.ring_bonds().is_empty());

        let mut component_query = QueryMol::from_str("CC").unwrap().edit();
        assert!(component_edit(
            &mut component_query,
            &mut SmallRng::seed_from_u64(88),
            0
        ));
        let component_query = component_query.into_query_mol().unwrap();
        assert!(component_query.component_count() >= 1);
    }

    #[test]
    fn atom_map_mutation_only_edits_existing_maps() {
        let mut editable = QueryMol::from_str("C").unwrap().edit();
        assert!(!mutate_atom_map(
            &mut editable,
            &mut SmallRng::seed_from_u64(91)
        ));

        let mut editable = QueryMol::from_str("[#6:1]~[#7]").unwrap().edit();
        assert!(mutate_atom_map(
            &mut editable,
            &mut SmallRng::seed_from_u64(92)
        ));
        let mutated = editable.into_query_mol().unwrap();
        let mapped_atoms = mutated
            .atoms()
            .iter()
            .filter(|atom| {
                matches!(
                    &atom.expr,
                    AtomExpr::Bracket(BracketExpr {
                        atom_map: Some(_),
                        ..
                    })
                )
            })
            .count();

        assert!(mapped_atoms <= 1);
    }

    #[test]
    fn random_numeric_query_reaches_all_supported_range_shapes() {
        let mut rng = SmallRng::seed_from_u64(92);
        let mut saw_closed = false;
        let mut saw_open_lower = false;
        let mut saw_open_upper = false;

        for _ in 0..4096 {
            let NumericQuery::Range(NumericRange { min, max }) =
                random_numeric_query(&mut rng, 1, 6)
            else {
                continue;
            };

            match (min, max) {
                (Some(_), Some(_)) => saw_closed = true,
                (None, Some(_)) => saw_open_lower = true,
                (Some(_), None) => saw_open_upper = true,
                (None, None) => unreachable!("range queries must keep at least one bound"),
            }
        }

        assert!(saw_closed);
        assert!(saw_open_lower);
        assert!(saw_open_upper);
    }

    #[test]
    fn direct_helpers_cover_remaining_mutation_edges() {
        let bracket = BracketExpr {
            tree: BracketExprTree::Primitive(AtomPrimitive::Charge(1)),
            atom_map: Some(3),
        };
        assert_eq!(
            atom_expr_to_bracket(AtomExpr::Bracket(bracket.clone())),
            bracket
        );

        let mut editable = QueryMol::from_str("[$(CC)]N").unwrap().edit();
        assert!(!mutate_recursive_query(
            &mut editable,
            &[],
            &mut SmallRng::seed_from_u64(100),
            MAX_RECURSIVE_QUERY_DEPTH,
            MutationDirection::Balanced,
        ));

        let mut editable = QueryMol::from_str("C").unwrap().edit();
        assert!(!mutate_component_grouping(
            &mut editable,
            &mut SmallRng::seed_from_u64(101),
        ));

        let mut editable = QueryMol::from_str("C").unwrap().edit();
        assert!(!ring_edit(&mut editable, &mut SmallRng::seed_from_u64(102)));

        let mut editable = QueryMol::from_str("CC").unwrap().edit();
        assert!(mutate_bond_constraints(
            &mut editable,
            &mut SmallRng::seed_from_u64(103),
        ));

        let regrouped = normalize_component_groups(vec![Some(4), None, Some(7), Some(4)]);
        assert_eq!(regrouped, vec![Some(0), None, Some(1), Some(0)]);
    }

    #[test]
    fn primitive_generalization_and_specialization_cover_remaining_cases() {
        let mut rng = SmallRng::seed_from_u64(104);

        assert!(matches!(
            generalize_atom_primitive(
                &AtomPrimitive::Symbol {
                    element: Element::C,
                    aromatic: true,
                },
                &mut rng,
                0,
            ),
            AtomPrimitive::AromaticAny
        ));
        assert!(matches!(
            generalize_atom_primitive(&AtomPrimitive::AtomicNumber(6), &mut rng, 0),
            AtomPrimitive::Wildcard
        ));
        assert!(matches!(
            generalize_atom_primitive(&AtomPrimitive::Charge(2), &mut rng, 0),
            AtomPrimitive::Charge(1)
        ));
        assert!(matches!(
            generalize_atom_primitive(&AtomPrimitive::Charge(-2), &mut rng, 0),
            AtomPrimitive::Charge(-1)
        ));

        assert!(matches!(
            specialize_atom_primitive(&AtomPrimitive::AtomicNumber(6), &mut rng, 0),
            AtomPrimitive::IsotopeWildcard(_)
        ));
        assert!(matches!(
            specialize_atom_primitive(
                &AtomPrimitive::Symbol {
                    element: Element::Fe,
                    aromatic: false,
                },
                &mut rng,
                0,
            ),
            AtomPrimitive::AtomicNumber(26)
        ));
        assert!(matches!(
            specialize_atom_primitive(&AtomPrimitive::Charge(1), &mut rng, 0),
            AtomPrimitive::Charge(2)
        ));
        assert!(matches!(
            specialize_atom_primitive(&AtomPrimitive::Charge(-1), &mut rng, 0),
            AtomPrimitive::Charge(-2)
        ));
        assert!(matches!(
            specialize_atom_primitive(
                &AtomPrimitive::RecursiveQuery(Box::new(QueryMol::from_str("[#6]").unwrap())),
                &mut rng,
                0
            ),
            AtomPrimitive::Wildcard
                | AtomPrimitive::AliphaticAny
                | AtomPrimitive::AromaticAny
                | AtomPrimitive::Symbol { .. }
                | AtomPrimitive::Isotope { .. }
                | AtomPrimitive::IsotopeWildcard(_)
                | AtomPrimitive::AtomicNumber(_)
                | AtomPrimitive::Degree(_)
                | AtomPrimitive::Connectivity(_)
                | AtomPrimitive::Valence(_)
                | AtomPrimitive::Hydrogen(_, _)
                | AtomPrimitive::RingMembership(_)
                | AtomPrimitive::RingSize(_)
                | AtomPrimitive::RingConnectivity(_)
                | AtomPrimitive::Hybridization(_)
                | AtomPrimitive::HeteroNeighbor(_)
                | AtomPrimitive::AliphaticHeteroNeighbor(_)
                | AtomPrimitive::Chirality(_)
                | AtomPrimitive::Charge(_)
        ));
    }

    #[test]
    fn random_helpers_reach_optional_and_symbol_variants() {
        let mut rng = SmallRng::seed_from_u64(105);
        let mut saw_bare_aromatic = false;
        let mut saw_bare_aliphatic = false;
        let mut saw_symbol_aromatic = false;
        let mut saw_symbol_aliphatic = false;
        let mut saw_optional_none = false;
        let mut saw_optional_some = false;
        let mut saw_elided = false;

        for _ in 0..2048 {
            match random_bare_atom_expr(&mut rng) {
                AtomExpr::Bare { aromatic: true, .. } => saw_bare_aromatic = true,
                AtomExpr::Bare {
                    aromatic: false, ..
                } => saw_bare_aliphatic = true,
                _ => {}
            }
            match random_symbol_primitive(&mut rng) {
                AtomPrimitive::Symbol { aromatic: true, .. } => saw_symbol_aromatic = true,
                AtomPrimitive::Symbol {
                    aromatic: false, ..
                } => saw_symbol_aliphatic = true,
                _ => {}
            }
            match random_optional_numeric_query(&mut rng, 0, 3) {
                None => saw_optional_none = true,
                Some(_) => saw_optional_some = true,
            }
            if matches!(random_bond_expr(&mut rng), BondExpr::Elided) {
                saw_elided = true;
            }
        }

        assert!(saw_bare_aromatic);
        assert!(saw_bare_aliphatic);
        assert!(saw_symbol_aromatic);
        assert!(saw_symbol_aliphatic);
        assert!(saw_optional_none);
        assert!(saw_optional_some);
        assert!(saw_elided);
    }

    #[test]
    fn widened_value_generators_reach_beyond_old_caps() {
        let mut rng = SmallRng::seed_from_u64(106);
        let mut saw_heavy_atomic_number = false;
        let mut saw_uncommon_symbol = false;
        let mut saw_large_isotope_wildcard = false;
        let mut saw_large_charge = false;

        for _ in 0..4096 {
            if random_atomic_number(&mut rng) > 53 {
                saw_heavy_atomic_number = true;
            }

            if random_isotope_wildcard(&mut rng) > 2095 {
                saw_large_isotope_wildcard = true;
            }

            if random_charge(&mut rng).abs() > 2 {
                saw_large_charge = true;
            }

            if let AtomPrimitive::Symbol { element, .. } = random_symbol_primitive(&mut rng)
                && element.atomic_number() > 53
            {
                saw_uncommon_symbol = true;
            }
        }

        assert!(saw_heavy_atomic_number);
        assert!(saw_uncommon_symbol);
        assert!(saw_large_isotope_wildcard);
        assert!(saw_large_charge);
    }

    #[test]
    fn reset_pool_restarts_from_seed_and_applies_variation() {
        let mutator = SmartsMutation {
            mutation_rate: 1.0,
            reset_pool: vec![SmartsGenome::from_smarts("[#6]").unwrap()],
            reset_probability: 1.0,
            max_mutation_steps: 1,
            attempt_budget: 128,
        };

        for seed in 0..16 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mutated = mutator.mutate(SmartsGenome::from_smarts("[#8]").unwrap(), &mut rng);
            assert_ne!(mutated.smarts(), "[#6]");
        }
    }

    #[test]
    fn mutation_can_apply_multi_step_structural_changes() {
        let mut rng = SmallRng::seed_from_u64(99);
        let mutator = SmartsMutation::new(1.0);
        let genome = SmartsGenome::from_smarts("[#6]~[#7]").unwrap();

        let mut observed_large_change = false;
        for _ in 0..128 {
            let mutated = mutator.mutate(genome.clone(), &mut rng);
            if mutated.is_valid()
                && mutated.smarts() != genome.smarts()
                && mutated.smarts_len().abs_diff(genome.smarts_len()) >= 2
            {
                observed_large_change = true;
                break;
            }
        }

        assert!(observed_large_change);
    }
}
