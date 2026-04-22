use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::str::FromStr;

use elements_rs::isotopes::{CarbonIsotope, NitrogenIsotope, OxygenIsotope};
use elements_rs::{AtomicNumber, Element, ElementVariant, Isotope};
use rand::Rng;
use rand::RngExt;
use rand::prelude::IndexedRandom;
use rand::seq::IteratorRandom;
use smarts_parser::{
    AtomExpr, AtomPrimitive, BondExpr, BondExprTree, BondPrimitive, BracketExpr, BracketExprTree,
    ComponentGroupId, EditableQueryMol, ExprPath, HydrogenKind, NumericQuery, NumericRange,
    QueryMol, add_atom_primitive, add_bond_primitive, normalize_bond_tree, remove_atom_primitive,
    remove_bond_primitive, replace_atom_primitive, replace_bond_primitive,
};
use smiles_parser::atom::bracketed::chirality::Chirality;
use smiles_parser::bond::Bond;

use crate::genome::SmartsGenome;
use crate::genome::limits::{MAX_PRIMITIVES_PER_BRACKET, MAX_SMARTS_COMPLEXITY};

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
        Self {
            mutation_rate,
            reset_pool: Vec::new(),
            reset_probability: 0.20,
            max_mutation_steps: 6,
            attempt_budget: 24,
        }
    }

    pub fn with_reset_pool(mutation_rate: f64, reset_pool: Vec<SmartsGenome>) -> Self {
        Self {
            mutation_rate,
            reset_pool,
            reset_probability: 0.20,
            max_mutation_steps: 6,
            attempt_budget: 24,
        }
    }

    pub fn mutate<R>(&self, genome: SmartsGenome, rng: &mut R) -> SmartsGenome
    where
        R: Rng + Sized,
    {
        if !rng.random_bool(self.mutation_rate) {
            return genome;
        }

        if !self.reset_pool.is_empty() && rng.random_bool(self.reset_probability) {
            let reset_idx = rng.random_range(0..self.reset_pool.len());
            return self.reset_pool[reset_idx].clone();
        }

        for _ in 0..self.attempt_budget {
            let mut editable = genome.query().edit();
            let mutation_steps = rng.random_range(2..=self.max_mutation_steps.max(2));
            let mut mutated = false;

            for _ in 0..mutation_steps {
                mutated |= apply_random_mutation(&mut editable, &self.reset_pool, rng, 0);
            }

            if !mutated {
                continue;
            }

            let Ok(query) = editable.into_query_mol() else {
                continue;
            };
            let Some(candidate) = genome_from_query(&query) else {
                continue;
            };
            return candidate;
        }

        genome
    }
}

fn genome_from_query(query: &QueryMol) -> Option<SmartsGenome> {
    let genome = SmartsGenome::from_query_mol(query);
    (genome.complexity() <= MAX_SMARTS_COMPLEXITY && genome.is_valid()).then_some(genome)
}

fn apply_random_mutation<R: Rng>(
    editable: &mut EditableQueryMol,
    reset_pool: &[SmartsGenome],
    rng: &mut R,
    recursive_depth: usize,
) -> bool {
    let roll: f64 = rng.random();

    if roll < 0.12 {
        mutate_atom_constraints(editable, rng, recursive_depth)
    } else if roll < 0.22 {
        generalize_specialize_atom(editable, rng, recursive_depth)
    } else if roll < 0.30 {
        mutate_atom_map(editable, rng, recursive_depth)
    } else if roll < 0.40 {
        mutate_recursive_query(editable, reset_pool, rng, recursive_depth)
    } else if roll < 0.52 {
        attach_leaf_atom(editable, rng)
    } else if roll < 0.60 {
        insert_atom_on_bond(editable, rng)
    } else if roll < 0.68 {
        remove_leaf_atom(editable, rng)
    } else if roll < 0.78 {
        mutate_bond_constraints(editable, rng)
    } else if roll < 0.86 {
        ring_edit(editable, rng)
    } else if roll < 0.92 {
        component_edit(editable, rng, recursive_depth)
    } else if roll < 0.97 {
        toggle_atom_not(editable, rng)
    } else {
        graft_seed_fragment(editable, reset_pool, rng)
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
                    let path = primitive_paths.choose(rng).unwrap();
                    replace_atom_primitive(
                        &mut expr,
                        path,
                        random_atom_primitive(rng, recursive_depth),
                    )
                    .is_ok()
                }
                2 if primitive_paths.len() > 1 => {
                    let path = primitive_paths.choose(rng).unwrap();
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
        _ => editable
            .replace_atom_expr(atom_id, random_atom_expr(rng, recursive_depth))
            .is_ok(),
    }
}

fn generalize_specialize_atom<R: Rng>(
    editable: &mut EditableQueryMol,
    rng: &mut R,
    recursive_depth: usize,
) -> bool {
    let (atom_id, atom_expr) = random_atom(editable, rng);

    let AtomExpr::Bracket(mut expr) = atom_expr else {
        return editable
            .replace_atom_expr(atom_id, random_atom_expr(rng, recursive_depth))
            .is_ok();
    };
    let primitive_paths = bracket_primitive_paths(&expr);
    let Some(path) = primitive_paths.choose(rng) else {
        return false;
    };
    let Some(BracketExprTree::Primitive(current)) = expr.tree.get(path) else {
        return false;
    };
    let replacement = if rng.random_bool(0.5) {
        generalize_atom_primitive(current, rng, recursive_depth)
    } else {
        specialize_atom_primitive(current, rng, recursive_depth)
    };

    replace_atom_primitive(&mut expr, path, replacement).is_ok()
        && editable
            .replace_atom_expr(atom_id, AtomExpr::Bracket(expr))
            .is_ok()
}

fn mutate_atom_map<R: Rng>(
    editable: &mut EditableQueryMol,
    rng: &mut R,
    recursive_depth: usize,
) -> bool {
    let (atom_id, atom_expr) = random_atom(editable, rng);
    let mut expr = match atom_expr {
        AtomExpr::Bracket(expr) => expr,
        other => atom_expr_to_bracket(other, recursive_depth),
    };

    expr.atom_map = match expr.atom_map {
        Some(_) if rng.random_bool(0.35) => None,
        _ => Some(rng.random_range(1..=MAX_ATOM_MAP)),
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
                let mut expr = atom_expr_to_bracket(other, recursive_depth);
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

    let (atom_id, path) = sites.choose(rng).unwrap().clone();
    let mut atom_expr = editable.as_query_mol().atom(atom_id).unwrap().expr.clone();
    let AtomExpr::Bracket(bracket) = &mut atom_expr else {
        return false;
    };
    let Some(BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(inner))) =
        bracket.tree.get_mut(&path)
    else {
        return false;
    };

    let mutated_inner = if rng.random_bool(0.30) {
        **inner = random_recursive_query(rng, recursive_depth + 1);
        true
    } else {
        let mut nested = inner.edit();
        apply_random_mutation(&mut nested, reset_pool, rng, recursive_depth + 1)
            && nested
                .into_query_mol()
                .map(|query| {
                    **inner = query;
                })
                .is_ok()
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

    let current = editable.as_query_mol().bond(bond_id).unwrap().expr.clone();
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
                    let path = primitive_paths.choose(rng).unwrap();
                    replace_bond_primitive(&mut tree, path, random_bond_primitive(rng)).is_ok()
                }
                2 if primitive_paths.len() > 1 => {
                    let path = primitive_paths.choose(rng).unwrap();
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
        return editable.open_ring(*ring_bonds.choose(rng).unwrap()).is_ok();
    }

    let mut candidates = Vec::new();
    for component in 0..snapshot.component_count() {
        let atoms = snapshot.component_atoms(component);
        for (idx, &left) in atoms.iter().enumerate() {
            for &right in &atoms[idx + 1..] {
                if snapshot.bond_between(left, right).is_none() {
                    candidates.push((left, right));
                }
            }
        }
    }

    let Some(&(left, right)) = candidates.choose(rng) else {
        return false;
    };
    editable
        .close_ring(left, right, random_bond_expr(rng))
        .is_ok()
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
        .filter(|&atom_id| {
            snapshot
                .component_atoms(snapshot.atom(atom_id).unwrap().component)
                .len()
                == 1
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

fn atom_expr_to_bracket(atom_expr: AtomExpr, _recursive_depth: usize) -> BracketExpr {
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
            .random_bool(0.15)
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
    match rng.random_range(0..20) {
        0 => AtomPrimitive::Wildcard,
        1 => AtomPrimitive::AliphaticAny,
        2 => AtomPrimitive::AromaticAny,
        3 => random_symbol_primitive(rng),
        4 => AtomPrimitive::Isotope {
            isotope: random_isotope(rng),
            aromatic: rng.random_bool(0.2),
        },
        5 => AtomPrimitive::IsotopeWildcard(random_isotope_wildcard(rng)),
        6 => AtomPrimitive::AtomicNumber(random_atomic_number(rng)),
        7 => AtomPrimitive::Degree(random_optional_numeric_query(rng, 0, MAX_DEGREE)),
        8 => AtomPrimitive::Connectivity(random_optional_numeric_query(rng, 0, MAX_CONNECTIVITY)),
        9 => AtomPrimitive::Valence(random_optional_numeric_query(rng, 0, MAX_VALENCE)),
        10 if recursive_depth < MAX_RECURSIVE_QUERY_DEPTH => AtomPrimitive::RecursiveQuery(
            Box::new(random_recursive_query(rng, recursive_depth + 1)),
        ),
        10 | 11 => AtomPrimitive::Hydrogen(
            if rng.random_bool(0.5) {
                HydrogenKind::Total
            } else {
                HydrogenKind::Implicit
            },
            random_optional_numeric_query(rng, 0, MAX_HYDROGEN_COUNT),
        ),
        12 => AtomPrimitive::RingMembership(random_optional_numeric_query(
            rng,
            0,
            MAX_RING_MEMBERSHIP,
        )),
        13 => AtomPrimitive::RingSize(random_optional_numeric_query(rng, 3, MAX_RING_SIZE)),
        14 => AtomPrimitive::RingConnectivity(random_optional_numeric_query(
            rng,
            0,
            MAX_RING_CONNECTIVITY,
        )),
        15 => AtomPrimitive::Hybridization(random_numeric_query(rng, 1, MAX_HYBRIDIZATION)),
        16 => AtomPrimitive::HeteroNeighbor(random_optional_numeric_query(
            rng,
            0,
            MAX_HETERO_NEIGHBOR,
        )),
        17 => AtomPrimitive::AliphaticHeteroNeighbor(random_optional_numeric_query(
            rng,
            0,
            MAX_ALIPHATIC_HETERO_NEIGHBOR,
        )),
        18 => AtomPrimitive::Chirality(random_chirality(rng)),
        _ => AtomPrimitive::Charge(random_charge(rng)),
    }
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
    match rng.random_range(0..9) {
        0 => BondPrimitive::Bond(Bond::Single),
        1 => BondPrimitive::Bond(Bond::Double),
        2 => BondPrimitive::Bond(Bond::Triple),
        3 => BondPrimitive::Bond(Bond::Quadruple),
        4 => BondPrimitive::Bond(Bond::Aromatic),
        5 => BondPrimitive::Bond(Bond::Up),
        6 => BondPrimitive::Bond(Bond::Down),
        7 => BondPrimitive::Any,
        _ => BondPrimitive::Ring,
    }
}

fn random_atomic_number<R: Rng>(rng: &mut R) -> u16 {
    rng.random_range(1..=MAX_ATOMIC_NUMBER)
}

fn random_symbol_primitive<R: Rng>(rng: &mut R) -> AtomPrimitive {
    let aromatic = rng.random_bool(0.30);
    let element = random_supported_element(rng, aromatic);
    AtomPrimitive::Symbol { element, aromatic }
}

fn random_supported_element<R: Rng>(rng: &mut R, aromatic: bool) -> Element {
    let symbol = if aromatic {
        *AROMATIC_ELEMENT_SYMBOLS.choose(rng).unwrap()
    } else {
        *ALL_ELEMENT_SYMBOLS.choose(rng).unwrap()
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
    rng.random_range(0..=MAX_ISOTOPE_WILDCARD)
}

fn random_charge<R: Rng>(rng: &mut R) -> i8 {
    rng.random_range(-MAX_ABS_CHARGE..=MAX_ABS_CHARGE)
}

fn random_chirality<R: Rng>(rng: &mut R) -> Chirality {
    match rng.random_range(0..7) {
        0 => Chirality::At,
        1 => Chirality::AtAt,
        2 => Chirality::try_th(rng.random_range(1..=2)).unwrap(),
        3 => Chirality::try_al(rng.random_range(1..=2)).unwrap(),
        4 => Chirality::try_sp(rng.random_range(1..=3)).unwrap(),
        5 => Chirality::try_tb(rng.random_range(1..=20)).unwrap(),
        _ => Chirality::try_oh(rng.random_range(1..=30)).unwrap(),
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

fn random_recursive_query<R: Rng>(rng: &mut R, recursive_depth: usize) -> QueryMol {
    let smarts = RECURSIVE_FRAGMENT_SMARTS.choose(rng).unwrap();
    let mut query = QueryMol::from_str(smarts).unwrap();
    if recursive_depth < MAX_RECURSIVE_QUERY_DEPTH && rng.random_bool(0.35) {
        let mut editable = query.edit();
        let _ = apply_random_mutation(&mut editable, &[], rng, recursive_depth);
        query = editable.into_query_mol().unwrap_or(query);
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
    fn genome_from_query_rejects_overly_complex_queries() {
        let over_limit = std::iter::repeat_n("[#6]", MAX_SMARTS_COMPLEXITY + 1)
            .collect::<Vec<_>>()
            .join("~");
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

        for _ in 0..1024 {
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
            0
        ));
        let query = editable.into_query_mol().unwrap();
        assert!(query.to_string().contains("$("));
        assert!(SmartsGenome::from_query_mol(&query).is_valid());
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
    fn atom_map_mutation_can_introduce_maps() {
        let mut editable = QueryMol::from_str("C").unwrap().edit();
        assert!(mutate_atom_map(
            &mut editable,
            &mut SmallRng::seed_from_u64(91),
            0
        ));
        let query = editable.into_query_mol().unwrap();
        assert!(query.to_string().contains(':'));
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
            atom_expr_to_bracket(AtomExpr::Bracket(bracket.clone()), 0),
            bracket
        );

        let mut editable = QueryMol::from_str("[$(CC)]N").unwrap().edit();
        assert!(!mutate_recursive_query(
            &mut editable,
            &[],
            &mut SmallRng::seed_from_u64(100),
            MAX_RECURSIVE_QUERY_DEPTH,
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
    fn reset_pool_can_return_seed_fragment() {
        let mut rng = SmallRng::seed_from_u64(1);
        let mutator = SmartsMutation::with_reset_pool(
            1.0,
            vec![
                SmartsGenome::from_smarts("[#6]").unwrap(),
                SmartsGenome::from_smarts("[#7]").unwrap(),
            ],
        );

        let mut observed_reset = false;
        for _ in 0..256 {
            let mutated = mutator.mutate(SmartsGenome::from_smarts("[#8]").unwrap(), &mut rng);
            if mutated.smarts() == "[#6]" || mutated.smarts() == "[#7]" {
                observed_reset = true;
                break;
            }
        }

        assert!(observed_reset);
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
                && mutated.complexity().abs_diff(genome.complexity()) >= 2
            {
                observed_large_change = true;
                break;
            }
        }

        assert!(observed_large_change);
    }
}
