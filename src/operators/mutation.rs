use alloc::boxed::Box;
use alloc::vec::Vec;

use rand::Rng;
use rand::seq::{IteratorRandom, SliceRandom};
use smarts_parser::{
    AtomExpr, AtomPrimitive, BondExpr, BondExprTree, BondPrimitive, BracketExpr, BracketExprTree,
    EditableQueryMol, ExprPath, HydrogenKind, NumericQuery, QueryMol, add_atom_primitive,
    remove_atom_primitive, replace_atom_primitive,
};
use smiles_parser::bond::Bond;

use crate::genome::SmartsGenome;
use crate::genome::limits::{MAX_PRIMITIVES_PER_BRACKET, MAX_SMARTS_COMPLEXITY};

const COMMON_ATOMIC_NUMBERS: &[u16] = &[6, 7, 8, 9, 15, 16, 17, 35, 53];

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
        if !rng.gen_bool(self.mutation_rate) {
            return genome;
        }

        if !self.reset_pool.is_empty() && rng.gen_bool(self.reset_probability) {
            let reset_idx = rng.gen_range(0..self.reset_pool.len());
            return self.reset_pool[reset_idx].clone();
        }

        for _ in 0..self.attempt_budget {
            let mut editable = genome.query().edit();
            let mutation_steps = rng.gen_range(2..=self.max_mutation_steps.max(2));
            let mut mutated = false;

            for _ in 0..mutation_steps {
                mutated |= apply_random_mutation(&mut editable, &self.reset_pool, rng);
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
) -> bool {
    let roll: f64 = rng.r#gen();

    if roll < 0.18 {
        mutate_atom_constraints(editable, rng)
    } else if roll < 0.33 {
        generalize_specialize_atom(editable, rng)
    } else if roll < 0.50 {
        attach_leaf_atom(editable, rng)
    } else if roll < 0.62 {
        insert_atom_on_bond(editable, rng)
    } else if roll < 0.74 {
        remove_leaf_atom(editable, rng)
    } else if roll < 0.86 {
        change_bond_expr(editable, rng)
    } else if roll < 0.94 {
        toggle_atom_not(editable, rng)
    } else {
        graft_seed_fragment(editable, reset_pool, rng)
    }
}

fn random_atom<R: Rng>(editable: &EditableQueryMol, rng: &mut R) -> (usize, AtomExpr) {
    let atoms = editable.as_query_mol().atoms();
    debug_assert!(!atoms.is_empty());
    let atom = &atoms[rng.gen_range(0..atoms.len())];
    (atom.id, atom.expr.clone())
}

fn random_atom_id<R: Rng>(editable: &EditableQueryMol, rng: &mut R) -> usize {
    let atoms = editable.as_query_mol().atoms();
    debug_assert!(!atoms.is_empty());
    atoms[rng.gen_range(0..atoms.len())].id
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

fn mutate_atom_constraints<R: Rng>(editable: &mut EditableQueryMol, rng: &mut R) -> bool {
    let (atom_id, atom_expr) = random_atom(editable, rng);

    match atom_expr {
        AtomExpr::Bracket(mut expr) => {
            let primitive_paths = bracket_primitive_paths(&expr);
            let mutated = match rng.gen_range(0..4) {
                0 if primitive_paths.len() < MAX_PRIMITIVES_PER_BRACKET => {
                    add_atom_primitive(&mut expr, random_atom_primitive(rng)).is_ok()
                }
                1 if !primitive_paths.is_empty() => {
                    let path = primitive_paths.choose(rng).unwrap();
                    replace_atom_primitive(&mut expr, path, random_atom_primitive(rng)).is_ok()
                }
                2 if primitive_paths.len() > 1 => {
                    let path = primitive_paths.choose(rng).unwrap();
                    remove_atom_primitive(&mut expr, path).is_ok()
                }
                _ => {
                    expr = random_bracket_expr(rng);
                    true
                }
            };

            mutated
                && editable
                    .replace_atom_expr(atom_id, AtomExpr::Bracket(expr))
                    .is_ok()
        }
        _ => editable
            .replace_atom_expr(atom_id, random_atom_expr(rng))
            .is_ok(),
    }
}

fn generalize_specialize_atom<R: Rng>(editable: &mut EditableQueryMol, rng: &mut R) -> bool {
    let (atom_id, atom_expr) = random_atom(editable, rng);

    let AtomExpr::Bracket(mut expr) = atom_expr else {
        return editable
            .replace_atom_expr(atom_id, random_atom_expr(rng))
            .is_ok();
    };
    let primitive_paths = bracket_primitive_paths(&expr);
    let Some(path) = primitive_paths.choose(rng) else {
        return false;
    };
    let Some(BracketExprTree::Primitive(current)) = expr.tree.get(path) else {
        return false;
    };
    let replacement = if rng.gen_bool(0.5) {
        generalize_atom_primitive(current, rng)
    } else {
        specialize_atom_primitive(current, rng)
    };

    replace_atom_primitive(&mut expr, path, replacement).is_ok()
        && editable
            .replace_atom_expr(atom_id, AtomExpr::Bracket(expr))
            .is_ok()
}

fn attach_leaf_atom<R: Rng>(editable: &mut EditableQueryMol, rng: &mut R) -> bool {
    let parent = random_atom_id(editable, rng);

    editable
        .attach_leaf(parent, random_bond_expr(rng), random_atom_expr(rng))
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
            random_atom_expr(rng),
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

fn change_bond_expr<R: Rng>(editable: &mut EditableQueryMol, rng: &mut R) -> bool {
    let Some(bond_id) = random_bond_id(editable, rng) else {
        return false;
    };

    editable
        .replace_bond_expr(bond_id, random_bond_expr(rng))
        .is_ok()
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

fn graft_seed_fragment<R: Rng>(
    editable: &mut EditableQueryMol,
    reset_pool: &[SmartsGenome],
    rng: &mut R,
) -> bool {
    if reset_pool.is_empty() {
        return false;
    }

    let parent = random_atom_id(editable, rng);
    let fragment_genome = &reset_pool[rng.gen_range(0..reset_pool.len())];

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

fn random_atom_expr<R: Rng>(rng: &mut R) -> AtomExpr {
    AtomExpr::Bracket(random_bracket_expr(rng))
}

fn random_bracket_expr<R: Rng>(rng: &mut R) -> BracketExpr {
    let mut expr = BracketExpr {
        tree: BracketExprTree::Primitive(random_atom_primitive(rng)),
        atom_map: None,
    };

    if rng.gen_bool(0.30) {
        let _ = add_atom_primitive(&mut expr, random_atom_primitive(rng));
    }

    expr
}

fn random_atom_primitive<R: Rng>(rng: &mut R) -> AtomPrimitive {
    match rng.gen_range(0..11) {
        0 => AtomPrimitive::Wildcard,
        1 => AtomPrimitive::AliphaticAny,
        2 => AtomPrimitive::AromaticAny,
        3 => AtomPrimitive::AtomicNumber(random_atomic_number(rng)),
        4 => AtomPrimitive::Degree(Some(NumericQuery::Exact(rng.gen_range(1..=4)))),
        5 => AtomPrimitive::Connectivity(Some(NumericQuery::Exact(rng.gen_range(1..=4)))),
        6 => AtomPrimitive::Hydrogen(
            HydrogenKind::Total,
            Some(NumericQuery::Exact(rng.gen_range(0..=3))),
        ),
        7 => AtomPrimitive::RingMembership(if rng.gen_bool(0.5) {
            Some(NumericQuery::Exact(rng.gen_range(0..=3)))
        } else {
            None
        }),
        8 => AtomPrimitive::RingSize(if rng.gen_bool(0.5) {
            Some(NumericQuery::Exact(*[5, 6, 7].choose(rng).unwrap()))
        } else {
            None
        }),
        9 => AtomPrimitive::RingConnectivity(if rng.gen_bool(0.5) {
            Some(NumericQuery::Exact(rng.gen_range(0..=3)))
        } else {
            None
        }),
        _ => AtomPrimitive::Charge(*[-1, 1, 2].choose(rng).unwrap()),
    }
}

fn generalize_atom_primitive<R: Rng>(primitive: &AtomPrimitive, rng: &mut R) -> AtomPrimitive {
    match primitive {
        AtomPrimitive::AtomicNumber(_) => AtomPrimitive::Wildcard,
        AtomPrimitive::Degree(Some(NumericQuery::Exact(value))) if *value > 1 => {
            AtomPrimitive::Degree(Some(NumericQuery::Exact(value - 1)))
        }
        AtomPrimitive::Connectivity(Some(NumericQuery::Exact(value))) if *value > 1 => {
            AtomPrimitive::Connectivity(Some(NumericQuery::Exact(value - 1)))
        }
        AtomPrimitive::Hydrogen(kind, Some(NumericQuery::Exact(value))) if *value < 3 => {
            AtomPrimitive::Hydrogen(*kind, Some(NumericQuery::Exact(value + 1)))
        }
        AtomPrimitive::RingMembership(Some(_)) => AtomPrimitive::RingMembership(None),
        AtomPrimitive::RingSize(Some(_)) => AtomPrimitive::RingSize(None),
        AtomPrimitive::RingConnectivity(Some(_)) => AtomPrimitive::RingConnectivity(None),
        AtomPrimitive::Charge(_) => AtomPrimitive::Wildcard,
        _ => random_atom_primitive(rng),
    }
}

fn specialize_atom_primitive<R: Rng>(primitive: &AtomPrimitive, rng: &mut R) -> AtomPrimitive {
    match primitive {
        AtomPrimitive::Wildcard | AtomPrimitive::AliphaticAny | AtomPrimitive::AromaticAny => {
            AtomPrimitive::AtomicNumber(random_atomic_number(rng))
        }
        AtomPrimitive::RingMembership(None) => {
            AtomPrimitive::RingMembership(Some(NumericQuery::Exact(rng.gen_range(1..=3))))
        }
        AtomPrimitive::RingSize(None) => {
            AtomPrimitive::RingSize(Some(NumericQuery::Exact(*[5, 6, 7].choose(rng).unwrap())))
        }
        AtomPrimitive::RingConnectivity(None) => {
            AtomPrimitive::RingConnectivity(Some(NumericQuery::Exact(rng.gen_range(1..=3))))
        }
        AtomPrimitive::Degree(Some(NumericQuery::Exact(value))) if *value < 4 => {
            AtomPrimitive::Degree(Some(NumericQuery::Exact(value + 1)))
        }
        AtomPrimitive::Connectivity(Some(NumericQuery::Exact(value))) if *value < 4 => {
            AtomPrimitive::Connectivity(Some(NumericQuery::Exact(value + 1)))
        }
        AtomPrimitive::Hydrogen(kind, Some(NumericQuery::Exact(value))) if *value > 0 => {
            AtomPrimitive::Hydrogen(*kind, Some(NumericQuery::Exact(value - 1)))
        }
        _ => random_atom_primitive(rng),
    }
}

fn random_bond_expr<R: Rng>(rng: &mut R) -> BondExpr {
    if rng.gen_bool(0.20) {
        BondExpr::Elided
    } else {
        BondExpr::Query(BondExprTree::Primitive(random_bond_primitive(rng)))
    }
}

fn random_bond_primitive<R: Rng>(rng: &mut R) -> BondPrimitive {
    match rng.gen_range(0..6) {
        0 => BondPrimitive::Bond(Bond::Single),
        1 => BondPrimitive::Bond(Bond::Double),
        2 => BondPrimitive::Bond(Bond::Triple),
        3 => BondPrimitive::Bond(Bond::Aromatic),
        4 => BondPrimitive::Any,
        _ => BondPrimitive::Ring,
    }
}

fn random_atomic_number<R: Rng>(rng: &mut R) -> u16 {
    *COMMON_ATOMIC_NUMBERS.choose(rng).unwrap()
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use std::string::ToString;
    use std::vec;

    use super::*;
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
            &mut SmallRng::seed_from_u64(3)
        ));

        let mut editable = query.edit();
        assert!(generalize_specialize_atom(
            &mut editable,
            &mut SmallRng::seed_from_u64(4)
        ));

        let mut editable = query.edit();
        assert!(!change_bond_expr(
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
            specialize_atom_primitive(&AtomPrimitive::Wildcard, &mut rng),
            AtomPrimitive::AtomicNumber(_)
        ));
        assert!(matches!(
            specialize_atom_primitive(&AtomPrimitive::RingMembership(None), &mut rng),
            AtomPrimitive::RingMembership(Some(_))
        ));
        assert!(matches!(
            specialize_atom_primitive(&AtomPrimitive::RingSize(None), &mut rng),
            AtomPrimitive::RingSize(Some(_))
        ));
        assert!(matches!(
            specialize_atom_primitive(&AtomPrimitive::RingConnectivity(None), &mut rng),
            AtomPrimitive::RingConnectivity(Some(_))
        ));
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
