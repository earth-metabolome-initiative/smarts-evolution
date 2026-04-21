use genevo::genetic::{Children, Parents};
use genevo::operator::{CrossoverOp, GeneticOperator};
use rand::Rng;
use rand::seq::SliceRandom;
use smarts_parser::QueryMol;

use crate::genome::SmartsGenome;
use crate::genome::limits::MAX_CROSSOVER_CHILD_COMPLEXITY;

/// SMARTS-aware crossover operator.
///
/// Uses graph-level subtree exchange through `smarts-parser`'s editable query
/// API. Each child is produced by replacing one chain-bond subtree in one
/// parent with a subtree cloned from the other parent.
#[derive(Clone, Debug)]
pub struct SmartsCrossover {
    pub crossover_rate: f64,
}

impl SmartsCrossover {
    pub fn new(crossover_rate: f64) -> Self {
        Self { crossover_rate }
    }
}

impl GeneticOperator for SmartsCrossover {
    fn name() -> String {
        "SmartsCrossover".to_string()
    }
}

impl CrossoverOp<SmartsGenome> for SmartsCrossover {
    fn crossover<R>(&self, parents: Parents<SmartsGenome>, rng: &mut R) -> Children<SmartsGenome>
    where
        R: Rng + Sized,
    {
        if parents.len() < 2 {
            return parents;
        }

        let parent_a = &parents[0];
        let parent_b = &parents[1];

        if !rng.gen_bool(self.crossover_rate) {
            return vec![parent_a.clone(), parent_b.clone()];
        }

        let query_a = parent_a.query();
        let query_b = parent_b.query();

        let Some((anchor_a, child_a, subtree_a)) = choose_subtree_cut(query_a, rng) else {
            return vec![parent_a.clone(), parent_b.clone()];
        };
        let Some((anchor_b, child_b, subtree_b)) = choose_subtree_cut(query_b, rng) else {
            return vec![parent_a.clone(), parent_b.clone()];
        };

        let c1 = build_spliced_child(query_a, anchor_a, child_a, query_b, child_b, &subtree_b)
            .unwrap_or_else(|| parent_a.clone());
        let c2 = build_spliced_child(query_b, anchor_b, child_b, query_a, child_a, &subtree_a)
            .unwrap_or_else(|| parent_b.clone());

        vec![c1, c2]
    }
}

fn choose_subtree_cut<R: Rng>(query: &QueryMol, rng: &mut R) -> Option<(usize, usize, Vec<usize>)> {
    let bond_ids = query.chain_bonds();
    let &bond_id = bond_ids.choose(rng)?;
    let bond = query.bond(bond_id)?;
    let (anchor, child) = if rng.gen_bool(0.5) {
        (bond.src, bond.dst)
    } else {
        (bond.dst, bond.src)
    };
    let subtree = query.rooted_subtree(child, Some(anchor));
    (!subtree.is_empty()).then_some((anchor, child, subtree))
}

fn build_spliced_child(
    recipient: &QueryMol,
    recipient_anchor: usize,
    recipient_child: usize,
    donor: &QueryMol,
    donor_child: usize,
    donor_subtree: &[usize],
) -> Option<SmartsGenome> {
    let fragment = donor.clone_subgraph(donor_subtree).ok()?;
    let fragment_root = donor_subtree
        .iter()
        .position(|&atom_id| atom_id == donor_child)?;

    let mut editable = recipient.edit();
    editable
        .replace_subtree(recipient_anchor, recipient_child, &fragment, fragment_root)
        .ok()?;
    let query = editable.into_query_mol().ok()?;
    genome_from_query(&query)
}

fn genome_from_query(query: &QueryMol) -> Option<SmartsGenome> {
    let genome = SmartsGenome::from_query_mol(query);
    (genome.complexity() <= MAX_CROSSOVER_CHILD_COMPLEXITY && genome.is_valid()).then_some(genome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn test_crossover_validity_rate() {
        let seeds = &[
            "[#6](=[#8])[#7]",
            "[#6;R]~[#6;R]~[#6;R]",
            "[#7;H2]~[#6]",
            "[#6;X3](=[#8])[#8]",
            "[#6]~[#7]~[#6]~[#8]",
        ];

        let mut rng = SmallRng::seed_from_u64(42);
        let crossover = SmartsCrossover::new(1.0); // Always crossover

        let mut valid = 0;
        let total = 500;

        for i in 0..total {
            let a = SmartsGenome::from_smarts(seeds[i % seeds.len()]).unwrap();
            let b = SmartsGenome::from_smarts(seeds[(i + 1) % seeds.len()]).unwrap();
            let parents = vec![a, b];
            let children = crossover.crossover(parents, &mut rng);
            for child in &children {
                if child.is_valid() {
                    valid += 1;
                }
            }
        }

        let rate = valid as f64 / (total * 2) as f64;
        assert!(
            rate >= 0.60,
            "Crossover validity rate {rate:.2} < 0.60 ({valid}/{})",
            total * 2
        );
    }

    #[test]
    fn crossover_name_and_disabled_rate_preserve_parents() {
        let parent_a = SmartsGenome::from_smarts("[#6]~[#7]").unwrap();
        let parent_b = SmartsGenome::from_smarts("[#8]~[#6]").unwrap();
        let mut rng = SmallRng::seed_from_u64(5);
        let crossover = SmartsCrossover::new(0.0);

        assert_eq!(SmartsCrossover::name(), "SmartsCrossover");
        assert_eq!(
            crossover.crossover(vec![parent_a.clone(), parent_b.clone()], &mut rng),
            vec![parent_a, parent_b]
        );
    }

    #[test]
    fn crossover_with_single_parent_returns_input() {
        let parent = SmartsGenome::from_smarts("[#6]~[#7]").unwrap();
        let mut rng = SmallRng::seed_from_u64(6);
        let crossover = SmartsCrossover::new(1.0);

        assert_eq!(
            crossover.crossover(vec![parent.clone()], &mut rng),
            vec![parent]
        );
    }

    #[test]
    fn crossover_returns_parents_without_swappable_subtrees() {
        let parent_a = SmartsGenome::from_smarts("[#6]").unwrap();
        let parent_b = SmartsGenome::from_smarts("[#7]").unwrap();

        let crossover = SmartsCrossover::new(1.0);
        let mut rng = SmallRng::seed_from_u64(123);
        let children = crossover.crossover(vec![parent_a.clone(), parent_b.clone()], &mut rng);

        assert_eq!(children[0].smarts(), parent_a.smarts());
        assert_eq!(children[1].smarts(), parent_b.smarts());
    }

    #[test]
    fn crossover_can_swap_graph_subtrees() {
        let parent_a = SmartsGenome::from_smarts("[#6]~[#7]~[#8]").unwrap();
        let parent_b = SmartsGenome::from_smarts("[#6](~[#17])~[#35]").unwrap();

        let crossover = SmartsCrossover::new(1.0);
        let mut rng = SmallRng::seed_from_u64(9);
        let children = crossover.crossover(vec![parent_a.clone(), parent_b.clone()], &mut rng);

        assert_eq!(children.len(), 2);
        assert!(children.iter().all(SmartsGenome::is_valid));
        assert!(
            children[0].smarts() != parent_a.smarts() || children[1].smarts() != parent_b.smarts()
        );
    }
}
