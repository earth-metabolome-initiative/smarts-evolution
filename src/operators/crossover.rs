use genevo::genetic::{Children, Parents};
use genevo::operator::{CrossoverOp, GeneticOperator};
use rand::Rng;

use crate::genome::genome::SmartsGenome;
use crate::genome::parser::MAX_CROSSOVER_CHILD_TOKENS;
use crate::genome::token::SmartsToken;

/// SMARTS-aware crossover operator.
///
/// Uses fragment exchange at atom-group boundaries:
/// 1. Identify atom groups in each parent (bracketed atoms or bare atoms)
/// 2. Pick a random crossover point in each parent at an atom-group boundary
/// 3. Exchange suffixes
/// 4. Validate both children structurally; replace invalid ones with parent clones
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

        // Find atom-group boundaries
        let bounds_a = atom_group_boundaries(&parent_a.tokens);
        let bounds_b = atom_group_boundaries(&parent_b.tokens);

        if bounds_a.len() < 2 || bounds_b.len() < 2 {
            return vec![parent_a.clone(), parent_b.clone()];
        }

        // Pick crossover points (not at 0 or end)
        let cut_a = bounds_a[rng.gen_range(1..bounds_a.len())];
        let cut_b = bounds_b[rng.gen_range(1..bounds_b.len())];

        // Child 1: prefix of A + suffix of B
        let mut child1_tokens: Vec<SmartsToken> = parent_a.tokens[..cut_a].to_vec();
        child1_tokens.extend_from_slice(&parent_b.tokens[cut_b..]);

        // Child 2: prefix of B + suffix of A
        let mut child2_tokens: Vec<SmartsToken> = parent_b.tokens[..cut_b].to_vec();
        child2_tokens.extend_from_slice(&parent_a.tokens[cut_a..]);

        if child1_tokens.len() > MAX_CROSSOVER_CHILD_TOKENS
            || child2_tokens.len() > MAX_CROSSOVER_CHILD_TOKENS
        {
            return vec![parent_a.clone(), parent_b.clone()];
        }

        let child1 = SmartsGenome::from_tokens(child1_tokens);
        let child2 = SmartsGenome::from_tokens(child2_tokens);

        // Validate and fall back to parents if invalid.
        let c1 = if child1.is_structurally_valid() {
            child1
        } else {
            parent_a.clone()
        };
        let c2 = if child2.is_structurally_valid() {
            child2
        } else {
            parent_b.clone()
        };

        vec![c1, c2]
    }
}

/// Find token positions that are atom-group boundaries.
///
/// A boundary is a position between two atom groups — after a `]` or bare atom
/// and before a bond, `(`, or `[`.
fn atom_group_boundaries(tokens: &[SmartsToken]) -> Vec<usize> {
    let mut boundaries = vec![0]; // Start is always a boundary
    let mut i = 0;

    while i < tokens.len() {
        match &tokens[i] {
            SmartsToken::OpenBracket => {
                // Skip to closing bracket
                while i < tokens.len() && !matches!(tokens[i], SmartsToken::CloseBracket) {
                    i += 1;
                }
                i += 1; // past ']'
                boundaries.push(i);
            }
            SmartsToken::Atom(_) => {
                i += 1;
                boundaries.push(i);
            }
            _ => {
                i += 1;
            }
        }
    }

    boundaries.dedup();
    // Only keep boundaries within range
    boundaries.retain(|&b| b <= tokens.len());
    boundaries
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::parser::MAX_CROSSOVER_CHILD_TOKENS;
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
                if child.is_valid_matcher() {
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
    fn crossover_rejects_children_above_token_cap() {
        let atom = vec![
            SmartsToken::OpenBracket,
            SmartsToken::Atom(crate::genome::token::AtomPrimitive::AtomicNumber(6)),
            SmartsToken::CloseBracket,
        ];
        let mut long_tokens = Vec::new();
        for _ in 0..(MAX_CROSSOVER_CHILD_TOKENS / 3 + 2) {
            long_tokens.extend(atom.iter().cloned());
        }
        let parent_a = SmartsGenome::from_tokens(long_tokens.clone());
        let parent_b = SmartsGenome::from_tokens(long_tokens);

        let crossover = SmartsCrossover::new(1.0);
        let mut rng = SmallRng::seed_from_u64(123);
        let children = crossover.crossover(vec![parent_a.clone(), parent_b.clone()], &mut rng);

        assert_eq!(children[0].smarts_string, parent_a.smarts_string);
        assert_eq!(children[1].smarts_string, parent_b.smarts_string);
    }
}
