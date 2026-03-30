use genevo::operator::{GeneticOperator, MutationOp};
use rand::Rng;
use rand::seq::SliceRandom;

use crate::genome::genome::SmartsGenome;
use crate::genome::parser::MAX_SMARTS_TOKENS;
use crate::genome::token::*;

use super::primitives::*;

/// SMARTS-aware mutation operator.
///
/// Mutation types and their probabilities:
/// - Toggle atom primitive (40%): add/remove/change a constraint in a bracketed atom
/// - Generalize/specialize (20%): move up/down the generality ladder
/// - Reset to seed fragment (10%): jump back to a known-good seed-like pattern
/// - Add atom (15%): extend pattern with new atom+bond
/// - Remove atom (10%): prune a terminal atom
/// - Change bond (10%): replace a bond type
/// - Toggle NOT (5%): add/remove `!` negation
#[derive(Clone, Debug)]
pub struct SmartsMutation {
    pub mutation_rate: f64,
    reset_pool: Vec<SmartsGenome>,
    reset_probability: f64,
}

impl SmartsMutation {
    pub fn new(mutation_rate: f64) -> Self {
        Self {
            mutation_rate,
            reset_pool: Vec::new(),
            reset_probability: 0.10,
        }
    }

    pub fn with_reset_pool(mutation_rate: f64, reset_pool: Vec<SmartsGenome>) -> Self {
        Self {
            mutation_rate,
            reset_pool,
            reset_probability: 0.10,
        }
    }
}

impl GeneticOperator for SmartsMutation {
    fn name() -> String {
        "SmartsMutation".to_string()
    }
}

impl MutationOp<SmartsGenome> for SmartsMutation {
    fn mutate<R>(&self, genome: SmartsGenome, rng: &mut R) -> SmartsGenome
    where
        R: Rng + Sized,
    {
        if !rng.gen_bool(self.mutation_rate) {
            return genome;
        }

        if !self.reset_pool.is_empty() && rng.gen_bool(self.reset_probability) {
            return self.reset_pool.choose(rng).cloned().unwrap_or(genome);
        }

        // Try up to 5 times to produce a structurally valid mutation.
        for _ in 0..5 {
            let mut tokens = genome.tokens.clone();
            let roll: f64 = rng.r#gen();

            let mutated = if roll < 0.40 {
                toggle_atom_primitive(&mut tokens, rng)
            } else if roll < 0.60 {
                generalize_specialize(&mut tokens, rng)
            } else if roll < 0.75 {
                add_atom(&mut tokens, rng)
            } else if roll < 0.85 {
                remove_atom(&mut tokens, rng)
            } else if roll < 0.95 {
                change_bond(&mut tokens, rng)
            } else {
                toggle_not(&mut tokens, rng)
            };

            if mutated {
                let candidate = SmartsGenome::from_tokens(tokens);
                if candidate.is_structurally_valid() {
                    return candidate;
                }
            }
        }

        // All attempts failed; return original
        genome
    }
}

/// Find indices of all atom primitives inside brackets.
fn find_bracket_atom_indices(tokens: &[SmartsToken]) -> Vec<usize> {
    let mut indices = Vec::new();
    let mut in_bracket = false;
    for (i, token) in tokens.iter().enumerate() {
        match token {
            SmartsToken::OpenBracket => in_bracket = true,
            SmartsToken::CloseBracket => in_bracket = false,
            SmartsToken::Atom(_) if in_bracket => indices.push(i),
            _ => {}
        }
    }
    indices
}

/// Find indices of all bond tokens.
fn find_bond_indices(tokens: &[SmartsToken]) -> Vec<usize> {
    tokens
        .iter()
        .enumerate()
        .filter_map(|(i, t)| matches!(t, SmartsToken::Bond(_)).then_some(i))
        .collect()
}

/// Find "atom groups" — ranges of tokens forming a single atom expression.
/// Returns (start, end) pairs where tokens[start..end] is one atom.
/// Bare atoms are single tokens; bracketed atoms span from `[` to `]`.
fn find_atom_groups(tokens: &[SmartsToken]) -> Vec<(usize, usize)> {
    let mut groups = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        match &tokens[i] {
            SmartsToken::OpenBracket => {
                let start = i;
                while i < tokens.len() && !matches!(tokens[i], SmartsToken::CloseBracket) {
                    i += 1;
                }
                if i < tokens.len() {
                    i += 1; // include ']'
                }
                groups.push((start, i));
            }
            SmartsToken::Atom(_) => {
                groups.push((i, i + 1));
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    groups
}

/// Toggle atom primitive: add, remove, or change a constraint inside a bracket.
fn toggle_atom_primitive<R: Rng>(tokens: &mut Vec<SmartsToken>, rng: &mut R) -> bool {
    let atom_indices = find_bracket_atom_indices(tokens);
    if atom_indices.is_empty() {
        return false;
    }

    let &idx = atom_indices.choose(rng).unwrap();

    match rng.gen_range(0..3) {
        0 => {
            // Change: replace the primitive with a random one
            tokens[idx] = SmartsToken::Atom(random_atom_primitive(rng));
            true
        }
        1 => {
            // Add: insert a new primitive after this one with `;` separator
            let new_prim = random_atom_primitive(rng);
            tokens.insert(idx + 1, SmartsToken::Atom(new_prim));
            tokens.insert(idx + 1, SmartsToken::Logic(LogicOp::AndLow));
            true
        }
        _ => {
            // Remove: remove this primitive (and preceding logic op if any)
            // Only if there are other primitives in this bracket
            let bracket_atoms: usize = atom_indices
                .iter()
                .filter(|&&ai| {
                    // Check same bracket
                    let before_idx = tokens[..idx]
                        .iter()
                        .rposition(|t| matches!(t, SmartsToken::OpenBracket));
                    let before_ai = tokens[..ai]
                        .iter()
                        .rposition(|t| matches!(t, SmartsToken::OpenBracket));
                    before_idx == before_ai
                })
                .count();

            if bracket_atoms > 1 {
                // Remove preceding logic op if it exists
                if idx > 0 && matches!(tokens[idx - 1], SmartsToken::Logic(_)) {
                    tokens.remove(idx);
                    tokens.remove(idx - 1);
                } else if idx + 1 < tokens.len() && matches!(tokens[idx + 1], SmartsToken::Logic(_))
                {
                    tokens.remove(idx + 1);
                    tokens.remove(idx);
                } else {
                    tokens.remove(idx);
                }
                true
            } else {
                // Only primitive in bracket — change instead of remove
                tokens[idx] = SmartsToken::Atom(random_atom_primitive(rng));
                true
            }
        }
    }
}

/// Generalize or specialize a random atom primitive.
fn generalize_specialize<R: Rng>(tokens: &mut Vec<SmartsToken>, rng: &mut R) -> bool {
    let atom_indices = find_bracket_atom_indices(tokens);
    if atom_indices.is_empty() {
        return false;
    }
    let &idx = atom_indices.choose(rng).unwrap();

    if let SmartsToken::Atom(prim) = &tokens[idx] {
        let new_prim = if rng.gen_bool(0.5) {
            generalize(prim, rng)
        } else {
            specialize(prim, rng)
        };
        tokens[idx] = SmartsToken::Atom(new_prim);
        true
    } else {
        false
    }
}

/// Add a new atom + bond at the end or as a branch.
fn add_atom<R: Rng>(tokens: &mut Vec<SmartsToken>, rng: &mut R) -> bool {
    let atom_groups = find_atom_groups(tokens);
    if atom_groups.is_empty() || tokens.len() >= MAX_SMARTS_TOKENS {
        return false;
    }

    let bond = SmartsToken::Bond(random_bond(rng));
    let new_atom = random_bracketed_atom(rng);

    if rng.gen_bool(0.6) {
        // Append at the end
        tokens.push(bond);
        tokens.extend(new_atom);
    } else {
        // Insert as a branch after a random atom group
        let &(_, end) = atom_groups.choose(rng).unwrap();
        let insert_pos = end.min(tokens.len());
        let mut branch = vec![SmartsToken::BranchOpen, bond];
        branch.extend(new_atom);
        branch.push(SmartsToken::BranchClose);
        for (j, tok) in branch.into_iter().enumerate() {
            tokens.insert(insert_pos + j, tok);
        }
    }

    true
}

/// Remove a terminal atom (last atom group or a branch).
fn remove_atom<R: Rng>(tokens: &mut Vec<SmartsToken>, rng: &mut R) -> bool {
    let atom_groups = find_atom_groups(tokens);
    if atom_groups.len() <= 1 {
        return false; // Don't remove the only atom
    }

    // Try removing the last atom group + preceding bond
    let &(start, end) = atom_groups.last().unwrap();

    // Check if there's a preceding bond
    let bond_start = if start > 0 && matches!(tokens[start - 1], SmartsToken::Bond(_)) {
        start - 1
    } else {
        start
    };

    tokens.drain(bond_start..end);
    prune_empty_branches(tokens);
    let _ = rng; // used for selection in more complex cases
    true
}

fn prune_empty_branches(tokens: &mut Vec<SmartsToken>) {
    let mut i = 0;
    while i + 1 < tokens.len() {
        if matches!(tokens[i], SmartsToken::BranchOpen)
            && matches!(tokens[i + 1], SmartsToken::BranchClose)
        {
            tokens.drain(i..=i + 1);
            i = i.saturating_sub(1);
        } else {
            i += 1;
        }
    }
}

/// Change a random bond type.
fn change_bond<R: Rng>(tokens: &mut Vec<SmartsToken>, rng: &mut R) -> bool {
    let bond_indices = find_bond_indices(tokens);
    if bond_indices.is_empty() {
        return false;
    }
    let &idx = bond_indices.choose(rng).unwrap();
    tokens[idx] = SmartsToken::Bond(random_bond(rng));
    true
}

/// Toggle NOT (`!`) on a random atom primitive inside a bracket.
fn toggle_not<R: Rng>(tokens: &mut Vec<SmartsToken>, rng: &mut R) -> bool {
    let atom_indices = find_bracket_atom_indices(tokens);
    if atom_indices.is_empty() {
        return false;
    }
    let &idx = atom_indices.choose(rng).unwrap();

    // Check if preceded by NOT
    if idx > 0 && matches!(tokens[idx - 1], SmartsToken::Logic(LogicOp::Not)) {
        // Remove the NOT
        tokens.remove(idx - 1);
    } else {
        // Insert NOT before the primitive
        tokens.insert(idx, SmartsToken::Logic(LogicOp::Not));
    }
    let _ = rng;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::display::tokens_to_smarts;
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
        let mutator = SmartsMutation::new(1.0); // Always mutate for testing

        let mut valid = 0;
        let total = 1000;

        for i in 0..total {
            let seed = seeds[i % seeds.len()];
            let genome = SmartsGenome::from_smarts(seed).unwrap();
            let mutated = mutator.mutate(genome, &mut rng);
            if mutated.is_valid_rdkit() {
                valid += 1;
            }
        }

        let rate = valid as f64 / total as f64;
        assert!(
            rate >= 0.80,
            "Mutation validity rate {rate:.2} < 0.80 ({valid}/{total})"
        );
    }

    #[test]
    fn remove_atom_prunes_empty_branch() {
        let mut rng = SmallRng::seed_from_u64(7);
        let mut tokens = SmartsGenome::from_smarts("[#6](~[#7])").unwrap().tokens;
        assert!(remove_atom(&mut tokens, &mut rng));
        assert_eq!(tokens_to_smarts(&tokens), "[#6]");
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
            if mutated.smarts_string == "[#6]" || mutated.smarts_string == "[#7]" {
                observed_reset = true;
                break;
            }
        }

        assert!(observed_reset);
    }
}
