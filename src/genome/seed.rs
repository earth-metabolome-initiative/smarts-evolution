use genevo::population::GenomeBuilder;
use rand::Rng;
use rand::seq::SliceRandom;

use super::genome::SmartsGenome;
use super::token::*;
use crate::data::rdkit_lock::with_rdkit_lock;

/// Builds initial SMARTS genomes for a population.
///
/// Uses three strategies:
/// - SMILES-derived seeds: extract atom-environment patterns from positive example SMILES
/// - PAP seeds: simple 1-3 atom SMARTS from a catalog of primitives
/// - Random seeds: randomly assemble small SMARTS patterns
pub struct SmartsGenomeBuilder {
    /// SMILES strings from positive examples (for deriving seed patterns).
    pub positive_smiles: Vec<String>,
}

impl SmartsGenomeBuilder {
    pub fn new(positive_smiles: Vec<String>) -> Self {
        Self { positive_smiles }
    }
}

impl GenomeBuilder<SmartsGenome> for SmartsGenomeBuilder {
    fn build_genome<R>(&self, index: usize, rng: &mut R) -> SmartsGenome
    where
        R: Rng + Sized,
    {
        // Mix strategies: 60% SMILES-derived, 25% PAP, 15% random
        let choice = index % 20;
        let genome = if choice < 12 && !self.positive_smiles.is_empty() {
            smiles_derived_seed(&self.positive_smiles, rng)
        } else if choice < 17 {
            pap_seed(rng)
        } else {
            random_seed(rng)
        };

        // Validate with RDKit; fall back to a safe pattern if invalid
        if genome.is_valid_rdkit() {
            genome
        } else {
            fallback_seed(rng)
        }
    }
}

/// Generate a SMARTS seed by converting a random positive SMILES to a basic atom pattern.
///
/// Takes a random SMILES, extracts 2-4 atoms with their bonds, and converts
/// element symbols to SMARTS atom primitives (e.g., `C` → `[#6]`).
fn smiles_derived_seed<R: Rng>(positive_smiles: &[String], rng: &mut R) -> SmartsGenome {
    let smiles = &positive_smiles[rng.gen_range(0..positive_smiles.len())];

    // Parse SMILES into RDKit and extract a simple substructure
    if let Some(canon) = with_rdkit_lock(|| {
        rdkit::ROMol::from_smiles(smiles)
            .ok()
            .map(|mol| mol.as_smiles())
    }) {
        // Take a prefix of the canonical SMILES (2-6 heavy atoms)
        let atom_count = rng.gen_range(2..=6);
        let smarts = smiles_prefix_to_smarts(&canon, atom_count);
        if let Ok(g) = SmartsGenome::from_smarts(&smarts) {
            if g.is_valid_rdkit() {
                return g;
            }
        }
    }

    // Fallback to PAP
    pap_seed(rng)
}

/// Convert the first N atoms of a SMILES to a basic SMARTS pattern.
fn smiles_prefix_to_smarts(smiles: &str, max_atoms: usize) -> String {
    let mut out = String::new();
    let mut atom_count = 0;
    let chars: Vec<char> = smiles.chars().collect();
    let mut i = 0;

    while i < chars.len() && atom_count < max_atoms {
        match chars[i] {
            // Bracketed atom — convert to atomic number form
            '[' => {
                // Find closing bracket
                let start = i;
                while i < chars.len() && chars[i] != ']' {
                    i += 1;
                }
                if i < chars.len() {
                    i += 1; // skip ']'
                }
                // Extract the bracketed content and pass through
                let bracket: String = chars[start..i].iter().collect();
                out.push_str(&bracket);
                atom_count += 1;
            }
            // Organic subset atoms — convert to [#N] form
            c @ ('C' | 'N' | 'O' | 'S' | 'P' | 'B' | 'F' | 'I') => {
                let (num, skip) = match c {
                    'C' if i + 1 < chars.len() && chars[i + 1] == 'l' => (17, 2),
                    'B' if i + 1 < chars.len() && chars[i + 1] == 'r' => (35, 2),
                    'C' => (6, 1),
                    'N' => (7, 1),
                    'O' => (8, 1),
                    'S' => (16, 1),
                    'P' => (15, 1),
                    'B' => (5, 1),
                    'F' => (9, 1),
                    'I' => (53, 1),
                    _ => unreachable!(),
                };
                out.push_str(&format!("[#{num}]"));
                i += skip;
                atom_count += 1;
            }
            c @ ('c' | 'n' | 'o' | 's' | 'p') => {
                let num = match c {
                    'c' => 6,
                    'n' => 7,
                    'o' => 8,
                    's' => 16,
                    'p' => 15,
                    _ => unreachable!(),
                };
                out.push_str(&format!("[#{num}]"));
                i += 1;
                atom_count += 1;
            }
            // Bonds — pass through as SMARTS bond
            '-' => {
                out.push('~');
                i += 1;
            }
            '=' => {
                out.push('=');
                i += 1;
            }
            '#' => {
                out.push('#');
                i += 1;
            }
            ':' => {
                out.push('~');
                i += 1;
            }
            // Branches
            '(' => {
                out.push('(');
                i += 1;
            }
            ')' => {
                out.push(')');
                i += 1;
            }
            // Ring closures and other characters — skip
            _ => {
                i += 1;
            }
        }
    }

    // Close any unmatched parentheses
    let open_parens = out.chars().filter(|&c| c == '(').count();
    let close_parens = out.chars().filter(|&c| c == ')').count();
    if open_parens > close_parens {
        for _ in 0..(open_parens - close_parens) {
            // Instead of adding dangling close-parens, strip the unmatched opens
            if let Some(pos) = out.rfind('(') {
                out.truncate(pos);
            }
        }
    }

    if out.is_empty() {
        "[#6]".to_string()
    } else {
        out
    }
}

/// Generate a simple Primitive Atom Pattern (PAP) seed.
/// These are 1-3 atom patterns with common constraints.
fn pap_seed<R: Rng>(rng: &mut R) -> SmartsGenome {
    let patterns = &[
        "[#6]",
        "[#7]",
        "[#8]",
        "[#16]",
        "[#6;R]",
        "[#7;R]",
        "[#6]=[#8]",
        "[#6](=[#8])[#7]",
        "[#6](=[#8])[#8]",
        "[#6;R]~[#6;R]",
        "[#7;H2]",
        "[#7;H1]",
        "[#8;H1]",
        "[#6;X4]",
        "[#6;X3]",
        "[#6]~[#6]~[#6]",
        "[#6](~[#6])(~[#6])~[#6]",
        "[#6;R;X3]",
        "[#7;+]",
        "[#8;-]",
        "[#6]=[#6]",
        "[#6]~[#7]~[#6]",
        "[#6]~[#8]~[#6]",
        "[#16](=[#8])(=[#8])",
        "[#6;r6]",
        "[#6;r5]",
        "[#7;r6]",
        "[#7;r5]",
    ];

    let pat = patterns.choose(rng).unwrap();
    SmartsGenome::from_smarts(pat).unwrap()
}

/// Generate a random small SMARTS pattern.
fn random_seed<R: Rng>(rng: &mut R) -> SmartsGenome {
    let atom_count = rng.gen_range(1..=4);
    let common_atoms: &[u8] = &[6, 7, 8, 16, 15, 9, 17, 35];

    let mut tokens = Vec::new();

    for i in 0..atom_count {
        // Optional bond between atoms
        if i > 0 {
            let bonds = &[BondToken::Single, BondToken::Double, BondToken::Any];
            tokens.push(SmartsToken::Bond(bonds.choose(rng).unwrap().clone()));
        }

        // Bracketed atom with random primitives
        tokens.push(SmartsToken::OpenBracket);
        let elem = common_atoms.choose(rng).unwrap();
        tokens.push(SmartsToken::Atom(AtomPrimitive::AtomicNumber(*elem)));

        // Optionally add a constraint
        if rng.gen_bool(0.4) {
            tokens.push(SmartsToken::Logic(LogicOp::AndLow));
            let constraint = match rng.gen_range(0..4) {
                0 => AtomPrimitive::RingMembership(None),
                1 => AtomPrimitive::HCount(rng.gen_range(0..=3)),
                2 => AtomPrimitive::Degree(rng.gen_range(1..=4)),
                _ => AtomPrimitive::Aromatic,
            };
            tokens.push(SmartsToken::Atom(constraint));
        }

        tokens.push(SmartsToken::CloseBracket);
    }

    SmartsGenome::from_tokens(tokens)
}

/// A guaranteed-valid fallback pattern.
fn fallback_seed<R: Rng>(rng: &mut R) -> SmartsGenome {
    let safe = &["[#6]", "[#7]", "[#8]", "[#6]~[#6]", "[#6]~[#7]"];
    let pat = safe.choose(rng).unwrap();
    SmartsGenome::from_smarts(pat).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn test_seed_generation_all_valid() {
        let smiles = vec![
            "c1ccccc1".to_string(),
            "CC(=O)O".to_string(),
            "CCN".to_string(),
            "c1ccncc1".to_string(),
            "CC(C)C(=O)C(=O)O".to_string(),
        ];
        let builder = SmartsGenomeBuilder::new(smiles);
        let mut rng = SmallRng::seed_from_u64(42);

        let mut valid = 0;
        let total = 100;
        for i in 0..total {
            let genome = builder.build_genome(i, &mut rng);
            if genome.is_valid_rdkit() {
                valid += 1;
            }
        }
        assert_eq!(
            valid, total,
            "Only {valid}/{total} genomes were valid SMARTS"
        );
    }
}
