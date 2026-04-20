#[cfg(feature = "std-io")]
use std::fs::File;
#[cfg(feature = "std-io")]
use std::io::{BufRead, BufReader};
#[cfg(feature = "std-io")]
use std::path::Path;

use genevo::population::GenomeBuilder;
use rand::Rng;
use rand::seq::SliceRandom;

use super::genome::SmartsGenome;
use super::token::*;

const STRATEGY_BUCKETS: usize = 20;
const CORPUS_BUCKETS: usize = 9;
const SMILES_BUCKETS: usize = 6;
const PAP_BUCKETS: usize = 3;

const BUILTIN_SEED_SMARTS: &[&str] = &[
    "[#6]",
    "[#7]",
    "[#8]",
    "[#15]",
    "[#16]",
    "[#9]",
    "[#17]",
    "[#35]",
    "[#53]",
    "[#6;X4]",
    "[#6;X3]",
    "[#6;R]",
    "[#7;R]",
    "[#8;R]",
    "[#6;r5]",
    "[#6;r6]",
    "[#7;r5]",
    "[#7;r6]",
    "[#6]=[#8]",
    "[#6]=[#6]",
    "[#6]~[#6]",
    "[#6]~[#7]",
    "[#6]~[#8]",
    "[#6]~[#16]",
    "[#6]~[#17]",
    "[#6]~[#35]",
    "[#6]~[#53]",
    "[#7;H2]",
    "[#7;H1]",
    "[#8;H1]",
    "[#7;+]",
    "[#8;-]",
    "[#6](=[#8])[#6]",
    "[#6](=[#8])[#7]",
    "[#6](=[#8])[#8]",
    "[#6](=[#8])[#16]",
    "[#6]~[#7]~[#6]",
    "[#6]~[#8]~[#6]",
    "[#6]~[#16]~[#6]",
    "[#6]~[#6]~[#6]",
    "[#6]~[#6]~[#8]",
    "[#6]~[#6]~[#7]",
    "[#6](~[#6])~[#6]",
    "[#6](~[#6])(~[#6])~[#6]",
    "[#6;R]~[#6;R]",
    "[#6;R]~[#6;R]~[#6;R]",
    "[#6;r6]~[#6;r6]",
    "[#6;r5]~[#7;r5]",
    "[#6;r6]~[#7;r6]",
    "[#16](=[#8])(=[#8])",
    "[#15](~[#8])(~[#8])~[#8]",
];

/// A curated corpus of known-reasonable SMARTS seeds.
///
/// This is a first-class fuzzing-style seed corpus, not just an incidental
/// hardcoded fallback list. The corpus can be built from shipped expert seeds,
/// user-provided files, prior evolved SMARTS, or any mixture of those sources.
#[derive(Clone, Debug, Default)]
pub struct SeedCorpus {
    seeds: Vec<SmartsGenome>,
}

impl SeedCorpus {
    pub fn builtin() -> Self {
        let mut corpus = Self::default();
        corpus
            .extend_from_smarts(BUILTIN_SEED_SMARTS.iter().copied())
            .expect("built-in SMARTS seed corpus must stay valid");
        corpus
    }

    pub fn from_smarts(smarts: Vec<String>) -> Result<Self, String> {
        let mut corpus = Self::default();
        corpus.extend_from_smarts(smarts)?;
        Ok(corpus)
    }

    #[cfg(feature = "std-io")]
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut corpus = Self::default();

        for (line_idx, line) in reader.lines().enumerate() {
            let raw_line = line?;
            let smarts = raw_line.trim();
            if smarts.is_empty() || smarts.starts_with('#') {
                continue;
            }

            corpus.insert_smarts(smarts).map_err(|error| {
                format!(
                    "invalid SMARTS seed at {}:{}: {error}",
                    path.display(),
                    line_idx + 1
                )
            })?;
        }

        Ok(corpus)
    }

    pub fn len(&self) -> usize {
        self.seeds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seeds.is_empty()
    }

    pub fn entries(&self) -> &[SmartsGenome] {
        &self.seeds
    }

    pub fn sample<R: Rng>(&self, rng: &mut R) -> Option<SmartsGenome> {
        self.seeds.choose(rng).cloned()
    }

    pub fn insert_smarts(&mut self, smarts: &str) -> Result<bool, String> {
        let genome = SmartsGenome::from_smarts(smarts)?;
        if !genome.is_valid_matcher() {
            return Err(format!("matcher rejected SMARTS '{smarts}'"));
        }
        Ok(self.insert_genome(genome))
    }

    pub fn extend(&mut self, other: SeedCorpus) -> usize {
        let mut inserted = 0usize;
        for genome in other.seeds {
            if self.insert_genome(genome) {
                inserted += 1;
            }
        }
        inserted
    }

    pub fn extend_from_smarts<I, S>(&mut self, smarts_iter: I) -> Result<usize, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut inserted = 0usize;
        for smarts in smarts_iter {
            if self.insert_smarts(smarts.as_ref())? {
                inserted += 1;
            }
        }
        Ok(inserted)
    }

    fn insert_genome(&mut self, genome: SmartsGenome) -> bool {
        if self
            .seeds
            .iter()
            .any(|existing| existing.smarts_string == genome.smarts_string)
        {
            return false;
        }
        self.seeds.push(genome);
        true
    }
}

/// Builds initial SMARTS genomes for a population.
///
/// Uses four strategies:
/// - seed corpus: built-in, file-backed, or caller-provided SMARTS
/// - SMILES-derived seeds: extract atom-environment patterns from positive example SMILES
/// - PAP seeds: simple 1-3 atom SMARTS from a catalog of primitives
/// - Random seeds: randomly assemble small SMARTS patterns
pub struct SmartsGenomeBuilder {
    /// SMILES strings from positive examples (for deriving seed patterns).
    pub positive_smiles: Vec<String>,
    /// Curated and/or user-provided SMARTS corpus.
    pub seed_corpus: SeedCorpus,
}

impl SmartsGenomeBuilder {
    pub fn new(positive_smiles: Vec<String>) -> Self {
        Self::with_seed_corpus(positive_smiles, SeedCorpus::builtin())
    }

    pub fn with_seed_corpus(positive_smiles: Vec<String>, seed_corpus: SeedCorpus) -> Self {
        Self {
            positive_smiles,
            seed_corpus,
        }
    }
}

impl GenomeBuilder<SmartsGenome> for SmartsGenomeBuilder {
    fn build_genome<R>(&self, index: usize, rng: &mut R) -> SmartsGenome
    where
        R: Rng + Sized,
    {
        let choice = index % STRATEGY_BUCKETS;
        let genome = if choice < CORPUS_BUCKETS && !self.seed_corpus.is_empty() {
            corpus_seed(&self.seed_corpus, rng)
        } else if choice < CORPUS_BUCKETS + SMILES_BUCKETS && !self.positive_smiles.is_empty() {
            smiles_derived_seed(&self.positive_smiles, rng)
        } else if choice < CORPUS_BUCKETS + SMILES_BUCKETS + PAP_BUCKETS {
            pap_seed(rng)
        } else {
            random_seed(rng)
        };

        if genome.is_valid_matcher() {
            genome
        } else {
            fallback_seed(rng)
        }
    }
}

fn corpus_seed<R: Rng>(seed_corpus: &SeedCorpus, rng: &mut R) -> SmartsGenome {
    seed_corpus.sample(rng).unwrap_or_else(|| pap_seed(rng))
}

/// Generate a SMARTS seed by converting a random positive SMILES to a basic atom pattern.
///
/// Takes a random SMILES, extracts 2-6 atoms with their bonds, and converts
/// element symbols to SMARTS atom primitives (e.g., `C` → `[#6]`).
fn smiles_derived_seed<R: Rng>(positive_smiles: &[String], rng: &mut R) -> SmartsGenome {
    let smiles = &positive_smiles[rng.gen_range(0..positive_smiles.len())];

    let atom_count = rng.gen_range(2..=6);
    let smarts = smiles_prefix_to_smarts(smiles, atom_count);
    if let Ok(g) = SmartsGenome::from_smarts(&smarts) {
        if g.is_valid_matcher() {
            return g;
        }
    }

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
            '[' => {
                let start = i;
                while i < chars.len() && chars[i] != ']' {
                    i += 1;
                }
                if i < chars.len() {
                    i += 1;
                }
                let bracket: String = chars[start..i].iter().collect();
                out.push_str(&bracket);
                atom_count += 1;
            }
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
            '(' => {
                out.push('(');
                i += 1;
            }
            ')' => {
                out.push(')');
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    let open_parens = out.chars().filter(|&c| c == '(').count();
    let close_parens = out.chars().filter(|&c| c == ')').count();
    if open_parens > close_parens {
        for _ in 0..(open_parens - close_parens) {
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
fn pap_seed<R: Rng>(rng: &mut R) -> SmartsGenome {
    let pat = BUILTIN_SEED_SMARTS.choose(rng).unwrap();
    SmartsGenome::from_smarts(pat).unwrap()
}

/// Generate a random small SMARTS pattern.
fn random_seed<R: Rng>(rng: &mut R) -> SmartsGenome {
    let atom_count = rng.gen_range(1..=4);
    let common_atoms: &[u8] = &[6, 7, 8, 16, 15, 9, 17, 35];

    let mut tokens = Vec::new();

    for i in 0..atom_count {
        if i > 0 {
            let bonds = &[BondToken::Single, BondToken::Double, BondToken::Any];
            tokens.push(SmartsToken::Bond(bonds.choose(rng).unwrap().clone()));
        }

        tokens.push(SmartsToken::OpenBracket);
        let elem = common_atoms.choose(rng).unwrap();
        tokens.push(SmartsToken::Atom(AtomPrimitive::AtomicNumber(*elem)));

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
    use std::collections::HashSet;
    #[cfg(feature = "std-io")]
    use std::io::Write;

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
            if genome.is_valid_matcher() {
                valid += 1;
            }
        }
        assert_eq!(
            valid, total,
            "Only {valid}/{total} genomes were valid SMARTS"
        );
    }

    #[test]
    fn builtin_seed_corpus_is_valid() {
        let corpus = SeedCorpus::builtin();
        assert!(corpus.len() >= BUILTIN_SEED_SMARTS.len());
        assert!(corpus.entries().iter().all(SmartsGenome::is_valid_matcher));
    }

    #[test]
    fn seed_corpus_from_smarts_vec_deduplicates_and_validates() {
        let corpus = SeedCorpus::from_smarts(vec![
            "[#6]".to_string(),
            "[#7]".to_string(),
            "[#6]".to_string(),
        ])
        .unwrap();

        let smarts: HashSet<_> = corpus
            .entries()
            .iter()
            .map(|genome| genome.smarts_string.as_str())
            .collect();

        assert_eq!(corpus.len(), 2);
        assert!(smarts.contains("[#6]"));
        assert!(smarts.contains("[#7]"));
    }

    #[cfg(feature = "std-io")]
    #[test]
    fn seed_corpus_file_ignores_comments_and_deduplicates() {
        let path = std::env::temp_dir().join(format!(
            "smarts-seed-corpus-{}-{}.sma",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(file, "# comment").unwrap();
        writeln!(file, "").unwrap();
        writeln!(file, "[#6]").unwrap();
        writeln!(file, "[#7]").unwrap();
        writeln!(file, "[#6]").unwrap();
        drop(file);

        let corpus = SeedCorpus::from_file(&path).unwrap();
        let smarts: HashSet<_> = corpus
            .entries()
            .iter()
            .map(|genome| genome.smarts_string.as_str())
            .collect();

        assert_eq!(corpus.len(), 2);
        assert!(smarts.contains("[#6]"));
        assert!(smarts.contains("[#7]"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn builder_can_draw_from_explicit_seed_corpus() {
        let corpus = SeedCorpus::from_smarts(vec![
            "[#6]~[#17]".to_string(),
            "[#6]~[#35]".to_string(),
        ])
        .unwrap();
        let builder = SmartsGenomeBuilder::with_seed_corpus(Vec::new(), corpus);
        let mut rng = SmallRng::seed_from_u64(7);

        let observed: HashSet<_> = (0..16)
            .map(|i| builder.build_genome(i, &mut rng).smarts_string)
            .collect();

        assert!(observed.contains("[#6]~[#17]") || observed.contains("[#6]~[#35]"));
    }
}
