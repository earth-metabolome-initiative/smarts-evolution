use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::convert::TryFrom;

use rand::Rng;
use rand::RngExt;
use rand::prelude::IndexedRandom;

use super::SmartsGenome;

const STRATEGY_BUCKETS: usize = 12;
const CORPUS_BUCKETS: usize = 8;
const SIMPLE_BUILTIN_BUCKETS: usize = 2;

const GENERIC_BUILTIN_SEED_SMARTS: &[&str] = &[
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

const CURATED_BUILTIN_SEED_SMARTS: &[&str] = &[
    "[#7;H2]~[#6]",
    "[#7;H1](~[#6])~[#6]",
    "[#7](~[#6])(~[#6])~[#6]",
    "[#6;r6]~[#7]",
    "[#6;r6]~[#8]",
    "[#6;r6]~[#17]",
    "[#6;r6]~[#35]",
    "[#6]~[#6]~[#6]~[#7]",
    "[#6;r6]~[#6]~[#6]~[#7]",
    "[#7]~[#6](=[#8])~[#7]",
    "[#8]~[#6](=[#8])~[#7]",
    "[#8]~[#6](=[#8])~[#8]",
    "[#16](=[#8])~[#6]",
    "[#16](=[#8])(=[#8])~[#6]",
    "[#16](=[#8])(=[#8])~[#7]",
    "[#6]#[#7]",
    "[#6]~[#6]#[#7]",
    "[#8]~[#6]~[#6]~[#8]",
    "[#6]~[#6]~[#6]~[#6](=[#8])[#8]",
    "[#15](=[#8])([#8])([#8])[#8]",
];

/// A curated corpus of known-reasonable SMARTS seeds.
///
/// This is a first-class fuzzing-style seed corpus, not just an incidental
/// hardcoded fallback list.
#[derive(Clone, Debug, Default)]
pub struct SeedCorpus {
    seeds: Vec<SmartsGenome>,
}

impl SeedCorpus {
    /// Build a corpus from the shipped built-in SMARTS fragments.
    pub fn builtin() -> Self {
        let mut corpus = Self::default();
        let inserted = corpus.extend_from_smarts(
            GENERIC_BUILTIN_SEED_SMARTS
                .iter()
                .chain(CURATED_BUILTIN_SEED_SMARTS.iter())
                .copied(),
        );
        debug_assert!(
            inserted.is_ok(),
            "built-in SMARTS seed corpus must stay valid"
        );
        corpus
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

    /// Insert one SMARTS string if it is valid and not already present.
    pub fn insert_smarts(&mut self, smarts: &str) -> Result<bool, String> {
        let genome = SmartsGenome::from_smarts(smarts)?;
        if !genome.is_valid() {
            return Err(format!("SMARTS exceeds structural limits: '{smarts}'"));
        }
        Ok(self.insert_genome(genome))
    }

    /// Extend this corpus with entries from another corpus.
    pub fn extend(&mut self, other: SeedCorpus) -> usize {
        let mut inserted = 0usize;
        for genome in other.seeds {
            if self.insert_genome(genome) {
                inserted += 1;
            }
        }
        inserted
    }

    /// Extend this corpus from an iterator of SMARTS-like strings.
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
            .any(|existing| existing.smarts() == genome.smarts())
        {
            return false;
        }
        self.seeds.push(genome);
        true
    }
}

impl<S> TryFrom<Vec<S>> for SeedCorpus
where
    S: AsRef<str>,
{
    type Error = String;

    fn try_from(value: Vec<S>) -> Result<Self, Self::Error> {
        let mut corpus = Self::default();
        corpus.extend_from_smarts(value)?;
        Ok(corpus)
    }
}

impl<'a, const N: usize> TryFrom<[&'a str; N]> for SeedCorpus {
    type Error = String;

    fn try_from(value: [&'a str; N]) -> Result<Self, Self::Error> {
        let mut corpus = Self::default();
        corpus.extend_from_smarts(value)?;
        Ok(corpus)
    }
}

/// Builds initial SMARTS genomes for a population.
pub struct SmartsGenomeBuilder {
    /// Curated and/or user-provided SMARTS corpus.
    seed_corpus: SeedCorpus,
}

impl SmartsGenomeBuilder {
    pub fn new(seed_corpus: SeedCorpus) -> Self {
        Self { seed_corpus }
    }

    pub fn build_genome<R>(&self, index: usize, rng: &mut R) -> SmartsGenome
    where
        R: Rng + Sized,
    {
        let choice = index % STRATEGY_BUCKETS;
        if choice < CORPUS_BUCKETS && !self.seed_corpus.is_empty() {
            corpus_seed(&self.seed_corpus, rng)
        } else if choice < CORPUS_BUCKETS + SIMPLE_BUILTIN_BUCKETS {
            builtin_seed(rng)
        } else {
            random_seed(rng)
        }
    }
}

fn corpus_seed<R: Rng>(seed_corpus: &SeedCorpus, rng: &mut R) -> SmartsGenome {
    seed_corpus.sample(rng).unwrap_or_else(|| builtin_seed(rng))
}

/// Generate one small built-in SMARTS seed.
fn builtin_seed<R: Rng>(rng: &mut R) -> SmartsGenome {
    let total = GENERIC_BUILTIN_SEED_SMARTS.len() + CURATED_BUILTIN_SEED_SMARTS.len();
    let idx = rng.random_range(0..total);
    let pat = if idx < GENERIC_BUILTIN_SEED_SMARTS.len() {
        GENERIC_BUILTIN_SEED_SMARTS[idx]
    } else {
        CURATED_BUILTIN_SEED_SMARTS[idx - GENERIC_BUILTIN_SEED_SMARTS.len()]
    };
    SmartsGenome::from_smarts(pat).unwrap()
}

/// Generate a random small SMARTS pattern.
fn random_seed<R: Rng>(rng: &mut R) -> SmartsGenome {
    let atom_count = rng.random_range(1..=4);
    let common_atoms: &[u8] = &[6, 7, 8, 16, 15, 9, 17, 35];
    let mut smarts = String::new();

    for i in 0..atom_count {
        if i > 0 {
            smarts.push_str(random_seed_bond(rng));
        }
        let elem = common_atoms.choose(rng).unwrap();
        smarts.push_str(&format!("[#{elem}"));
        if rng.random_bool(0.4) {
            let constraint = match rng.random_range(0..3) {
                0 => "R".to_string(),
                1 => format!("H{}", rng.random_range(0..=3)),
                _ => format!("D{}", rng.random_range(1..=4)),
            };
            smarts.push(';');
            smarts.push_str(&constraint);
        }
        smarts.push(']');
    }

    SmartsGenome::from_smarts(&smarts).unwrap()
}

fn random_seed_bond<R: Rng>(rng: &mut R) -> &'static str {
    ["-", "=", "~"].choose(rng).unwrap()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::vec;

    use super::*;
    use crate::genome::limits::MAX_SMARTS_COMPLEXITY;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn test_seed_generation_all_valid() {
        let builder = SmartsGenomeBuilder::new(SeedCorpus::builtin());
        let mut rng = SmallRng::seed_from_u64(42);

        let mut valid = 0;
        let total = 100;
        for i in 0..total {
            let genome = builder.build_genome(i, &mut rng);
            if genome.is_valid() {
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
        assert!(
            corpus.len() >= GENERIC_BUILTIN_SEED_SMARTS.len() + CURATED_BUILTIN_SEED_SMARTS.len()
        );
        assert!(corpus.entries().iter().all(SmartsGenome::is_valid));
    }

    #[test]
    fn builtin_seed_corpus_contains_curated_motifs() {
        let corpus = SeedCorpus::builtin();
        let smarts: HashSet<_> = corpus
            .entries()
            .iter()
            .map(|genome| genome.smarts())
            .collect();

        assert!(smarts.contains("[#7](~[#6])(~[#6])~[#6]"));
        assert!(smarts.contains("[#16](=[#8])(=[#8])~[#7]"));
        assert!(smarts.contains("[#6;r6]~[#6]~[#6]~[#7]"));
    }

    #[test]
    fn seed_corpus_from_smarts_vec_deduplicates_and_validates() {
        let corpus = SeedCorpus::try_from(vec![
            "[#6]".to_string(),
            "[#7]".to_string(),
            "[#6]".to_string(),
        ])
        .unwrap();

        let smarts: HashSet<_> = corpus
            .entries()
            .iter()
            .map(|genome| genome.smarts())
            .collect();

        assert_eq!(corpus.len(), 2);
        assert!(smarts.contains("[#6]"));
        assert!(smarts.contains("[#7]"));
    }

    #[test]
    fn seed_corpus_try_from_vec_of_strs_deduplicates_and_validates() {
        let corpus = SeedCorpus::try_from(vec!["[#6]", "[#7]", "[#6]"]).unwrap();

        let smarts: HashSet<_> = corpus
            .entries()
            .iter()
            .map(|genome| genome.smarts())
            .collect();

        assert_eq!(corpus.len(), 2);
        assert!(smarts.contains("[#6]"));
        assert!(smarts.contains("[#7]"));
    }

    #[test]
    fn seed_corpus_try_from_array_of_strs_deduplicates_and_validates() {
        let corpus = SeedCorpus::try_from(["[#6]", "[#7]", "[#6]"]).unwrap();

        let smarts: HashSet<_> = corpus
            .entries()
            .iter()
            .map(|genome| genome.smarts())
            .collect();

        assert_eq!(corpus.len(), 2);
        assert!(smarts.contains("[#6]"));
        assert!(smarts.contains("[#7]"));
    }

    #[test]
    fn seed_corpus_sample_extend_and_invalid_insertions_behave() {
        let mut corpus = SeedCorpus::default();
        let too_large = std::iter::repeat_n("[#6]", MAX_SMARTS_COMPLEXITY + 1)
            .collect::<Vec<_>>()
            .join("~");

        assert!(corpus.is_empty());
        assert!(corpus.sample(&mut SmallRng::seed_from_u64(11)).is_none());
        assert!(corpus.insert_smarts("not smarts").is_err());
        assert!(corpus.insert_smarts(&too_large).is_err());

        assert!(corpus.insert_smarts("[#6]").unwrap());
        assert!(!corpus.is_empty());

        let inserted = corpus.extend(SeedCorpus::try_from(["[#6]", "[#7]"]).unwrap());
        assert_eq!(inserted, 1);
        assert_eq!(corpus.len(), 2);
        assert!(corpus.sample(&mut SmallRng::seed_from_u64(12)).is_some());
    }

    #[test]
    fn builder_can_draw_from_explicit_seed_corpus() {
        let corpus = SeedCorpus::try_from(["[#6]~[#17]", "[#6]~[#35]"]).unwrap();
        let builder = SmartsGenomeBuilder::new(corpus);
        let mut rng = SmallRng::seed_from_u64(7);

        let observed: HashSet<_> = (0..16)
            .map(|i| builder.build_genome(i, &mut rng).smarts().to_string())
            .collect();

        assert!(observed.contains("[#6]~[#17]") || observed.contains("[#6]~[#35]"));
    }

    #[test]
    fn builder_without_corpus_falls_back_to_builtin_and_random_seeds() {
        let builder = SmartsGenomeBuilder::new(SeedCorpus::default());
        let mut rng = SmallRng::seed_from_u64(19);

        for i in 0..32 {
            let genome = builder.build_genome(i, &mut rng);
            assert!(genome.is_valid());
            assert!(!genome.smarts().is_empty());
        }
    }
}
