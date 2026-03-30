use genevo::genetic::Genotype;

use super::display::tokens_to_smarts;
use super::parser::{parse_and_validate_smarts, validate_smarts_tokens};
use super::token::SmartsToken;
use crate::data::rdkit_lock::with_rdkit_lock;

/// A SMARTS pattern genome for evolutionary optimization.
///
/// Maintains both a token representation (for genetic operators) and
/// a cached string representation (for RDKit evaluation).
#[derive(Clone, Debug)]
pub struct SmartsGenome {
    pub tokens: Vec<SmartsToken>,
    pub smarts_string: String,
}

impl SmartsGenome {
    /// Create a genome from tokens.
    pub fn from_tokens(tokens: Vec<SmartsToken>) -> Self {
        let smarts_string = tokens_to_smarts(&tokens);
        Self {
            tokens,
            smarts_string,
        }
    }

    /// Create a genome from a SMARTS string.
    pub fn from_smarts(smarts: &str) -> Result<Self, String> {
        let tokens = parse_and_validate_smarts(smarts)?;
        Ok(Self {
            smarts_string: smarts.to_string(),
            tokens,
        })
    }

    /// Rebuild the cached string from tokens.
    pub fn rebuild_string(&mut self) {
        self.smarts_string = tokens_to_smarts(&self.tokens);
    }

    /// Validate this genome using only the local token-level checks.
    ///
    /// This is the hot-path validator for breeding. It rejects obviously bad
    /// token sequences without calling into RDKit.
    #[inline]
    pub fn is_structurally_valid(&self) -> bool {
        !self.smarts_string.is_empty() && validate_smarts_tokens(&self.tokens).is_ok()
    }

    /// Validate this genome structurally and with RDKit.
    ///
    /// Use this only on colder paths. Breeding should prefer
    /// `is_structurally_valid()` to avoid reparsing SMARTS in RDKit for every
    /// offspring candidate.
    #[inline]
    pub fn is_valid_rdkit(&self) -> bool {
        if !self.is_structurally_valid() {
            return false;
        }

        with_rdkit_lock(|| rdkit::RWMol::from_smarts(&self.smarts_string).is_ok())
    }
}

impl PartialEq for SmartsGenome {
    fn eq(&self, other: &Self) -> bool {
        self.smarts_string == other.smarts_string
    }
}

impl Genotype for SmartsGenome {
    type Dna = SmartsToken;
}

impl std::fmt::Display for SmartsGenome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.smarts_string)
    }
}
