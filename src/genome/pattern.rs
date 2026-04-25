use alloc::string::{String, ToString};
use alloc::sync::Arc;
use core::fmt;
use core::str::FromStr;
use smarts_rs::QueryMol;

use super::limits::MAX_SMARTS_LEN;

/// A SMARTS pattern genome for evolutionary optimization.
///
/// Stores the parsed SMARTS query and its canonical string form.
#[derive(Clone, Debug)]
pub struct SmartsGenome {
    query: QueryMol,
    smarts_string: Arc<str>,
}

impl SmartsGenome {
    /// Create a genome from a SMARTS string.
    ///
    /// # Examples
    ///
    /// ```
    /// use smarts_evolution::SmartsGenome;
    ///
    /// let genome = SmartsGenome::from_smarts("[#6](=[#8])[#7]").unwrap();
    /// assert!(genome.is_valid());
    /// assert_eq!(genome.to_string(), "[#6]([#7])=[#8]");
    /// ```
    pub fn from_smarts(smarts: &str) -> Result<Self, String> {
        let query = QueryMol::from_str(smarts).map_err(|error| error.to_string())?;
        Ok(Self::from_query(query))
    }

    /// Create a genome from one parsed SMARTS query.
    pub fn from_query_mol(query: &QueryMol) -> Self {
        Self::from_query(query.clone())
    }

    fn from_query(query: QueryMol) -> Self {
        let query = query.canonicalize();
        let smarts_string: Arc<str> = query.to_string().into();
        Self {
            query,
            smarts_string,
        }
    }

    /// Returns the length of the canonical SMARTS string.
    #[inline]
    pub fn smarts_len(&self) -> usize {
        self.smarts_string.len()
    }

    #[inline]
    pub fn query(&self) -> &QueryMol {
        &self.query
    }

    #[inline]
    pub fn smarts(&self) -> &str {
        &self.smarts_string
    }

    #[inline]
    pub(crate) fn smarts_shared(&self) -> &Arc<str> {
        &self.smarts_string
    }

    /// Validate this genome using simple structural limits.
    #[inline]
    pub fn is_valid(&self) -> bool {
        !self.smarts_string.is_empty() && self.smarts_len() <= MAX_SMARTS_LEN
    }
}

impl PartialEq for SmartsGenome {
    fn eq(&self, other: &Self) -> bool {
        self.smarts_string == other.smarts_string
    }
}

impl fmt::Display for SmartsGenome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.smarts_string)
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;
    use crate::genome::over_limit_smarts_fixture;

    #[test]
    fn genome_round_trips_branched_smarts() {
        let smarts = "[#6](=[#8])[#7]";
        let genome = SmartsGenome::from_smarts(smarts).unwrap();

        assert_eq!(genome.smarts(), "[#6]([#7])=[#8]");
        assert_eq!(genome.smarts_len(), genome.smarts().len());
        assert!(genome.is_valid());
    }

    #[test]
    fn genome_from_query_mol_canonicalizes_query_and_display() {
        let query = QueryMol::from_str("[#6]~[#7]~[#8]").unwrap();
        let genome = SmartsGenome::from_query_mol(&query);

        assert!(genome.query().is_canonical());
        assert_eq!(genome.smarts(), query.to_canonical_smarts());
        assert_eq!(genome.query().to_string(), query.to_canonical_smarts());
        assert!(genome.is_valid());
    }

    #[test]
    fn genome_equality_and_display_use_canonical_smarts() {
        let left = SmartsGenome::from_smarts("[#6]~[#7]").unwrap();
        let right = SmartsGenome::from_smarts("[#7]~[#6]").unwrap();

        assert_eq!(left, right);
        assert_eq!(left.to_string(), "[#6]~[#7]");
    }

    #[test]
    fn genome_canonicalizes_equivalent_atom_orderings() {
        let left = SmartsGenome::from_smarts("[#6](=[#8])[#7]").unwrap();
        let right = SmartsGenome::from_smarts("[#8]=[#6]([#7])").unwrap();

        assert_eq!(left, right);
        assert_eq!(left.smarts(), "[#6]([#7])=[#8]");
        assert!(left.query().is_canonical());
    }

    #[test]
    fn genome_uses_upstream_canonicalization_for_double_negation() {
        let direct = SmartsGenome::from_smarts("[!!#6]").unwrap();
        let nested = SmartsGenome::from_smarts("[$([!!#6])]").unwrap();
        let bond = SmartsGenome::from_smarts("[#6]!!@[#7]").unwrap();

        assert_eq!(direct.smarts(), "[#6]");
        assert!(!nested.smarts().contains("!!"));
        assert_eq!(bond.smarts(), "[#6]@[#7]");
        assert!(direct.query().is_canonical());
        assert!(nested.query().is_canonical());
        assert!(bond.query().is_canonical());
    }

    #[test]
    fn genome_uses_upstream_canonicalization_for_boolean_simplifications() {
        for (source, expected) in [
            ("[#6&#6]", "[#6]"),
            ("[#6;#6]", "[#6]"),
            ("[#6,#6]", "[#6]"),
            ("[!#6&!#6]", "[!#6]"),
            ("[#6,!#6]", "*"),
            ("[#6&!#6]", "[!*]"),
            ("[#6]-&~[#7]", "[#6]-[#7]"),
            ("[$([!!#6;*])]", "[#6]"),
        ] {
            let genome = SmartsGenome::from_smarts(source).unwrap();

            assert_eq!(genome.smarts(), expected, "{source}");
            assert!(genome.query().is_canonical(), "{source}");
        }
    }

    #[test]
    fn genome_rejects_queries_past_length_limit() {
        let over_limit = over_limit_smarts_fixture();
        let genome = SmartsGenome::from_smarts(&over_limit).unwrap();

        assert!(genome.smarts_len() > MAX_SMARTS_LEN);
        assert!(!genome.is_valid());
    }
}
