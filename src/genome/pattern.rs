use alloc::rc::Rc;
use alloc::string::{String, ToString};
use core::fmt;
use core::str::FromStr;
use smarts_parser::QueryMol;

use super::limits::MAX_SMARTS_COMPLEXITY;

/// A SMARTS pattern genome for evolutionary optimization.
///
/// Stores the parsed SMARTS query and its canonical string form.
#[derive(Clone, Debug)]
pub struct SmartsGenome {
    query: QueryMol,
    smarts_string: Rc<str>,
    complexity: usize,
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
    /// assert_eq!(genome.to_string(), "[#6](=[#8])[#7]");
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
        let smarts_string: Rc<str> = query.to_string().into();
        Self {
            complexity: query.atom_count() + query.bonds().len(),
            query,
            smarts_string,
        }
    }

    /// Returns one cheap structural size metric.
    ///
    /// This is intentionally coarse. It is used for size limits and progress
    /// reporting, not as a canonical measure of SMARTS semantics.
    #[inline]
    pub fn complexity(&self) -> usize {
        self.complexity
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
    pub(crate) fn smarts_shared(&self) -> &Rc<str> {
        &self.smarts_string
    }

    /// Validate this genome using simple structural limits.
    #[inline]
    pub fn is_valid(&self) -> bool {
        !self.smarts_string.is_empty() && self.complexity <= MAX_SMARTS_COMPLEXITY
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
    use std::vec::Vec;

    use super::*;

    #[test]
    fn genome_round_trips_complex_smarts() {
        let smarts = "[#6](=[#8])[#7]";
        let genome = SmartsGenome::from_smarts(smarts).unwrap();

        assert_eq!(genome.smarts(), smarts);
        assert_eq!(genome.complexity(), 5);
        assert!(genome.is_valid());
    }

    #[test]
    fn genome_from_query_mol_preserves_query() {
        let query = QueryMol::from_str("[#6]~[#7]~[#8]").unwrap();
        let genome = SmartsGenome::from_query_mol(&query);

        assert_eq!(genome.query().to_string(), query.to_string());
        assert!(genome.is_valid());
    }

    #[test]
    fn genome_equality_and_display_use_canonical_smarts() {
        let left = SmartsGenome::from_smarts("[#6]~[#7]").unwrap();
        let right = SmartsGenome::from_query_mol(&QueryMol::from_str("[#6]~[#7]").unwrap());

        assert_eq!(left, right);
        assert_eq!(left.to_string(), "[#6]~[#7]");
    }

    #[test]
    fn genome_rejects_queries_past_structural_limit() {
        let over_limit = std::iter::repeat_n("[#6]", MAX_SMARTS_COMPLEXITY + 1)
            .collect::<Vec<_>>()
            .join("~");
        let genome = SmartsGenome::from_smarts(&over_limit).unwrap();

        assert!(!genome.is_valid());
    }
}
