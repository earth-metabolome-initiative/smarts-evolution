//! SMARTS genome types, limits, and seed corpus support.

pub mod compatibility;
pub mod limits;
pub mod pattern;
pub mod seed;

pub use pattern::SmartsGenome;

#[cfg(test)]
pub(crate) fn over_limit_smarts_fixture() -> alloc::string::String {
    let mut atoms = alloc::vec::Vec::new();
    loop {
        atoms.push("[#6]");
        let smarts = atoms.join(".");
        let genome = SmartsGenome::from_smarts(&smarts).unwrap();
        if genome.smarts_len() > limits::MAX_SMARTS_LEN {
            return smarts;
        }
    }
}
