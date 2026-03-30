use rand::Rng;
use rand::seq::SliceRandom;

use crate::genome::token::*;

/// Common elements used in organic chemistry, by atomic number.
pub const COMMON_ELEMENTS: &[u8] = &[6, 7, 8, 16, 15, 9, 17, 35, 53, 5];

/// Elements that can be aromatic in SMARTS.
pub const AROMATIC_ELEMENTS: &[u8] = &[6, 7, 8, 16, 15];

/// Pick a random element atomic number.
pub fn random_element<R: Rng>(rng: &mut R) -> u8 {
    *COMMON_ELEMENTS.choose(rng).unwrap()
}

/// Pick a random atom primitive suitable for use inside brackets.
pub fn random_atom_primitive<R: Rng>(rng: &mut R) -> AtomPrimitive {
    match rng.gen_range(0..10) {
        0 => AtomPrimitive::Any,
        1 => AtomPrimitive::Aromatic,
        2 => AtomPrimitive::Aliphatic,
        3 => AtomPrimitive::AtomicNumber(random_element(rng)),
        4 => AtomPrimitive::Degree(rng.gen_range(1..=4)),
        5 => AtomPrimitive::TotalConnections(rng.gen_range(1..=4)),
        6 => AtomPrimitive::HCount(rng.gen_range(0..=3)),
        7 => AtomPrimitive::RingMembership(if rng.gen_bool(0.5) {
            Some(rng.gen_range(0..=3))
        } else {
            None
        }),
        8 => AtomPrimitive::SmallestRing(if rng.gen_bool(0.5) {
            Some(*[5u8, 6, 7].choose(rng).unwrap())
        } else {
            None
        }),
        _ => AtomPrimitive::Valence(rng.gen_range(1..=4)),
    }
}

/// Generalize an atom primitive (make it match more molecules).
pub fn generalize<R: Rng>(prim: &AtomPrimitive, rng: &mut R) -> AtomPrimitive {
    match prim {
        // Symbol → AtomicNumber (drops implicit valence constraint)
        AtomPrimitive::Symbol(n) => AtomPrimitive::AtomicNumber(*n),
        AtomPrimitive::AromaticSymbol(n) => AtomPrimitive::AtomicNumber(*n),
        // AtomicNumber → Any
        AtomPrimitive::AtomicNumber(_) => AtomPrimitive::Any,
        // Specific ring size → any ring
        AtomPrimitive::SmallestRing(Some(_)) => AtomPrimitive::SmallestRing(None),
        AtomPrimitive::RingMembership(Some(_)) => AtomPrimitive::RingMembership(None),
        // Ring → Any
        AtomPrimitive::RingMembership(None) => AtomPrimitive::Any,
        AtomPrimitive::SmallestRing(None) => AtomPrimitive::Any,
        // Specific degree → wider range
        AtomPrimitive::Degree(d) if *d > 1 => AtomPrimitive::Degree(d - 1),
        AtomPrimitive::TotalConnections(x) if *x > 1 => AtomPrimitive::TotalConnections(x - 1),
        // HCount: increase (matches more)
        AtomPrimitive::HCount(h) if *h < 4 => AtomPrimitive::HCount(h + 1),
        // Already very general → random primitive
        _ => random_atom_primitive(rng),
    }
}

/// Specialize an atom primitive (make it match fewer molecules).
pub fn specialize<R: Rng>(prim: &AtomPrimitive, rng: &mut R) -> AtomPrimitive {
    match prim {
        // Any → random element
        AtomPrimitive::Any => AtomPrimitive::AtomicNumber(random_element(rng)),
        // Aromatic → specific aromatic element
        AtomPrimitive::Aromatic => {
            AtomPrimitive::AromaticSymbol(*AROMATIC_ELEMENTS.choose(rng).unwrap())
        }
        // Aliphatic → specific element
        AtomPrimitive::Aliphatic => AtomPrimitive::Symbol(random_element(rng)),
        // AtomicNumber → Symbol (adds implicit valence)
        AtomPrimitive::AtomicNumber(n) => AtomPrimitive::Symbol(*n),
        // Any ring → specific ring size
        AtomPrimitive::RingMembership(None) => {
            AtomPrimitive::RingMembership(Some(rng.gen_range(1..=3)))
        }
        AtomPrimitive::SmallestRing(None) => {
            AtomPrimitive::SmallestRing(Some(*[5u8, 6, 7].choose(rng).unwrap()))
        }
        // Increase degree
        AtomPrimitive::Degree(d) if *d < 4 => AtomPrimitive::Degree(d + 1),
        AtomPrimitive::TotalConnections(x) if *x < 4 => AtomPrimitive::TotalConnections(x + 1),
        // Decrease H count (matches fewer)
        AtomPrimitive::HCount(h) if *h > 0 => AtomPrimitive::HCount(h - 1),
        _ => random_atom_primitive(rng),
    }
}

/// Pick a random bond token.
pub fn random_bond<R: Rng>(rng: &mut R) -> BondToken {
    let bonds = [
        BondToken::Single,
        BondToken::Double,
        BondToken::Triple,
        BondToken::AromaticBond,
        BondToken::Any,
    ];
    bonds.choose(rng).unwrap().clone()
}

/// Generate a random bracketed atom (e.g., `[#6]`, `[#7;R]`).
pub fn random_bracketed_atom<R: Rng>(rng: &mut R) -> Vec<SmartsToken> {
    let mut tokens = vec![SmartsToken::OpenBracket];
    tokens.push(SmartsToken::Atom(AtomPrimitive::AtomicNumber(
        random_element(rng),
    )));
    if rng.gen_bool(0.3) {
        tokens.push(SmartsToken::Logic(LogicOp::AndLow));
        tokens.push(SmartsToken::Atom(random_atom_primitive(rng)));
    }
    tokens.push(SmartsToken::CloseBracket);
    tokens
}
