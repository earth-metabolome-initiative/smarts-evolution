//! SMARTS compatibility profiles for downstream search engines.

use alloc::string::ToString;

use elements_rs::{AtomicNumber, Element, ElementVariant, MassNumber};
use smarts_rs::{
    AtomExpr, AtomPrimitive, BondExpr, BondExprTree, BondPrimitive, BracketExprTree, HydrogenKind,
    NumericQuery, QueryMol,
};
use smiles_parser::atom::bracketed::chirality::Chirality;
use smiles_parser::bond::Bond;

pub(crate) const PUBCHEM_MAX_ATOM_MAP: u32 = 999;
pub(crate) const PUBCHEM_MIN_ATOMIC_NUMBER: u16 = 1;
pub(crate) const PUBCHEM_MAX_ATOMIC_NUMBER: u16 = 118;
pub(crate) const PUBCHEM_MAX_ISOTOPE_MASS_NUMBER: u16 = 255;

/// Compatibility profile for generated SMARTS.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SmartsCompatibilityMode {
    /// Allow every SMARTS construct supported by `smarts-rs`.
    #[default]
    Full,
    /// Restrict generated SMARTS to a conservative PubChem-oriented subset.
    PubChem,
}

impl SmartsCompatibilityMode {
    #[inline]
    pub const fn pubchem_compatible(self) -> bool {
        matches!(self, Self::PubChem)
    }

    #[inline]
    pub(crate) fn allows_query(self, query: &QueryMol) -> bool {
        match self {
            Self::Full => true,
            Self::PubChem => pubchem_query_is_compatible(query),
        }
    }
}

pub(crate) fn pubchem_query_is_compatible(query: &QueryMol) -> bool {
    let serialized = query.to_string();
    query
        .atoms()
        .iter()
        .all(|atom| pubchem_atom_expr_is_compatible(&atom.expr))
        && query
            .bonds()
            .iter()
            .all(|bond| pubchem_bond_expr_is_compatible(&bond.expr))
        && pubchem_serialized_smarts_is_compatible(&serialized)
}

fn pubchem_atom_expr_is_compatible(expr: &AtomExpr) -> bool {
    match expr {
        AtomExpr::Wildcard => true,
        AtomExpr::Bare { element, .. } => pubchem_element_symbol_is_compatible(*element),
        AtomExpr::Bracket(expr) => {
            expr.atom_map
                .is_none_or(|atom_map| atom_map <= PUBCHEM_MAX_ATOM_MAP)
                && pubchem_bracket_tree_is_compatible(&expr.tree)
        }
    }
}

fn pubchem_bracket_tree_is_compatible(tree: &BracketExprTree) -> bool {
    pubchem_bracket_tree_is_compatible_at_negation_depth(tree, false)
}

fn pubchem_bracket_tree_is_compatible_at_negation_depth(
    tree: &BracketExprTree,
    negated: bool,
) -> bool {
    match tree {
        BracketExprTree::Primitive(primitive) => pubchem_atom_primitive_is_compatible(primitive),
        BracketExprTree::Not(inner) => {
            !negated
                && !bracket_tree_contains_wildcard_primitive(inner)
                && pubchem_bracket_tree_is_compatible_at_negation_depth(inner, true)
        }
        BracketExprTree::HighAnd(children)
        | BracketExprTree::Or(children)
        | BracketExprTree::LowAnd(children) => children
            .iter()
            .all(|child| pubchem_bracket_tree_is_compatible_at_negation_depth(child, negated)),
    }
}

pub(crate) fn bracket_tree_contains_wildcard_primitive(tree: &BracketExprTree) -> bool {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::Wildcard) => true,
        BracketExprTree::Primitive(_) => false,
        BracketExprTree::Not(inner) => bracket_tree_contains_wildcard_primitive(inner),
        BracketExprTree::HighAnd(children)
        | BracketExprTree::Or(children)
        | BracketExprTree::LowAnd(children) => children
            .iter()
            .any(bracket_tree_contains_wildcard_primitive),
    }
}

fn pubchem_atom_primitive_is_compatible(primitive: &AtomPrimitive) -> bool {
    match primitive {
        AtomPrimitive::Wildcard
        | AtomPrimitive::AliphaticAny
        | AtomPrimitive::AromaticAny
        | AtomPrimitive::Charge(_) => true,
        AtomPrimitive::AtomicNumber(atomic_number) => {
            (PUBCHEM_MIN_ATOMIC_NUMBER..=PUBCHEM_MAX_ATOMIC_NUMBER).contains(atomic_number)
        }
        AtomPrimitive::Symbol { element, .. } => pubchem_element_symbol_is_compatible(*element),
        AtomPrimitive::Chirality(chirality) => pubchem_chirality_is_compatible(*chirality),
        AtomPrimitive::Hybridization(_)
        | AtomPrimitive::HeteroNeighbor(_)
        | AtomPrimitive::AliphaticHeteroNeighbor(_) => false,
        AtomPrimitive::Isotope { isotope, .. } => {
            isotope.mass_number() <= PUBCHEM_MAX_ISOTOPE_MASS_NUMBER
                && pubchem_element_symbol_is_compatible(isotope.element())
        }
        AtomPrimitive::IsotopeWildcard(mass_number) => {
            *mass_number <= PUBCHEM_MAX_ISOTOPE_MASS_NUMBER
        }
        AtomPrimitive::Degree(query)
        | AtomPrimitive::Connectivity(query)
        | AtomPrimitive::RingMembership(query)
        | AtomPrimitive::RingSize(query)
        | AtomPrimitive::RingConnectivity(query) => {
            pubchem_optional_numeric_query_is_compatible(*query)
        }
        AtomPrimitive::Valence(query) => query.is_some_and(pubchem_numeric_query_is_compatible),
        AtomPrimitive::Hydrogen(HydrogenKind::Total, query) => {
            pubchem_optional_numeric_query_is_compatible(*query)
        }
        AtomPrimitive::Hydrogen(HydrogenKind::Implicit, _) => false,
        AtomPrimitive::RecursiveQuery(query) => pubchem_recursive_query_is_compatible(query),
    }
}

pub(crate) fn pubchem_recursive_query_is_compatible(query: &QueryMol) -> bool {
    query.component_count() == 1
        && query.component_groups().iter().all(Option::is_none)
        && pubchem_query_is_compatible(query)
}

pub(crate) fn pubchem_element_symbol_is_compatible(element: Element) -> bool {
    let atomic_number: u16 = element.atomic_number().into();
    !matches!(atomic_number, 104 | 106 | 111 | 114 | 115 | 116 | 117 | 118)
}

fn pubchem_chirality_is_compatible(chirality: Chirality) -> bool {
    matches!(
        chirality,
        Chirality::At | Chirality::AtAt | Chirality::TH(_) | Chirality::SP(_)
    )
}

fn pubchem_optional_numeric_query_is_compatible(query: Option<NumericQuery>) -> bool {
    query.is_none_or(pubchem_numeric_query_is_compatible)
}

fn pubchem_numeric_query_is_compatible(query: NumericQuery) -> bool {
    matches!(query, NumericQuery::Exact(_))
}

pub(crate) fn pubchem_bond_expr_is_compatible(expr: &BondExpr) -> bool {
    match expr {
        BondExpr::Elided => true,
        BondExpr::Query(tree) => pubchem_bond_tree_is_compatible(tree),
    }
}

fn pubchem_bond_tree_is_compatible(tree: &BondExprTree) -> bool {
    pubchem_bond_tree_is_compatible_at_negation_depth(tree, false)
}

fn pubchem_bond_tree_is_compatible_at_negation_depth(tree: &BondExprTree, negated: bool) -> bool {
    match tree {
        BondExprTree::Primitive(primitive) => pubchem_bond_primitive_is_compatible(*primitive),
        BondExprTree::Not(inner) => {
            !negated && pubchem_bond_tree_is_compatible_at_negation_depth(inner, true)
        }
        BondExprTree::HighAnd(children)
        | BondExprTree::Or(children)
        | BondExprTree::LowAnd(children) => children
            .iter()
            .all(|child| pubchem_bond_tree_is_compatible_at_negation_depth(child, negated)),
    }
}

fn pubchem_bond_primitive_is_compatible(primitive: BondPrimitive) -> bool {
    !matches!(primitive, BondPrimitive::Bond(Bond::Quadruple))
}

fn pubchem_serialized_smarts_is_compatible(smarts: &str) -> bool {
    let bytes = smarts.as_bytes();

    for index in 0..bytes.len() {
        let tail = &bytes[index..];
        if tail.starts_with(b"[@") || tail.starts_with(b"[!@") {
            return false;
        }

        if tail.starts_with(b"@AL")
            && index
                .checked_sub(1)
                .and_then(|previous| bytes.get(previous))
                .is_some_and(|previous| matches!(previous, b'[' | b'&' | b';' | b',' | b'!'))
        {
            return false;
        }

        if tail.starts_with(b"[-") {
            let Some(next) = bytes.get(index + 2) else {
                continue;
            };
            if !next.is_ascii_digit() && *next != b'-' {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;
    use alloc::vec;
    use core::str::FromStr;

    use super::*;

    #[test]
    fn pubchem_compatibility_accepts_observed_pubchem_features() {
        for smarts in [
            "[#6;X3](=[#8])[$([#7])]",
            "[#6:1]",
            "[#6:999]",
            "[Db]",
            "[#118]",
            "[13*]",
            "C[C@H](F)O",
            "F/C=C/F",
            "([#8].[#8])",
            // smarts-rs normalizes double negation before the GA stores genomes.
            "[!!#6]",
            "[#6&!!X3]",
            "[#6]!!#[#7]",
        ] {
            let query = QueryMol::from_str(smarts).unwrap();

            assert!(pubchem_query_is_compatible(&query), "{smarts}");
        }
    }

    #[test]
    fn pubchem_compatibility_rejects_observed_pubchem_failures() {
        for smarts in [
            "[D{2-4}]",
            "[$([D{2-4}])]",
            "[65535*]",
            "[^2]",
            "[z2]",
            "[Z2]",
            "[#0]",
            "[#119]",
            "[#6:1000]",
            "[c;h1]",
            "[C@TB1](F)(Cl)Br",
            "[Rf]",
            "[Og]",
            "[$([#6]-[#8].[#7])]",
            "[$(([#6].[#8]))]",
            "[!*]",
            "[v]",
            "[!v]",
            "[v,D]",
            "[-]",
            "[-&N]",
            "[-;N]",
            "[-,N]",
            "[-:1]",
            "[@]",
            "[@&#6]",
            "[@;#6]",
            "[@,C]",
            "[!@]",
            "[!@@]",
            "[C@AL2]",
            "[C&@AL2]",
            "[C;@AL2]",
            "[C,!@AL2]",
        ] {
            let query = QueryMol::from_str(smarts).unwrap();

            assert!(!pubchem_query_is_compatible(&query), "{smarts}");
        }

        let mut editable = QueryMol::from_str("[#6]-[#6]").unwrap().edit();
        editable
            .replace_bond_expr(
                0,
                BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Bond(
                    Bond::Quadruple,
                ))),
            )
            .unwrap();
        let query = editable.into_query_mol().unwrap();

        assert!(!pubchem_query_is_compatible(&query), "quadruple bond");

        let nested_bond_negation =
            BondExpr::Query(BondExprTree::Not(Box::new(BondExprTree::HighAnd(vec![
                BondExprTree::Primitive(BondPrimitive::Bond(Bond::Triple)),
                BondExprTree::Not(Box::new(BondExprTree::Primitive(BondPrimitive::Bond(
                    Bond::Double,
                )))),
            ]))));
        assert!(
            !pubchem_bond_expr_is_compatible(&nested_bond_negation),
            "nested bond negation"
        );
    }

    #[test]
    fn pubchem_compatibility_allows_useful_negative_charge_forms() {
        for smarts in [
            "[O-]",
            "[#8-]",
            "[N&-]",
            "[N;-]",
            "[*-]",
            "[--]",
            "[!-]",
            "[#6@]",
            "[A&@]",
            "[C@H](F)O",
            "[C!@H](F)O",
            "[v2]",
            "[!v12&A]",
        ] {
            let query = QueryMol::from_str(smarts).unwrap();

            assert!(pubchem_query_is_compatible(&query), "{smarts}");
        }
    }
}
