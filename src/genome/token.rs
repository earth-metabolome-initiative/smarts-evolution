/// Primitive atom properties that can appear inside a SMARTS bracketed atom `[...]`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AtomPrimitive {
    /// Any atom: `*`
    Any,
    /// Aromatic: `a`
    Aromatic,
    /// Aliphatic: `A`
    Aliphatic,
    /// Atomic number: `#6` for carbon, etc.
    AtomicNumber(u8),
    /// Organic subset element symbol (aliphatic): `C`, `N`, `O`, `S`, `P`, `F`, `Cl`, `Br`, `I`
    Symbol(u8), // atomic number
    /// Aromatic element symbol: `c`, `n`, `o`, `s`, `p`
    AromaticSymbol(u8), // atomic number
    /// Total degree (connections): `D`, `D2`, etc.
    Degree(u8),
    /// Total connections (including H): `X`, `X3`, etc.
    TotalConnections(u8),
    /// Hydrogen count: `H`, `H0`, `H2`, etc.
    HCount(u8),
    /// Ring membership count: `R`, `R0`, `R2`, etc.
    RingMembership(Option<u8>),
    /// Smallest ring size: `r`, `r5`, `r6`, etc.
    SmallestRing(Option<u8>),
    /// Valence: `v`, `v4`, etc.
    Valence(u8),
    /// Formal charge: `+`, `+2`, `-`, `-1`, etc.
    Charge(i8),
    /// Ring bond count: `x`, `x2`, etc.
    RingBondCount(Option<u8>),
}

/// Logical operators within a SMARTS atom expression.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LogicOp {
    /// High-precedence AND: `&`
    And,
    /// OR: `,`
    Or,
    /// Low-precedence AND: `;`
    AndLow,
    /// NOT: `!`
    Not,
}

/// Bond types in SMARTS.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BondToken {
    /// Single bond: `-`
    Single,
    /// Double bond: `=`
    Double,
    /// Triple bond: `#`
    Triple,
    /// Aromatic bond: `:`
    AromaticBond,
    /// Any bond: `~`
    Any,
    /// Ring bond: `@`
    Ring,
}

/// A token in a SMARTS pattern.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SmartsToken {
    /// Open bracket: `[`
    OpenBracket,
    /// Close bracket: `]`
    CloseBracket,
    /// An atom primitive (inside brackets or bare)
    Atom(AtomPrimitive),
    /// A logical operator
    Logic(LogicOp),
    /// A bond between atoms
    Bond(BondToken),
    /// Branch open: `(`
    BranchOpen,
    /// Branch close: `)`
    BranchClose,
    /// Ring closure digit: `1`-`9`
    RingClosure(u8),
}

/// Common organic subset elements that can appear bare (outside brackets).
pub const ORGANIC_SUBSET: &[(u8, &str, &str)] = &[
    (5, "B", "b"),
    (6, "C", "c"),
    (7, "N", "n"),
    (8, "O", "o"),
    (9, "F", ""),
    (15, "P", "p"),
    (16, "S", "s"),
    (17, "Cl", ""),
    (35, "Br", ""),
    (53, "I", ""),
];

/// Map atomic number to aliphatic symbol.
pub fn atomic_number_to_symbol(num: u8) -> Option<&'static str> {
    ORGANIC_SUBSET
        .iter()
        .find(|(n, _, _)| *n == num)
        .map(|(_, sym, _)| *sym)
        .filter(|s| !s.is_empty())
}

/// Map atomic number to aromatic symbol.
pub fn atomic_number_to_aromatic(num: u8) -> Option<&'static str> {
    ORGANIC_SUBSET
        .iter()
        .find(|(n, _, _)| *n == num)
        .map(|(_, _, ar)| *ar)
        .filter(|s| !s.is_empty())
}
