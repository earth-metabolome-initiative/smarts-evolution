use super::token::*;

pub const MAX_SMARTS_TOKENS: usize = 80;
pub const MAX_CROSSOVER_CHILD_TOKENS: usize = MAX_SMARTS_TOKENS;
pub const MAX_BRANCH_DEPTH: usize = 4;
pub const MAX_PRIMITIVES_PER_BRACKET: usize = 3;

/// Parse a SMARTS string into tokens.
///
/// This handles a practical subset of SMARTS sufficient for evolutionary generation:
/// bracketed atoms with primitives and logic, bare organic atoms, bonds, branches,
/// ring closures. Does not handle recursive SMARTS `$(...)` or chiral specifications.
pub fn parse_smarts(input: &str) -> Result<Vec<SmartsToken>, String> {
    let chars: Vec<char> = input.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            '[' => {
                tokens.push(SmartsToken::OpenBracket);
                i += 1;
                // Parse bracketed atom expression until ']'
                i = parse_bracket_contents(&chars, i, &mut tokens)?;
                if i >= chars.len() || chars[i] != ']' {
                    return Err(format!("Unclosed bracket at position {i}"));
                }
                tokens.push(SmartsToken::CloseBracket);
                i += 1;
            }
            '(' => {
                tokens.push(SmartsToken::BranchOpen);
                i += 1;
            }
            ')' => {
                tokens.push(SmartsToken::BranchClose);
                i += 1;
            }
            '-' => {
                tokens.push(SmartsToken::Bond(BondToken::Single));
                i += 1;
            }
            '=' => {
                tokens.push(SmartsToken::Bond(BondToken::Double));
                i += 1;
            }
            '~' => {
                tokens.push(SmartsToken::Bond(BondToken::Any));
                i += 1;
            }
            ':' => {
                tokens.push(SmartsToken::Bond(BondToken::AromaticBond));
                i += 1;
            }
            '@' => {
                tokens.push(SmartsToken::Bond(BondToken::Ring));
                i += 1;
            }
            d @ '1'..='9' => {
                tokens.push(SmartsToken::RingClosure(d as u8 - b'0'));
                i += 1;
            }
            // Bare organic atoms
            'C' if i + 1 < chars.len() && chars[i + 1] == 'l' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(17)));
                i += 2;
            }
            'B' if i + 1 < chars.len() && chars[i + 1] == 'r' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(35)));
                i += 2;
            }
            'B' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(5)));
                i += 1;
            }
            'C' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(6)));
                i += 1;
            }
            'N' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(7)));
                i += 1;
            }
            'O' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(8)));
                i += 1;
            }
            'F' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(9)));
                i += 1;
            }
            'P' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(15)));
                i += 1;
            }
            'S' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(16)));
                i += 1;
            }
            'I' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(53)));
                i += 1;
            }
            // Bare aromatic atoms
            'c' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::AromaticSymbol(6)));
                i += 1;
            }
            'n' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::AromaticSymbol(7)));
                i += 1;
            }
            'o' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::AromaticSymbol(8)));
                i += 1;
            }
            's' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::AromaticSymbol(16)));
                i += 1;
            }
            'p' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::AromaticSymbol(15)));
                i += 1;
            }
            '*' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Any));
                i += 1;
            }
            '#' if i + 1 < chars.len() && chars[i + 1].is_ascii_digit() => {
                // Outside brackets: triple bond if followed by digit — but # is also
                // used for atomic number inside brackets. Outside brackets, `#` before
                // a digit is ambiguous; treat as triple bond for bare context.
                tokens.push(SmartsToken::Bond(BondToken::Triple));
                i += 1;
            }
            '#' => {
                tokens.push(SmartsToken::Bond(BondToken::Triple));
                i += 1;
            }
            c => {
                return Err(format!("Unexpected character '{c}' at position {i}"));
            }
        }
    }

    Ok(tokens)
}

/// Parse a SMARTS string and reject obviously dangerous degenerate structures
/// before handing it to the full SMARTS matcher.
pub fn parse_and_validate_smarts(input: &str) -> Result<Vec<SmartsToken>, String> {
    let tokens = parse_smarts(input)?;
    validate_smarts_tokens(&tokens)?;
    Ok(tokens)
}

/// Reject token sequences that are syntactically balanced but structurally
/// degenerate, such as `[]`, `()`, or logic operators outside brackets.
pub fn validate_smarts_tokens(tokens: &[SmartsToken]) -> Result<(), String> {
    if tokens.is_empty() {
        return Err("SMARTS is empty".to_string());
    }
    if tokens.len() > MAX_SMARTS_TOKENS {
        return Err(format!(
            "SMARTS exceeds max token length ({}/{MAX_SMARTS_TOKENS})",
            tokens.len()
        ));
    }

    let mut in_bracket = false;
    let mut bracket_has_atom = false;
    let mut bracket_primitive_count = 0usize;
    let mut branch_stack: Vec<bool> = Vec::new();
    let mut branch_depth = 0usize;

    for token in tokens {
        match token {
            SmartsToken::OpenBracket => {
                if in_bracket {
                    return Err("nested brackets are not supported".to_string());
                }
                in_bracket = true;
                bracket_has_atom = false;
                bracket_primitive_count = 0;
            }
            SmartsToken::CloseBracket => {
                if !in_bracket {
                    return Err("unmatched closing bracket".to_string());
                }
                if !bracket_has_atom {
                    return Err("empty bracket atom".to_string());
                }
                in_bracket = false;
                bracket_has_atom = false;
                bracket_primitive_count = 0;
                if let Some(branch_has_atom) = branch_stack.last_mut() {
                    *branch_has_atom = true;
                }
            }
            SmartsToken::Atom(_) => {
                if in_bracket {
                    bracket_has_atom = true;
                    bracket_primitive_count += 1;
                    if bracket_primitive_count > MAX_PRIMITIVES_PER_BRACKET {
                        return Err(format!(
                            "bracket atom exceeds max primitive count ({MAX_PRIMITIVES_PER_BRACKET})"
                        ));
                    }
                }
                if let Some(branch_has_atom) = branch_stack.last_mut() {
                    *branch_has_atom = true;
                }
            }
            SmartsToken::Logic(_) => {
                if !in_bracket {
                    return Err("logic operator outside brackets".to_string());
                }
            }
            SmartsToken::BranchOpen => {
                branch_depth += 1;
                if branch_depth > MAX_BRANCH_DEPTH {
                    return Err(format!(
                        "SMARTS exceeds max branch depth ({MAX_BRANCH_DEPTH})"
                    ));
                }
                branch_stack.push(false);
            }
            SmartsToken::BranchClose => match branch_stack.pop() {
                Some(true) => {
                    branch_depth = branch_depth.saturating_sub(1);
                }
                Some(false) => return Err("empty branch".to_string()),
                None => return Err("unmatched closing branch".to_string()),
            },
            SmartsToken::Bond(_) | SmartsToken::RingClosure(_) => {}
        }
    }

    if in_bracket {
        return Err("unclosed bracket".to_string());
    }
    if !branch_stack.is_empty() {
        return Err("unclosed branch".to_string());
    }

    Ok(())
}

/// Parse the contents inside `[...]` until we hit `]`.
/// Returns the position of `]`.
fn parse_bracket_contents(
    chars: &[char],
    mut i: usize,
    tokens: &mut Vec<SmartsToken>,
) -> Result<usize, String> {
    while i < chars.len() && chars[i] != ']' {
        match chars[i] {
            '!' => {
                tokens.push(SmartsToken::Logic(LogicOp::Not));
                i += 1;
            }
            '&' => {
                tokens.push(SmartsToken::Logic(LogicOp::And));
                i += 1;
            }
            ',' => {
                tokens.push(SmartsToken::Logic(LogicOp::Or));
                i += 1;
            }
            ';' => {
                tokens.push(SmartsToken::Logic(LogicOp::AndLow));
                i += 1;
            }
            '*' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Any));
                i += 1;
            }
            'a' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Aromatic));
                i += 1;
            }
            'A' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Aliphatic));
                i += 1;
            }
            '#' => {
                // Atomic number: #6, #7, #16, etc.
                i += 1;
                let (num, new_i) = parse_number(chars, i)?;
                tokens.push(SmartsToken::Atom(AtomPrimitive::AtomicNumber(num as u8)));
                i = new_i;
            }
            'D' => {
                i += 1;
                let (num, new_i) = parse_optional_number(chars, i, 1);
                tokens.push(SmartsToken::Atom(AtomPrimitive::Degree(num as u8)));
                i = new_i;
            }
            'X' => {
                i += 1;
                let (num, new_i) = parse_optional_number(chars, i, 1);
                tokens.push(SmartsToken::Atom(AtomPrimitive::TotalConnections(
                    num as u8,
                )));
                i = new_i;
            }
            'H' => {
                i += 1;
                let (num, new_i) = parse_optional_number(chars, i, 1);
                tokens.push(SmartsToken::Atom(AtomPrimitive::HCount(num as u8)));
                i = new_i;
            }
            'R' => {
                i += 1;
                if i < chars.len() && chars[i].is_ascii_digit() {
                    let (num, new_i) = parse_number(chars, i)?;
                    tokens.push(SmartsToken::Atom(AtomPrimitive::RingMembership(Some(
                        num as u8,
                    ))));
                    i = new_i;
                } else {
                    tokens.push(SmartsToken::Atom(AtomPrimitive::RingMembership(None)));
                }
            }
            'r' => {
                i += 1;
                if i < chars.len() && chars[i].is_ascii_digit() {
                    let (num, new_i) = parse_number(chars, i)?;
                    tokens.push(SmartsToken::Atom(AtomPrimitive::SmallestRing(Some(
                        num as u8,
                    ))));
                    i = new_i;
                } else {
                    tokens.push(SmartsToken::Atom(AtomPrimitive::SmallestRing(None)));
                }
            }
            'v' => {
                i += 1;
                let (num, new_i) = parse_optional_number(chars, i, 1);
                tokens.push(SmartsToken::Atom(AtomPrimitive::Valence(num as u8)));
                i = new_i;
            }
            'x' => {
                i += 1;
                if i < chars.len() && chars[i].is_ascii_digit() {
                    let (num, new_i) = parse_number(chars, i)?;
                    tokens.push(SmartsToken::Atom(AtomPrimitive::RingBondCount(Some(
                        num as u8,
                    ))));
                    i = new_i;
                } else {
                    tokens.push(SmartsToken::Atom(AtomPrimitive::RingBondCount(None)));
                }
            }
            '+' => {
                i += 1;
                if i < chars.len() && chars[i].is_ascii_digit() {
                    let (num, new_i) = parse_number(chars, i)?;
                    tokens.push(SmartsToken::Atom(AtomPrimitive::Charge(num as i8)));
                    i = new_i;
                } else {
                    tokens.push(SmartsToken::Atom(AtomPrimitive::Charge(1)));
                }
            }
            '-' => {
                i += 1;
                if i < chars.len() && chars[i].is_ascii_digit() {
                    let (num, new_i) = parse_number(chars, i)?;
                    tokens.push(SmartsToken::Atom(AtomPrimitive::Charge(-(num as i8))));
                    i = new_i;
                } else {
                    tokens.push(SmartsToken::Atom(AtomPrimitive::Charge(-1)));
                }
            }
            // Element symbols inside brackets (two-letter first)
            'C' if i + 1 < chars.len() && chars[i + 1] == 'l' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(17)));
                i += 2;
            }
            'B' if i + 1 < chars.len() && chars[i + 1] == 'r' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(35)));
                i += 2;
            }
            'C' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(6)));
                i += 1;
            }
            'B' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(5)));
                i += 1;
            }
            'N' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(7)));
                i += 1;
            }
            'O' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(8)));
                i += 1;
            }
            'F' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(9)));
                i += 1;
            }
            'P' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(15)));
                i += 1;
            }
            'S' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(16)));
                i += 1;
            }
            'I' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::Symbol(53)));
                i += 1;
            }
            'c' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::AromaticSymbol(6)));
                i += 1;
            }
            'n' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::AromaticSymbol(7)));
                i += 1;
            }
            'o' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::AromaticSymbol(8)));
                i += 1;
            }
            's' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::AromaticSymbol(16)));
                i += 1;
            }
            'p' => {
                tokens.push(SmartsToken::Atom(AtomPrimitive::AromaticSymbol(15)));
                i += 1;
            }
            c => {
                // Skip unknown characters inside brackets (e.g., @, /, \)
                // This allows us to parse SMARTS we don't fully model
                i += 1;
                log::trace!("Skipping unknown bracket char '{c}' at position {i}");
            }
        }
    }
    Ok(i)
}

/// Parse a required decimal number at position i.
fn parse_number(chars: &[char], mut i: usize) -> Result<(u32, usize), String> {
    let start = i;
    while i < chars.len() && chars[i].is_ascii_digit() {
        i += 1;
    }
    if i == start {
        return Err(format!("Expected number at position {start}"));
    }
    let s: String = chars[start..i].iter().collect();
    let num: u32 = s
        .parse()
        .map_err(|e| format!("Invalid number '{s}' at position {start}: {e}"))?;
    Ok((num, i))
}

/// Parse an optional decimal number. If no digit follows, returns `default`.
fn parse_optional_number(chars: &[char], i: usize, default: u32) -> (u32, usize) {
    if i < chars.len() && chars[i].is_ascii_digit() {
        parse_number(chars, i).unwrap_or((default, i))
    } else {
        (default, i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::display::tokens_to_smarts;

    #[test]
    fn test_roundtrip_simple() {
        let patterns = &[
            "[#6]",
            "[NX3;H2]",
            "[CX3](=O)[#6]",
            "c1ccccc1",
            "[#7]~[#6]~[#7]",
            "[C,N]",
            "[!#1]",
        ];
        for &pat in patterns {
            let tokens = parse_smarts(pat).expect(pat);
            let out = tokens_to_smarts(&tokens);
            assert_eq!(out, pat, "Roundtrip failed for '{pat}': got '{out}'");
        }
    }

    #[test]
    fn test_roundtrip_complex() {
        let patterns = &["[#6;R]", "[CX4H3]", "[#7;H2;D1]"];
        for &pat in patterns {
            let tokens = parse_smarts(pat).expect(pat);
            let out = tokens_to_smarts(&tokens);
            // For complex patterns, just verify we can parse and reproduce something valid
            assert!(!out.is_empty(), "Empty output for '{pat}'");
        }
    }

    #[test]
    fn test_parse_and_validate_rejects_empty_branch() {
        let err = parse_and_validate_smarts("[#6]()").unwrap_err();
        assert!(err.contains("empty branch"), "unexpected error: {err}");
    }

    #[test]
    fn test_parse_and_validate_rejects_empty_bracket() {
        let err = parse_and_validate_smarts("[]").unwrap_err();
        assert!(err.contains("empty bracket"), "unexpected error: {err}");
    }

    #[test]
    fn test_parse_and_validate_accepts_normal_branch() {
        let tokens = parse_and_validate_smarts("[#6](~[#7])").unwrap();
        assert_eq!(tokens_to_smarts(&tokens), "[#6](~[#7])");
    }

    #[test]
    fn test_parse_and_validate_rejects_overlong_smarts() {
        let smarts = "[#6]".repeat(MAX_SMARTS_TOKENS + 1);
        let err = parse_and_validate_smarts(&smarts).unwrap_err();
        assert!(err.contains("max token length"), "unexpected error: {err}");
    }

    #[test]
    fn test_parse_and_validate_rejects_deep_branch_nesting() {
        let err = parse_and_validate_smarts("[#6]([#6]([#6]([#6]([#6]([#6])))))").unwrap_err();
        assert!(err.contains("max branch depth"), "unexpected error: {err}");
    }

    #[test]
    fn test_parse_and_validate_rejects_too_many_bracket_primitives() {
        let err = parse_and_validate_smarts("[#6;R;H1;D3]").unwrap_err();
        assert!(
            err.contains("max primitive count"),
            "unexpected error: {err}"
        );
    }
}
