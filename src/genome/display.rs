use std::fmt;

use super::token::*;

/// Format a slice of SMARTS tokens into a SMARTS string.
pub fn tokens_to_smarts(tokens: &[SmartsToken]) -> String {
    let mut out = String::with_capacity(tokens.len() * 2);
    for token in tokens {
        write_token(&mut out, token);
    }
    out
}

fn write_token(out: &mut String, token: &SmartsToken) {
    match token {
        SmartsToken::OpenBracket => out.push('['),
        SmartsToken::CloseBracket => out.push(']'),
        SmartsToken::Atom(prim) => write_atom_primitive(out, prim),
        SmartsToken::Logic(op) => write_logic(out, op),
        SmartsToken::Bond(b) => write_bond(out, b),
        SmartsToken::BranchOpen => out.push('('),
        SmartsToken::BranchClose => out.push(')'),
        SmartsToken::RingClosure(d) => {
            fmt::Write::write_fmt(out, format_args!("{d}")).unwrap();
        }
    }
}

fn write_atom_primitive(out: &mut String, prim: &AtomPrimitive) {
    match prim {
        AtomPrimitive::Any => out.push('*'),
        AtomPrimitive::Aromatic => out.push('a'),
        AtomPrimitive::Aliphatic => out.push('A'),
        AtomPrimitive::AtomicNumber(n) => {
            fmt::Write::write_fmt(out, format_args!("#{n}")).unwrap();
        }
        AtomPrimitive::Symbol(n) => {
            if let Some(sym) = atomic_number_to_symbol(*n) {
                out.push_str(sym);
            } else {
                fmt::Write::write_fmt(out, format_args!("#{n}")).unwrap();
            }
        }
        AtomPrimitive::AromaticSymbol(n) => {
            if let Some(sym) = atomic_number_to_aromatic(*n) {
                out.push_str(sym);
            } else {
                fmt::Write::write_fmt(out, format_args!("#{n}")).unwrap();
            }
        }
        AtomPrimitive::Degree(d) => {
            fmt::Write::write_fmt(out, format_args!("D{d}")).unwrap();
        }
        AtomPrimitive::TotalConnections(x) => {
            fmt::Write::write_fmt(out, format_args!("X{x}")).unwrap();
        }
        AtomPrimitive::HCount(h) => {
            fmt::Write::write_fmt(out, format_args!("H{h}")).unwrap();
        }
        AtomPrimitive::RingMembership(None) => out.push('R'),
        AtomPrimitive::RingMembership(Some(r)) => {
            fmt::Write::write_fmt(out, format_args!("R{r}")).unwrap();
        }
        AtomPrimitive::SmallestRing(None) => out.push('r'),
        AtomPrimitive::SmallestRing(Some(r)) => {
            fmt::Write::write_fmt(out, format_args!("r{r}")).unwrap();
        }
        AtomPrimitive::Valence(v) => {
            fmt::Write::write_fmt(out, format_args!("v{v}")).unwrap();
        }
        AtomPrimitive::Charge(c) => {
            if *c > 0 {
                out.push('+');
                if *c > 1 {
                    fmt::Write::write_fmt(out, format_args!("{c}")).unwrap();
                }
            } else if *c < 0 {
                out.push('-');
                if *c < -1 {
                    fmt::Write::write_fmt(out, format_args!("{}", c.abs())).unwrap();
                }
            }
        }
        AtomPrimitive::RingBondCount(None) => out.push('x'),
        AtomPrimitive::RingBondCount(Some(x)) => {
            fmt::Write::write_fmt(out, format_args!("x{x}")).unwrap();
        }
    }
}

fn write_logic(out: &mut String, op: &LogicOp) {
    match op {
        LogicOp::And => out.push('&'),
        LogicOp::Or => out.push(','),
        LogicOp::AndLow => out.push(';'),
        LogicOp::Not => out.push('!'),
    }
}

fn write_bond(out: &mut String, b: &BondToken) {
    match b {
        BondToken::Single => out.push('-'),
        BondToken::Double => out.push('='),
        BondToken::Triple => out.push('#'),
        BondToken::AromaticBond => out.push(':'),
        BondToken::Any => out.push('~'),
        BondToken::Ring => out.push('@'),
    }
}
