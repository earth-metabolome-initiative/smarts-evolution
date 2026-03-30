use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, info, warn};
use rdkit::ROMol;

use super::compound::Compound;
use super::rdkit_lock::with_rdkit_lock;

/// Warm up RDKit's lazy singletons (PeriodicTable, etc.) on the main thread.
/// Must be called before long-running RDKit usage. See:
/// https://github.com/rdkit/rdkit/issues/5746
pub fn warmup_rdkit() {
    with_rdkit_lock(|| {
        let _ = ROMol::from_smiles("C");
        let _ = rdkit::RWMol::from_smarts("[#6]");
        let query = rdkit::RWMol::from_smarts("[#6]").unwrap().to_ro_mol();
        let mol = ROMol::from_smiles("CC").unwrap();
        let params = rdkit::SubstructMatchParameters::default();
        let _ = rdkit::substruct_match(&mol, &query, &params);
    });
    debug!("RDKit singletons warmed up");
}

/// Parse SMILES strings once to validate them, populating `compound.parsed`.
///
/// This stays single-threaded because RDKit parsing is not safely reentrant in
/// a shared process through the current Rust FFI bindings.
pub fn parse_molecules(compounds: &mut [Compound]) {
    let total = compounds.len();
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{msg} [{bar:30.cyan/dim}] {pos}/{len} molecules ({eta} remaining)",
        )
        .unwrap()
        .progress_chars("=> "),
    );
    pb.set_message("Parsing molecules");

    let inputs: Vec<(u64, String)> = compounds
        .iter()
        .map(|c| (c.cid, c.smiles.clone()))
        .collect();

    let parsed_flags: Vec<bool> = with_rdkit_lock(|| {
        inputs
            .iter()
            .map(|(cid, smiles)| {
                let parsed = match ROMol::from_smiles(smiles) {
                    Ok(_) => true,
                    Err(_) => {
                        warn!("Failed to parse SMILES for cid={cid}: {smiles}");
                        false
                    }
                };
                pb.inc(1);
                parsed
            })
            .collect()
    });

    let mut ok = 0usize;
    for (compound, parsed) in compounds.iter_mut().zip(parsed_flags) {
        if parsed {
            ok += 1;
        }
        compound.parsed = parsed;
    }
    pb.finish_with_message(format!("Parsing molecules complete ({ok}/{total})"));
    info!("Parsed {ok}/{total} molecules successfully");
}
