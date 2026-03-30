use std::io::BufRead;
use std::path::Path;

use log::{info, warn};
use serde::Deserialize;

use super::compound::Compound;
use super::io::open_jsonl_reader;

/// The taxonomy level names for ClassyFire, in order from coarsest to finest.
pub const LEVELS: &[&str] = &[
    "kingdom",
    "superclass",
    "class",
    "subclass",
    "direct_parent",
];

#[derive(Deserialize)]
struct Record {
    cid: u64,
    classyfire: ClassyFireData,
}

#[derive(Deserialize)]
struct ClassyFireData {
    smiles: String,
    kingdom: Option<TaxNode>,
    superclass: Option<TaxNode>,
    class: Option<TaxNode>,
    subclass: Option<TaxNode>,
    direct_parent: Option<TaxNode>,
}

#[derive(Deserialize)]
struct TaxNode {
    name: Option<String>,
}

/// Load compounds from a ClassyFire `.jsonl` or `.jsonl.zst` file.
///
/// Returns the list of compounds and the level names.
/// Compounds whose SMILES is empty are skipped.
pub fn load(path: &Path) -> Result<Vec<Compound>, Box<dyn std::error::Error>> {
    let reader = open_jsonl_reader(path)?;

    let mut compounds = Vec::new();
    let mut skipped = 0usize;

    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let record: Record = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                warn!("Skipping line {}: {e}", line_no + 1);
                skipped += 1;
                continue;
            }
        };

        let cf = &record.classyfire;
        if cf.smiles.is_empty() {
            skipped += 1;
            continue;
        }

        let nodes = [
            &cf.kingdom,
            &cf.superclass,
            &cf.class,
            &cf.subclass,
            &cf.direct_parent,
        ];
        let labels: Vec<Vec<String>> = nodes
            .iter()
            .map(|opt| {
                opt.as_ref()
                    .and_then(|n| n.name.clone())
                    .filter(|s| !s.is_empty())
                    .into_iter()
                    .collect()
            })
            .collect();

        compounds.push(Compound {
            cid: record.cid,
            smiles: cf.smiles.clone(),
            parsed: false,
            labels,
        });
    }

    info!(
        "ClassyFire: loaded {} compounds ({} skipped) from {}",
        compounds.len(),
        skipped,
        path.display()
    );
    Ok(compounds)
}
