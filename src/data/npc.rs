use std::io::{BufRead, BufReader};
use std::path::Path;

use log::{info, warn};
use serde::Deserialize;

use super::compound::Compound;

/// The taxonomy level names for NPC, in order from coarsest to finest.
pub const LEVELS: &[&str] = &["pathway", "superclass", "class"];

#[derive(Deserialize)]
struct Record {
    cid: u64,
    smiles: String,
    #[serde(default)]
    pathway_results: Vec<String>,
    #[serde(default)]
    superclass_results: Vec<String>,
    #[serde(default)]
    class_results: Vec<String>,
}

/// Load compounds from an NPC `.jsonl` file.
///
/// Entries where any of pathway_results, superclass_results, or class_results
/// is empty are discarded.
pub fn load(path: &Path) -> Result<Vec<Compound>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

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

        if record.smiles.is_empty()
            || record.pathway_results.is_empty()
            || record.superclass_results.is_empty()
            || record.class_results.is_empty()
        {
            skipped += 1;
            continue;
        }

        let labels = vec![
            record.pathway_results,
            record.superclass_results,
            record.class_results,
        ];

        compounds.push(Compound {
            cid: record.cid,
            smiles: record.smiles,
            parsed: false,
            labels,
        });
    }

    info!(
        "NPC: loaded {} compounds ({} skipped) from {}",
        compounds.len(),
        skipped,
        path.display()
    );
    Ok(compounds)
}
