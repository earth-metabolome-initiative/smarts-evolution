use std::io::BufRead;
use std::path::Path;

use log::{info, warn};
use serde::Deserialize;
use serde_json::{Map, Value};

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
struct RawZenodoRecord {
    #[serde(rename = "cid")]
    _cid: u64,
    classyfire: Map<String, Value>,
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
    let mut raw_rows_without_smiles = 0usize;

    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let record = match serde_json::from_str::<Record>(&line) {
            Ok(r) => r,
            Err(error) => match serde_json::from_str::<RawZenodoRecord>(&line) {
                Ok(raw) => {
                    if !raw.classyfire.contains_key("smiles") {
                        raw_rows_without_smiles += 1;
                    } else {
                        warn!("Skipping line {}: {error}", line_no + 1);
                    }
                    skipped += 1;
                    continue;
                }
                Err(_) => {
                    warn!("Skipping line {}: {error}", line_no + 1);
                    skipped += 1;
                    continue;
                }
            },
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

    if compounds.is_empty() && raw_rows_without_smiles > 0 {
        return Err(format!(
            "{} appears to be a raw weekly ClassyFire Zenodo snapshot without SMILES; omit --path so smarts-evolution can prepare the labeled-SMILES dataset automatically",
            path.display()
        )
        .into());
    }

    Ok(compounds)
}
