use std::io::BufRead;
use std::path::Path;

use log::{info, warn};
use serde::Deserialize;

use super::compound::Compound;
use super::io::open_jsonl_reader;

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

/// Load compounds from an NPC `.jsonl` or `.jsonl.zst` file.
///
/// Entries where any of pathway_results, superclass_results, or class_results
/// is empty are discarded.
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

#[cfg(test)]
mod tests {
    use std::fs::{self, File};
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::load;

    fn unique_temp_path(name: &str, extension: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "smarts-evolution-npc-{name}-{}-{nanos}.{extension}",
            std::process::id()
        ))
    }

    #[test]
    fn load_accepts_zstd_compressed_jsonl() {
        let path = unique_temp_path("compressed", "jsonl.zst");
        let file = File::create(&path).unwrap();
        let mut encoder = zstd::Encoder::new(file, 0).unwrap();
        writeln!(
            encoder,
            "{{\"cid\":1,\"smiles\":\"CC\",\"pathway_results\":[\"PathA\"],\"superclass_results\":[\"SuperA\"],\"class_results\":[\"ClassA\"]}}"
        )
        .unwrap();
        writeln!(
            encoder,
            "{{\"cid\":2,\"smiles\":\"N\",\"pathway_results\":[\"PathB\"],\"superclass_results\":[],\"class_results\":[\"ClassB\"]}}"
        )
        .unwrap();
        encoder.finish().unwrap();

        let compounds = load(&path).unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!(compounds.len(), 1);
        assert_eq!(compounds[0].cid, 1);
        assert_eq!(compounds[0].smiles, "CC");
        assert_eq!(compounds[0].labels[0], vec!["PathA".to_string()]);
        assert_eq!(compounds[0].labels[1], vec!["SuperA".to_string()]);
        assert_eq!(compounds[0].labels[2], vec!["ClassA".to_string()]);
    }
}
