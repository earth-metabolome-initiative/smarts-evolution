use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Open a `.jsonl` or `.jsonl.zst` dataset as a buffered line reader.
pub fn open_jsonl_reader(path: &Path) -> Result<Box<dyn BufRead>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;

    if path.extension().and_then(|ext| ext.to_str()) == Some("zst") {
        let decoder = zstd::Decoder::new(file)?;
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}
