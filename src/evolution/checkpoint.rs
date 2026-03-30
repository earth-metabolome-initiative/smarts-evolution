use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Checkpoint for a single node's evolution.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NodeCheckpoint {
    pub node_id: usize,
    pub node_name: String,
    pub level: usize,
    pub generation: u64,
    pub best_smarts: String,
    #[serde(default)]
    pub best_mcc: f64,
    pub population: Vec<String>,
}

/// Full checkpoint containing all evolved nodes.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct FullCheckpoint {
    pub nodes: HashMap<usize, NodeCheckpoint>,
}

impl FullCheckpoint {
    /// Load from a JSON file.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let checkpoint: Self = serde_json::from_str(&data)?;
        Ok(checkpoint)
    }

    /// Save to a JSON file.
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }
}
