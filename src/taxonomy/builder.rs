use std::collections::{HashMap, HashSet};

use super::dag::TaxonomyDag;
use super::node::TaxonomyNode;
use crate::data::compound::Compound;

/// Build a TaxonomyDag from loaded compounds and their level names.
///
/// Edges are inferred from co-occurrence: if a compound has label A at level i
/// and label B at level i+1, we create an edge A → B. For multi-label compounds,
/// all combinations between adjacent levels produce edges (the taxonomy is a DAG).
pub fn build_dag(
    compounds: &[Compound],
    level_names: &[&str],
) -> Result<TaxonomyDag, Box<dyn std::error::Error>> {
    let num_levels = level_names.len();

    // Step 1: Collect all unique labels and assign dense IDs.
    // Key: (level, name) → node_id
    let mut name_to_id: HashMap<(usize, String), usize> = HashMap::new();
    let mut nodes: Vec<TaxonomyNode> = Vec::new();

    for compound in compounds {
        for (level, labels) in compound.labels.iter().enumerate().take(num_levels) {
            for label in labels {
                if !name_to_id.contains_key(&(level, label.clone())) {
                    let id = nodes.len();
                    name_to_id.insert((level, label.clone()), id);
                    nodes.push(TaxonomyNode {
                        id,
                        name: label.clone(),
                        level,
                        compound_indices: Vec::new(),
                    });
                }
            }
        }
    }

    // Step 2: Populate compound_indices for each node.
    for (compound_idx, compound) in compounds.iter().enumerate() {
        for (level, labels) in compound.labels.iter().enumerate().take(num_levels) {
            for label in labels {
                if let Some(&node_id) = name_to_id.get(&(level, label.clone())) {
                    nodes[node_id].compound_indices.push(compound_idx);
                }
            }
        }
    }

    // Step 3: Infer parent→child edges from adjacent-level co-occurrence.
    let mut edges: HashSet<(usize, usize)> = HashSet::new();

    for compound in compounds {
        for level in 0..num_levels.saturating_sub(1) {
            let parent_labels = &compound.labels[level];
            let child_labels = compound.labels.get(level + 1).cloned().unwrap_or_default();

            for parent_name in parent_labels {
                for child_name in &child_labels {
                    if let (Some(&pid), Some(&cid)) = (
                        name_to_id.get(&(level, parent_name.clone())),
                        name_to_id.get(&(level + 1, child_name.clone())),
                    ) {
                        edges.insert((pid, cid));
                    }
                }
            }
        }
    }

    let edges_vec: Vec<(usize, usize)> = edges.into_iter().collect();

    TaxonomyDag::new(nodes, edges_vec, level_names)
}
