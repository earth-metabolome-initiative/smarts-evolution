use std::collections::HashSet;

use rand::Rng;

use super::stratified::stratified_k_fold;
use crate::taxonomy::dag::TaxonomyDag;

/// Per-node fold assignments for k-fold cross-validation.
pub struct FoldSplits {
    /// Number of folds.
    pub k: usize,
    /// For each fold i, `fold_indices[i]` is the set of compound indices in that fold.
    pub fold_indices: Vec<Vec<usize>>,
}

/// Minimum number of positive examples for a node to be eligible for evolution.
pub const MIN_POSITIVE_EXAMPLES: usize = 20;

impl FoldSplits {
    /// Build k-fold stratified splits from the taxonomy DAG.
    ///
    /// Stratification is based on the most specific (deepest) label each compound carries.
    /// For multi-label datasets, iterative stratification is used.
    pub fn build<R: Rng>(dag: &TaxonomyDag, k: usize, rng: &mut R) -> Self {
        // For each compound, collect its labels at the finest level.
        // We stratify on the finest level because that automatically balances coarser levels.
        let finest_level = dag.level_names.len() - 1;

        // Build per-compound label list at finest level
        let num_compounds = dag
            .nodes
            .iter()
            .flat_map(|n| n.compound_indices.iter())
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        let mut labels_per_compound: Vec<Vec<String>> = vec![Vec::new(); num_compounds];
        for node in &dag.nodes {
            if node.level == finest_level {
                for &ci in &node.compound_indices {
                    labels_per_compound[ci].push(node.name.clone());
                }
            }
        }

        // Compounds without a finest-level label get a synthetic label based on
        // their deepest available label (to avoid unlabeled strata)
        for (ci, labels) in labels_per_compound.iter_mut().enumerate() {
            if labels.is_empty() {
                // Find deepest label for this compound
                let mut deepest = String::from("__unlabeled__");
                let mut deepest_level = 0;
                for node in &dag.nodes {
                    if node.level >= deepest_level && node.compound_indices.contains(&ci) {
                        deepest_level = node.level;
                        deepest = format!("__level{}_{}", node.level, node.name);
                    }
                }
                labels.push(deepest);
            }
        }

        let fold_indices = stratified_k_fold(&labels_per_compound, k, rng);

        FoldSplits { k, fold_indices }
    }

    /// For a given node, get the (train_indices, test_indices) for fold `fold_idx`.
    ///
    /// `positive_set` is the set of compound indices belonging to this node.
    /// `candidate_set` is the set of compound indices to consider (e.g. those matching parent SMARTS).
    ///
    /// Returns (train, test) where each is a Vec of compound indices,
    /// and a boolean `is_positive` per index.
    pub fn train_test_for_node(
        &self,
        fold_idx: usize,
        positive_set: &HashSet<usize>,
        candidate_set: &HashSet<usize>,
    ) -> (Vec<usize>, Vec<bool>, Vec<usize>, Vec<bool>) {
        let mut train_indices = Vec::new();
        let mut train_labels = Vec::new();
        let mut test_indices = Vec::new();
        let mut test_labels = Vec::new();

        for (fi, fold) in self.fold_indices.iter().enumerate() {
            for &ci in fold {
                if !candidate_set.contains(&ci) {
                    continue;
                }
                let is_pos = positive_set.contains(&ci);
                if fi == fold_idx {
                    test_indices.push(ci);
                    test_labels.push(is_pos);
                } else {
                    train_indices.push(ci);
                    train_labels.push(is_pos);
                }
            }
        }

        (train_indices, train_labels, test_indices, test_labels)
    }

    /// Check if a node has enough positive examples for evolution.
    pub fn is_eligible(node_compound_count: usize) -> bool {
        node_compound_count >= MIN_POSITIVE_EXAMPLES
    }
}
