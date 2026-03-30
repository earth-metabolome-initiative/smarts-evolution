use std::collections::HashMap;

use geometric_traits::prelude::*;

use super::node::TaxonomyNode;

/// A taxonomy DAG backed by geometric-traits.
pub struct TaxonomyDag {
    /// The underlying directed graph.
    pub graph: DiGraph<usize>,
    /// Nodes indexed by their dense ID.
    pub nodes: Vec<TaxonomyNode>,
    /// Parent node IDs for each node.
    parents: Vec<Vec<usize>>,
    /// Lookup: (level, name) → node_id.
    pub name_to_id: HashMap<(usize, String), usize>,
    /// Topological order (parent before child).
    pub topological_order: Vec<usize>,
    /// Level names (e.g. ["kingdom", "superclass", "class", ...]).
    pub level_names: Vec<String>,
}

impl TaxonomyDag {
    /// Build from nodes and directed edges (parent_id → child_id).
    pub fn new(
        nodes: Vec<TaxonomyNode>,
        mut edges: Vec<(usize, usize)>,
        level_names: &[&str],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let num_nodes = nodes.len();
        let mut parents = vec![Vec::new(); num_nodes];

        for &(parent_id, child_id) in &edges {
            parents[child_id].push(parent_id);
        }
        for parent_ids in &mut parents {
            parent_ids.sort_unstable();
            parent_ids.dedup();
        }

        // Build name_to_id lookup
        let mut name_to_id = HashMap::with_capacity(num_nodes);
        for node in &nodes {
            name_to_id.insert((node.level, node.name.clone()), node.id);
        }

        // CSR format requires edges sorted by (source, destination)
        edges.sort();

        // Build the geometric-traits DiGraph
        let node_ids: Vec<usize> = (0..num_nodes).collect();
        let vocab: SortedVec<usize> = GenericVocabularyBuilder::default()
            .expected_number_of_symbols(num_nodes)
            .symbols(node_ids.into_iter().enumerate())
            .build()
            .map_err(|e| format!("Vocabulary build error: {e:?}"))?;

        let edge_matrix: SquareCSR2D<CSR2D<usize, usize, usize>> = DiEdgesBuilder::default()
            .expected_number_of_edges(edges.len())
            .expected_shape(num_nodes)
            .edges(edges.into_iter())
            .build()
            .map_err(|e| format!("Edge build error: {e:?}"))?;

        let graph: DiGraph<usize> = DiGraph::from((vocab, edge_matrix));

        // Run Kahn's topological sort to validate DAG and get ordering
        let topo = graph
            .edges()
            .matrix()
            .kahn()
            .map_err(|e| format!("Topological sort failed (cycle?): {e:?}"))?;

        Ok(Self {
            graph,
            nodes,
            parents,
            name_to_id,
            topological_order: topo,
            level_names: level_names.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Number of nodes in the DAG.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get nodes at a given hierarchy level.
    pub fn nodes_at_level(&self, level: usize) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| n.level == level)
            .map(|n| n.id)
            .collect()
    }

    /// Get parent node IDs for a given node.
    pub fn parents_of(&self, node_id: usize) -> &[usize] {
        &self.parents[node_id]
    }

    /// Get the occurrence count per node (number of compounds in each node).
    /// Used as input for InformationContent and Lin similarity.
    pub fn occurrences(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .map(|n| n.compound_indices.len())
            .collect()
    }
}
