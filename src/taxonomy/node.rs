/// A node in the taxonomy DAG.
#[derive(Clone, Debug)]
pub struct TaxonomyNode {
    /// Dense integer ID (matches the graph node index).
    pub id: usize,
    /// Human-readable name (e.g. "Organic compounds").
    pub name: String,
    /// Hierarchy level (0 = coarsest, e.g. kingdom/pathway).
    pub level: usize,
    /// Indices into the compound array for compounds belonging to this node.
    pub compound_indices: Vec<usize>,
}
