/// A compound with its parsed molecule and taxonomy labels.
///
/// `labels` is indexed by taxonomy level (0 = coarsest, e.g. kingdom/pathway).
/// Each entry is a Vec because some datasets are multi-label at a given level.
/// An empty inner Vec means the compound has no label at that level.
pub struct Compound {
    pub cid: u64,
    pub smiles: String,
    pub parsed: bool,
    pub labels: Vec<Vec<String>>,
}
