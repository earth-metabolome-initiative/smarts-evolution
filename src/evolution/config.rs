use std::path::PathBuf;

/// Configuration for the evolutionary algorithm.
#[derive(Clone, Debug)]
pub struct EvolutionConfig {
    /// Population size per node.
    pub population_size: usize,
    /// Maximum generations per node.
    pub generation_limit: u64,
    /// Mutation rate (probability of mutating each offspring).
    pub mutation_rate: f64,
    /// Crossover rate (probability of crossing over each parent pair).
    pub crossover_rate: f64,
    /// Fraction of population selected as parents.
    pub selection_ratio: f64,
    /// Tournament size for selection.
    pub tournament_size: usize,
    /// Fixed number of elites preserved unchanged each generation.
    pub elite_count: usize,
    /// Fraction of each generation replaced by fresh random immigrants.
    pub random_immigrant_ratio: f64,
    /// Stop evolution for a node if no improvement for this many generations.
    pub stagnation_limit: u64,
    /// Save checkpoint every N generations.
    pub checkpoint_interval: u64,
    /// Directory for checkpoint files.
    pub checkpoint_dir: PathBuf,
    /// Number of evaluator worker processes.
    pub worker_processes: usize,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 200,
            generation_limit: 500,
            mutation_rate: 0.15,
            crossover_rate: 0.7,
            selection_ratio: 0.5,
            tournament_size: 3,
            elite_count: 4,
            random_immigrant_ratio: 0.10,
            stagnation_limit: 50,
            checkpoint_interval: 25,
            checkpoint_dir: PathBuf::from("checkpoints"),
            worker_processes: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::EvolutionConfig;

    #[test]
    fn default_worker_processes_follow_available_parallelism() {
        let config = EvolutionConfig::default();
        let expected = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        assert_eq!(config.worker_processes, expected);
    }
}
