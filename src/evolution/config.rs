/// Configuration for the evolutionary algorithm.
#[derive(Clone, Debug)]
pub struct EvolutionConfig {
    /// Population size per task.
    pub population_size: usize,
    /// Maximum generations per task.
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
    /// Stop evolution for a task if no improvement for this many generations.
    pub stagnation_limit: u64,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 200,
            generation_limit: 500,
            mutation_rate: 0.85,
            crossover_rate: 0.7,
            selection_ratio: 0.5,
            tournament_size: 3,
            elite_count: 4,
            random_immigrant_ratio: 0.10,
            stagnation_limit: 50,
        }
    }
}
