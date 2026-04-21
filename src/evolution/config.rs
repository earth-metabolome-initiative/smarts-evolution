const DEFAULT_POPULATION_SIZE: usize = 200;
const DEFAULT_GENERATION_LIMIT: u64 = 500;
const DEFAULT_MUTATION_RATE: f64 = 0.85;
const DEFAULT_CROSSOVER_RATE: f64 = 0.7;
const DEFAULT_SELECTION_RATIO: f64 = 0.5;
const DEFAULT_TOURNAMENT_SIZE: usize = 3;
const DEFAULT_ELITE_COUNT: usize = 4;
const DEFAULT_RANDOM_IMMIGRANT_RATIO: f64 = 0.10;
const DEFAULT_STAGNATION_LIMIT: u64 = 50;

/// Configuration for the evolutionary algorithm.
#[derive(Clone, Debug)]
pub struct EvolutionConfig {
    population_size: usize,
    generation_limit: u64,
    mutation_rate: f64,
    crossover_rate: f64,
    selection_ratio: f64,
    tournament_size: usize,
    elite_count: usize,
    random_immigrant_ratio: f64,
    stagnation_limit: u64,
}

/// Fluent builder for [`EvolutionConfig`].
#[derive(Clone, Debug)]
pub struct EvolutionConfigBuilder {
    population_size: usize,
    generation_limit: u64,
    mutation_rate: f64,
    crossover_rate: f64,
    selection_ratio: f64,
    tournament_size: usize,
    elite_count: usize,
    random_immigrant_ratio: f64,
    stagnation_limit: u64,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self::builder().finish()
    }
}

impl Default for EvolutionConfigBuilder {
    fn default() -> Self {
        Self {
            population_size: DEFAULT_POPULATION_SIZE,
            generation_limit: DEFAULT_GENERATION_LIMIT,
            mutation_rate: DEFAULT_MUTATION_RATE,
            crossover_rate: DEFAULT_CROSSOVER_RATE,
            selection_ratio: DEFAULT_SELECTION_RATIO,
            tournament_size: DEFAULT_TOURNAMENT_SIZE,
            elite_count: DEFAULT_ELITE_COUNT,
            random_immigrant_ratio: DEFAULT_RANDOM_IMMIGRANT_RATIO,
            stagnation_limit: DEFAULT_STAGNATION_LIMIT,
        }
    }
}

impl EvolutionConfig {
    /// Start building one evolution configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use smarts_evolution::EvolutionConfig;
    ///
    /// let config = EvolutionConfig::builder()
    ///     .population_size(32)
    ///     .generation_limit(10)
    ///     .stagnation_limit(5)
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(config.population_size(), 32);
    /// assert!(config.validate().is_ok());
    /// ```
    pub fn builder() -> EvolutionConfigBuilder {
        EvolutionConfigBuilder::default()
    }

    /// Validate that the configuration is internally consistent.
    pub fn validate(&self) -> Result<(), String> {
        if self.population_size == 0 {
            return Err("population_size must be greater than zero".into());
        }
        if self.generation_limit == 0 {
            return Err("generation_limit must be greater than zero".into());
        }
        if self.stagnation_limit == 0 {
            return Err("stagnation_limit must be greater than zero".into());
        }
        if !(0.0..=1.0).contains(&self.mutation_rate) {
            return Err("mutation_rate must be between 0.0 and 1.0".into());
        }
        if !(0.0..=1.0).contains(&self.crossover_rate) {
            return Err("crossover_rate must be between 0.0 and 1.0".into());
        }
        if !(0.0..=1.0).contains(&self.selection_ratio) || self.selection_ratio == 0.0 {
            return Err("selection_ratio must be greater than 0.0 and at most 1.0".into());
        }
        if self.tournament_size == 0 {
            return Err("tournament_size must be greater than zero".into());
        }
        if !(0.0..=1.0).contains(&self.random_immigrant_ratio) {
            return Err("random_immigrant_ratio must be between 0.0 and 1.0".into());
        }
        if self.elite_count > self.population_size {
            return Err("elite_count must not exceed population_size".into());
        }
        Ok(())
    }

    pub fn population_size(&self) -> usize {
        self.population_size
    }

    pub fn generation_limit(&self) -> u64 {
        self.generation_limit
    }

    pub fn mutation_rate(&self) -> f64 {
        self.mutation_rate
    }

    pub fn crossover_rate(&self) -> f64 {
        self.crossover_rate
    }

    pub fn selection_ratio(&self) -> f64 {
        self.selection_ratio
    }

    pub fn tournament_size(&self) -> usize {
        self.tournament_size
    }

    pub fn elite_count(&self) -> usize {
        self.elite_count
    }

    pub fn random_immigrant_ratio(&self) -> f64 {
        self.random_immigrant_ratio
    }

    pub fn stagnation_limit(&self) -> u64 {
        self.stagnation_limit
    }
}

impl EvolutionConfigBuilder {
    pub fn population_size(mut self, population_size: usize) -> Self {
        self.population_size = population_size;
        self
    }

    pub fn generation_limit(mut self, generation_limit: u64) -> Self {
        self.generation_limit = generation_limit;
        self
    }

    pub fn mutation_rate(mut self, mutation_rate: f64) -> Self {
        self.mutation_rate = mutation_rate;
        self
    }

    pub fn crossover_rate(mut self, crossover_rate: f64) -> Self {
        self.crossover_rate = crossover_rate;
        self
    }

    pub fn selection_ratio(mut self, selection_ratio: f64) -> Self {
        self.selection_ratio = selection_ratio;
        self
    }

    pub fn tournament_size(mut self, tournament_size: usize) -> Self {
        self.tournament_size = tournament_size;
        self
    }

    pub fn elite_count(mut self, elite_count: usize) -> Self {
        self.elite_count = elite_count;
        self
    }

    pub fn random_immigrant_ratio(mut self, random_immigrant_ratio: f64) -> Self {
        self.random_immigrant_ratio = random_immigrant_ratio;
        self
    }

    pub fn stagnation_limit(mut self, stagnation_limit: u64) -> Self {
        self.stagnation_limit = stagnation_limit;
        self
    }

    pub fn build(self) -> Result<EvolutionConfig, String> {
        let config = self.finish();
        config.validate()?;
        Ok(config)
    }

    fn finish(self) -> EvolutionConfig {
        EvolutionConfig {
            population_size: self.population_size,
            generation_limit: self.generation_limit,
            mutation_rate: self.mutation_rate,
            crossover_rate: self.crossover_rate,
            selection_ratio: self.selection_ratio,
            tournament_size: self.tournament_size,
            elite_count: self.elite_count,
            random_immigrant_ratio: self.random_immigrant_ratio,
            stagnation_limit: self.stagnation_limit,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        assert!(EvolutionConfig::default().validate().is_ok());
    }

    #[test]
    fn validate_rejects_population_size_zero() {
        let config = EvolutionConfig::builder().population_size(0).build();
        assert_eq!(
            config.unwrap_err(),
            "population_size must be greater than zero"
        );
    }

    #[test]
    fn validate_rejects_generation_limit_zero() {
        let config = EvolutionConfig::builder().generation_limit(0).build();
        assert_eq!(
            config.unwrap_err(),
            "generation_limit must be greater than zero"
        );
    }

    #[test]
    fn validate_rejects_stagnation_limit_zero() {
        let config = EvolutionConfig::builder().stagnation_limit(0).build();
        assert_eq!(
            config.unwrap_err(),
            "stagnation_limit must be greater than zero"
        );
    }

    #[test]
    fn validate_rejects_invalid_probabilities_and_sizes() {
        let invalid_cases = [
            (
                EvolutionConfig::builder().mutation_rate(-0.1).build(),
                "mutation_rate must be between 0.0 and 1.0",
            ),
            (
                EvolutionConfig::builder().crossover_rate(1.1).build(),
                "crossover_rate must be between 0.0 and 1.0",
            ),
            (
                EvolutionConfig::builder().selection_ratio(0.0).build(),
                "selection_ratio must be greater than 0.0 and at most 1.0",
            ),
            (
                EvolutionConfig::builder().tournament_size(0).build(),
                "tournament_size must be greater than zero",
            ),
            (
                EvolutionConfig::builder()
                    .random_immigrant_ratio(1.1)
                    .build(),
                "random_immigrant_ratio must be between 0.0 and 1.0",
            ),
            (
                EvolutionConfig::builder()
                    .elite_count(EvolutionConfig::default().population_size() + 1)
                    .build(),
                "elite_count must not exceed population_size",
            ),
        ];

        for (config, expected) in invalid_cases {
            assert_eq!(config.unwrap_err(), expected);
        }
    }
}
