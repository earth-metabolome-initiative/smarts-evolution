use alloc::format;
use alloc::string::String;
use core::time::Duration;

use crate::genome::limits::MAX_SMARTS_COMPLEXITY;

const DEFAULT_POPULATION_SIZE: usize = 200;
const DEFAULT_GENERATION_LIMIT: u64 = 500;
const DEFAULT_MUTATION_RATE: f64 = 0.85;
const DEFAULT_CROSSOVER_RATE: f64 = 0.7;
const DEFAULT_SELECTION_RATIO: f64 = 0.5;
const DEFAULT_TOURNAMENT_SIZE: usize = 3;
const DEFAULT_ELITE_COUNT: usize = 4;
const DEFAULT_RANDOM_IMMIGRANT_RATIO: f64 = 0.10;
const DEFAULT_STAGNATION_LIMIT: u64 = 50;
const DEFAULT_FITNESS_CACHE_CAPACITY: usize = 100_000;
const DEFAULT_SLOW_EVALUATION_LOG_THRESHOLD: Duration = Duration::from_secs(30);
const DEFAULT_MAX_EVALUATION_SMARTS_COMPLEXITY: usize = 1_536;

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
    rng_seed: Option<u64>,
    fitness_cache_capacity: usize,
    max_evaluation_smarts_complexity: usize,
    max_evaluation_smarts_len: Option<usize>,
    slow_evaluation_log_threshold: Option<Duration>,
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
    rng_seed: Option<u64>,
    fitness_cache_capacity: usize,
    max_evaluation_smarts_complexity: usize,
    max_evaluation_smarts_len: Option<usize>,
    slow_evaluation_log_threshold: Option<Duration>,
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
            rng_seed: None,
            fitness_cache_capacity: DEFAULT_FITNESS_CACHE_CAPACITY,
            max_evaluation_smarts_complexity: DEFAULT_MAX_EVALUATION_SMARTS_COMPLEXITY,
            max_evaluation_smarts_len: None,
            slow_evaluation_log_threshold: Some(DEFAULT_SLOW_EVALUATION_LOG_THRESHOLD),
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
        if self.max_evaluation_smarts_complexity == 0 {
            return Err("max_evaluation_smarts_complexity must be greater than zero".into());
        }
        if self.max_evaluation_smarts_complexity > MAX_SMARTS_COMPLEXITY {
            return Err(format!(
                "max_evaluation_smarts_complexity must not exceed {MAX_SMARTS_COMPLEXITY}"
            ));
        }
        if self.max_evaluation_smarts_len == Some(0) {
            return Err("max_evaluation_smarts_len must be greater than zero".into());
        }
        if self
            .slow_evaluation_log_threshold
            .is_some_and(|threshold| threshold.is_zero())
        {
            return Err("slow_evaluation_log_threshold must be greater than zero".into());
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

    pub fn rng_seed(&self) -> Option<u64> {
        self.rng_seed
    }

    pub fn fitness_cache_capacity(&self) -> usize {
        self.fitness_cache_capacity
    }

    pub fn max_evaluation_smarts_complexity(&self) -> usize {
        self.max_evaluation_smarts_complexity
    }

    pub fn max_evaluation_smarts_len(&self) -> Option<usize> {
        self.max_evaluation_smarts_len
    }

    pub fn slow_evaluation_log_threshold(&self) -> Option<Duration> {
        self.slow_evaluation_log_threshold
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

    pub fn rng_seed(mut self, rng_seed: u64) -> Self {
        self.rng_seed = Some(rng_seed);
        self
    }

    pub fn fitness_cache_capacity(mut self, fitness_cache_capacity: usize) -> Self {
        self.fitness_cache_capacity = fitness_cache_capacity;
        self
    }

    pub fn max_evaluation_smarts_complexity(mut self, max_complexity: usize) -> Self {
        self.max_evaluation_smarts_complexity = max_complexity;
        self
    }

    pub fn max_evaluation_smarts_len(mut self, max_len: usize) -> Self {
        self.max_evaluation_smarts_len = Some(max_len);
        self
    }

    pub fn no_max_evaluation_smarts_len(mut self) -> Self {
        self.max_evaluation_smarts_len = None;
        self
    }

    pub fn slow_evaluation_log_threshold(mut self, threshold: Duration) -> Self {
        self.slow_evaluation_log_threshold = Some(threshold);
        self
    }

    pub fn disable_slow_evaluation_logging(mut self) -> Self {
        self.slow_evaluation_log_threshold = None;
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
            rng_seed: self.rng_seed,
            fitness_cache_capacity: self.fitness_cache_capacity,
            max_evaluation_smarts_complexity: self.max_evaluation_smarts_complexity,
            max_evaluation_smarts_len: self.max_evaluation_smarts_len,
            slow_evaluation_log_threshold: self.slow_evaluation_log_threshold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SeedCorpus, SmartsGenome};

    #[test]
    fn default_config_is_valid() {
        let config = EvolutionConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(
            config.max_evaluation_smarts_complexity(),
            DEFAULT_MAX_EVALUATION_SMARTS_COMPLEXITY
        );
        assert!(config.max_evaluation_smarts_complexity() < MAX_SMARTS_COMPLEXITY);
        assert_eq!(config.max_evaluation_smarts_len(), None);
        assert_eq!(
            config.slow_evaluation_log_threshold(),
            Some(DEFAULT_SLOW_EVALUATION_LOG_THRESHOLD)
        );
    }

    #[test]
    fn default_complexity_threshold_has_seed_headroom_and_rejects_known_pathology() {
        let config = EvolutionConfig::default();
        let default_limit = config.max_evaluation_smarts_complexity();
        let builtin_max = SeedCorpus::builtin()
            .entries()
            .iter()
            .map(SmartsGenome::complexity)
            .max()
            .unwrap();
        let pathology = SmartsGenome::from_smarts("[!r6&$([#6,#7])].[R]").unwrap();

        assert!(builtin_max * 10 < default_limit);
        assert!(pathology.complexity() > default_limit);
        assert!(pathology.complexity() < MAX_SMARTS_COMPLEXITY);
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
        let expected_max_complexity_error =
            format!("max_evaluation_smarts_complexity must not exceed {MAX_SMARTS_COMPLEXITY}");
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
            (
                EvolutionConfig::builder()
                    .max_evaluation_smarts_complexity(0)
                    .build(),
                "max_evaluation_smarts_complexity must be greater than zero",
            ),
            (
                EvolutionConfig::builder()
                    .max_evaluation_smarts_complexity(MAX_SMARTS_COMPLEXITY + 1)
                    .build(),
                expected_max_complexity_error.as_str(),
            ),
            (
                EvolutionConfig::builder()
                    .max_evaluation_smarts_len(0)
                    .build(),
                "max_evaluation_smarts_len must be greater than zero",
            ),
            (
                EvolutionConfig::builder()
                    .slow_evaluation_log_threshold(Duration::ZERO)
                    .build(),
                "slow_evaluation_log_threshold must be greater than zero",
            ),
        ];

        for (config, expected) in invalid_cases {
            assert_eq!(config.unwrap_err(), expected);
        }
    }

    #[test]
    fn builder_round_trips_seed_and_cache_capacity() {
        let config = EvolutionConfig::builder()
            .rng_seed(7)
            .fitness_cache_capacity(1234)
            .max_evaluation_smarts_complexity(128)
            .max_evaluation_smarts_len(256)
            .slow_evaluation_log_threshold(Duration::from_millis(250))
            .build()
            .unwrap();

        assert_eq!(config.rng_seed(), Some(7));
        assert_eq!(config.fitness_cache_capacity(), 1234);
        assert_eq!(config.max_evaluation_smarts_complexity(), 128);
        assert_eq!(config.max_evaluation_smarts_len(), Some(256));
        assert_eq!(
            config.slow_evaluation_log_threshold(),
            Some(Duration::from_millis(250))
        );

        let config = EvolutionConfig::builder()
            .max_evaluation_smarts_len(10)
            .no_max_evaluation_smarts_len()
            .disable_slow_evaluation_logging()
            .build()
            .unwrap();
        assert_eq!(config.max_evaluation_smarts_len(), None);
        assert_eq!(config.slow_evaluation_log_threshold(), None);
    }
}
