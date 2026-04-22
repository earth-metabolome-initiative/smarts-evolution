use serde::{Deserialize, Serialize};

const DEFAULT_POPULATION_SIZE: usize = 200;
const DEFAULT_GENERATION_LIMIT: u64 = 500;
const DEFAULT_MUTATION_RATE: f64 = 0.85;
const DEFAULT_CROSSOVER_RATE: f64 = 0.7;
const DEFAULT_SELECTION_RATIO: f64 = 0.5;
const DEFAULT_TOURNAMENT_SIZE: usize = 3;
const DEFAULT_ELITE_COUNT: usize = 4;
const DEFAULT_RANDOM_IMMIGRANT_RATIO: f64 = 0.10;
const DEFAULT_STAGNATION_LIMIT: u64 = 50;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvolutionConfigInput {
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

impl Default for EvolutionConfigInput {
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

impl EvolutionConfigInput {
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

    pub fn population_size_mut(mut self, population_size: usize) -> Self {
        self.population_size = population_size;
        self
    }

    pub fn generation_limit_mut(mut self, generation_limit: u64) -> Self {
        self.generation_limit = generation_limit;
        self
    }

    pub fn mutation_rate_mut(mut self, mutation_rate: f64) -> Self {
        self.mutation_rate = mutation_rate;
        self
    }

    pub fn crossover_rate_mut(mut self, crossover_rate: f64) -> Self {
        self.crossover_rate = crossover_rate;
        self
    }

    pub fn selection_ratio_mut(mut self, selection_ratio: f64) -> Self {
        self.selection_ratio = selection_ratio;
        self
    }

    pub fn tournament_size_mut(mut self, tournament_size: usize) -> Self {
        self.tournament_size = tournament_size;
        self
    }

    pub fn elite_count_mut(mut self, elite_count: usize) -> Self {
        self.elite_count = elite_count;
        self
    }

    pub fn random_immigrant_ratio_mut(mut self, random_immigrant_ratio: f64) -> Self {
        self.random_immigrant_ratio = random_immigrant_ratio;
        self
    }

    pub fn stagnation_limit_mut(mut self, stagnation_limit: u64) -> Self {
        self.stagnation_limit = stagnation_limit;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunRequest {
    run_id: u64,
    positive_smiles: String,
    negative_smiles: String,
    seed_smarts: String,
    config: EvolutionConfigInput,
    top_k: usize,
}

impl RunRequest {
    pub fn new(
        run_id: u64,
        positive_smiles: impl Into<String>,
        negative_smiles: impl Into<String>,
        seed_smarts: impl Into<String>,
        config: EvolutionConfigInput,
        top_k: usize,
    ) -> Self {
        Self {
            run_id,
            positive_smiles: positive_smiles.into(),
            negative_smiles: negative_smiles.into(),
            seed_smarts: seed_smarts.into(),
            config,
            top_k: top_k.max(1),
        }
    }

    pub fn run_id(&self) -> u64 {
        self.run_id
    }

    pub fn positive_smiles(&self) -> &str {
        &self.positive_smiles
    }

    pub fn negative_smiles(&self) -> &str {
        &self.negative_smiles
    }

    pub fn seed_smarts(&self) -> &str {
        &self.seed_smarts
    }

    pub fn config(&self) -> &EvolutionConfigInput {
        &self.config
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankedCandidate {
    smarts: String,
    mcc: f64,
    complexity: usize,
}

impl RankedCandidate {
    pub fn new(smarts: impl Into<String>, mcc: f64, complexity: usize) -> Self {
        Self {
            smarts: smarts.into(),
            mcc,
            complexity,
        }
    }

    pub fn smarts(&self) -> &str {
        &self.smarts
    }

    pub fn mcc(&self) -> f64 {
        self.mcc
    }

    pub fn complexity(&self) -> usize {
        self.complexity
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StartupUpdate {
    run_id: u64,
    label: String,
    completed: usize,
    total: usize,
}

impl StartupUpdate {
    pub fn new(run_id: u64, label: impl Into<String>, completed: usize, total: usize) -> Self {
        Self {
            run_id,
            label: label.into(),
            completed,
            total: total.max(1),
        }
    }

    pub fn run_id(&self) -> u64 {
        self.run_id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn completed(&self) -> usize {
        self.completed
    }

    pub fn total(&self) -> usize {
        self.total
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RunStatus {
    Running,
    Stagnated,
    Completed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProgressUpdate {
    run_id: u64,
    generation: u64,
    generation_limit: u64,
    status: RunStatus,
    best: RankedCandidate,
    leaders: Vec<RankedCandidate>,
    unique_count: usize,
    total_count: usize,
    duplicate_count: usize,
    cache_hits: usize,
    lead_complexity: usize,
    average_complexity: f64,
    stagnation: u64,
}

impl ProgressUpdate {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run_id: u64,
        generation: u64,
        generation_limit: u64,
        status: RunStatus,
        best: RankedCandidate,
        leaders: Vec<RankedCandidate>,
        unique_count: usize,
        total_count: usize,
        duplicate_count: usize,
        cache_hits: usize,
        lead_complexity: usize,
        average_complexity: f64,
        stagnation: u64,
    ) -> Self {
        Self {
            run_id,
            generation,
            generation_limit,
            status,
            best,
            leaders,
            unique_count,
            total_count,
            duplicate_count,
            cache_hits,
            lead_complexity,
            average_complexity,
            stagnation,
        }
    }

    pub fn run_id(&self) -> u64 {
        self.run_id
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn generation_limit(&self) -> u64 {
        self.generation_limit
    }

    pub fn status(&self) -> RunStatus {
        self.status
    }

    pub fn best(&self) -> &RankedCandidate {
        &self.best
    }

    pub fn leaders(&self) -> &[RankedCandidate] {
        &self.leaders
    }

    pub fn unique_count(&self) -> usize {
        self.unique_count
    }

    pub fn total_count(&self) -> usize {
        self.total_count
    }

    pub fn duplicate_count(&self) -> usize {
        self.duplicate_count
    }

    pub fn cache_hits(&self) -> usize {
        self.cache_hits
    }

    pub fn lead_complexity(&self) -> usize {
        self.lead_complexity
    }

    pub fn average_complexity(&self) -> f64 {
        self.average_complexity
    }

    pub fn stagnation(&self) -> u64 {
        self.stagnation
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompletedRun {
    run_id: u64,
    best: RankedCandidate,
    leaders: Vec<RankedCandidate>,
    generations: u64,
}

impl CompletedRun {
    pub fn new(
        run_id: u64,
        best: RankedCandidate,
        leaders: Vec<RankedCandidate>,
        generations: u64,
    ) -> Self {
        Self {
            run_id,
            best,
            leaders,
            generations,
        }
    }

    pub fn run_id(&self) -> u64 {
        self.run_id
    }

    pub fn best(&self) -> &RankedCandidate {
        &self.best
    }

    pub fn leaders(&self) -> &[RankedCandidate] {
        &self.leaders
    }

    pub fn generations(&self) -> u64 {
        self.generations
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FatalResponse {
    run_id: u64,
    message: String,
}

impl FatalResponse {
    pub fn new(run_id: u64, message: impl Into<String>) -> Self {
        Self {
            run_id,
            message: message.into(),
        }
    }

    pub fn run_id(&self) -> u64 {
        self.run_id
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum WorkerRequest {
    Run(RunRequest),
    Stop { run_id: u64 },
    Resume { run_id: u64 },
}

impl WorkerRequest {
    pub fn run_id(&self) -> u64 {
        match self {
            Self::Run(request) => request.run_id(),
            Self::Stop { run_id } | Self::Resume { run_id } => *run_id,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum WorkerResponse {
    Ready,
    Startup(StartupUpdate),
    Progress(ProgressUpdate),
    Complete(CompletedRun),
    Fatal(FatalResponse),
}

impl WorkerResponse {
    pub fn run_id(&self) -> u64 {
        match self {
            Self::Ready => 0,
            Self::Startup(startup) => startup.run_id(),
            Self::Progress(progress) => progress.run_id(),
            Self::Complete(result) => result.run_id(),
            Self::Fatal(error) => error.run_id(),
        }
    }
}
