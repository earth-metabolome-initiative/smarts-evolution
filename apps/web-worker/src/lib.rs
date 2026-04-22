//! Dedicated worker for browser-side SMARTS evolution runs.

use std::str::FromStr;

use js_sys::global;
use smarts_evolution::{
    EvolutionConfig, EvolutionProgress, EvolutionStatus, EvolutionTask, FoldData, FoldSample,
    RankedSmarts, SeedCorpus, TaskResult, evolve_task_with_progress,
};
use smarts_evolution_web_protocol::{
    CompletedRun, EvolutionConfigInput, FatalResponse, ProgressUpdate, RankedCandidate, RunRequest,
    RunStatus, StartupUpdate, WorkerRequest, WorkerResponse,
};
use smarts_validator::PreparedTarget;
use smiles_parser::Smiles;
use wasm_bindgen::{JsCast, JsValue, closure::Closure, prelude::wasm_bindgen};
use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};

const STARTUP_PROGRESS_BATCH: usize = 16;

/// Starts the dedicated evolution worker.
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    let scope = worker_scope();
    let onmessage = Closure::wrap(Box::new(move |event: MessageEvent| {
        let request = match serde_wasm_bindgen::from_value::<WorkerRequest>(event.data()) {
            Ok(request) => request,
            Err(error) => {
                let _ = post_response(&WorkerResponse::Fatal(FatalResponse::new(
                    0,
                    format!("invalid worker request: {error}"),
                )));
                return;
            }
        };

        match request {
            WorkerRequest::Run(request) => {
                if let Err(message) = run_request(request.clone()) {
                    let _ = post_response(&WorkerResponse::Fatal(FatalResponse::new(
                        request.run_id(),
                        message,
                    )));
                }
            }
        }
    }) as Box<dyn FnMut(MessageEvent)>);

    scope.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
    post_response(&WorkerResponse::Ready)?;
    onmessage.forget();
    Ok(())
}

fn run_request(request: RunRequest) -> Result<(), String> {
    let run_id = request.run_id();
    let mut startup = StartupReporter::new(run_id, startup_total(&request));
    startup.post_initial("Queued run")?;
    startup.advance(1, "Validating configuration")?;
    let config = build_config(request.config())?;
    let seed_corpus = build_seed_corpus(request.seed_smarts(), &mut startup)?;
    let task = build_task(&request, &mut startup)?;
    startup.advance(1, "Preparing evaluation fold")?;
    startup.advance(1, "Starting genetic search")?;
    let leaderboard_size = request.top_k().max(1);

    let result =
        evolve_task_with_progress(&task, &config, &seed_corpus, leaderboard_size, |snapshot| {
            let response =
                WorkerResponse::Progress(progress_update_from_snapshot(run_id, &snapshot));
            let _ = post_response(&response);
        })
        .map_err(|error| error.to_string())?;

    post_response(&WorkerResponse::Complete(completed_run_from_result(
        run_id, &result,
    )))
    .map_err(js_error)
}

fn build_config(input: &EvolutionConfigInput) -> Result<EvolutionConfig, String> {
    EvolutionConfig::builder()
        .population_size(input.population_size())
        .generation_limit(input.generation_limit())
        .mutation_rate(input.mutation_rate())
        .crossover_rate(input.crossover_rate())
        .selection_ratio(input.selection_ratio())
        .tournament_size(input.tournament_size())
        .elite_count(input.elite_count())
        .random_immigrant_ratio(input.random_immigrant_ratio())
        .stagnation_limit(input.stagnation_limit())
        .build()
}

fn build_task(
    request: &RunRequest,
    startup: &mut StartupReporter,
) -> Result<EvolutionTask, String> {
    let positives = parse_samples(request.positive_smiles(), true, "positive", startup)?;
    let negatives = parse_samples(request.negative_smiles(), false, "negative", startup)?;
    if positives.is_empty() {
        return Err("at least one positive SMILES is required".to_string());
    }
    if negatives.is_empty() {
        return Err("at least one negative SMILES is required".to_string());
    }

    let mut samples = Vec::with_capacity(positives.len() + negatives.len());
    samples.extend(positives);
    samples.extend(negatives);
    Ok(EvolutionTask::new(
        format!("web-run-{}", request.run_id()),
        vec![FoldData::new(samples)],
    ))
}

fn parse_samples(
    input: &str,
    is_positive: bool,
    label: &str,
    startup: &mut StartupReporter,
) -> Result<Vec<FoldSample>, String> {
    let total = count_nonempty_lines(input);
    let mut samples = Vec::with_capacity(total);
    let mut seen = 0usize;
    let mut reported = 0usize;
    for (line_idx, line) in input.lines().enumerate() {
        let smiles = line.trim();
        if smiles.is_empty() {
            continue;
        }

        seen += 1;
        let parsed = Smiles::from_str(smiles)
            .map_err(|error| format!("invalid {label} SMILES at line {}: {error}", line_idx + 1))?;
        let target = PreparedTarget::new(parsed);
        samples.push(if is_positive {
            FoldSample::positive(target)
        } else {
            FoldSample::negative(target)
        });

        if should_report_progress(seen, total) {
            startup.advance(
                seen.saturating_sub(reported),
                format!("Parsing {label} SMILES {seen}/{total}"),
            )?;
            reported = seen;
        }
    }
    Ok(samples)
}

fn build_seed_corpus(input: &str, startup: &mut StartupReporter) -> Result<SeedCorpus, String> {
    if input.trim().is_empty() {
        startup.advance(1, "Loading built-in seed corpus")?;
        return Ok(SeedCorpus::builtin());
    }

    let mut corpus = SeedCorpus::default();
    let total = count_nonempty_noncomment_lines(input);
    let mut seen = 0usize;
    let mut reported = 0usize;
    for (line_idx, line) in input.lines().enumerate() {
        let smarts = line.trim();
        if smarts.is_empty() || smarts.starts_with('#') {
            continue;
        }
        seen += 1;
        corpus
            .insert_smarts(smarts)
            .map_err(|error| format!("invalid SMARTS seed at line {}: {error}", line_idx + 1))?;
        if should_report_progress(seen, total) {
            startup.advance(
                seen.saturating_sub(reported),
                format!("Validating seed SMARTS {seen}/{total}"),
            )?;
            reported = seen;
        }
    }
    if corpus.is_empty() {
        return Err("seed corpus is empty after filtering blank lines".to_string());
    }
    Ok(corpus)
}

fn progress_update_from_snapshot(run_id: u64, snapshot: &EvolutionProgress) -> ProgressUpdate {
    ProgressUpdate::new(
        run_id,
        snapshot.generation(),
        snapshot.generation_limit(),
        run_status(snapshot.status()),
        ranked_candidate(snapshot.best()),
        snapshot.leaders().iter().map(ranked_candidate).collect(),
        snapshot.unique_count(),
        snapshot.total_count(),
        snapshot.duplicate_count(),
        snapshot.cache_hits(),
        snapshot.lead_complexity(),
        snapshot.average_complexity(),
        snapshot.stagnation(),
    )
}

fn completed_run_from_result(run_id: u64, result: &TaskResult) -> CompletedRun {
    CompletedRun::new(
        run_id,
        RankedCandidate::new(
            result.best_smarts(),
            result.best_mcc(),
            result
                .leaders()
                .first()
                .map(RankedSmarts::complexity)
                .unwrap_or(0),
        ),
        result.leaders().iter().map(ranked_candidate).collect(),
        result.generations(),
    )
}

fn ranked_candidate(candidate: &RankedSmarts) -> RankedCandidate {
    RankedCandidate::new(candidate.smarts(), candidate.mcc(), candidate.complexity())
}

fn run_status(status: EvolutionStatus) -> RunStatus {
    match status {
        EvolutionStatus::Running => RunStatus::Running,
        EvolutionStatus::Stagnated => RunStatus::Stagnated,
        EvolutionStatus::Completed => RunStatus::Completed,
    }
}

fn post_response(response: &WorkerResponse) -> Result<(), JsValue> {
    let payload = serde_wasm_bindgen::to_value(response)
        .map_err(|error| JsValue::from_str(&format!("invalid worker response: {error}")))?;
    worker_scope().post_message(&payload)
}

fn worker_scope() -> DedicatedWorkerGlobalScope {
    global().unchecked_into::<DedicatedWorkerGlobalScope>()
}

fn js_error(error: JsValue) -> String {
    error
        .as_string()
        .unwrap_or_else(|| "unknown JavaScript worker error".to_string())
}

fn startup_total(request: &RunRequest) -> usize {
    1 + count_nonempty_lines(request.positive_smiles())
        + count_nonempty_lines(request.negative_smiles())
        + count_seed_units(request.seed_smarts())
        + 2
}

fn count_nonempty_lines(input: &str) -> usize {
    input.lines().filter(|line| !line.trim().is_empty()).count()
}

fn count_nonempty_noncomment_lines(input: &str) -> usize {
    input
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !trimmed.starts_with('#')
        })
        .count()
}

fn count_seed_units(input: &str) -> usize {
    let count = count_nonempty_noncomment_lines(input);
    count.max(1)
}

fn should_report_progress(completed: usize, total: usize) -> bool {
    completed == total || completed <= 4 || completed.is_multiple_of(STARTUP_PROGRESS_BATCH)
}

struct StartupReporter {
    run_id: u64,
    total: usize,
    completed: usize,
}

impl StartupReporter {
    fn new(run_id: u64, total: usize) -> Self {
        Self {
            run_id,
            total: total.max(1),
            completed: 0,
        }
    }

    fn post_initial(&self, label: impl Into<String>) -> Result<(), String> {
        post_response(&WorkerResponse::Startup(StartupUpdate::new(
            self.run_id,
            label,
            self.completed,
            self.total,
        )))
        .map_err(js_error)
    }

    fn advance(&mut self, amount: usize, label: impl Into<String>) -> Result<(), String> {
        self.completed = (self.completed + amount).min(self.total);
        post_response(&WorkerResponse::Startup(StartupUpdate::new(
            self.run_id,
            label,
            self.completed,
            self.total,
        )))
        .map_err(js_error)
    }
}

#[cfg(test)]
mod tests {
    use super::{count_seed_units, should_report_progress, startup_total};
    use smarts_evolution_web_protocol::{EvolutionConfigInput, RunRequest};

    #[test]
    fn progress_reporting_is_dense_at_start_then_sparse() {
        assert!(should_report_progress(1, 40));
        assert!(should_report_progress(4, 40));
        assert!(!should_report_progress(5, 40));
        assert!(should_report_progress(16, 40));
        assert!(should_report_progress(40, 40));
    }

    #[test]
    fn seed_units_track_real_seed_entries() {
        assert_eq!(count_seed_units(""), 1);
        assert_eq!(count_seed_units("\n# comment\n"), 1);
        assert_eq!(count_seed_units("[#6]\n[#7]\n"), 2);
    }

    #[test]
    fn startup_total_uses_real_dataset_counts() {
        let request = RunRequest::new(
            7,
            "CCO\nN\n",
            "O\nC=C\n",
            "",
            EvolutionConfigInput::default(),
            10,
        );

        assert_eq!(startup_total(&request), 8);
    }
}
