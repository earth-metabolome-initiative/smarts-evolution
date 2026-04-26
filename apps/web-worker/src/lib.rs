//! Dedicated worker for browser-side SMARTS evolution runs.
#![cfg_attr(test, allow(clippy::unwrap_used))]

use std::cell::RefCell;
use std::str::FromStr;
use std::time::Duration;

#[cfg(target_arch = "wasm32")]
use js_sys::Date;
use js_sys::global;
use smarts_evolution::{
    EvolutionConfig, EvolutionEvaluationProgress, EvolutionProgress, EvolutionSession,
    EvolutionStatus, EvolutionTask, FoldData, FoldSample, RankedSmarts, SeedCorpus, TaskResult,
};
use smarts_evolution_web_protocol::{
    CompletedRun, EvaluationUpdate, EvolutionConfigInput, FatalResponse, ProgressUpdate,
    RankedCandidate, RunRequest, RunStatus, StartupUpdate, WorkerRequest, WorkerResponse,
};
use smarts_rs::PreparedTarget;
use smiles_parser::Smiles;
use wasm_bindgen::{JsCast, JsValue, closure::Closure, prelude::wasm_bindgen};
use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};

const STARTUP_PROGRESS_BATCH: usize = 16;
const WORKER_MATCH_TIME_LIMIT: Duration = Duration::from_secs(1);

thread_local! {
    static WORKER_STATE: RefCell<WorkerState> = RefCell::new(WorkerState::default());
}

#[derive(Default)]
struct WorkerState {
    active_run: Option<ActiveRun>,
}

struct ActiveRun {
    run_id: u64,
    session: EvolutionSession,
    paused: bool,
    scheduled: bool,
}

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
                clear_active_run();
                if let Err(message) = start_run(request.clone()) {
                    let _ = post_response(&WorkerResponse::Fatal(FatalResponse::new(
                        request.run_id(),
                        message,
                    )));
                }
            }
            WorkerRequest::Stop { run_id } => pause_active_run(run_id),
            WorkerRequest::Resume { run_id } => {
                if let Err(message) = resume_active_run(run_id) {
                    let _ = post_response(&WorkerResponse::Fatal(FatalResponse::new(
                        run_id, message,
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

fn start_run(request: RunRequest) -> Result<(), String> {
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
    let session =
        EvolutionSession::new(&task, &config, &seed_corpus, leaderboard_size)
            .map_err(|error| error.to_string())?;

    WORKER_STATE.with(|state| {
        state.borrow_mut().active_run = Some(ActiveRun {
            run_id,
            session,
            paused: false,
            scheduled: false,
        });
    });
    schedule_active_run()
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
        .pubchem_compatible_smarts(input.pubchem_compatible_smarts())
        .match_time_limit(WORKER_MATCH_TIME_LIMIT)
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
        let parsed = parse_worker_smiles(smiles, label, line_idx + 1)?;
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

fn parse_worker_smiles(smiles: &str, label: &str, line_number: usize) -> Result<Smiles, String> {
    Smiles::from_str(smiles)
        .map(|parsed| parsed.canonicalize())
        .map_err(|error| format!("invalid {label} SMILES at line {line_number}: {error}"))
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
        snapshot.match_timeout_count(),
        snapshot.lead_smarts_len(),
        snapshot.average_smarts_len(),
        snapshot.stagnation(),
    )
}

fn evaluation_update_from_snapshot(
    run_id: u64,
    snapshot: &EvolutionEvaluationProgress,
) -> EvaluationUpdate {
    EvaluationUpdate::new(
        run_id,
        snapshot.generation(),
        snapshot.generation_limit(),
        snapshot.completed(),
        snapshot.total(),
    )
}

fn completed_run_from_result(run_id: u64, result: &TaskResult) -> CompletedRun {
    CompletedRun::new(
        run_id,
        RankedCandidate::new(
            result.best_smarts(),
            result.best_mcc(),
            result.best_smarts_len(),
            result.best_coverage_score(),
        ),
        result.leaders().iter().map(ranked_candidate).collect(),
        result.generations(),
    )
}

fn clear_active_run() {
    WORKER_STATE.with(|state| {
        state.borrow_mut().active_run = None;
    });
}

fn pause_active_run(run_id: u64) {
    WORKER_STATE.with(|state| {
        let mut state = state.borrow_mut();
        if let Some(active_run) = state.active_run.as_mut()
            && active_run.run_id == run_id
        {
            active_run.paused = true;
        }
    });
}

fn resume_active_run(run_id: u64) -> Result<(), String> {
    let should_schedule = WORKER_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(active_run) = state.active_run.as_mut() else {
            return false;
        };
        if active_run.run_id != run_id {
            return false;
        }
        active_run.paused = false;
        !active_run.scheduled
    });

    if should_schedule {
        schedule_active_run()?;
    }
    Ok(())
}

fn schedule_active_run() -> Result<(), String> {
    let should_schedule = WORKER_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(active_run) = state.active_run.as_mut() else {
            return false;
        };
        if active_run.paused || active_run.scheduled {
            return false;
        }
        active_run.scheduled = true;
        true
    });

    if !should_schedule {
        return Ok(());
    }

    let callback = Closure::once_into_js(drive_active_run);
    worker_scope()
        .set_timeout_with_callback_and_timeout_and_arguments_0(callback.unchecked_ref(), 0)
        .map_err(js_error)?;
    Ok(())
}

fn drive_active_run() {
    enum StepOutcome {
        Idle,
        Advanced {
            progress: Box<ProgressUpdate>,
            result: Option<CompletedRun>,
        },
        Fatal(FatalResponse),
    }

    let outcome = WORKER_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(mut active_run) = state.active_run.take() else {
            return StepOutcome::Idle;
        };
        active_run.scheduled = false;

        if active_run.paused {
            state.active_run = Some(active_run);
            return StepOutcome::Idle;
        }

        let run_id = active_run.run_id;
        let on_evaluation_progress = |progress: EvolutionEvaluationProgress| {
            if should_report_evaluation_progress(progress.completed(), progress.total()) {
                let _ = post_response(&WorkerResponse::Evaluation(
                    evaluation_update_from_snapshot(run_id, &progress),
                ));
            }
        };
        let progress = {
            #[cfg(target_arch = "wasm32")]
            {
                active_run.session.step_with_evaluation_progress_and_clock(
                    &worker_now_ms,
                    on_evaluation_progress,
                )
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                active_run
                    .session
                    .step_with_evaluation_progress(on_evaluation_progress)
            }
        };
        let Some(progress) = progress else {
            return StepOutcome::Fatal(FatalResponse::new(
                active_run.run_id,
                "paused evolution session ended unexpectedly",
            ));
        };

        let progress = progress_update_from_snapshot(active_run.run_id, &progress);
        let result = active_run
            .session
            .take_result()
            .map(|result| completed_run_from_result(active_run.run_id, &result));
        if result.is_none() {
            state.active_run = Some(active_run);
        }
        StepOutcome::Advanced {
            progress: Box::new(progress),
            result,
        }
    });

    match outcome {
        StepOutcome::Idle => {}
        StepOutcome::Advanced { progress, result } => {
            let _ = post_response(&WorkerResponse::Progress(*progress));
            if let Some(result) = result {
                let _ = post_response(&WorkerResponse::Complete(result));
            } else {
                let _ = schedule_active_run();
            }
        }
        StepOutcome::Fatal(error) => {
            let _ = post_response(&WorkerResponse::Fatal(error));
        }
    }
}

fn ranked_candidate(candidate: &RankedSmarts) -> RankedCandidate {
    RankedCandidate::new(
        candidate.smarts(),
        candidate.mcc(),
        candidate.smarts_len(),
        candidate.coverage_score(),
    )
}

const fn run_status(status: EvolutionStatus) -> RunStatus {
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

#[cfg(target_arch = "wasm32")]
fn worker_now_ms() -> f64 {
    worker_scope()
        .performance()
        .map_or_else(Date::now, |performance| performance.now())
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

const fn should_report_progress(completed: usize, total: usize) -> bool {
    completed == total || completed <= 4 || completed.is_multiple_of(STARTUP_PROGRESS_BATCH)
}

const fn should_report_evaluation_progress(completed: usize, total: usize) -> bool {
    completed == 0 || completed == total || completed <= 4 || completed.is_multiple_of(8)
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
    use std::time::Duration;

    use super::{
        build_config, count_seed_units, parse_worker_smiles, should_report_progress, startup_total,
    };
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

    #[test]
    fn worker_uses_short_match_time_limit() {
        let config = build_config(&EvolutionConfigInput::default()).unwrap();

        assert_eq!(config.match_time_limit(), Some(Duration::from_secs(1)));
    }

    #[test]
    fn worker_canonicalizes_smiles_lines() {
        let parsed = parse_worker_smiles("OC", "positive", 1).unwrap();
        assert!(parsed.is_canonical());
        assert_eq!(parsed.to_string(), "CO");
    }
}
