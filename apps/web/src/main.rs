mod icons;

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::str::FromStr;

use dioxus::prelude::*;
use icons::{AppIcon, app_icon};
use smiles_parser::Smiles;
use smarts_evolution_web_protocol::{
    CompletedRun, EvaluationUpdate, EvolutionConfigInput, ProgressUpdate, RankedCandidate,
    RunRequest, StartupUpdate, WorkerRequest,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, JsValue, closure::Closure};
#[cfg(target_arch = "wasm32")]
use web_sys::{
    CanvasRenderingContext2d, ErrorEvent, HtmlCanvasElement, MessageEvent, MouseEvent, Worker,
    WorkerOptions, WorkerType,
};
#[cfg(target_arch = "wasm32")]
use smarts_evolution_web_protocol::WorkerResponse;

#[cfg(target_arch = "wasm32")]
const WORKER_SCRIPT: &str = "/generated/evolution-worker.js";
const LEADERBOARD_LIMIT: usize = 100;
const PAGE_SIZE_OPTIONS: [usize; 3] = [10, 25, 50];
const PLOT_WIDTH: f64 = 560.0;
const PLOT_HEIGHT: f64 = 180.0;
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
const PLOT_LEFT_PAD: f64 = 44.0;
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
const PLOT_RIGHT_PAD: f64 = 14.0;
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
const PLOT_TOP_PAD: f64 = 16.0;
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
const PLOT_BOTTOM_PAD: f64 = 30.0;

#[derive(Clone, Copy)]
struct ExamplePreset {
    label: &'static str,
    icon: AppIcon,
    positive_smiles: &'static str,
    negative_smiles: &'static str,
}

#[derive(Clone, Copy)]
struct NumericFieldSpec {
    label: &'static str,
    id: &'static str,
    icon: AppIcon,
    note: Option<&'static str>,
}

impl ExamplePreset {
    fn positive_smiles(self) -> String {
        self.positive_smiles.trim().to_string()
    }

    fn negative_smiles(self) -> String {
        self.negative_smiles.trim().to_string()
    }
}

const EXAMPLE_PRESETS: [ExamplePreset; 5] = [
    ExamplePreset {
        label: "Amphetamines",
        icon: AppIcon::Bolt,
        positive_smiles: include_str!(
            "../examples/amphetamines_and_derivatives_positive.smiles"
        ),
        negative_smiles: include_str!(
            "../examples/amphetamines_and_derivatives_negative.smiles"
        ),
    },
    ExamplePreset {
        label: "Flavonoids",
        icon: AppIcon::Leaf,
        positive_smiles: include_str!(
            "../examples/flavonoids_positive.smiles"
        ),
        negative_smiles: include_str!(
            "../examples/flavonoids_negative.smiles"
        ),
    },
    ExamplePreset {
        label: "Fatty acids",
        icon: AppIcon::Droplet,
        positive_smiles: include_str!(
            "../examples/fatty_acids_and_conjugates_positive.smiles"
        ),
        negative_smiles: include_str!(
            "../examples/fatty_acids_and_conjugates_negative.smiles"
        ),
    },
    ExamplePreset {
        label: "Penicillins",
        icon: AppIcon::Capsules,
        positive_smiles: include_str!(
            "../examples/penicillins_positive.smiles"
        ),
        negative_smiles: include_str!(
            "../examples/penicillins_negative.smiles"
        ),
    },
    ExamplePreset {
        label: "Steroids",
        icon: AppIcon::Dumbbell,
        positive_smiles: include_str!(
            "../examples/steroids_and_steroid_derivatives_positive.smiles"
        ),
        negative_smiles: include_str!(
            "../examples/steroids_and_steroid_derivatives_negative.smiles"
        ),
    },
];

#[derive(Clone, PartialEq)]
struct RunDraft {
    positive_smiles: String,
    negative_smiles: String,
    seed_smarts: String,
    population_size: String,
    generation_limit: String,
    mutation_rate: String,
    crossover_rate: String,
    selection_ratio: String,
    tournament_size: String,
    elite_count: String,
    random_immigrant_ratio: String,
    stagnation_limit: String,
}

impl Default for RunDraft {
    fn default() -> Self {
        let defaults = EvolutionConfigInput::default();
        Self {
            positive_smiles: String::new(),
            negative_smiles: String::new(),
            seed_smarts: String::new(),
            population_size: defaults.population_size().to_string(),
            generation_limit: defaults.generation_limit().to_string(),
            mutation_rate: defaults.mutation_rate().to_string(),
            crossover_rate: defaults.crossover_rate().to_string(),
            selection_ratio: defaults.selection_ratio().to_string(),
            tournament_size: defaults.tournament_size().to_string(),
            elite_count: defaults.elite_count().to_string(),
            random_immigrant_ratio: defaults.random_immigrant_ratio().to_string(),
            stagnation_limit: defaults.stagnation_limit().to_string(),
        }
    }
}

#[derive(Clone, Default)]
struct DraftValidation {
    positive_smiles: Option<String>,
    negative_smiles: Option<String>,
    population_size: Option<String>,
    generation_limit: Option<String>,
    mutation_rate: Option<String>,
    crossover_rate: Option<String>,
    selection_ratio: Option<String>,
    tournament_size: Option<String>,
    elite_count: Option<String>,
    random_immigrant_ratio: Option<String>,
    stagnation_limit: Option<String>,
}

impl DraftValidation {
    fn has_errors(&self) -> bool {
        self.positive_smiles.is_some()
            || self.negative_smiles.is_some()
            || self.population_size.is_some()
            || self.generation_limit.is_some()
            || self.mutation_rate.is_some()
            || self.crossover_rate.is_some()
            || self.selection_ratio.is_some()
            || self.tournament_size.is_some()
            || self.elite_count.is_some()
            || self.random_immigrant_ratio.is_some()
            || self.stagnation_limit.is_some()
    }

    fn first_error(&self) -> Option<&str> {
        self.positive_smiles
            .as_deref()
            .or(self.negative_smiles.as_deref())
            .or(self.population_size.as_deref())
            .or(self.generation_limit.as_deref())
            .or(self.mutation_rate.as_deref())
            .or(self.crossover_rate.as_deref())
            .or(self.selection_ratio.as_deref())
            .or(self.tournament_size.as_deref())
            .or(self.elite_count.as_deref())
            .or(self.random_immigrant_ratio.as_deref())
            .or(self.stagnation_limit.as_deref())
    }

    fn first_config_error(&self) -> Option<&str> {
        self.population_size
            .as_deref()
            .or(self.generation_limit.as_deref())
            .or(self.mutation_rate.as_deref())
            .or(self.crossover_rate.as_deref())
            .or(self.selection_ratio.as_deref())
            .or(self.tournament_size.as_deref())
            .or(self.elite_count.as_deref())
            .or(self.random_immigrant_ratio.as_deref())
            .or(self.stagnation_limit.as_deref())
    }
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RunPhase {
    Idle,
    Running,
    Completed,
    Stopped,
    Failed,
}

#[derive(Clone, PartialEq)]
struct ProgressPoint {
    generation: u64,
    best_mcc: f64,
    best: RankedCandidate,
    stagnation: u64,
    leaders: Vec<RankedCandidate>,
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, PartialEq)]
struct ScatterPlotPoint {
    x: f64,
    y: f64,
    generation: u64,
    plot_width: f64,
    candidate: RankedCandidate,
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, PartialEq)]
struct ScatterPlotData {
    points: Vec<ScatterPlotPoint>,
    x_ticks: Vec<(f64, String)>,
    y_ticks: Vec<(f64, String)>,
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, PartialEq)]
struct ScatterTooltip {
    x: f64,
    y: f64,
    generation: u64,
    plot_width: f64,
    candidate: RankedCandidate,
}

#[derive(Clone, PartialEq)]
struct RunView {
    phase: RunPhase,
    startup: Option<StartupUpdate>,
    evaluation: Option<EvaluationUpdate>,
    progress: Option<ProgressUpdate>,
    result: Option<CompletedRun>,
    history: Vec<ProgressPoint>,
    message: Option<String>,
}

impl Default for RunView {
    fn default() -> Self {
        Self {
            phase: RunPhase::Idle,
            startup: None,
            evaluation: None,
            progress: None,
            result: None,
            history: Vec::new(),
            message: None,
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
impl RunView {
    fn starting(run_id: u64) -> Self {
        Self {
            phase: RunPhase::Running,
            startup: Some(StartupUpdate::new(run_id, "Launching worker", 0, 1)),
            evaluation: None,
            progress: None,
            result: None,
            history: Vec::new(),
            message: Some("Evolution run started.".to_string()),
        }
    }

    fn apply_startup(&mut self, startup: StartupUpdate) {
        if self.phase != RunPhase::Stopped {
            self.phase = RunPhase::Running;
            self.message = None;
        }
        self.startup = Some(startup);
        self.evaluation = None;
        self.progress = None;
        self.result = None;
    }

    fn apply_evaluation(&mut self, evaluation: EvaluationUpdate) {
        if self.phase != RunPhase::Stopped {
            self.phase = RunPhase::Running;
            self.message = None;
        }
        self.startup = None;
        self.evaluation = Some(evaluation);
        self.result = None;
    }

    fn apply_progress(&mut self, progress: ProgressUpdate) {
        let was_stopped = self.phase == RunPhase::Stopped;
        if !was_stopped {
            self.phase = RunPhase::Running;
        }
        self.startup = None;
        self.evaluation = None;
        self.push_progress_point(&progress);
        self.progress = Some(progress);
        self.result = None;
        self.message = if was_stopped {
            Some("Evolution paused. Resume continues from the last completed generation.".to_string())
        } else {
            None
        };
    }

    fn finish(&mut self, result: CompletedRun) {
        self.phase = RunPhase::Completed;
        self.startup = None;
        self.evaluation = None;
        self.result = Some(result);
        self.message = Some("Evolution run completed.".to_string());
    }

    fn fail(&mut self, message: String) {
        self.phase = RunPhase::Failed;
        self.startup = None;
        self.evaluation = None;
        self.message = Some(message);
    }

    fn stop(&mut self) {
        self.phase = RunPhase::Stopped;
        self.message = Some(if self.progress.is_some() {
            "Pausing after the current generation.".to_string()
        } else {
            "Run paused by the user.".to_string()
        });
    }

    fn resume(&mut self) {
        self.phase = RunPhase::Running;
        self.message = Some("Evolution run resumed.".to_string());
    }

    fn push_progress_point(&mut self, progress: &ProgressUpdate) {
        let point = ProgressPoint {
            generation: progress.generation(),
            best_mcc: progress.best().mcc(),
            best: progress.best().clone(),
            stagnation: progress.stagnation(),
            leaders: progress.leaders().to_vec(),
        };
        if let Some(last) = self.history.last_mut()
            && last.generation == point.generation
        {
            *last = point;
            return;
        }
        self.history.push(point);
    }
}

#[cfg(target_arch = "wasm32")]
struct EvolutionWorker {
    worker: Worker,
    _onmessage: Closure<dyn FnMut(MessageEvent)>,
    _onerror: Closure<dyn FnMut(ErrorEvent)>,
}

#[cfg(target_arch = "wasm32")]
impl EvolutionWorker {
    fn new(
        mut run_view: Signal<RunView>,
        active_run_id: Rc<Cell<u64>>,
        pending_request: Rc<RefCell<Option<RunRequest>>>,
    ) -> Result<Self, String> {
        let options = WorkerOptions::new();
        options.set_type(WorkerType::Module);
        let worker = Worker::new_with_options(WORKER_SCRIPT, &options)
            .map_err(|error| format!("failed to start worker: {}", js_error(error)))?;

        let message_worker = worker.clone();
        let onmessage_run_id = active_run_id.clone();
        let onmessage_pending_request = pending_request.clone();
        let onmessage = Closure::wrap(Box::new(move |event: MessageEvent| {
            let response = match serde_wasm_bindgen::from_value::<WorkerResponse>(event.data()) {
                Ok(response) => response,
                Err(error) => {
                    run_view
                        .write()
                        .fail(format!("failed to decode worker response: {error}"));
                    return;
                }
            };

            if response.run_id() != 0 && response.run_id() != onmessage_run_id.get() {
                return;
            }

            match response {
                WorkerResponse::Ready => {
                    let maybe_request = onmessage_pending_request.borrow_mut().take();
                    if let Some(request) = maybe_request {
                        let payload =
                            match serde_wasm_bindgen::to_value(&WorkerRequest::Run(request)) {
                                Ok(payload) => payload,
                                Err(error) => {
                                    run_view
                                        .write()
                                        .fail(format!("failed to encode worker request: {error}"));
                                    return;
                                }
                            };
                        if let Err(error) = message_worker.post_message(&payload) {
                            run_view.write().fail(format!(
                                "failed to post worker request: {}",
                                js_error(error)
                            ));
                        }
                    }
                }
                WorkerResponse::Startup(startup) => run_view.write().apply_startup(startup),
                WorkerResponse::Evaluation(evaluation) => {
                    run_view.write().apply_evaluation(evaluation)
                }
                WorkerResponse::Progress(progress) => run_view.write().apply_progress(progress),
                WorkerResponse::Complete(result) => run_view.write().finish(result),
                WorkerResponse::Fatal(error) => run_view.write().fail(error.message().to_string()),
            }
        }) as Box<dyn FnMut(MessageEvent)>);
        worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));

        let onerror = Closure::wrap(Box::new(move |event: ErrorEvent| {
            run_view
                .write()
                .fail(format!("worker crashed: {}", event.message()));
        }) as Box<dyn FnMut(ErrorEvent)>);
        worker.set_onerror(Some(onerror.as_ref().unchecked_ref()));

        Ok(Self {
            worker,
            _onmessage: onmessage,
            _onerror: onerror,
        })
    }
    fn terminate(&self) {
        self.worker.terminate();
    }

    fn post_request(&self, request: &WorkerRequest) -> Result<(), String> {
        let payload = serde_wasm_bindgen::to_value(request)
            .map_err(|error| format!("failed to encode worker request: {error}"))?;
        self.worker
            .post_message(&payload)
            .map_err(|error| format!("failed to post worker request: {}", js_error(error)))
    }
}

#[cfg(target_arch = "wasm32")]
impl Drop for EvolutionWorker {
    fn drop(&mut self) {
        self.worker.terminate();
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct EvolutionWorker;

#[cfg(not(target_arch = "wasm32"))]
impl EvolutionWorker {
    fn new(
        _: Signal<RunView>,
        _: Rc<Cell<u64>>,
        _: Rc<RefCell<Option<RunRequest>>>,
    ) -> Result<Self, String> {
        Err("the web app must run on wasm32".to_string())
    }
    fn terminate(&self) {}
    fn post_request(&self, _: &WorkerRequest) -> Result<(), String> {
        Err("the web app must run on wasm32".to_string())
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for EvolutionWorker {
    fn drop(&mut self) {}
}

#[derive(Clone)]
struct EvolutionWorkerController {
    run_view: Signal<RunView>,
    worker_slot: Rc<RefCell<Option<EvolutionWorker>>>,
    active_run_id: Rc<Cell<u64>>,
    pending_request: Rc<RefCell<Option<RunRequest>>>,
    last_request: Rc<RefCell<Option<RunRequest>>>,
}

impl EvolutionWorkerController {
    fn next_run_id(&self) -> u64 {
        self.active_run_id.get() + 1
    }

    fn start(&self, request: RunRequest) -> Result<(), String> {
        let run_id = request.run_id();
        self.active_run_id.set(run_id);
        self.terminate();
        *self.last_request.borrow_mut() = Some(request.clone());
        *self.pending_request.borrow_mut() = Some(request);

        let worker = EvolutionWorker::new(
            self.run_view,
            self.active_run_id.clone(),
            self.pending_request.clone(),
        )?;

        let mut run_view = self.run_view;
        run_view.set(RunView::starting(run_id));
        *self.worker_slot.borrow_mut() = Some(worker);
        Ok(())
    }

    fn stop(&self) {
        let should_pause_in_place = {
            let run_view = (self.run_view)();
            run_view.phase == RunPhase::Running && run_view.progress.is_some()
        };

        if should_pause_in_place {
            let stop_result = self
                .worker_slot
                .borrow()
                .as_ref()
                .map(|worker| {
                    worker.post_request(&WorkerRequest::Stop {
                        run_id: self.active_run_id.get(),
                    })
                })
                .unwrap_or_else(|| Err("no active worker".to_string()));
            if stop_result.is_err() {
                self.terminate();
                *self.pending_request.borrow_mut() = None;
            }
        } else {
            self.terminate();
            *self.pending_request.borrow_mut() = None;
        }
        let mut run_view = self.run_view;
        run_view.write().stop();
    }

    fn resume(&self) -> Result<(), String> {
        let run_id = self.active_run_id.get();
        if let Some(worker) = self.worker_slot.borrow().as_ref() {
            worker.post_request(&WorkerRequest::Resume { run_id })?;
            let mut run_view = self.run_view;
            run_view.write().resume();
            return Ok(());
        }

        let request = self
            .last_request
            .borrow()
            .clone()
            .ok_or_else(|| "no stopped run is available to resume".to_string())?;
        self.start(request)
    }

    fn terminate(&self) {
        if let Some(worker) = self.worker_slot.borrow_mut().take() {
            worker.terminate();
        }
    }
}

fn use_evolution_worker(run_view: Signal<RunView>) -> EvolutionWorkerController {
    use_hook(move || EvolutionWorkerController {
        run_view,
        worker_slot: Rc::new(RefCell::new(None::<EvolutionWorker>)),
        active_run_id: Rc::new(Cell::new(0_u64)),
        pending_request: Rc::new(RefCell::new(None::<RunRequest>)),
        last_request: Rc::new(RefCell::new(None::<RunRequest>)),
    })
}

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    let mut draft = use_signal(RunDraft::default);
    let mut run_view = use_signal(RunView::default);
    let setup_visible = use_signal(|| true);
    let worker_controller = use_evolution_worker(run_view);
    let leaderboard_page_size = use_signal(|| PAGE_SIZE_OPTIONS[0]);
    let leaderboard_page = use_signal(|| 0usize);

    let run_view_value = run_view();
    let setup_is_visible = setup_visible();
    let draft_value = if setup_is_visible { Some(draft()) } else { None };
    let validation_cache =
        use_hook(|| RefCell::new(None::<(RunDraft, DraftValidation)>));
    let validation = if let Some(draft_value) = draft_value.as_ref() {
        let mut cache = validation_cache.borrow_mut();
        let recompute = match cache.as_ref() {
            Some((cached_draft, _)) => cached_draft != draft_value,
            None => true,
        };
        if recompute {
            *cache = Some((draft_value.clone(), validate_draft(draft_value)));
        }
        cache
            .as_ref()
            .map(|(_, validation)| validation.clone())
            .unwrap_or_default()
    } else {
        DraftValidation::default()
    };
    let leaders = current_leaders(&run_view_value);
    let best = current_best(&run_view_value);
    let last_stagnation = run_view_value.history.last().map(|point| point.stagnation);
    let page_size = leaderboard_page_size();
    let page_index = leaderboard_page();
    let page_window = leaderboard_page_window(leaders.len(), page_size, page_index);
    let visible_leaders = &leaders[page_window.start..page_window.end];
    let phase = run_view_value.phase;
    let run_message_class = if phase == RunPhase::Failed {
        "message message-error"
    } else {
        "message"
    };
    let is_running = phase == RunPhase::Running;
    let can_start = !is_running && !validation.has_errors();
    let positive_smiles_count = draft_value
        .as_ref()
        .map(|draft_value| smiles_entry_count(&draft_value.positive_smiles))
        .unwrap_or(0);
    let negative_smiles_count = draft_value
        .as_ref()
        .map(|draft_value| smiles_entry_count(&draft_value.negative_smiles))
        .unwrap_or(0);
    let draft_value = draft_value.unwrap_or_default();
    let has_run_started = run_view_value.startup.is_some()
        || run_view_value.evaluation.is_some()
        || run_view_value.progress.is_some()
        || run_view_value.result.is_some();
    let layout_class = "layout layout-results";

    let mut start_run_view = run_view;
    let mut start_draft = draft;
    let start_worker = worker_controller.clone();
    let mut hide_setup = setup_visible;
    let mut reset_leaderboard_page = leaderboard_page;

    let stop_worker = worker_controller.clone();
    let resume_worker = worker_controller.clone();
    let mut reopen_setup = setup_visible;
    let mut set_leaderboard_page_size = leaderboard_page_size;
    let mut set_leaderboard_page = leaderboard_page;

    rsx! {
        main { class: "page",
            header { class: "hero",
                div {
                    p { class: "eyebrow", "earth metabolome initiative" }
                    h1 { "SMARTS Evolution" }
                }
                p { class: "hero-copy",
                    "Generate SMARTS from positive and negative SMILES sets."
                }
            }

            section { class: "{layout_class}",
                if setup_is_visible {
                    div { class: "column",
                        section { class: "panel",
                            div { class: "panel-head",
                                div {
                                    div { class: "panel-title",
                                        {app_icon(AppIcon::ListCheck)}
                                        h2 { "Run Setup" }
                                    }
                                    p { class: "panel-copy",
                                        "Provide molecules the pattern should match, molecules it should reject, optional starting motifs, and search settings."
                                    }
                                }
                                button {
                                    class: "button button-primary",
                                    disabled: !can_start,
                                    onclick: move |_| {
                                        let next_run_id = start_worker.next_run_id();

                                        let request = match build_run_request(next_run_id, &start_draft()) {
                                            Ok(request) => request,
                                            Err(message) => {
                                                start_run_view.write().fail(message);
                                                return;
                                            }
                                        };

                                        let normalized_positive = request.positive_smiles().to_string();
                                        let normalized_negative = request.negative_smiles().to_string();
                                        {
                                            let mut next = start_draft.write();
                                            next.positive_smiles = normalized_positive;
                                            next.negative_smiles = normalized_negative;
                                        }

                                        if let Err(message) = start_worker.start(request) {
                                            start_run_view.write().fail(message);
                                            return;
                                        }

                                        hide_setup.set(false);
                                        reset_leaderboard_page.set(0);
                                    },
                                    {app_icon(AppIcon::Play)}
                                    span { "{start_button_label(phase)}" }
                                }
                            }

                            div { class: "example-presets",
                                p { class: "example-label", "Examples" }
                                div { class: "example-buttons",
                                    for preset in EXAMPLE_PRESETS {
                                        button {
                                            key: "{preset.label}",
                                            class: "button button-secondary button-example",
                                            disabled: is_running,
                                            onclick: move |_| {
                                                let mut next = draft.write();
                                                next.positive_smiles = preset.positive_smiles();
                                                next.negative_smiles = preset.negative_smiles();
                                                next.seed_smarts.clear();
                                            },
                                            {app_icon(preset.icon)}
                                            span { "{preset.label}" }
                                        }
                                    }
                                }
                            }

                            div { class: "textarea-grid triple",
                                div { class: if validation.positive_smiles.is_some() { "field field-invalid" } else { "field" },
                                    label { r#for: "positive-smiles",
                                        span { class: "field-label-row",
                                            {app_icon(AppIcon::FaceSmile)}
                                            span { "Positive SMILES" }
                                            span { class: "field-count", "{smiles_count_label(positive_smiles_count)}" }
                                        }
                                    }
                                    textarea {
                                        id: "positive-smiles",
                                        class: if validation.positive_smiles.is_some() { "input-invalid" } else { "" },
                                        disabled: is_running,
                                        value: draft_value.positive_smiles.clone(),
                                        oninput: move |event| {
                                            draft.write().positive_smiles = event.value();
                                        },
                                    }
                                    p { class: "field-note",
                                        "Examples the SMARTS should match."
                                    }
                                    if let Some(error) = validation.positive_smiles.as_ref() {
                                        p { class: "field-error", "{error}" }
                                    }
                                }

                                div { class: if validation.negative_smiles.is_some() { "field field-invalid" } else { "field" },
                                    label { r#for: "negative-smiles",
                                        span { class: "field-label-row",
                                            {app_icon(AppIcon::FaceFrown)}
                                            span { "Negative SMILES" }
                                            span { class: "field-count", "{smiles_count_label(negative_smiles_count)}" }
                                        }
                                    }
                                    textarea {
                                        id: "negative-smiles",
                                        class: if validation.negative_smiles.is_some() { "input-invalid" } else { "" },
                                        disabled: is_running,
                                        value: draft_value.negative_smiles.clone(),
                                        oninput: move |event| {
                                            draft.write().negative_smiles = event.value();
                                        },
                                    }
                                    p { class: "field-note",
                                        "Examples the SMARTS should avoid matching."
                                    }
                                    if let Some(error) = validation.negative_smiles.as_ref() {
                                        p { class: "field-error", "{error}" }
                                    }
                                }

                                div { class: "field",
                                    label { r#for: "seed-smarts",
                                        span { class: "field-label-row",
                                            {app_icon(AppIcon::Seedling)}
                                            span { "Seed SMARTS" }
                                        }
                                    }
                                    textarea {
                                        id: "seed-smarts",
                                        disabled: is_running,
                                        value: draft_value.seed_smarts.clone(),
                                        oninput: move |event| {
                                            draft.write().seed_smarts = event.value();
                                        },
                                    }
                                    p { class: "field-note",
                                        "Starting motifs. If empty, a default corpus will be used."
                                    }
                                }
                            }

                            div { class: "config-grid",
                                {numeric_field(NumericFieldSpec { label: "Population", id: "population-size", icon: AppIcon::PeopleGroup, note: Some("Population size per generation; larger populations increase search breadth but raise evaluation cost.") }, draft_value.population_size.clone(), validation.population_size.as_deref(), is_running, move |value| {
                                    draft.write().population_size = value;
                                })}
                                {numeric_field(NumericFieldSpec { label: "Generations", id: "generation-limit", icon: AppIcon::Timeline, note: Some("Upper bound on the number of evolutionary generations.") }, draft_value.generation_limit.clone(), validation.generation_limit.as_deref(), is_running, move |value| {
                                    draft.write().generation_limit = value;
                                })}
                                {numeric_field(NumericFieldSpec { label: "Mutation rate", id: "mutation-rate", icon: AppIcon::WandMagicSparkles, note: Some("Per-offspring mutation probability.") }, draft_value.mutation_rate.clone(), validation.mutation_rate.as_deref(), is_running, move |value| {
                                    draft.write().mutation_rate = value;
                                })}
                                {numeric_field(NumericFieldSpec { label: "Crossover rate", id: "crossover-rate", icon: AppIcon::CodeBranch, note: Some("Probability of crossover by subtree splicing between two parents.") }, draft_value.crossover_rate.clone(), validation.crossover_rate.as_deref(), is_running, move |value| {
                                    draft.write().crossover_rate = value;
                                })}
                                {numeric_field(NumericFieldSpec { label: "Selection ratio", id: "selection-ratio", icon: AppIcon::Filter, note: Some("Fraction of the population retained in the parent-selection pool.") }, draft_value.selection_ratio.clone(), validation.selection_ratio.as_deref(), is_running, move |value| {
                                    draft.write().selection_ratio = value;
                                })}
                                {numeric_field(NumericFieldSpec { label: "Tournament size", id: "tournament-size", icon: AppIcon::Trophy, note: Some("Number of candidates sampled per tournament during parent selection.") }, draft_value.tournament_size.clone(), validation.tournament_size.as_deref(), is_running, move |value| {
                                    draft.write().tournament_size = value;
                                })}
                                {numeric_field(NumericFieldSpec { label: "Elite count", id: "elite-count", icon: AppIcon::Crown, note: Some("Number of elite candidates preserved unchanged by elitist reinsertion.") }, draft_value.elite_count.clone(), validation.elite_count.as_deref(), is_running, move |value| {
                                    draft.write().elite_count = value;
                                })}
                                {numeric_field(NumericFieldSpec { label: "Immigrant ratio", id: "immigrant-ratio", icon: AppIcon::Shuffle, note: Some("Fraction of each generation replaced by random immigrants.") }, draft_value.random_immigrant_ratio.clone(), validation.random_immigrant_ratio.as_deref(), is_running, move |value| {
                                    draft.write().random_immigrant_ratio = value;
                                })}
                                {numeric_field(NumericFieldSpec { label: "Stagnation limit", id: "stagnation-limit", icon: AppIcon::HourglassHalf, note: Some("Terminate after this many generations without improvement in the incumbent best candidate.") }, draft_value.stagnation_limit.clone(), validation.stagnation_limit.as_deref(), is_running, move |value| {
                                    draft.write().stagnation_limit = value;
                                })}
                            }

                            if let Some(error) = validation.first_config_error() {
                                p { class: "message message-error", "{error}" }
                            }
                        }
                    }
                }

                if has_run_started {
                    div { class: "column",
                        section { class: "panel",
                            div { class: "panel-head",
                                div {
                                    div { class: "panel-title",
                                        {app_icon(AppIcon::ChartLine)}
                                        h2 { "Run Snapshot" }
                                    }
                                    p { class: "panel-copy",
                                        "Best SMARTS and search state."
                                    }
                                }
                                div { class: "controls",
                                    if is_running {
                                        button {
                                            class: "button button-danger",
                                            onclick: move |_| {
                                                stop_worker.stop();
                                            },
                                            {app_icon(AppIcon::Stop)}
                                            span { "Pause" }
                                        }
                                    } else if phase == RunPhase::Stopped {
                                        button {
                                            class: "button button-primary",
                                            onclick: move |_| {
                                                if let Err(message) = resume_worker.resume() {
                                                    run_view.write().fail(message);
                                                }
                                            },
                                            {app_icon(AppIcon::Play)}
                                            span { "Resume" }
                                        }
                                        button {
                                            class: "button button-secondary",
                                            onclick: move |_| {
                                                reopen_setup.set(true);
                                            },
                                            {app_icon(AppIcon::Sliders)}
                                            span { "Edit Setup" }
                                        }
                                    } else if !setup_is_visible {
                                        button {
                                            class: "button button-secondary",
                                            onclick: move |_| {
                                                reopen_setup.set(true);
                                            },
                                            {app_icon(AppIcon::Sliders)}
                                            span { "Edit Setup" }
                                        }
                                    }
                                }
                            }
                            if let Some(message) = run_view_value.message.as_ref() {
                                p {
                                    class: run_message_class,
                                    "{message}"
                                }
                            }

                            if let Some(progress) = run_view_value.progress.as_ref() {
                                div { class: "progress-stack",
                                    {progress_meter(
                                        "Generation progress",
                                        progress.generation() as f64,
                                        progress.generation_limit() as f64,
                                        format!("{}/{}", progress.generation(), progress.generation_limit()),
                                        "progress-fill",
                                    )}
                                    if let Some(evaluation) = run_view_value.evaluation.as_ref() {
                                        {progress_meter(
                                            format!("Scoring generation {}", evaluation.generation()),
                                            evaluation.completed() as f64,
                                            evaluation.total() as f64,
                                            format!("{}/{} SMARTS", evaluation.completed(), evaluation.total()),
                                            "progress-fill",
                                        )}
                                    }
                                }
                            } else if let Some(startup) = run_view_value.startup.as_ref() {
                                div { class: "progress-stack",
                                    {progress_meter(
                                        startup.label(),
                                        startup.completed() as f64,
                                        startup.total() as f64,
                                        format!(
                                            "{:.0}%",
                                            (startup.completed() as f64 / startup.total().max(1) as f64) * 100.0,
                                        ),
                                        "progress-fill",
                                    )}
                                }
                            } else if let Some(evaluation) = run_view_value.evaluation.as_ref() {
                                div { class: "progress-stack",
                                    {progress_meter(
                                        "Generation progress",
                                        evaluation.generation().saturating_sub(1) as f64,
                                        evaluation.generation_limit() as f64,
                                        format!(
                                            "{}/{}",
                                            evaluation.generation().saturating_sub(1),
                                            evaluation.generation_limit(),
                                        ),
                                        "progress-fill",
                                    )}
                                    {progress_meter(
                                        format!("Scoring generation {}", evaluation.generation()),
                                        evaluation.completed() as f64,
                                        evaluation.total() as f64,
                                        format!("{}/{} SMARTS", evaluation.completed(), evaluation.total()),
                                        "progress-fill",
                                    )}
                                }
                            }

                            if let Some(best) = best {
                                div { class: "stats-grid",
                                    {stat_card("Best MCC", format!("{:.3}", best.mcc()))}
                                    {stat_card("Size", best.complexity().to_string())}
                                    if let Some(stagnation) = last_stagnation {
                                        {stat_card("No improvement", stagnation.to_string())}
                                    }
                                }
                                div { class: "field",
                                    label { "Best SMARTS" }
                                    p { class: "message stat-mono", "{best.smarts()}" }
                                }
                                if !run_view_value.history.is_empty() {
                                    div {
                                        key: "plots-{run_view_value.history.len()}-{run_view_value.history.last().map(|point| point.generation).unwrap_or(0)}",
                                        MccGenerationImprovementPlot {
                                            history: run_view_value.history.clone(),
                                        }
                                    }
                                }
                            }
                        }

                        section { class: "panel leaderboard-card",
                            div { class: "leaderboard-head",
                                div {
                                    div { class: "panel-title panel-title-compact",
                                        {app_icon(AppIcon::Trophy)}
                                        h3 { "Candidates" }
                                    }
                                    p { class: "leaderboard-meta",
                                        if let Some(progress) = run_view_value.progress.as_ref() {
                                            "Generation {progress.generation()} of {progress.generation_limit()}"
                                        } else if let Some(evaluation) = run_view_value.evaluation.as_ref() {
                                            "Scoring generation {evaluation.generation()} of {evaluation.generation_limit()}"
                                        } else if let Some(result) = run_view_value.result.as_ref() {
                                            "Leaderboard after {result.generations()} generations"
                                        } else {
                                            "Waiting for generation 1"
                                        }
                                    }
                                }
                                if !leaders.is_empty() {
                                    div { class: "leaderboard-tools",
                                        label { r#for: "leaderboard-page-size", "Rows" }
                                        select {
                                            id: "leaderboard-page-size",
                                            value: "{page_window.page_size}",
                                            onchange: move |event| {
                                                if let Ok(value) = event.value().parse::<usize>() {
                                                    set_leaderboard_page_size.set(value);
                                                    set_leaderboard_page.set(0);
                                                }
                                            },
                                            for option in PAGE_SIZE_OPTIONS {
                                                option {
                                                    key: "{option}",
                                                    value: "{option}",
                                                    selected: option == page_window.page_size,
                                                    "{option}"
                                                }
                                            }
                                        }
                                        span { class: "leaderboard-page-label",
                                            "{page_window.start + 1}-{page_window.end} of {leaders.len()}"
                                        }
                                        button {
                                            class: "button button-secondary button-compact",
                                            disabled: page_window.page == 0,
                                            onclick: move |_| {
                                                if page_window.page > 0 {
                                                    set_leaderboard_page.set(page_window.page - 1);
                                                }
                                            },
                                            {app_icon(AppIcon::ChevronLeft)}
                                        }
                                        button {
                                            class: "button button-secondary button-compact",
                                            disabled: page_window.page + 1 >= page_window.page_count,
                                            onclick: move |_| {
                                                if page_window.page + 1 < page_window.page_count {
                                                    set_leaderboard_page.set(page_window.page + 1);
                                                }
                                            },
                                            {app_icon(AppIcon::ChevronRight)}
                                        }
                                    }
                                }
                            }

                            if leaders.is_empty() {
                                div { class: "empty-state",
                                    "The leaderboard will populate after the first scored generation."
                                }
                            } else {
                                table { class: "leaderboard-table",
                                    thead {
                                        tr {
                                            th { "#" }
                                            th { "SMARTS" }
                                            th { "MCC" }
                                            th { "Size" }
                                        }
                                    }
                                    tbody {
                                        for (index, candidate) in visible_leaders.iter().enumerate() {
                                            tr {
                                                key: "{candidate.smarts()}",
                                                td { "{page_window.start + index + 1}" }
                                                td { class: "smarts-cell", "{candidate.smarts()}" }
                                                td { {format!("{:.3}", candidate.mcc())} }
                                                td { "{candidate.complexity()}" }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn numeric_field(
    spec: NumericFieldSpec,
    value: String,
    error: Option<&str>,
    disabled: bool,
    mut on_input: impl FnMut(String) + 'static,
) -> Element {
    rsx! {
        div {
            class: if error.is_some() { "field field-inline field-invalid" } else { "field field-inline" },
            div { class: "field-inline-row",
                label { r#for: "{spec.id}", class: "field-inline-label",
                    span { class: "field-label-row",
                        {app_icon(spec.icon)}
                        span { "{spec.label}" }
                    }
                }
                input {
                    id: "{spec.id}",
                    r#type: "text",
                    inputmode: "decimal",
                    aria_label: "{spec.label}",
                    class: if error.is_some() { "field-inline-input input-invalid" } else { "field-inline-input" },
                    disabled: disabled,
                    value: "{value}",
                    oninput: move |event| on_input(event.value()),
                }
            }
            if let Some(note) = spec.note {
                p { class: "field-note field-note-inline", "{note}" }
            }
            if let Some(error) = error {
                p { class: "field-error", "{error}" }
            }
        }
    }
}

fn stat_card(label: impl Into<String>, value: impl Into<String>) -> Element {
    let label = label.into();
    let value = value.into();
    rsx! {
        div { class: "stat-card",
            span { class: "stat-label", "{label}" }
            span { class: "stat-value", "{value}" }
        }
    }
}

fn candidate_tooltip(tooltip: &ScatterTooltip) -> Element {
    rsx! {
        div {
            class: "plot-tooltip",
            style: "{scatter_tooltip_style(tooltip)}",
            div { class: "plot-tooltip-row",
                span { class: "plot-tooltip-label", "Generation" }
                span { class: "plot-tooltip-value", {tooltip.generation.to_string()} }
            }
            div { class: "plot-tooltip-row",
                span { class: "plot-tooltip-label", "MCC" }
                span { class: "plot-tooltip-value", {format!("{:.3}", tooltip.candidate.mcc())} }
            }
            div { class: "plot-tooltip-row",
                span { class: "plot-tooltip-label", "Size" }
                span { class: "plot-tooltip-value", {tooltip.candidate.complexity().to_string()} }
            }
            p { class: "plot-tooltip-smarts", {tooltip.candidate.smarts()} }
        }
    }
}

#[component]
fn MccGenerationImprovementPlot(history: Vec<ProgressPoint>) -> Element {
    let hover = use_signal(|| None::<ScatterTooltip>);
    let hover_value = hover();
    #[cfg(target_arch = "wasm32")]
    let move_slot = use_hook(|| Rc::new(RefCell::new(None::<Closure<dyn FnMut(MouseEvent)>>)));
    #[cfg(target_arch = "wasm32")]
    let leave_slot = use_hook(|| Rc::new(RefCell::new(None::<Closure<dyn FnMut(MouseEvent)>>)));

    #[cfg(target_arch = "wasm32")]
    {
        let history_for_effect = history.clone();
        let hover_signal = hover;
        let move_slot = move_slot.clone();
        let leave_slot = leave_slot.clone();
        use_effect(move || {
            let Some((_, width, height)) = canvas_context("mcc-generation-improvement-canvas") else {
                return;
            };
            let plot_data =
                build_mcc_generation_improvement_data_for_size(&history_for_effect, width, height);
            draw_mcc_generation_improvement_canvas("mcc-generation-improvement-canvas", &plot_data);
            bind_scatter_hover_handlers(
                "mcc-generation-improvement-canvas",
                plot_data.points.clone(),
                hover_signal,
                move_slot.clone(),
                leave_slot.clone(),
            );
        });
    }

    rsx! {
        div { class: "plot-card",
            div { class: "plot-head",
                span { class: "stat-label", "MCC improvements by generation" }
                span { class: "plot-meta", "One point per new incumbent." }
            }
            div { class: "plot-canvas-shell",
                canvas {
                    id: "mcc-generation-improvement-canvas",
                    class: "plot-canvas",
                    width: "{PLOT_WIDTH as u32}",
                    height: "{PLOT_HEIGHT as u32}",
                }
                if let Some(tooltip) = hover_value {
                    {candidate_tooltip(&tooltip)}
                }
            }
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn linear_mcc_range(history: &[ProgressPoint]) -> (f64, f64) {
    let observed_min = history
        .iter()
        .flat_map(|point| point.leaders.iter().map(RankedCandidate::mcc))
        .chain(history.iter().map(|point| point.best_mcc))
        .fold(f64::INFINITY, f64::min);
    let observed_max = history
        .iter()
        .flat_map(|point| point.leaders.iter().map(RankedCandidate::mcc))
        .chain(history.iter().map(|point| point.best_mcc))
        .fold(f64::NEG_INFINITY, f64::max);
    let span = (observed_max - observed_min).abs();
    let padding = if span < 1e-6 {
        0.05
    } else {
        (span * 0.12).max(0.01)
    };
    (
        (observed_min - padding).max(-1.0),
        (observed_max + padding).min(1.0),
    )
}

fn linear_mcc_ticks(min_mcc: f64, max_mcc: f64) -> Vec<(f64, String)> {
    (0..5)
        .map(|index| {
            let fraction = index as f64 / 4.0;
            let value = max_mcc - fraction * (max_mcc - min_mcc);
            (value, format!("{value:.3}"))
        })
        .collect()
}

fn scatter_tooltip_style(tooltip: &ScatterTooltip) -> String {
    let transform = if tooltip.x > tooltip.plot_width - 220.0 {
        "translate(calc(-100% - 12px), -50%)"
    } else {
        "translate(12px, -50%)"
    };
    format!(
        "left: {:.1}px; top: {:.1}px; transform: {};",
        tooltip.x, tooltip.y, transform
    )
}

fn compare_ranked_candidates(left: &RankedCandidate, right: &RankedCandidate) -> std::cmp::Ordering {
    right
        .mcc()
        .partial_cmp(&left.mcc())
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.smarts().len().cmp(&right.smarts().len()))
        .then_with(|| left.smarts().cmp(right.smarts()))
}

fn ranked_candidate_is_better(candidate: &RankedCandidate, incumbent: &RankedCandidate) -> bool {
    compare_ranked_candidates(candidate, incumbent).is_lt()
}

#[cfg(target_arch = "wasm32")]
fn bind_scatter_hover_handlers(
    canvas_id: &str,
    points: Vec<ScatterPlotPoint>,
    mut hover: Signal<Option<ScatterTooltip>>,
    move_slot: Rc<RefCell<Option<Closure<dyn FnMut(MouseEvent)>>>>,
    leave_slot: Rc<RefCell<Option<Closure<dyn FnMut(MouseEvent)>>>>,
) {
    let Some((_, _, _)) = canvas_context(canvas_id) else {
        return;
    };
    let document = match web_sys::window().and_then(|window| window.document()) {
        Some(document) => document,
        None => return,
    };
    let Some(canvas) = document.get_element_by_id(canvas_id) else {
        return;
    };
    let Ok(canvas) = canvas.dyn_into::<HtmlCanvasElement>() else {
        return;
    };

    let move_points = points.clone();
    let onmove = Closure::wrap(Box::new(move |event: MouseEvent| {
        let mouse_x = f64::from(event.offset_x());
        let mouse_y = f64::from(event.offset_y());
        hover.set(find_hovered_scatter_point(&move_points, mouse_x, mouse_y));
    }) as Box<dyn FnMut(MouseEvent)>);
    canvas.set_onmousemove(Some(onmove.as_ref().unchecked_ref()));
    *move_slot.borrow_mut() = Some(onmove);

    let onleave = Closure::wrap(Box::new(move |_event: MouseEvent| {
        hover.set(None);
    }) as Box<dyn FnMut(MouseEvent)>);
    canvas.set_onmouseleave(Some(onleave.as_ref().unchecked_ref()));
    *leave_slot.borrow_mut() = Some(onleave);
}

#[cfg(target_arch = "wasm32")]
fn find_hovered_scatter_point(
    points: &[ScatterPlotPoint],
    mouse_x: f64,
    mouse_y: f64,
) -> Option<ScatterTooltip> {
    let mut best: Option<(f64, &ScatterPlotPoint)> = None;
    for point in points {
        let dx = point.x - mouse_x;
        let dy = point.y - mouse_y;
        let distance_sq = dx * dx + dy * dy;
        if distance_sq <= 64.0 {
            match best {
                Some((best_distance_sq, _)) if distance_sq >= best_distance_sq => {}
                _ => best = Some((distance_sq, point)),
            }
        }
    }
    best.map(|(_, point)| ScatterTooltip {
        x: point.x,
        y: point.y,
        generation: point.generation,
        plot_width: point.plot_width,
        candidate: point.candidate.clone(),
    })
}

#[cfg(target_arch = "wasm32")]
fn draw_mcc_generation_improvement_canvas(canvas_id: &str, history: &ScatterPlotData) {
    let Some((ctx, width, height)) = canvas_context(canvas_id) else {
        return;
    };
    clear_plot(&ctx, width, height);

    if history.points.is_empty() {
        return;
    }

    draw_axes(&ctx, width, height, &history.y_ticks);
    let bottom_y = height - PLOT_BOTTOM_PAD;
    draw_x_ticks(&ctx, bottom_y, height, &history.x_ticks);

    if history.points.len() >= 2 {
        ctx.begin_path();
        ctx.set_stroke_style_str("#0f7a68");
        ctx.set_line_width(2.0);
        for (index, point) in history.points.iter().enumerate() {
            if index == 0 {
                ctx.move_to(point.x, point.y);
            } else {
                ctx.line_to(point.x, point.y);
            }
        }
        ctx.stroke();
    }

    ctx.set_fill_style_str("#0f7a68");
    for point in &history.points {
        ctx.begin_path();
        let _ = ctx.arc(point.x, point.y, 3.2, 0.0, std::f64::consts::TAU);
        ctx.fill();
    }
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn build_mcc_generation_improvement_data_for_size(
    history: &[ProgressPoint],
    width: f64,
    height: f64,
) -> ScatterPlotData {
    if history.is_empty() {
        return ScatterPlotData {
            points: Vec::new(),
            x_ticks: Vec::new(),
            y_ticks: Vec::new(),
        };
    }

    let (min_mcc, max_mcc) = linear_mcc_range(history);
    let y_ticks = linear_mcc_ticks(min_mcc, max_mcc);
    let usable_width = width - PLOT_LEFT_PAD - PLOT_RIGHT_PAD;
    let usable_height = height - PLOT_TOP_PAD - PLOT_BOTTOM_PAD;
    let min_generation = history.first().map(|point| point.generation).unwrap_or(1) as f64;
    let max_generation = history.last().map(|point| point.generation).unwrap_or(1) as f64;
    let x_for = |generation: u64| {
        if history.len() == 1 || (max_generation - min_generation).abs() < f64::EPSILON {
            PLOT_LEFT_PAD + usable_width / 2.0
        } else {
            PLOT_LEFT_PAD
                + ((generation as f64 - min_generation) / (max_generation - min_generation))
                    * usable_width
        }
    };
    let y_for = |mcc: f64| {
        PLOT_TOP_PAD + (1.0 - ((mcc - min_mcc) / (max_mcc - min_mcc).max(1e-9))) * usable_height
    };

    let x_ticks: Vec<(f64, String)> = if history.len() == 1 {
        vec![(PLOT_LEFT_PAD + usable_width / 2.0, history[0].generation.to_string())]
    } else {
        let tick_count = history.len().min(5);
        (0..tick_count)
            .map(|tick_index| {
                let history_index = tick_index * (history.len() - 1) / (tick_count - 1);
                (
                    x_for(history[history_index].generation),
                    history[history_index].generation.to_string(),
                )
            })
            .collect()
    };

    let mut incumbent: Option<&RankedCandidate> = None;
    let points = history
        .iter()
        .filter_map(|point| {
            if incumbent
                .map(|best| ranked_candidate_is_better(&point.best, best))
                .unwrap_or(true)
            {
                incumbent = Some(&point.best);
                Some(ScatterPlotPoint {
                    x: x_for(point.generation),
                    y: y_for(point.best_mcc),
                    generation: point.generation,
                    plot_width: width,
                    candidate: point.best.clone(),
                })
            } else {
                None
            }
        })
        .collect();

    ScatterPlotData {
        points,
        x_ticks,
        y_ticks,
    }
}

#[cfg(target_arch = "wasm32")]
fn canvas_context(canvas_id: &str) -> Option<(CanvasRenderingContext2d, f64, f64)> {
    let window = web_sys::window()?;
    let document = window.document()?;
    let canvas = document.get_element_by_id(canvas_id)?;
    let canvas: HtmlCanvasElement = canvas.dyn_into().ok()?;
    let css_width = f64::from(canvas.client_width().max(1));
    let css_height = f64::from(canvas.client_height().max(1));
    let device_pixel_ratio = window.device_pixel_ratio().max(1.0);
    let backing_width = (css_width * device_pixel_ratio).round().max(1.0) as u32;
    let backing_height = (css_height * device_pixel_ratio).round().max(1.0) as u32;
    if canvas.width() != backing_width {
        canvas.set_width(backing_width);
    }
    if canvas.height() != backing_height {
        canvas.set_height(backing_height);
    }
    let context = canvas
        .get_context("2d")
        .ok()??
        .dyn_into::<CanvasRenderingContext2d>()
        .ok()?;
    let _ = context.set_transform(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    let _ = context.scale(device_pixel_ratio, device_pixel_ratio);
    Some((context, css_width, css_height))
}

#[cfg(target_arch = "wasm32")]
fn clear_plot(ctx: &CanvasRenderingContext2d, width: f64, height: f64) {
    ctx.clear_rect(0.0, 0.0, width, height);
    ctx.set_fill_style_str("#fffaf4");
    ctx.fill_rect(0.0, 0.0, width, height);
}

#[cfg(target_arch = "wasm32")]
fn draw_axes(ctx: &CanvasRenderingContext2d, width: f64, height: f64, y_ticks: &[(f64, String)]) {
    let usable_height = height - PLOT_TOP_PAD - PLOT_BOTTOM_PAD;
    let min_mcc = y_ticks.last().map(|(value, _)| *value).unwrap_or(-1.0);
    let max_mcc = y_ticks.first().map(|(value, _)| *value).unwrap_or(1.0);
    let y_for = |mcc: f64| {
        PLOT_TOP_PAD + (1.0 - ((mcc - min_mcc) / (max_mcc - min_mcc).max(1e-9))) * usable_height
    };
    let bottom_y = height - PLOT_BOTTOM_PAD;

    ctx.set_stroke_style_str("rgba(19, 38, 48, 0.18)");
    ctx.set_line_width(1.0);
    for (value, _) in y_ticks {
        let y = y_for(*value);
        ctx.begin_path();
        ctx.move_to(PLOT_LEFT_PAD, y);
        ctx.line_to(width - PLOT_RIGHT_PAD, y);
        ctx.stroke();
    }

    ctx.set_stroke_style_str("rgba(19, 38, 48, 0.28)");
    ctx.begin_path();
    ctx.move_to(PLOT_LEFT_PAD, PLOT_TOP_PAD);
    ctx.line_to(PLOT_LEFT_PAD, bottom_y);
    ctx.line_to(width - PLOT_RIGHT_PAD, bottom_y);
    ctx.stroke();

    ctx.set_fill_style_str("#56656d");
    ctx.set_font("10px SFMono-Regular, Iosevka, Menlo, Consolas, monospace");
    for (value, label) in y_ticks {
        let y = y_for(*value);
        let _ = ctx.fill_text(label, 0.0, y + 4.0);
    }
}

#[cfg(target_arch = "wasm32")]
fn draw_x_ticks(
    ctx: &CanvasRenderingContext2d,
    bottom_y: f64,
    height: f64,
    x_ticks: &[(f64, String)],
) {
    ctx.set_stroke_style_str("rgba(19, 38, 48, 0.28)");
    ctx.set_fill_style_str("#56656d");
    ctx.set_font("11px SFMono-Regular, Iosevka, Menlo, Consolas, monospace");
    for (x, label) in x_ticks {
        ctx.begin_path();
        ctx.move_to(*x, bottom_y);
        ctx.line_to(*x, bottom_y + 6.0);
        ctx.stroke();
        let _ = ctx.fill_text(label, *x - 12.0, height - 8.0);
    }
}

fn progress_meter(
    label: impl Into<String>,
    current: f64,
    max: f64,
    detail: impl Into<String>,
    fill_class: &'static str,
) -> Element {
    let label = label.into();
    let detail = detail.into();
    let bounded_max = max.max(1.0);
    let percent = ((current / bounded_max) * 100.0).clamp(0.0, 100.0);
    let width_style = format!("width: {percent:.3}%;");

    rsx! {
        div { class: "progress-meter",
            div { class: "progress-head",
                span { class: "stat-label", "{label}" }
                span { class: "progress-detail", "{detail}" }
            }
            div { class: "progress-track",
                div {
                    class: "{fill_class}",
                    style: "{width_style}",
                }
            }
        }
    }
}

fn current_leaders(view: &RunView) -> &[RankedCandidate] {
    if let Some(result) = view.result.as_ref() {
        result.leaders()
    } else if let Some(progress) = view.progress.as_ref() {
        progress.leaders()
    } else {
        &[]
    }
}

fn current_best(view: &RunView) -> Option<&RankedCandidate> {
    if let Some(result) = view.result.as_ref() {
        Some(result.best())
    } else {
        view.progress.as_ref().map(ProgressUpdate::best)
    }
}

fn start_button_label(phase: RunPhase) -> &'static str {
    match phase {
        RunPhase::Idle => "Start",
        RunPhase::Completed => "Restart",
        RunPhase::Stopped => "Restart",
        RunPhase::Failed => "Retry",
        RunPhase::Running => "Start",
    }
}

fn validate_draft(draft: &RunDraft) -> DraftValidation {
    let mut validation = DraftValidation {
        positive_smiles: validate_smiles_lines("positive SMILES", &draft.positive_smiles),
        negative_smiles: validate_smiles_lines("negative SMILES", &draft.negative_smiles),
        ..DraftValidation::default()
    };

    let population_size = validate_usize_value("Population", &draft.population_size, true);
    validation.population_size = population_size.error.clone();

    let generation_limit = validate_u64_value("Generations", &draft.generation_limit, true);
    validation.generation_limit = generation_limit.error.clone();

    let mutation_rate = validate_probability_value("Mutation rate", &draft.mutation_rate, false);
    validation.mutation_rate = mutation_rate.error.clone();

    let crossover_rate =
        validate_probability_value("Crossover rate", &draft.crossover_rate, false);
    validation.crossover_rate = crossover_rate.error.clone();

    let selection_ratio =
        validate_probability_value("Selection ratio", &draft.selection_ratio, true);
    validation.selection_ratio = selection_ratio.error.clone();

    let tournament_size = validate_usize_value("Tournament size", &draft.tournament_size, true);
    validation.tournament_size = tournament_size.error.clone();

    let elite_count = validate_usize_value("Elite count", &draft.elite_count, false);
    validation.elite_count = elite_count.error.clone();

    let random_immigrant_ratio = validate_probability_value(
        "Immigrant ratio",
        &draft.random_immigrant_ratio,
        false,
    );
    validation.random_immigrant_ratio = random_immigrant_ratio.error.clone();

    let stagnation_limit = validate_u64_value("Stagnation limit", &draft.stagnation_limit, true);
    validation.stagnation_limit = stagnation_limit.error.clone();

    if validation.population_size.is_none()
        && validation.elite_count.is_none()
        && let (Some(population_size), Some(elite_count)) =
            (population_size.value, elite_count.value)
        && elite_count > population_size
    {
        validation.elite_count = Some("Elite count must not exceed population.".to_string());
    }

    validation
}

fn build_run_request(run_id: u64, draft: &RunDraft) -> Result<RunRequest, String> {
    let validation = validate_draft(draft);
    if let Some(error) = validation.first_error() {
        return Err(error.to_string());
    }

    let normalized_positive_smiles = compact_smiles_lines("positive SMILES", &draft.positive_smiles)?;
    let normalized_negative_smiles = compact_smiles_lines("negative SMILES", &draft.negative_smiles)?;

    let config = EvolutionConfigInput::default()
        .population_size_mut(parse_usize("population size", &draft.population_size)?)
        .generation_limit_mut(parse_u64("generation limit", &draft.generation_limit)?)
        .mutation_rate_mut(parse_f64("mutation rate", &draft.mutation_rate)?)
        .crossover_rate_mut(parse_f64("crossover rate", &draft.crossover_rate)?)
        .selection_ratio_mut(parse_f64("selection ratio", &draft.selection_ratio)?)
        .tournament_size_mut(parse_usize("tournament size", &draft.tournament_size)?)
        .elite_count_mut(parse_usize("elite count", &draft.elite_count)?)
        .random_immigrant_ratio_mut(parse_f64(
            "random immigrant ratio",
            &draft.random_immigrant_ratio,
        )?)
        .stagnation_limit_mut(parse_u64("stagnation limit", &draft.stagnation_limit)?);

    Ok(RunRequest::new(
        run_id,
        normalized_positive_smiles,
        normalized_negative_smiles,
        draft.seed_smarts.clone(),
        config,
        LEADERBOARD_LIMIT,
    ))
}

struct LeaderboardPage {
    page: usize,
    page_count: usize,
    page_size: usize,
    start: usize,
    end: usize,
}

fn leaderboard_page_window(
    total: usize,
    requested_page_size: usize,
    requested_page: usize,
) -> LeaderboardPage {
    let page_size = requested_page_size.max(1);
    let page_count = total.div_ceil(page_size).max(1);
    let page = requested_page.min(page_count - 1);
    let start = (page * page_size).min(total);
    let end = (start + page_size).min(total);
    LeaderboardPage {
        page,
        page_count,
        page_size,
        start,
        end,
    }
}

fn validate_smiles_lines(label: &str, value: &str) -> Option<String> {
    let mut has_any = false;

    for (line_idx, line) in value.lines().enumerate() {
        let smiles = line.trim();
        if smiles.is_empty() {
            continue;
        }

        has_any = true;
        if let Err(error) = Smiles::from_str(smiles) {
            return Some(format!(
                "{label} has invalid SMILES at line {}: {error}",
                line_idx + 1
            ));
        }
    }

    if has_any {
        None
    } else {
        Some(format!("{label} cannot be empty."))
    }
}

fn compact_smiles_lines(label: &str, value: &str) -> Result<String, String> {
    let mut normalized = Vec::new();

    for line in value.lines() {
        let smiles = line.trim();
        if smiles.is_empty() {
            continue;
        }

        normalized.push(smiles.to_string());
    }

    if normalized.is_empty() {
        Err(format!("{label} cannot be empty."))
    } else {
        Ok(normalized.join("\n"))
    }
}

fn smiles_entry_count(value: &str) -> usize {
    value.lines().filter(|line| !line.trim().is_empty()).count()
}

fn smiles_count_label(count: usize) -> String {
    let suffix = if count == 1 { "entry" } else { "entries" };
    format!("{count} {suffix}")
}

struct ParsedValue<T> {
    value: Option<T>,
    error: Option<String>,
}

fn validate_usize_value(label: &str, value: &str, require_positive: bool) -> ParsedValue<usize> {
    match value.trim().parse::<usize>() {
        Ok(parsed) if require_positive && parsed == 0 => ParsedValue {
            value: Some(parsed),
            error: Some(format!("{label} must be greater than zero.")),
        },
        Ok(parsed) => ParsedValue {
            value: Some(parsed),
            error: None,
        },
        Err(error) => ParsedValue {
            value: None,
            error: Some(format!("{label} is not a valid integer: {error}")),
        },
    }
}

fn validate_u64_value(label: &str, value: &str, require_positive: bool) -> ParsedValue<u64> {
    match value.trim().parse::<u64>() {
        Ok(parsed) if require_positive && parsed == 0 => ParsedValue {
            value: Some(parsed),
            error: Some(format!("{label} must be greater than zero.")),
        },
        Ok(parsed) => ParsedValue {
            value: Some(parsed),
            error: None,
        },
        Err(error) => ParsedValue {
            value: None,
            error: Some(format!("{label} is not a valid integer: {error}")),
        },
    }
}

fn validate_probability_value(label: &str, value: &str, require_positive: bool) -> ParsedValue<f64> {
    match value.trim().parse::<f64>() {
        Ok(parsed) if !(0.0..=1.0).contains(&parsed) => ParsedValue {
            value: Some(parsed),
            error: Some(format!("{label} must be between 0 and 1.")),
        },
        Ok(parsed) if require_positive && parsed == 0.0 => ParsedValue {
            value: Some(parsed),
            error: Some(format!("{label} must be greater than zero.")),
        },
        Ok(parsed) => ParsedValue {
            value: Some(parsed),
            error: None,
        },
        Err(error) => ParsedValue {
            value: None,
            error: Some(format!("{label} is not a valid number: {error}")),
        },
    }
}

fn parse_usize(label: &str, value: &str) -> Result<usize, String> {
    value
        .trim()
        .parse::<usize>()
        .map_err(|error| format!("invalid {label}: {error}"))
}

fn parse_u64(label: &str, value: &str) -> Result<u64, String> {
    value
        .trim()
        .parse::<u64>()
        .map_err(|error| format!("invalid {label}: {error}"))
}

fn parse_f64(label: &str, value: &str) -> Result<f64, String> {
    value
        .trim()
        .parse::<f64>()
        .map_err(|error| format!("invalid {label}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::{
        DraftValidation, EXAMPLE_PRESETS, RankedCandidate, build_run_request,
        compact_smiles_lines, compare_ranked_candidates, smiles_count_label, smiles_entry_count,
        validate_smiles_lines,
    };

    #[test]
    fn baked_in_examples_are_smiles_only() {
        for preset in EXAMPLE_PRESETS {
            for (smiles, expected_count) in
                [(preset.positive_smiles(), 2000usize), (preset.negative_smiles(), 2000usize)]
            {
                let first = smiles.lines().next().unwrap();
                assert!(!first.contains('\t'));
                assert!(!first.chars().all(|ch| ch.is_ascii_digit()));
                assert_eq!(smiles.lines().count(), expected_count);
            }
        }
    }

    #[test]
    fn ranked_candidate_order_prefers_shorter_text_ties() {
        let longer = RankedCandidate::new("[#6]", 0.8124, 1);
        let shorter = RankedCandidate::new("[N]", 0.8124, 1);
        assert!(compare_ranked_candidates(&shorter, &longer).is_lt());
    }

    #[test]
    fn ranked_candidate_order_prefers_higher_mcc() {
        let lower_mcc = RankedCandidate::new("[#6]", 0.8000, 1);
        let higher_mcc = RankedCandidate::new("[#6]", 0.9000, 1);

        assert!(compare_ranked_candidates(&higher_mcc, &lower_mcc).is_lt());
    }

    #[test]
    fn config_summary_error_does_not_repeat_smiles_errors() {
        let validation = DraftValidation {
            positive_smiles: Some("positive SMILES cannot be empty.".to_string()),
            population_size: Some("Population must be greater than zero.".to_string()),
            ..DraftValidation::default()
        };

        assert_eq!(
            validation.first_error(),
            Some("positive SMILES cannot be empty.")
        );
        assert_eq!(
            validation.first_config_error(),
            Some("Population must be greater than zero.")
        );
    }

    #[test]
    fn invalid_smiles_line_is_reported_before_start() {
        let error = validate_smiles_lines("positive SMILES", "CCO\nnot-a-smiles\nCCN");
        assert!(error.is_some());
        assert!(
            error
                .unwrap()
                .starts_with("positive SMILES has invalid SMILES at line 2:")
        );
    }

    #[test]
    fn compact_smiles_lines_trims_and_compacts_input() {
        let normalized = compact_smiles_lines("positive SMILES", " CCO \n\n N \n").unwrap();
        assert_eq!(normalized, "CCO\nN");
    }

    #[test]
    fn build_run_request_compacts_smiles_before_dispatch() {
        let draft = super::RunDraft {
            positive_smiles: " CCO \n\n N \n".to_string(),
            negative_smiles: " O \n C=C ".to_string(),
            ..super::RunDraft::default()
        };

        let request = build_run_request(7, &draft).unwrap();
        assert_eq!(request.positive_smiles(), "CCO\nN");
        assert_eq!(request.negative_smiles(), "O\nC=C");
    }

    #[test]
    fn smiles_entry_counts_ignore_blank_lines() {
        assert_eq!(smiles_entry_count("CCO\n\nN\n"), 2);
        assert_eq!(smiles_count_label(1), "1 entry");
        assert_eq!(smiles_count_label(2), "2 entries");
    }
}

#[cfg(target_arch = "wasm32")]
fn js_error(error: JsValue) -> String {
    error
        .as_string()
        .unwrap_or_else(|| "unknown JavaScript worker error".to_string())
}
