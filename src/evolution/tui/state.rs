use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::time::Duration;
use std::time::Instant;

use ratatui::prelude::Rect;

use super::TuiEvolutionError;
use super::layout::{DashboardLayout, contains_position};
use super::metric::{TuiMetric, YAxisMode};
use super::platform::current_process_resident_memory_bytes;
use super::render::ratio;
use super::worker::{WorkerEvent, WorkerOutcome};
use crate::evolution::runner::{EvolutionProgress, EvolutionStatus, RankedSmarts, TaskResult};

const BUTTON_FEEDBACK_DURATION: Duration = Duration::from_millis(450);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DashboardStatus {
    Waiting,
    Running,
    Pausing,
    Paused,
    Stopping,
    Completed,
    Stopped,
    Failed,
}

impl DashboardStatus {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Waiting => "WAITING",
            Self::Running => "RUNNING",
            Self::Pausing => "PAUSING",
            Self::Paused => "PAUSED",
            Self::Stopping => "STOPPING",
            Self::Completed => "COMPLETED",
            Self::Stopped => "STOPPED",
            Self::Failed => "FAILED",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DashboardPhase {
    Startup,
    Evaluating,
    Mutating,
    GenerationComplete,
    Pausing,
    Paused,
    Stopping,
    Completed,
    Failed,
}

impl DashboardPhase {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Startup => "startup",
            Self::Evaluating => "evaluating",
            Self::Mutating => "mutating",
            Self::GenerationComplete => "generation complete",
            Self::Pausing => "pausing",
            Self::Paused => "paused",
            Self::Stopping => "stopping",
            Self::Completed => "complete",
            Self::Failed => "failed",
        }
    }

    pub(super) fn short_label(self) -> &'static str {
        match self {
            Self::Startup => "start",
            Self::Evaluating => "eval",
            Self::Mutating => "mut",
            Self::GenerationComplete => "gen",
            Self::Pausing => "pause",
            Self::Paused => "pause",
            Self::Stopping => "stop",
            Self::Completed => "done",
            Self::Failed => "fail",
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct HistoryPoint {
    pub(super) generation: u64,
    pub(super) mcc: f64,
    pub(super) coverage: f64,
    pub(super) stagnation: u64,
    pub(super) unique_ratio: f64,
    pub(super) timeouts: usize,
    pub(super) lead_len: usize,
    pub(super) average_len: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ChangePointSummary {
    pub(super) generation: u64,
    pub(super) smarts: String,
    pub(super) mcc: f64,
    pub(super) smarts_len: usize,
    pub(super) coverage_score: f64,
}

impl ChangePointSummary {
    pub(super) fn from_candidate(generation: u64, candidate: &CandidateSummary) -> Self {
        Self {
            generation,
            smarts: candidate.smarts.clone(),
            mcc: candidate.mcc,
            smarts_len: candidate.smarts_len,
            coverage_score: candidate.coverage_score,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct HoveredChangePoint {
    pub(super) point: ChangePointSummary,
    pub(super) marker_x: u16,
    pub(super) marker_y: u16,
}

#[derive(Clone, Debug)]
pub(super) struct CandidateSummary {
    pub(super) smarts: String,
    pub(super) mcc: f64,
    pub(super) smarts_len: usize,
    pub(super) coverage_score: f64,
}

impl CandidateSummary {
    pub(super) fn from_ranked(candidate: &RankedSmarts) -> Self {
        Self {
            smarts: candidate.smarts().to_string(),
            mcc: candidate.mcc(),
            smarts_len: candidate.smarts_len(),
            coverage_score: candidate.coverage_score(),
        }
    }

    pub(super) fn from_result(result: &TaskResult) -> Self {
        Self {
            smarts: result.best_smarts().to_string(),
            mcc: result.best_mcc(),
            smarts_len: result.best_smarts_len(),
            coverage_score: result.best_coverage_score(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct ClickRegions {
    pub(super) layout: Option<DashboardLayout>,
    pub(super) metric_area: Option<Rect>,
    pub(super) plot_area: Option<Rect>,
    pub(super) best_area: Option<Rect>,
    pub(super) footer_prev: Option<Rect>,
    pub(super) footer_pause: Option<Rect>,
    pub(super) footer_best: Option<Rect>,
    pub(super) footer_next: Option<Rect>,
    pub(super) footer_stop: Option<Rect>,
    pub(super) footer_help: Option<Rect>,
    pub(super) help_overlay: Option<Rect>,
    pub(super) best_overlay: Option<Rect>,
    pub(super) best_copy: Option<Rect>,
    pub(super) change_copy: Option<Rect>,
    pub(super) y_axis_mode: Option<Rect>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FooterControl {
    PreviousMetric,
    PauseResume,
    BestSmarts,
    NextMetric,
    Stop,
    Help,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CopyControl {
    BestSmarts,
    ChangePoint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ButtonId {
    Footer(FooterControl),
    Copy(CopyControl),
    YAxisMode,
}

impl FooterControl {
    pub(super) const ALL: [Self; 6] = [
        Self::PreviousMetric,
        Self::PauseResume,
        Self::BestSmarts,
        Self::NextMetric,
        Self::Stop,
        Self::Help,
    ];
}

impl CopyControl {
    pub(super) const ALL: [Self; 2] = [Self::BestSmarts, Self::ChangePoint];
}

impl ButtonId {
    pub(super) const ALL: [Self; 9] = [
        Self::Copy(CopyControl::BestSmarts),
        Self::Copy(CopyControl::ChangePoint),
        Self::YAxisMode,
        Self::Footer(FooterControl::PreviousMetric),
        Self::Footer(FooterControl::PauseResume),
        Self::Footer(FooterControl::BestSmarts),
        Self::Footer(FooterControl::NextMetric),
        Self::Footer(FooterControl::Stop),
        Self::Footer(FooterControl::Help),
    ];
}

impl ClickRegions {
    pub(super) fn reset_footer_buttons(&mut self) {
        for control in FooterControl::ALL {
            self.set_footer_button(control, None);
        }
    }

    pub(super) fn set_button(&mut self, button: ButtonId, region: Option<Rect>) {
        match button {
            ButtonId::Footer(control) => self.set_footer_button(control, region),
            ButtonId::Copy(control) => match control {
                CopyControl::BestSmarts => self.best_copy = region,
                CopyControl::ChangePoint => self.change_copy = region,
            },
            ButtonId::YAxisMode => self.y_axis_mode = region,
        }
    }

    pub(super) fn button(&self, button: ButtonId) -> Option<Rect> {
        match button {
            ButtonId::Footer(control) => self.footer_button(control),
            ButtonId::Copy(control) => match control {
                CopyControl::BestSmarts => self.best_copy,
                CopyControl::ChangePoint => self.change_copy,
            },
            ButtonId::YAxisMode => self.y_axis_mode,
        }
    }

    pub(super) fn set_footer_button(&mut self, control: FooterControl, region: Option<Rect>) {
        match control {
            FooterControl::PreviousMetric => self.footer_prev = region,
            FooterControl::PauseResume => self.footer_pause = region,
            FooterControl::BestSmarts => self.footer_best = region,
            FooterControl::NextMetric => self.footer_next = region,
            FooterControl::Stop => self.footer_stop = region,
            FooterControl::Help => self.footer_help = region,
        }
    }

    pub(super) fn footer_button(&self, control: FooterControl) -> Option<Rect> {
        match control {
            FooterControl::PreviousMetric => self.footer_prev,
            FooterControl::PauseResume => self.footer_pause,
            FooterControl::BestSmarts => self.footer_best,
            FooterControl::NextMetric => self.footer_next,
            FooterControl::Stop => self.footer_stop,
            FooterControl::Help => self.footer_help,
        }
    }

    pub(super) fn contains(region: Option<Rect>, column: u16, row: u16) -> bool {
        region.is_some_and(|area| contains_position(area, column, row))
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ButtonFeedback {
    pub(super) button: ButtonId,
    pub(super) until: Instant,
}

impl ButtonFeedback {
    pub(super) fn new(button: ButtonId) -> Self {
        Self {
            button,
            until: Instant::now() + BUTTON_FEEDBACK_DURATION,
        }
    }

    pub(super) fn is_active_for(self, button: ButtonId) -> bool {
        self.button == button && Instant::now() < self.until
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RuntimeMetrics {
    pub(super) rayon_threads: usize,
    pub(super) resident_memory_bytes: Option<u64>,
}

impl RuntimeMetrics {
    pub(super) fn current() -> Self {
        Self {
            rayon_threads: rayon::current_num_threads(),
            resident_memory_bytes: current_process_resident_memory_bytes(),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct DashboardState {
    pub(super) task_id: String,
    pub(super) generation: u64,
    pub(super) generation_limit: u64,
    pub(super) status: DashboardStatus,
    pub(super) phase: DashboardPhase,
    pub(super) phase_generation: u64,
    pub(super) phase_completed: usize,
    pub(super) phase_total: usize,
    pub(super) runtime_metrics: RuntimeMetrics,
    pub(super) current_mcc: Option<f64>,
    pub(super) current_limit_exceeded: bool,
    pub(super) generation_best_mcc: Option<f64>,
    pub(super) best: Option<CandidateSummary>,
    pub(super) history: Vec<HistoryPoint>,
    pub(super) change_points: Vec<ChangePointSummary>,
    pub(super) selected_metric: TuiMetric,
    pub(super) y_axis_mode: YAxisMode,
    pub(super) hovered_metric: Option<TuiMetric>,
    pub(super) hovered_button: Option<ButtonId>,
    pub(super) hovered_change_point: Option<HoveredChangePoint>,
    pub(super) button_feedback: Option<ButtonFeedback>,
    pub(super) started_at: Instant,
    pub(super) last_event: Option<String>,
    pub(super) show_help: bool,
    pub(super) show_best_smarts: bool,
    pub(super) best_smarts_width: usize,
    pub(super) pending_clipboard: Option<String>,
    pub(super) click_regions: ClickRegions,
}

impl DashboardState {
    pub(super) fn new(best_smarts_width: usize) -> Self {
        Self {
            task_id: "waiting".into(),
            generation: 0,
            generation_limit: 1,
            status: DashboardStatus::Waiting,
            phase: DashboardPhase::Startup,
            phase_generation: 0,
            phase_completed: 0,
            phase_total: 1,
            runtime_metrics: RuntimeMetrics::current(),
            current_mcc: None,
            current_limit_exceeded: false,
            generation_best_mcc: None,
            best: None,
            history: Vec::new(),
            change_points: Vec::new(),
            selected_metric: TuiMetric::Mcc,
            y_axis_mode: YAxisMode::Zoomed,
            hovered_metric: None,
            hovered_button: None,
            hovered_change_point: None,
            button_feedback: None,
            started_at: Instant::now(),
            last_event: None,
            show_help: false,
            show_best_smarts: false,
            best_smarts_width,
            pending_clipboard: None,
            click_regions: ClickRegions::default(),
        }
    }

    pub(super) fn clear_hover(&mut self) {
        self.hovered_metric = None;
        self.hovered_button = None;
        self.hovered_change_point = None;
    }

    pub(super) fn close_overlays(&mut self) {
        self.show_help = false;
        self.show_best_smarts = false;
    }

    pub(super) fn toggle_best_smarts_overlay(&mut self) {
        if self.best.is_some() {
            self.show_best_smarts = !self.show_best_smarts;
            self.show_help = false;
        }
    }

    pub(super) fn show_best_smarts_overlay(&mut self) {
        if self.best.is_some() {
            self.show_best_smarts = true;
            self.show_help = false;
        }
    }

    pub(super) fn select_metric(&mut self, metric: TuiMetric) {
        self.selected_metric = metric;
        self.clear_hover();
    }

    pub(super) fn select_previous_metric(&mut self) {
        self.select_metric(self.selected_metric.previous());
    }

    pub(super) fn select_next_metric(&mut self) {
        self.select_metric(self.selected_metric.next());
    }

    pub(super) fn apply_worker_event(&mut self, event: WorkerEvent) -> WorkerOutcome {
        match event {
            WorkerEvent::Started {
                task_id,
                generation_limit,
            } => {
                self.task_id = task_id;
                self.generation_limit = generation_limit.max(1);
                self.status = DashboardStatus::Running;
                self.phase = DashboardPhase::Startup;
                self.started_at = Instant::now();
                self.current_mcc = None;
                self.current_limit_exceeded = false;
                self.last_event = None;
                WorkerOutcome::Continue
            }
            WorkerEvent::Evaluation(progress) => {
                self.phase = DashboardPhase::Evaluating;
                self.phase_generation = progress.generation();
                self.phase_completed = progress.completed();
                self.phase_total = progress.total();
                self.current_mcc = progress.last_mcc();
                self.current_limit_exceeded = progress.last_limit_exceeded().unwrap_or(false);
                self.generation_best_mcc = progress.generation_best_mcc();
                let best = CandidateSummary::from_ranked(progress.incumbent_best());
                self.best = Some(best);
                WorkerOutcome::Continue
            }
            WorkerEvent::Offspring(progress) => {
                self.phase = DashboardPhase::Mutating;
                self.phase_generation = progress.generation();
                self.phase_completed = progress.completed();
                self.phase_total = progress.total();
                WorkerOutcome::Continue
            }
            WorkerEvent::Generation(progress) => {
                self.apply_generation(progress);
                WorkerOutcome::Continue
            }
            WorkerEvent::Paused => {
                self.status = DashboardStatus::Paused;
                self.phase = DashboardPhase::Paused;
                self.phase_completed = 0;
                self.phase_total = 1;
                self.last_event = Some("paused at generation boundary".into());
                WorkerOutcome::Continue
            }
            WorkerEvent::Resumed => {
                self.status = DashboardStatus::Running;
                self.phase = DashboardPhase::Startup;
                self.last_event = Some("resumed".into());
                WorkerOutcome::Continue
            }
            WorkerEvent::Finished(result) => match result {
                Ok(result) => {
                    self.status = DashboardStatus::Completed;
                    self.phase = DashboardPhase::Completed;
                    self.generation = result.generations();
                    self.best = Some(CandidateSummary::from_result(&result));
                    WorkerOutcome::Finished(result)
                }
                Err(error) => {
                    self.status = DashboardStatus::Failed;
                    self.phase = DashboardPhase::Failed;
                    self.last_event = Some(error.to_string());
                    WorkerOutcome::Failed(TuiEvolutionError::Evolution(error))
                }
            },
            WorkerEvent::Stopped => {
                self.status = DashboardStatus::Stopped;
                self.phase = DashboardPhase::Stopping;
                self.last_event = Some("stopped at generation boundary".into());
                WorkerOutcome::Stopped
            }
        }
    }

    pub(super) fn apply_generation(&mut self, progress: EvolutionProgress) {
        if !matches!(
            self.status,
            DashboardStatus::Stopping | DashboardStatus::Pausing | DashboardStatus::Paused
        ) {
            self.status = match progress.status() {
                EvolutionStatus::Running => DashboardStatus::Running,
                EvolutionStatus::Stagnated | EvolutionStatus::Completed => {
                    DashboardStatus::Completed
                }
            };
        }
        self.phase = DashboardPhase::GenerationComplete;
        self.generation = progress.generation();
        self.generation_limit = progress.generation_limit().max(1);
        let best_so_far = CandidateSummary::from_ranked(progress.best_so_far());
        self.record_change_point(progress.generation(), &best_so_far);
        self.best = Some(best_so_far);
        self.current_mcc = None;
        self.current_limit_exceeded = false;
        self.generation_best_mcc = Some(progress.best().mcc());
        let point = HistoryPoint {
            generation: progress.generation(),
            mcc: progress.best_so_far().mcc(),
            coverage: progress.best_so_far().coverage_score(),
            stagnation: progress.stagnation(),
            unique_ratio: ratio(progress.unique_count(), progress.total_count()),
            timeouts: progress.match_timeout_count(),
            lead_len: progress.lead_smarts_len(),
            average_len: progress.average_smarts_len(),
        };
        if let Some(last) = self.history.last_mut()
            && last.generation == point.generation
        {
            *last = point;
            return;
        }
        self.history.push(point);
    }

    pub(super) fn record_change_point(&mut self, generation: u64, candidate: &CandidateSummary) {
        let point = ChangePointSummary::from_candidate(generation, candidate);
        if let Some(last) = self.change_points.last_mut() {
            if last.generation == generation {
                *last = point;
                return;
            }
            if last.smarts == point.smarts {
                return;
            }
        }
        self.change_points.push(point);
    }

    pub(super) fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    pub(super) fn best_mcc(&self) -> Option<f64> {
        self.best.as_ref().map(|candidate| candidate.mcc)
    }

    pub(super) fn best_coverage(&self) -> Option<f64> {
        self.best.as_ref().map(|candidate| candidate.coverage_score)
    }
}
