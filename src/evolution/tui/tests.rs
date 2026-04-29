use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::str::FromStr;
use core::time::Duration;
use std::io;
use std::sync::mpsc;

use crate::evolution::runner::{EvolutionError, EvolutionSession};
use crate::{EvolutionConfig, EvolutionTask, FoldData, FoldSample, SeedCorpus};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{self, MouseButton, MouseEvent, MouseEventKind};
use ratatui::prelude::{Color, Modifier, Rect};
use smarts_rs::PreparedTarget;
use smiles_parser::Smiles;

use super::input::*;
use super::layout::*;
use super::metric::*;
use super::platform::*;
use super::render::*;
use super::state::*;
use super::worker::*;
use super::{TuiEvolutionDashboard, TuiEvolutionError, is_terminal_error};

fn sample_state(metric: TuiMetric) -> DashboardState {
    let mut state = DashboardState::new(36);
    state.task_id = "class:259:Fatty nitriles".into();
    state.generation_limit = 800;
    state.generation = 3;
    state.status = DashboardStatus::Running;
    state.phase = DashboardPhase::Evaluating;
    state.phase_generation = 4;
    state.phase_completed = 128;
    state.phase_total = 2048;
    state.runtime_metrics = RuntimeMetrics {
        rayon_threads: 8,
        resident_memory_bytes: Some(512 * 1024 * 1024),
    };
    state.selected_metric = metric;
    state.current_mcc = Some(-1.0);
    state.generation_best_mcc = Some(0.42);
    state.best = Some(CandidateSummary {
        smarts: "[#6](=[#8])-[#8]-[#6]-[#6]-[#6]".into(),
        mcc: 0.5,
        smarts_len: 34,
        coverage_score: 0.6,
    });
    state.history = vec![
        HistoryPoint {
            generation: 1,
            mcc: 0.1,
            coverage: 0.2,
            stagnation: 0,
            unique_ratio: 1.0,
            timeouts: 0,
            lead_len: 12,
            average_len: 10.0,
        },
        HistoryPoint {
            generation: 2,
            mcc: 0.3,
            coverage: 0.4,
            stagnation: 0,
            unique_ratio: 0.95,
            timeouts: 1,
            lead_len: 18,
            average_len: 14.0,
        },
        HistoryPoint {
            generation: 3,
            mcc: 0.5,
            coverage: 0.6,
            stagnation: 1,
            unique_ratio: 0.90,
            timeouts: 0,
            lead_len: 24,
            average_len: 19.0,
        },
    ];
    state.change_points = vec![
        ChangePointSummary {
            generation: 1,
            smarts: "[#6]".into(),
            mcc: 0.1,
            smarts_len: 4,
            coverage_score: 0.2,
        },
        ChangePointSummary {
            generation: 3,
            smarts: "[#6](=[#8])-[#8]-[#6]-[#6]-[#6]".into(),
            mcc: 0.5,
            smarts_len: 34,
            coverage_score: 0.6,
        },
    ];
    state
}

fn render_to_string(state: &DashboardState, width: u16, height: u16) -> String {
    let mut state = state.clone();
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| render_dashboard(frame, &mut state))
        .unwrap();
    terminal.backend().to_string()
}

fn render_state(mut state: DashboardState, width: u16, height: u16) -> DashboardState {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| render_dashboard(frame, &mut state))
        .unwrap();
    state
}

fn render_to_backend(
    mut state: DashboardState,
    width: u16,
    height: u16,
) -> (DashboardState, TestBackend) {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| render_dashboard(frame, &mut state))
        .unwrap();
    (state, terminal.backend().clone())
}

fn mouse_event(kind: MouseEventKind, column: u16, row: u16) -> MouseEvent {
    MouseEvent {
        kind,
        column,
        row,
        modifiers: event::KeyModifiers::empty(),
    }
}

fn copy_button_probe(area: Rect) -> (u16, u16) {
    (
        area.x.saturating_add(1).min(area.right().saturating_sub(1)),
        area.y,
    )
}

fn prepared(smiles: &str) -> PreparedTarget {
    PreparedTarget::new(Smiles::from_str(smiles).unwrap())
}

fn tiny_tui_task() -> EvolutionTask {
    EvolutionTask::new(
        "tui-worker-test",
        vec![FoldData::new(vec![
            FoldSample::positive(prepared("CN")),
            FoldSample::positive(prepared("CCN")),
            FoldSample::negative(prepared("CC")),
            FoldSample::negative(prepared("CO")),
        ])],
    )
}

fn tiny_tui_config(generation_limit: u64) -> EvolutionConfig {
    EvolutionConfig::builder()
        .population_size(4)
        .generation_limit(generation_limit)
        .stagnation_limit(generation_limit + 2)
        .mutation_rate(0.0)
        .crossover_rate(0.0)
        .rng_seed(11)
        .disable_match_time_limit()
        .build()
        .unwrap()
}

fn tiny_tui_seed_corpus() -> SeedCorpus {
    SeedCorpus::try_from(["[#6]", "[#7]", "[#6]~[#7]", "[#6]~[#8]"]).unwrap()
}

fn tiny_tui_session(generation_limit: u64) -> EvolutionSession {
    let task = tiny_tui_task();
    let config = tiny_tui_config(generation_limit);
    let seeds = tiny_tui_seed_corpus();
    EvolutionSession::new(&task, &config, &seeds, 2).unwrap()
}

#[test]
fn facade_public_helpers_round_trip() {
    let dashboard = TuiEvolutionDashboard::default()
        .with_draw_interval(Duration::from_millis(250))
        .with_best_smarts_width(44);
    assert_eq!(dashboard.draw_interval, Duration::from_millis(250));
    assert_eq!(dashboard.best_smarts_width, 44);

    let evolution_error = TuiEvolutionError::Evolution(EvolutionError::InvalidConfig("bad".into()));
    assert!(evolution_error.to_string().contains("bad"));
    assert!(TuiEvolutionError::Stopped.to_string().contains("stopped"));
    assert!(
        TuiEvolutionError::WorkerDisconnected
            .to_string()
            .contains("disconnected")
    );
    assert!(
        TuiEvolutionError::WorkerPanicked
            .to_string()
            .contains("panicked")
    );

    let terminal_error = TuiEvolutionError::from(io::Error::other("no tty"));
    assert!(terminal_error.to_string().contains("terminal error"));
    assert!(is_terminal_error(&Some(Err(TuiEvolutionError::Terminal(
        io::Error::other("no tty"),
    )))));
    assert!(!is_terminal_error(&Some(Err(TuiEvolutionError::Stopped))));
}

#[test]
fn worker_emits_progress_and_state_consumes_real_events() -> Result<(), Box<dyn std::error::Error>>
{
    let generation_limit = 2;
    let session = tiny_tui_session(generation_limit);
    let (event_tx, event_rx) = mpsc::channel();
    let (_control_tx, control_rx) = mpsc::channel();
    let worker = spawn_worker(session, event_tx, control_rx);
    let mut state = DashboardState::new(36);
    let mut saw_started = false;
    let mut saw_evaluation = false;
    let mut saw_offspring = false;
    let mut saw_generation = false;

    loop {
        let event = event_rx.recv_timeout(Duration::from_secs(5))?;
        match &event {
            WorkerEvent::Started {
                task_id,
                generation_limit: started_generation_limit,
            } => {
                saw_started = true;
                assert_eq!(task_id, "tui-worker-test");
                assert_eq!(*started_generation_limit, generation_limit);
            }
            WorkerEvent::Evaluation(progress) => {
                saw_evaluation = true;
                assert_eq!(progress.task_id(), "tui-worker-test");
                assert!((1..=generation_limit).contains(&progress.generation()));
                assert!(progress.completed() <= progress.total());
            }
            WorkerEvent::Offspring(progress) => {
                saw_offspring = true;
                assert_eq!(progress.task_id(), "tui-worker-test");
                assert!((1..=generation_limit).contains(&progress.generation()));
                assert!(progress.completed() <= progress.total());
            }
            WorkerEvent::Generation(progress) => {
                saw_generation = true;
                assert_eq!(progress.task_id(), "tui-worker-test");
                assert!((1..=generation_limit).contains(&progress.generation()));
            }
            WorkerEvent::Finished(Ok(result)) => {
                assert_eq!(result.task_id(), "tui-worker-test");
            }
            WorkerEvent::Finished(Err(error)) => {
                return Err(format!("unexpected worker failure: {error}").into());
            }
            WorkerEvent::Paused | WorkerEvent::Resumed | WorkerEvent::Stopped => {
                return Err("unexpected control event".into());
            }
        }

        match state.apply_worker_event(event) {
            WorkerOutcome::Continue => {}
            WorkerOutcome::Finished(result) => {
                assert_eq!(result.task_id(), "tui-worker-test");
                break;
            }
            WorkerOutcome::Failed(error) => {
                return Err(format!("unexpected dashboard failure: {error}").into());
            }
            WorkerOutcome::Stopped => return Err("worker stopped unexpectedly".into()),
        }
    }

    worker
        .join()
        .map_err(|_| io::Error::other("worker panicked"))?;
    assert!(saw_started);
    assert!(saw_evaluation);
    assert!(saw_offspring);
    assert!(saw_generation);
    assert_eq!(state.status, DashboardStatus::Completed);
    assert_eq!(state.phase, DashboardPhase::Completed);
    assert_eq!(state.generation, generation_limit);
    assert!(state.best.is_some());
    assert!(!state.history.is_empty());
    assert!(!state.change_points.is_empty());

    Ok(())
}

#[test]
fn worker_honors_pause_then_resume() -> Result<(), Box<dyn std::error::Error>> {
    let session = tiny_tui_session(1);
    let (event_tx, event_rx) = mpsc::channel();
    let (control_tx, control_rx) = mpsc::channel();
    control_tx.send(WorkerControl::Pause)?;
    let worker = spawn_worker(session, event_tx, control_rx);

    let mut saw_started = false;
    let mut saw_paused = false;
    while !saw_paused {
        match event_rx.recv_timeout(Duration::from_secs(5))? {
            WorkerEvent::Started { .. } => saw_started = true,
            WorkerEvent::Paused => saw_paused = true,
            other => return Err(format!("unexpected event before pause: {other:?}").into()),
        }
    }

    control_tx.send(WorkerControl::Resume)?;
    let mut saw_resumed = false;
    loop {
        match event_rx.recv_timeout(Duration::from_secs(5))? {
            WorkerEvent::Resumed => saw_resumed = true,
            WorkerEvent::Finished(Ok(_)) => break,
            WorkerEvent::Finished(Err(error)) => {
                return Err(format!("unexpected worker failure: {error}").into());
            }
            WorkerEvent::Started { .. }
            | WorkerEvent::Evaluation(_)
            | WorkerEvent::Offspring(_)
            | WorkerEvent::Generation(_) => {}
            WorkerEvent::Paused | WorkerEvent::Stopped => {
                return Err("unexpected pause/stop event after resume".into());
            }
        }
    }

    worker
        .join()
        .map_err(|_| io::Error::other("worker panicked"))?;
    assert!(saw_started);
    assert!(saw_paused);
    assert!(saw_resumed);
    Ok(())
}

#[test]
fn worker_can_stop_while_paused() -> Result<(), Box<dyn std::error::Error>> {
    let session = tiny_tui_session(2);
    let (event_tx, event_rx) = mpsc::channel();
    let (control_tx, control_rx) = mpsc::channel();
    control_tx.send(WorkerControl::Pause)?;
    let worker = spawn_worker(session, event_tx, control_rx);

    loop {
        match event_rx.recv_timeout(Duration::from_secs(5))? {
            WorkerEvent::Started { .. } => {}
            WorkerEvent::Paused => break,
            other => return Err(format!("unexpected event before pause: {other:?}").into()),
        }
    }

    control_tx.send(WorkerControl::Stop)?;
    assert!(matches!(
        event_rx.recv_timeout(Duration::from_secs(5))?,
        WorkerEvent::Stopped
    ));
    worker
        .join()
        .map_err(|_| io::Error::other("worker panicked"))?;
    Ok(())
}

#[test]
fn worker_stops_when_control_channel_drops_while_paused() -> Result<(), Box<dyn std::error::Error>>
{
    let session = tiny_tui_session(2);
    let (event_tx, event_rx) = mpsc::channel();
    let (control_tx, control_rx) = mpsc::channel();
    control_tx.send(WorkerControl::Pause)?;
    let worker = spawn_worker(session, event_tx, control_rx);

    loop {
        match event_rx.recv_timeout(Duration::from_secs(5))? {
            WorkerEvent::Started { .. } => {}
            WorkerEvent::Paused => break,
            other => return Err(format!("unexpected event before pause: {other:?}").into()),
        }
    }

    drop(control_tx);
    assert!(matches!(
        event_rx.recv_timeout(Duration::from_secs(5))?,
        WorkerEvent::Stopped
    ));
    worker
        .join()
        .map_err(|_| io::Error::other("worker panicked"))?;
    Ok(())
}

#[test]
fn state_applies_simple_worker_events_and_overlay_toggles() {
    let mut state = DashboardState::new(36);
    assert!(matches!(
        state.apply_worker_event(WorkerEvent::Started {
            task_id: "task".into(),
            generation_limit: 0,
        }),
        WorkerOutcome::Continue
    ));
    assert_eq!(state.task_id, "task");
    assert_eq!(state.generation_limit, 1);
    assert_eq!(state.status, DashboardStatus::Running);

    assert!(matches!(
        state.apply_worker_event(WorkerEvent::Paused),
        WorkerOutcome::Continue
    ));
    assert_eq!(state.status, DashboardStatus::Paused);
    assert_eq!(state.phase, DashboardPhase::Paused);

    assert!(matches!(
        state.apply_worker_event(WorkerEvent::Resumed),
        WorkerOutcome::Continue
    ));
    assert_eq!(state.status, DashboardStatus::Running);
    assert_eq!(state.phase, DashboardPhase::Startup);

    state.show_help = true;
    state.show_best_smarts = true;
    state.close_overlays();
    assert!(!state.show_help);
    assert!(!state.show_best_smarts);
    state.toggle_best_smarts_overlay();
    assert!(!state.show_best_smarts);

    assert!(matches!(
        state.apply_worker_event(WorkerEvent::Stopped),
        WorkerOutcome::Stopped
    ));
    assert_eq!(state.status, DashboardStatus::Stopped);

    let failure = EvolutionError::InvalidConfig("bad".into());
    assert!(matches!(
        state.apply_worker_event(WorkerEvent::Finished(Err(failure))),
        WorkerOutcome::Failed(TuiEvolutionError::Evolution(_))
    ));
    assert_eq!(state.status, DashboardStatus::Failed);
    assert_eq!(state.phase, DashboardPhase::Failed);
}

#[test]
fn input_edge_branches_keep_state_consistent() {
    let mut help_state = sample_state(TuiMetric::Mcc);
    help_state.show_help = true;
    help_state.hovered_metric = Some(TuiMetric::Coverage);
    update_hover_state(&mut help_state, 0, 0);
    assert_eq!(help_state.hovered_metric, None);

    help_state = render_state(help_state, 100, 30);
    let help_overlay = help_state.click_regions.help_overlay.unwrap();
    handle_left_click(
        &mut help_state,
        help_overlay.x + 1,
        help_overlay.y + 1,
        None,
    );
    assert!(help_state.show_help);
    handle_left_click(&mut help_state, 0, 0, None);
    assert!(!help_state.show_help);

    let mut best_state = sample_state(TuiMetric::Mcc);
    best_state.show_best_smarts = true;
    best_state = render_state(best_state, 100, 30);
    let best_copy = best_state.click_regions.best_copy.unwrap();
    update_hover_state(&mut best_state, best_copy.x, best_copy.y);
    assert_eq!(
        best_state.hovered_button,
        Some(ButtonId::Copy(CopyControl::BestSmarts))
    );

    let mut disabled_stop = sample_state(TuiMetric::Mcc);
    disabled_stop.status = DashboardStatus::Completed;
    let (control_tx, control_rx) = mpsc::channel();
    request_stop(&mut disabled_stop, Some(&control_tx));
    assert!(control_rx.try_recv().is_err());
    assert_eq!(disabled_stop.status, DashboardStatus::Completed);

    let mut completed = disabled_stop.clone();
    toggle_pause(&mut completed, Some(&control_tx));
    assert_eq!(completed.status, DashboardStatus::Completed);
    assert!(control_rx.try_recv().is_err());

    let mut without_best = DashboardState::new(36);
    handle_footer_control(&mut without_best, FooterControl::BestSmarts, None);
    assert!(!without_best.show_best_smarts);
    request_best_smarts_copy(&mut without_best);
    assert!(without_best.pending_clipboard.is_none());
    request_change_point_smarts_copy(&mut without_best);
    assert!(without_best.pending_clipboard.is_none());

    assert_eq!(metric_at_position(&DashboardState::new(36), 0, 0), None);
    let mut rendered = render_state(sample_state(TuiMetric::Mcc), 100, 30);
    assert_eq!(metric_at_position(&rendered, 0, 0), None);
    let previous_metric = rendered.selected_metric;
    handle_metric_scroll(&mut rendered, 0, 0, 1);
    assert_eq!(rendered.selected_metric, previous_metric);

    let minimal = render_state(sample_state(TuiMetric::Mcc), 60, 18);
    let plot = minimal.click_regions.plot_area.unwrap();
    assert_eq!(
        change_point_at_position(&minimal, plot.x + 1, plot.y + 1),
        None
    );
    assert!(!hovered_change_point_tooltip_contains(&minimal, 0, 0));
}

#[test]
fn state_labels_and_change_point_update_branches_are_covered() {
    assert_eq!(DashboardStatus::Waiting.label(), "WAITING");
    assert_eq!(DashboardStatus::Running.label(), "RUNNING");
    assert_eq!(DashboardStatus::Pausing.label(), "PAUSING");
    assert_eq!(DashboardStatus::Paused.label(), "PAUSED");
    assert_eq!(DashboardStatus::Stopping.label(), "STOPPING");
    assert_eq!(DashboardStatus::Completed.label(), "COMPLETED");
    assert_eq!(DashboardStatus::Stopped.label(), "STOPPED");
    assert_eq!(DashboardStatus::Failed.label(), "FAILED");
    assert_eq!(DashboardPhase::Startup.label(), "startup");
    assert_eq!(DashboardPhase::Evaluating.label(), "evaluating");
    assert_eq!(DashboardPhase::Mutating.label(), "mutating");
    assert_eq!(
        DashboardPhase::GenerationComplete.label(),
        "generation complete"
    );
    assert_eq!(DashboardPhase::Pausing.label(), "pausing");
    assert_eq!(DashboardPhase::Paused.label(), "paused");
    assert_eq!(DashboardPhase::Stopping.label(), "stopping");
    assert_eq!(DashboardPhase::Completed.label(), "complete");
    assert_eq!(DashboardPhase::Failed.label(), "failed");
    assert_eq!(DashboardPhase::Startup.short_label(), "start");
    assert_eq!(DashboardPhase::Evaluating.short_label(), "eval");
    assert_eq!(DashboardPhase::Mutating.short_label(), "mut");
    assert_eq!(DashboardPhase::GenerationComplete.short_label(), "gen");
    assert_eq!(DashboardPhase::Pausing.short_label(), "pause");
    assert_eq!(DashboardPhase::Paused.short_label(), "pause");
    assert_eq!(DashboardPhase::Stopping.short_label(), "stop");
    assert_eq!(DashboardPhase::Completed.short_label(), "done");
    assert_eq!(DashboardPhase::Failed.short_label(), "fail");

    let mut state = DashboardState::new(36);
    assert!(state.elapsed() <= Duration::from_secs(1));
    assert_eq!(state.best_mcc(), None);
    assert_eq!(state.best_coverage(), None);

    let first = CandidateSummary {
        smarts: "[#6]".into(),
        mcc: 0.1,
        smarts_len: 4,
        coverage_score: 0.2,
    };
    let replacement = CandidateSummary {
        smarts: "[#7]".into(),
        mcc: 0.2,
        smarts_len: 4,
        coverage_score: 0.3,
    };
    state.record_change_point(1, &first);
    state.record_change_point(1, &replacement);
    assert_eq!(state.change_points.len(), 1);
    assert_eq!(state.change_points[0].smarts, "[#7]");

    state.best = Some(replacement);
    state.show_help = true;
    state.toggle_best_smarts_overlay();
    assert!(state.show_best_smarts);
    assert!(!state.show_help);
    assert_eq!(state.best_mcc(), Some(0.2));
    assert_eq!(state.best_coverage(), Some(0.3));
    state.toggle_best_smarts_overlay();
    assert!(!state.show_best_smarts);

    let mut selectable = sample_state(TuiMetric::Coverage);
    selectable.hovered_metric = Some(TuiMetric::Mcc);
    selectable.select_metric(TuiMetric::Unique);
    assert_eq!(selectable.selected_metric, TuiMetric::Unique);
    assert_eq!(selectable.hovered_metric, None);
}

#[test]
fn facade_evolve_methods_return_session_errors_before_terminal_setup() {
    let config = tiny_tui_config(1);
    let seeds = tiny_tui_seed_corpus();
    let dashboard = TuiEvolutionDashboard::default();

    let empty_task = EvolutionTask::new("empty", Vec::new());
    assert!(matches!(
        empty_task.evolve_with_tui(&config, &seeds),
        Err(TuiEvolutionError::Evolution(EvolutionError::EmptyFolds))
    ));

    let empty_task = EvolutionTask::new("empty", Vec::new());
    assert!(matches!(
        empty_task.evolve_with_tui_dashboard(&config, &seeds, 1, dashboard.clone()),
        Err(TuiEvolutionError::Evolution(EvolutionError::EmptyFolds))
    ));

    let empty_task = EvolutionTask::new("empty-owned", Vec::new());
    assert!(matches!(
        empty_task.evolve_owned_with_tui(&config, &seeds),
        Err(TuiEvolutionError::Evolution(EvolutionError::EmptyFolds))
    ));

    let empty_task = EvolutionTask::new("empty-owned", Vec::new());
    assert!(matches!(
        empty_task.evolve_owned_with_tui_dashboard(&config, &seeds, 1, dashboard),
        Err(TuiEvolutionError::Evolution(EvolutionError::EmptyFolds))
    ));
}

#[test]
fn input_direct_button_and_clipboard_paths_are_covered() {
    let mut state = sample_state(TuiMetric::Mcc);
    handle_button_click(&mut state, ButtonId::Copy(CopyControl::BestSmarts), None);
    assert_eq!(
        state.pending_clipboard.as_deref(),
        Some("[#6](=[#8])-[#8]-[#6]-[#6]-[#6]")
    );

    state.pending_clipboard = None;
    state.hovered_change_point = Some(HoveredChangePoint {
        point: state.change_points[1].clone(),
        marker_x: 10,
        marker_y: 5,
    });
    handle_button_click(&mut state, ButtonId::Copy(CopyControl::ChangePoint), None);
    assert_eq!(
        state.pending_clipboard.as_deref(),
        Some("[#6](=[#8])-[#8]-[#6]-[#6]-[#6]")
    );

    copy_pending_clipboard_request_with(&mut state, |_| Ok(()));
    assert_eq!(state.pending_clipboard, None);
    assert_eq!(
        state.last_event.as_deref(),
        Some("copied SMARTS to clipboard")
    );
    copy_pending_clipboard_request_with(&mut state, |_| Ok(()));
    state.pending_clipboard = Some("[#6]".into());
    copy_pending_clipboard_request_with(&mut state, |_| Err(io::Error::other("clipboard denied")));
    assert_eq!(
        state.last_event.as_deref(),
        Some("clipboard copy failed: clipboard denied")
    );

    handle_button_click(&mut state, ButtonId::YAxisMode, None);
    assert_eq!(state.y_axis_mode, YAxisMode::Absolute);
    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Up(MouseButton::Left), 0, 0),
        None,
    );
    assert_eq!(state.y_axis_mode, YAxisMode::Absolute);

    let mut with_hover = DashboardState::new(36);
    with_hover.hovered_change_point = Some(HoveredChangePoint {
        point: state.change_points[0].clone(),
        marker_x: 0,
        marker_y: 0,
    });
    assert!(!hovered_change_point_tooltip_contains(&with_hover, 0, 0));
}

#[test]
fn render_helper_edge_cases_are_stable() {
    assert_eq!(metric_tab_separator_count(0), 0);
    assert_eq!(fit_text_to_width("abcdef", 0), "");
    assert_eq!(fit_text_to_width("abcdef", 3), "abc");
    assert_eq!(distributed_text_line(&[], 10), "");
    assert_eq!(distributed_text_line(&[String::from("x")], 0), "");
    assert_eq!(spread_item_offsets(10, &[]), Vec::<u16>::new());
    assert_eq!(spread_item_offsets(10, &[4]), vec![3]);
    assert_eq!(spread_item_offsets(5, &[3, 3, 3]), vec![0, 4, 5]);
    assert_eq!(centered_text_line("abc", 0), "");

    assert_eq!(
        change_point_tooltip_area(Rect::new(0, 0, 20, 4), 1, 1),
        None
    );
    let lower_right = change_point_tooltip_area(Rect::new(10, 10, 80, 30), 80, 35).unwrap();
    assert!(lower_right.x < 80);
    assert!(lower_right.y < 35);
    let upper_left = change_point_tooltip_area(Rect::new(10, 10, 80, 30), 12, 12).unwrap();
    assert!(upper_left.x >= 10);
    assert!(upper_left.y > 12);

    assert_eq!(x_bounds(&[(2.0, 0.5)]), (2.0, 3.0));
    assert_eq!(zoomed_y_bounds(&[], 0.0, 1.0), (0.0, 1.0));
    let (lower, upper) = zoomed_y_bounds(&[(1.0, 0.5), (2.0, 0.5)], 0.0, 1.0);
    assert!(lower < 0.5 && upper > 0.5);
    assert_eq!(
        data_point_screen_position(Rect::new(0, 0, 0, 10), 0.0, 1.0, 0.0, 1.0, 0.5, 0.5),
        None
    );

    let waiting = DashboardState::new(36);
    assert_eq!(eta_text(&waiting), "---");
    assert_eq!(fmt_duration(Duration::from_secs(3661)), "01:01:01");
    assert_eq!(truncate_middle("abcdef", 10), "abcdef");
}

#[test]
fn renders_waiting_sparkline_without_history() {
    let output = render_to_string(&DashboardState::new(36), 60, 18);
    assert!(output.contains("Waiting for MCC"));
}

#[test]
fn renders_each_metric_tab() {
    for metric in TuiMetric::ALL {
        let output = render_to_string(&sample_state(metric), 100, 30);
        assert!(output.contains(metric.title()));
        assert!(output.contains("smarts-evolution"));
    }
}

#[test]
fn renders_too_small_message() {
    let output = render_to_string(&sample_state(TuiMetric::Mcc), 40, 10);
    assert!(output.contains("terminal too small"));
}

#[test]
fn renders_compact_layout() {
    let output = render_to_string(&sample_state(TuiMetric::Coverage), 80, 24);
    assert!(output.contains("Coverage best-so-far"));
    assert!(output.contains("[l ←]"));
}

#[test]
fn render_records_mouse_click_regions() {
    let state = render_state(sample_state(TuiMetric::Mcc), 100, 30);
    assert_eq!(state.click_regions.layout, Some(DashboardLayout::Full));
    assert!(state.click_regions.metric_area.is_some());
    assert!(state.click_regions.plot_area.is_some());
    assert!(state.click_regions.best_area.is_some());
    assert!(state.click_regions.footer_prev.is_some());
    assert!(state.click_regions.footer_pause.is_some());
    assert!(state.click_regions.footer_best.is_some());
    assert!(state.click_regions.footer_next.is_some());
    assert!(state.click_regions.footer_stop.is_some());
    assert!(state.click_regions.footer_help.is_some());
}

#[test]
fn header_lines_spread_content_across_width() {
    let state = sample_state(TuiMetric::Mcc);
    let full_line = one_line_header(Rect::new(0, 0, 150, 3), &state);
    assert_eq!(full_line.chars().count(), block_inner_width(150));
    assert!(full_line.contains("class:259:Fatty nitriles"));
    assert!(full_line.contains("gen   3/800"));
    assert!(full_line.contains("elapsed"));

    let compact_lines = two_line_header(Rect::new(0, 0, 80, 4), &state, DashboardLayout::Compact);
    assert_eq!(compact_lines.len(), 2);
    for line in compact_lines {
        assert_eq!(line.width(), block_inner_width(80));
    }
}

#[test]
fn footer_controls_spread_across_width() {
    let state = render_state(sample_state(TuiMetric::Mcc), 100, 30);
    let previous = state.click_regions.footer_prev.unwrap();
    let pause = state.click_regions.footer_pause.unwrap();
    let best = state.click_regions.footer_best.unwrap();
    let next = state.click_regions.footer_next.unwrap();
    let stop = state.click_regions.footer_stop.unwrap();
    let help = state.click_regions.footer_help.unwrap();
    let segments = footer_segments(&state, DashboardLayout::Full);
    let widths = segments
        .iter()
        .map(|segment| segment.label.chars().count() as u16)
        .collect::<Vec<_>>();
    let starts = spread_item_offsets(100, &widths);

    assert_eq!(previous.x, 0);
    assert!(pause.x > previous.x.saturating_add(previous.width));
    assert!(best.x > pause.x.saturating_add(pause.width));
    assert!(next.x > best.x.saturating_add(best.width));
    assert!(stop.x > next.x.saturating_add(next.width));
    assert!(help.x > stop.x.saturating_add(stop.width));
    assert_eq!(
        starts.last().copied().unwrap() + widths.last().copied().unwrap(),
        100
    );
}

#[test]
fn records_change_points_only_when_best_smarts_changes() {
    let mut state = DashboardState::new(36);
    let first = CandidateSummary {
        smarts: "[#6]".into(),
        mcc: 0.1,
        smarts_len: 4,
        coverage_score: 0.2,
    };
    let second = CandidateSummary {
        smarts: "[#6]-[#8]".into(),
        mcc: 0.4,
        smarts_len: 9,
        coverage_score: 0.5,
    };

    state.record_change_point(1, &first);
    state.record_change_point(2, &first);
    state.record_change_point(3, &second);

    assert_eq!(state.change_points.len(), 2);
    assert_eq!(state.change_points[0].generation, 1);
    assert_eq!(state.change_points[1].generation, 3);
    assert_eq!(state.change_points[1].smarts, "[#6]-[#8]");
}

#[test]
fn hover_on_change_point_shows_smarts_tooltip() {
    let mut state = render_state(sample_state(TuiMetric::Mcc), 100, 30);
    let (x, y) = change_point_marker_position(&state, 3);

    handle_mouse_event(&mut state, mouse_event(MouseEventKind::Moved, x, y), None);

    let hovered = state.hovered_change_point.as_ref().unwrap();
    assert_eq!(hovered.point.generation, 3);
    assert_eq!(hovered.point.smarts, "[#6](=[#8])-[#8]-[#6]-[#6]-[#6]");

    let output = render_to_string(&state, 100, 30);
    assert!(output.contains("SMARTS change"));
    assert!(output.contains(copy_change_point_smarts_label()));
    assert!(output.contains("MCC  0.500"));
    assert!(output.contains("[#6](=[#8])-[#8]-[#6]-[#6]-[#6]"));
}

#[test]
fn smarts_boxes_spread_metrics_and_center_smarts() {
    let state = sample_state(TuiMetric::Mcc);
    let point = &state.change_points[1];
    let header = smarts_box_metric_header(
        Some(point.generation),
        point.mcc,
        point.coverage_score,
        point.smarts_len,
        56,
    );
    let best_header = smarts_box_metric_header(None, 0.5, 0.6, 34, 56);
    let centered_smarts = centered_text_line("[#6]", 20);

    assert_eq!(header.chars().count(), 56);
    assert!(header.starts_with("gen 3"));
    assert!(header.contains("MCC  0.500"));
    assert!(header.contains("cov 0.600"));
    assert!(header.ends_with("len 34"));
    assert_eq!(best_header.chars().count(), 56);
    assert!(best_header.starts_with("MCC  0.500"));
    assert!(best_header.contains("cov 0.600"));
    assert!(best_header.ends_with("len 34"));
    assert_eq!(centered_smarts, "        [#6]");
}

#[test]
fn change_point_tooltip_stays_visible_when_hovered() {
    let mut state = render_state(sample_state(TuiMetric::Mcc), 100, 30);
    let plot_area = state.click_regions.plot_area.unwrap();
    let (x, y) = change_point_marker_position(&state, 3);

    handle_mouse_event(&mut state, mouse_event(MouseEventKind::Moved, x, y), None);
    let tooltip = change_point_tooltip_area(plot_area, x, y).unwrap();
    let tooltip_x = tooltip.x.saturating_add(2).min(tooltip.right() - 1);
    let tooltip_y = tooltip.y.saturating_add(1).min(tooltip.bottom() - 1);

    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Moved, tooltip_x, tooltip_y),
        None,
    );

    let hovered = state.hovered_change_point.as_ref().unwrap();
    assert_eq!(hovered.point.generation, 3);
    let output = render_to_string(&state, 100, 30);
    assert!(output.contains("SMARTS change"));
}

#[test]
fn change_point_tooltip_copy_button_requests_clipboard_copy() {
    let mut state = render_state(sample_state(TuiMetric::Mcc), 100, 30);
    let (x, y) = change_point_marker_position(&state, 3);

    handle_mouse_event(&mut state, mouse_event(MouseEventKind::Moved, x, y), None);
    state = render_state(state, 100, 30);
    let copy = state.click_regions.change_copy.unwrap();
    let (copy_x, copy_y) = copy_button_probe(copy);

    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Moved, copy_x, copy_y),
        None,
    );
    assert_eq!(
        state.hovered_button,
        Some(ButtonId::Copy(CopyControl::ChangePoint))
    );
    let (_, backend) = render_to_backend(state.clone(), 100, 30);
    assert!(
        backend.buffer()[(copy_x, copy_y)]
            .modifier
            .contains(Modifier::UNDERLINED)
    );

    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Down(MouseButton::Left), copy_x, copy_y),
        None,
    );

    assert_eq!(
        state.pending_clipboard.as_deref(),
        Some("[#6](=[#8])-[#8]-[#6]-[#6]-[#6]")
    );
    assert_eq!(
        state.last_event.as_deref(),
        Some("copying change-point SMARTS to clipboard")
    );
    let (_, backend) = render_to_backend(state, 100, 30);
    assert_eq!(backend.buffer()[(copy_x, copy_y)].fg, Color::Green);
}

#[test]
fn metric_tabs_use_equal_width_slots() {
    let state = render_state(sample_state(TuiMetric::Mcc), 100, 30);
    let area = state.click_regions.metric_area.unwrap();
    let slots = metric_tab_slots(area, DashboardLayout::Full);
    assert_eq!(slots.len(), TuiMetric::ALL.len());

    let min_width = slots.iter().map(|(_, _, width)| *width).min().unwrap();
    let max_width = slots.iter().map(|(_, _, width)| *width).max().unwrap();
    assert!(max_width.saturating_sub(min_width) <= 1);

    let first = slots.first().unwrap();
    let last = slots.last().unwrap();
    assert_eq!(first.1, area.x + 1);
    assert_eq!(
        last.1.saturating_add(last.2),
        area.x.saturating_add(area.width).saturating_sub(1)
    );
}

#[test]
fn mcc_and_coverage_y_bounds_support_absolute_and_zoomed_modes() {
    let points = vec![(1.0, 0.4), (2.0, 0.5)];

    assert_eq!(
        y_bounds(TuiMetric::Mcc, &points, YAxisMode::Absolute),
        (-1.0, 1.0)
    );
    let (mcc_min, mcc_max) = y_bounds(TuiMetric::Mcc, &points, YAxisMode::Zoomed);
    assert!(mcc_min > -1.0);
    assert!(mcc_max < 1.0);
    assert!(mcc_min < 0.4 && mcc_max > 0.5);

    assert_eq!(
        y_bounds(TuiMetric::Coverage, &points, YAxisMode::Absolute),
        (0.0, 1.0)
    );
    let (coverage_min, coverage_max) = y_bounds(TuiMetric::Coverage, &points, YAxisMode::Zoomed);
    assert!(coverage_min > 0.0);
    assert!(coverage_max < 1.0);
    assert!(coverage_min < 0.4 && coverage_max > 0.5);
}

#[test]
fn plot_y_axis_mode_button_toggles_and_underlines_on_hover() {
    let mut state = render_state(sample_state(TuiMetric::Mcc), 100, 30);
    let button = state.click_regions.y_axis_mode.unwrap();
    let (button_x, button_y) = copy_button_probe(button);
    assert_eq!(state.y_axis_mode, YAxisMode::Zoomed);

    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Moved, button_x, button_y),
        None,
    );
    assert_eq!(state.hovered_button, Some(ButtonId::YAxisMode));
    let (_, backend) = render_to_backend(state.clone(), 100, 30);
    assert!(
        backend.buffer()[(button_x, button_y)]
            .modifier
            .contains(Modifier::UNDERLINED)
    );

    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Down(MouseButton::Left), button_x, button_y),
        None,
    );
    assert_eq!(state.y_axis_mode, YAxisMode::Absolute);
    let (_, backend) = render_to_backend(state.clone(), 100, 30);
    assert_eq!(backend.buffer()[(button_x, button_y)].fg, Color::Green);
    let output = render_to_string(&state, 100, 30);
    assert!(output.contains("[z: abs]"));
}

#[test]
fn plot_y_axis_mode_button_is_only_for_mcc_and_coverage() {
    let mut state = sample_state(TuiMetric::Unique);
    state = render_state(state, 100, 30);
    assert!(state.click_regions.y_axis_mode.is_none());

    toggle_y_axis_mode(&mut state);
    assert_eq!(state.y_axis_mode, YAxisMode::Zoomed);
}

fn change_point_marker_position(state: &DashboardState, generation: u64) -> (u16, u16) {
    let area = state.click_regions.plot_area.unwrap();
    let layout = state.click_regions.layout.unwrap();
    let metric = state.selected_metric;
    let points = metric_points(metric, &state.history);
    let (x_min, x_max) = x_bounds(&points);
    let (y_min, y_max) = y_bounds(metric, &points, state.y_axis_mode);
    let graph_area = chart_graph_area(area, layout, x_min, y_min, y_max).unwrap();
    let history = state
        .history
        .iter()
        .find(|point| point.generation == generation)
        .unwrap();
    data_point_screen_position(
        graph_area,
        x_min,
        x_max,
        y_min,
        y_max,
        generation as f64,
        metric_value(metric, history),
    )
    .unwrap()
}

#[test]
fn mouse_click_selects_metric_tab() {
    let mut state = render_state(sample_state(TuiMetric::Mcc), 100, 30);
    let area = state.click_regions.metric_area.unwrap();
    let (_, start, _) = metric_hit_spans(&state, area)
        .into_iter()
        .find(|(metric, _, _)| *metric == TuiMetric::Coverage)
        .unwrap();

    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Down(MouseButton::Left), start, area.y + 1),
        None,
    );

    assert_eq!(state.selected_metric, TuiMetric::Coverage);
}

#[test]
fn footer_metric_buttons_switch_metric() {
    let mut state = render_state(sample_state(TuiMetric::Mcc), 100, 30);
    let next = state.click_regions.footer_next.unwrap();

    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Moved, next.x, next.y),
        None,
    );
    assert_eq!(
        state.hovered_button,
        Some(ButtonId::Footer(FooterControl::NextMetric))
    );
    let (_, backend) = render_to_backend(state.clone(), 100, 30);
    assert!(
        backend.buffer()[(next.x, next.y)]
            .modifier
            .contains(Modifier::UNDERLINED)
    );

    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Down(MouseButton::Left), next.x, next.y),
        None,
    );
    assert_eq!(state.selected_metric, TuiMetric::Coverage);
    let (_, backend) = render_to_backend(state.clone(), 100, 30);
    assert_eq!(backend.buffer()[(next.x, next.y)].fg, Color::Green);

    state = render_state(state, 100, 30);
    let previous = state.click_regions.footer_prev.unwrap();
    handle_mouse_event(
        &mut state,
        mouse_event(
            MouseEventKind::Down(MouseButton::Left),
            previous.x,
            previous.y,
        ),
        None,
    );
    assert_eq!(state.selected_metric, TuiMetric::Mcc);
}

#[test]
fn footer_pause_button_sends_pause_and_play_button_sends_resume() {
    let mut state = render_state(sample_state(TuiMetric::Mcc), 100, 30);
    let pause = state.click_regions.footer_pause.unwrap();
    let (control_tx, control_rx) = mpsc::channel();

    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Down(MouseButton::Left), pause.x, pause.y),
        Some(&control_tx),
    );
    assert_eq!(control_rx.try_recv(), Ok(WorkerControl::Pause));
    assert_eq!(state.status, DashboardStatus::Pausing);
    assert_eq!(pause_button_label(&state), "[p … pausing]");

    state = render_state(state, 100, 30);
    let play = state.click_regions.footer_pause.unwrap();
    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Down(MouseButton::Left), play.x, play.y),
        Some(&control_tx),
    );
    assert_eq!(control_rx.try_recv(), Ok(WorkerControl::Resume));
    assert_eq!(state.status, DashboardStatus::Running);
    assert_eq!(pause_button_label(&state), "[p ▮▮ pause]");
}

#[test]
fn footer_stop_and_help_buttons_are_clickable() {
    let mut state = render_state(sample_state(TuiMetric::Mcc), 100, 30);
    let stop = state.click_regions.footer_stop.unwrap();
    let help = state.click_regions.footer_help.unwrap();
    let (control_tx, control_rx) = mpsc::channel();

    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Down(MouseButton::Left), stop.x, stop.y),
        Some(&control_tx),
    );
    assert_eq!(control_rx.try_recv(), Ok(WorkerControl::Stop));
    assert_eq!(state.status, DashboardStatus::Stopping);
    assert_eq!(
        state.last_event.as_deref(),
        Some("stop requested; waiting for generation boundary")
    );
    let (_, backend) = render_to_backend(state.clone(), 100, 30);
    assert_eq!(backend.buffer()[(stop.x, stop.y)].fg, Color::Green);

    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Down(MouseButton::Left), help.x, help.y),
        None,
    );
    assert!(state.show_help);
}

#[test]
fn footer_best_button_opens_best_overlay() {
    let mut state = render_state(sample_state(TuiMetric::Mcc), 100, 30);
    let best = state.click_regions.footer_best.unwrap();

    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Down(MouseButton::Left), best.x, best.y),
        None,
    );

    assert!(state.show_best_smarts);
}

#[test]
fn worker_controls_emit_pause_resume_and_stop_events() {
    let (control_tx, control_rx) = mpsc::channel();
    let (event_tx, event_rx) = mpsc::channel();
    let mut paused = false;

    control_tx.send(WorkerControl::Pause).unwrap();
    assert!(!process_worker_controls(
        &mut paused,
        &control_rx,
        &event_tx
    ));
    assert!(paused);
    assert!(matches!(event_rx.try_recv(), Ok(WorkerEvent::Paused)));

    control_tx.send(WorkerControl::Resume).unwrap();
    assert!(!process_worker_controls(
        &mut paused,
        &control_rx,
        &event_tx
    ));
    assert!(!paused);
    assert!(matches!(event_rx.try_recv(), Ok(WorkerEvent::Resumed)));

    control_tx.send(WorkerControl::Stop).unwrap();
    assert!(process_worker_controls(&mut paused, &control_rx, &event_tx));
    assert!(matches!(event_rx.try_recv(), Ok(WorkerEvent::Stopped)));
}

#[test]
fn mouse_hover_underlines_metric_tab() {
    let mut state = render_state(sample_state(TuiMetric::Mcc), 100, 30);
    let area = state.click_regions.metric_area.unwrap();
    let (_, start, _) = metric_hit_spans(&state, area)
        .into_iter()
        .find(|(metric, _, _)| *metric == TuiMetric::Coverage)
        .unwrap();

    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Moved, start, area.y + 1),
        None,
    );
    assert_eq!(state.hovered_metric, Some(TuiMetric::Coverage));

    let (_, backend) = render_to_backend(state, 100, 30);
    assert!(
        backend.buffer()[(start, area.y + 1)]
            .modifier
            .contains(Modifier::UNDERLINED)
    );
}

#[test]
fn mouse_hover_outside_metric_tabs_clears_hover() {
    let mut state = render_state(sample_state(TuiMetric::Mcc), 100, 30);
    let area = state.click_regions.metric_area.unwrap();
    let (_, start, _) = metric_hit_spans(&state, area)
        .into_iter()
        .find(|(metric, _, _)| *metric == TuiMetric::Coverage)
        .unwrap();

    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Moved, start, area.y + 1),
        None,
    );
    handle_mouse_event(&mut state, mouse_event(MouseEventKind::Moved, 0, 0), None);

    assert_eq!(state.hovered_metric, None);
}

#[test]
fn mouse_scroll_over_plot_switches_metric() {
    let mut state = render_state(sample_state(TuiMetric::Mcc), 100, 30);
    let plot = state.click_regions.plot_area.unwrap();

    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::ScrollDown, plot.x + 1, plot.y + 1),
        None,
    );
    assert_eq!(state.selected_metric, TuiMetric::Coverage);

    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::ScrollUp, plot.x + 1, plot.y + 1),
        None,
    );
    assert_eq!(state.selected_metric, TuiMetric::Mcc);
}

#[test]
fn mouse_click_best_area_opens_and_outside_click_closes_overlay() {
    let mut state = render_state(sample_state(TuiMetric::Mcc), 100, 30);
    let best_area = state.click_regions.best_area.unwrap();

    handle_mouse_event(
        &mut state,
        mouse_event(
            MouseEventKind::Down(MouseButton::Left),
            best_area.x + 1,
            best_area.y,
        ),
        None,
    );
    assert!(state.show_best_smarts);

    state = render_state(state, 100, 30);
    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Down(MouseButton::Left), 0, 0),
        None,
    );
    assert!(!state.show_best_smarts);
}

#[test]
fn best_overlay_copy_button_requests_clipboard_copy() {
    let mut state = sample_state(TuiMetric::Mcc);
    state.show_best_smarts = true;
    state = render_state(state, 100, 30);
    let overlay = state.click_regions.best_overlay.unwrap();
    let copy = state.click_regions.best_copy.unwrap();
    assert_eq!(copy.y, overlay.y);
    assert_eq!(
        copy.x.saturating_add(copy.width),
        overlay.x.saturating_add(overlay.width).saturating_sub(1)
    );
    let (copy_x, copy_y) = copy_button_probe(copy);

    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Moved, copy_x, copy_y),
        None,
    );
    assert_eq!(
        state.hovered_button,
        Some(ButtonId::Copy(CopyControl::BestSmarts))
    );
    let (_, backend) = render_to_backend(state.clone(), 100, 30);
    assert!(
        backend.buffer()[(copy_x, copy_y)]
            .modifier
            .contains(Modifier::UNDERLINED)
    );

    handle_mouse_event(
        &mut state,
        mouse_event(MouseEventKind::Down(MouseButton::Left), copy_x, copy_y),
        None,
    );

    assert_eq!(
        state.pending_clipboard.as_deref(),
        Some("[#6](=[#8])-[#8]-[#6]-[#6]-[#6]")
    );
    assert_eq!(
        state.last_event.as_deref(),
        Some("copying best SMARTS to clipboard")
    );
    assert!(state.show_best_smarts);
    let (_, backend) = render_to_backend(state, 100, 30);
    assert_eq!(backend.buffer()[(copy_x, copy_y)].fg, Color::Green);
}

#[test]
fn osc52_clipboard_sequence_encodes_payload() {
    assert_eq!(osc52_clipboard_sequence("C=O"), "\x1b]52;c;Qz1P\x07");
    let mut output = Vec::new();
    write_clipboard_sequence(&mut output, "C=O").unwrap();
    assert_eq!(String::from_utf8(output).unwrap(), "\x1b]52;c;Qz1P\x07");
}

#[test]
fn header_fit_uses_width_dependent_line_count() {
    let state = sample_state(TuiMetric::Mcc);
    assert_eq!(HeaderFit::for_width(150, &state), HeaderFit::OneLine);
    assert_eq!(HeaderFit::for_width(80, &state), HeaderFit::TwoLine);
}

#[test]
fn header_phase_labels_are_space_padded() {
    let mut state = sample_state(TuiMetric::Mcc);
    state.phase = DashboardPhase::Evaluating;
    let evaluating_suffix = one_line_header_suffix(&state);

    state.phase = DashboardPhase::Mutating;
    let mutating_suffix = one_line_header_suffix(&state);

    assert!(evaluating_suffix.contains(&padded_phase_label(
        DashboardPhase::Evaluating.label(),
        FULL_PHASE_LABEL_WIDTH
    )));
    assert!(mutating_suffix.contains(&padded_phase_label(
        DashboardPhase::Mutating.label(),
        FULL_PHASE_LABEL_WIDTH
    )));
    assert_eq!(
        evaluating_suffix.chars().count(),
        mutating_suffix.chars().count()
    );
    assert_eq!(
        header_phase_label(DashboardPhase::Evaluating, DashboardLayout::Compact)
            .chars()
            .count(),
        header_phase_label(DashboardPhase::Mutating, DashboardLayout::Compact)
            .chars()
            .count()
    );
}

#[test]
fn header_numbers_are_space_padded() {
    let mut state = sample_state(TuiMetric::Mcc);
    state.generation = 3;
    state.generation_limit = 800;
    state.phase_completed = 7;
    state.phase_total = 2048;
    let short_suffix = one_line_header_suffix(&state);

    state.generation = 33;
    state.phase_completed = 77;
    let longer_suffix = one_line_header_suffix(&state);

    assert!(short_suffix.contains("gen   3/800"));
    assert!(short_suffix.contains(&format!(
        "{}    7/2048",
        padded_phase_label(DashboardPhase::Evaluating.label(), FULL_PHASE_LABEL_WIDTH)
    )));
    assert!(short_suffix.contains("best MCC  0.500"));
    assert_eq!(short_suffix.chars().count(), longer_suffix.chars().count());
}

#[test]
fn status_mcc_values_are_sign_padded() {
    let mut negative = sample_state(TuiMetric::Mcc);
    negative.current_mcc = Some(-1.0);
    negative.generation_best_mcc = Some(-0.25);

    let mut positive = negative.clone();
    positive.current_mcc = Some(0.5);
    positive.generation_best_mcc = Some(0.25);

    let mut timed_out = negative.clone();
    timed_out.current_limit_exceeded = true;

    let negative_line = status_metric_line(&negative, DashboardLayout::Compact, 92);
    let positive_line = status_metric_line(&positive, DashboardLayout::Compact, 92);
    let timed_out_line = status_metric_line(&timed_out, DashboardLayout::Compact, 92);

    assert!(negative_line.contains("cur -1.000"));
    assert!(negative_line.contains("gen -0.250"));
    assert!(positive_line.contains("cur  0.500"));
    assert!(positive_line.contains("gen  0.250"));
    assert!(timed_out_line.contains("cur    OOT"));
    assert!(timed_out_line.contains("gen -0.250"));
    assert!(timed_out_line.contains("thr  8"));
    assert!(timed_out_line.contains("rss 512.0MiB"));
    assert_eq!(negative_line.chars().count(), 92);
    assert_eq!(negative_line.chars().count(), positive_line.chars().count());
    assert_eq!(
        negative_line.chars().count(),
        timed_out_line.chars().count()
    );
}

#[test]
fn formats_runtime_memory_estimates() {
    assert_eq!(fmt_memory_bytes(512), "512B");
    assert_eq!(fmt_memory_bytes(1536), "1.5KiB");
    assert_eq!(fmt_memory_bytes(512 * 1024 * 1024), "512.0MiB");
    assert_eq!(fmt_memory_bytes(1024 * 1024 * 1024), "1.0GiB");
    assert_eq!(parse_vm_rss_line("VmRSS:\t  12345 kB"), Some(12_641_280));
    assert_eq!(parse_vm_rss_line("VmSize:\t  12345 kB"), None);
}

#[test]
fn renders_minimal_layout() {
    let output = render_to_string(&sample_state(TuiMetric::Mcc), 60, 18);
    assert!(output.contains("Metric [◆ MCC]"));
    assert!(output.contains("[l ←]"));
    assert!(!output.contains("terminal too small"));
}

#[test]
fn truncates_long_smarts() {
    let long = "[#6]-[#6]-[#6]-[#6]-[#6]-[#6]-[#6]-[#6]-[#6]-[#6]";
    let truncated = truncate_smarts(long, 16);
    assert_eq!(truncated, "[#6]-[#6]-[#6...");
}

#[test]
fn renders_full_best_smarts_overlay() {
    let mut state = sample_state(TuiMetric::Mcc);
    state.show_best_smarts = true;
    let output = render_to_string(&state, 100, 30);
    assert!(output.contains("Best SMARTS"));
    assert!(output.contains(copy_best_smarts_label()));
    assert!(output.contains("[#6](=[#8])-[#8]-[#6]-[#6]-[#6]"));
}

#[test]
fn clamps_phase_progress_ratio_inputs() {
    assert_eq!(ratio(10, 0), 10.0);
    let mut state = sample_state(TuiMetric::Mcc);
    state.phase_completed = 10;
    state.phase_total = 0;
    let output = render_to_string(&state, 100, 30);
    assert!(output.contains("10/1"));
}

#[test]
fn metric_navigation_wraps() {
    assert_eq!(TuiMetric::Mcc.previous(), TuiMetric::Length);
    assert_eq!(TuiMetric::Length.next(), TuiMetric::Mcc);
}
