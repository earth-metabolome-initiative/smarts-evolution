use alloc::format;
use alloc::string::String;
use core::time::Duration;
use std::io::{self, Write};
use std::sync::mpsc::Sender;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use ratatui::crossterm::event::{
    self, Event, KeyCode, KeyEventKind, MouseButton, MouseEvent, MouseEventKind,
};

use super::TuiEvolutionError;
use super::layout::{DashboardLayout, contains_position};
use super::metric::TuiMetric;
use super::render::{
    change_point_tooltip_area, chart_graph_area, data_point_screen_position, metric_hit_spans,
    metric_points, metric_value, stop_button_enabled, x_bounds, y_bounds,
};
use super::state::*;
use super::worker::WorkerControl;

pub(super) fn handle_terminal_input(
    state: &mut DashboardState,
    control_tx: &Sender<WorkerControl>,
    stop_requested: &mut bool,
) -> Result<(), TuiEvolutionError> {
    while event::poll(Duration::ZERO)? {
        match event::read()? {
            Event::Key(key) => {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Left | KeyCode::Char('l') => {
                        state.select_previous_metric();
                    }
                    KeyCode::Right | KeyCode::Char('r') => {
                        state.select_next_metric();
                    }
                    KeyCode::Enter | KeyCode::Char('b') if state.best.is_some() => {
                        state.toggle_best_smarts_overlay();
                    }
                    KeyCode::Char('p') => toggle_pause(state, Some(control_tx)),
                    KeyCode::Char('z') => toggle_y_axis_mode(state),
                    KeyCode::Char('?') => toggle_help(state),
                    KeyCode::Esc => {
                        state.close_overlays();
                    }
                    KeyCode::Char('q') if !*stop_requested => {
                        request_stop(state, Some(control_tx));
                        if state.status == DashboardStatus::Stopping {
                            *stop_requested = true;
                        }
                    }
                    _ => {}
                }
            }
            Event::Mouse(mouse) => handle_mouse_event(state, mouse, Some(control_tx)),
            _ => {}
        }
    }
    Ok(())
}

pub(super) fn handle_mouse_event(
    state: &mut DashboardState,
    mouse: MouseEvent,
    control_tx: Option<&Sender<WorkerControl>>,
) {
    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            update_hover_state(state, mouse.column, mouse.row);
            handle_left_click(state, mouse.column, mouse.row, control_tx)
        }
        MouseEventKind::Moved | MouseEventKind::Drag(_) => {
            update_hover_state(state, mouse.column, mouse.row)
        }
        MouseEventKind::ScrollUp => handle_metric_scroll(state, mouse.column, mouse.row, -1),
        MouseEventKind::ScrollDown => handle_metric_scroll(state, mouse.column, mouse.row, 1),
        _ => {}
    }
}

pub(super) fn update_hover_state(state: &mut DashboardState, column: u16, row: u16) {
    if state.show_help {
        state.clear_hover();
        return;
    }
    if state.show_best_smarts {
        state.clear_hover();
        state.hovered_button = copy_control_at_position(state, column, row).map(ButtonId::Copy);
        return;
    }
    state.hovered_button = button_at_position(state, column, row);
    state.hovered_metric = metric_at_position(state, column, row);
    if hovered_change_point_tooltip_contains(state, column, row) {
        return;
    }
    state.hovered_change_point = change_point_at_position(state, column, row);
}

pub(super) fn hovered_change_point_tooltip_contains(
    state: &DashboardState,
    column: u16,
    row: u16,
) -> bool {
    let Some(hovered) = state.hovered_change_point.as_ref() else {
        return false;
    };
    let Some(plot_area) = state.click_regions.plot_area else {
        return false;
    };
    ClickRegions::contains(
        change_point_tooltip_area(plot_area, hovered.marker_x, hovered.marker_y),
        column,
        row,
    )
}

pub(super) fn copy_control_at_position(
    state: &DashboardState,
    column: u16,
    row: u16,
) -> Option<CopyControl> {
    CopyControl::ALL.into_iter().find(|control| {
        ClickRegions::contains(
            state.click_regions.button(ButtonId::Copy(*control)),
            column,
            row,
        )
    })
}

pub(super) fn button_at_position(
    state: &DashboardState,
    column: u16,
    row: u16,
) -> Option<ButtonId> {
    ButtonId::ALL
        .into_iter()
        .find(|button| ClickRegions::contains(state.click_regions.button(*button), column, row))
}

pub(super) fn mark_button_pressed(state: &mut DashboardState, button: ButtonId) {
    state.button_feedback = Some(ButtonFeedback::new(button));
}

pub(super) fn handle_button_click(
    state: &mut DashboardState,
    button: ButtonId,
    control_tx: Option<&Sender<WorkerControl>>,
) {
    match button {
        ButtonId::Copy(CopyControl::BestSmarts) => request_best_smarts_copy(state),
        ButtonId::Copy(CopyControl::ChangePoint) => request_change_point_smarts_copy(state),
        ButtonId::YAxisMode => toggle_y_axis_mode(state),
        ButtonId::Footer(control) => handle_footer_control(state, control, control_tx),
    }
}

pub(super) fn handle_left_click(
    state: &mut DashboardState,
    column: u16,
    row: u16,
    control_tx: Option<&Sender<WorkerControl>>,
) {
    if state.show_help {
        if !ClickRegions::contains(state.click_regions.help_overlay, column, row) {
            state.show_help = false;
        }
        return;
    }

    if state.show_best_smarts {
        if ClickRegions::contains(state.click_regions.best_copy, column, row) {
            request_best_smarts_copy(state);
            return;
        }
        if !ClickRegions::contains(state.click_regions.best_overlay, column, row) {
            state.show_best_smarts = false;
        }
        return;
    }

    if let Some(button) = button_at_position(state, column, row) {
        handle_button_click(state, button, control_tx);
        return;
    }

    if let Some(metric) = metric_at_position(state, column, row) {
        state.select_metric(metric);
        return;
    }

    if state.best.is_some() && ClickRegions::contains(state.click_regions.best_area, column, row) {
        state.show_best_smarts_overlay();
    }
}

pub(super) fn request_best_smarts_copy(state: &mut DashboardState) {
    if let Some(best) = state.best.as_ref() {
        state.pending_clipboard = Some(best.smarts.clone());
        mark_button_pressed(state, ButtonId::Copy(CopyControl::BestSmarts));
        state.last_event = Some("copying best SMARTS to clipboard".into());
    }
}

pub(super) fn request_change_point_smarts_copy(state: &mut DashboardState) {
    if let Some(hovered) = state.hovered_change_point.as_ref() {
        state.pending_clipboard = Some(hovered.point.smarts.clone());
        mark_button_pressed(state, ButtonId::Copy(CopyControl::ChangePoint));
        state.last_event = Some("copying change-point SMARTS to clipboard".into());
    }
}

pub(super) fn toggle_y_axis_mode(state: &mut DashboardState) {
    if state.selected_metric.supports_y_axis_mode() {
        state.y_axis_mode = state.y_axis_mode.toggle();
        state.hovered_change_point = None;
        mark_button_pressed(state, ButtonId::YAxisMode);
    }
}

pub(super) fn copy_pending_clipboard_request(state: &mut DashboardState) {
    copy_pending_clipboard_request_with(state, copy_text_to_clipboard);
}

pub(super) fn copy_pending_clipboard_request_with(
    state: &mut DashboardState,
    mut copy_text: impl FnMut(&str) -> io::Result<()>,
) {
    let Some(text) = state.pending_clipboard.take() else {
        return;
    };
    match copy_text(&text) {
        Ok(()) => state.last_event = Some("copied SMARTS to clipboard".into()),
        Err(error) => state.last_event = Some(format!("clipboard copy failed: {error}")),
    }
}

pub(super) fn copy_text_to_clipboard(text: &str) -> io::Result<()> {
    let mut stdout = io::stdout();
    write_clipboard_sequence(&mut stdout, text)
}

pub(super) fn write_clipboard_sequence(writer: &mut impl Write, text: &str) -> io::Result<()> {
    let sequence = osc52_clipboard_sequence(text);
    writer.write_all(sequence.as_bytes())?;
    writer.flush()
}

pub(super) fn osc52_clipboard_sequence(text: &str) -> String {
    format!("\x1b]52;c;{}\x07", BASE64_STANDARD.encode(text.as_bytes()))
}

pub(super) fn handle_footer_control(
    state: &mut DashboardState,
    control: FooterControl,
    control_tx: Option<&Sender<WorkerControl>>,
) {
    match control {
        FooterControl::PreviousMetric => {
            mark_button_pressed(state, ButtonId::Footer(control));
            state.select_previous_metric();
        }
        FooterControl::NextMetric => {
            mark_button_pressed(state, ButtonId::Footer(control));
            state.select_next_metric();
        }
        FooterControl::PauseResume => toggle_pause(state, control_tx),
        FooterControl::BestSmarts if state.best.is_some() => {
            mark_button_pressed(state, ButtonId::Footer(control));
            state.show_best_smarts_overlay();
        }
        FooterControl::BestSmarts => {}
        FooterControl::Stop => request_stop(state, control_tx),
        FooterControl::Help => toggle_help(state),
    }
}

pub(super) fn request_stop(state: &mut DashboardState, control_tx: Option<&Sender<WorkerControl>>) {
    if !stop_button_enabled(state) {
        return;
    }
    mark_button_pressed(state, ButtonId::Footer(FooterControl::Stop));
    if let Some(control_tx) = control_tx {
        let _ = control_tx.send(WorkerControl::Stop);
    }
    state.status = DashboardStatus::Stopping;
    state.phase = DashboardPhase::Stopping;
    state.last_event = Some("stop requested; waiting for generation boundary".into());
}

pub(super) fn toggle_help(state: &mut DashboardState) {
    mark_button_pressed(state, ButtonId::Footer(FooterControl::Help));
    state.show_help = !state.show_help;
}

pub(super) fn toggle_pause(state: &mut DashboardState, control_tx: Option<&Sender<WorkerControl>>) {
    match state.status {
        DashboardStatus::Pausing | DashboardStatus::Paused => {
            mark_button_pressed(state, ButtonId::Footer(FooterControl::PauseResume));
            if let Some(control_tx) = control_tx {
                let _ = control_tx.send(WorkerControl::Resume);
            }
            state.status = DashboardStatus::Running;
            state.phase = DashboardPhase::Startup;
            state.last_event = Some("resume requested".into());
        }
        DashboardStatus::Waiting | DashboardStatus::Running => {
            mark_button_pressed(state, ButtonId::Footer(FooterControl::PauseResume));
            if let Some(control_tx) = control_tx {
                let _ = control_tx.send(WorkerControl::Pause);
            }
            state.status = DashboardStatus::Pausing;
            state.phase = DashboardPhase::Pausing;
            state.last_event = Some("pause requested; waiting for generation boundary".into());
        }
        DashboardStatus::Stopping
        | DashboardStatus::Completed
        | DashboardStatus::Stopped
        | DashboardStatus::Failed => {}
    }
}

pub(super) fn handle_metric_scroll(
    state: &mut DashboardState,
    column: u16,
    row: u16,
    direction: i8,
) {
    let regions = state.click_regions;
    let over_metric = ClickRegions::contains(regions.metric_area, column, row);
    let over_plot = ClickRegions::contains(regions.plot_area, column, row);
    if !over_metric && !over_plot {
        return;
    }
    if direction < 0 {
        state.select_previous_metric();
    } else {
        state.select_next_metric();
    }
}

pub(super) fn metric_at_position(
    state: &DashboardState,
    column: u16,
    row: u16,
) -> Option<TuiMetric> {
    let area = state.click_regions.metric_area?;
    if !contains_position(area, column, row) {
        return None;
    }
    metric_hit_spans(state, area)
        .into_iter()
        .find_map(|(metric, start, end)| (column >= start && column < end).then_some(metric))
}

pub(super) fn change_point_at_position(
    state: &DashboardState,
    column: u16,
    row: u16,
) -> Option<HoveredChangePoint> {
    let area = state.click_regions.plot_area?;
    let layout = state.click_regions.layout.unwrap_or(DashboardLayout::Full);
    if layout == DashboardLayout::Minimal
        || area.height < 8
        || !contains_position(area, column, row)
    {
        return None;
    }
    let metric = state.selected_metric;
    let points = metric_points(metric, &state.history);
    if points.is_empty() {
        return None;
    }
    let (x_min, x_max) = x_bounds(&points);
    let (y_min, y_max) = y_bounds(metric, &points, state.y_axis_mode);
    let graph_area = chart_graph_area(area, layout, x_min, y_min, y_max)?;
    state
        .change_points
        .iter()
        .filter_map(|point| {
            let history = state
                .history
                .iter()
                .find(|history| history.generation == point.generation)?;
            let (marker_x, marker_y) = data_point_screen_position(
                graph_area,
                x_min,
                x_max,
                y_min,
                y_max,
                point.generation as f64,
                metric_value(metric, history),
            )?;
            let dx = column.abs_diff(marker_x);
            let dy = row.abs_diff(marker_y);
            (dx <= 1 && dy <= 1).then_some(HoveredChangePoint {
                point: point.clone(),
                marker_x,
                marker_y,
            })
        })
        .min_by_key(|hovered| column.abs_diff(hovered.marker_x) + row.abs_diff(hovered.marker_y))
}
