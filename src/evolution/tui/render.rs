use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::time::Duration;

use ratatui::prelude::*;
use ratatui::symbols;
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Clear, Dataset, Gauge, GraphType, Paragraph, Sparkline, Wrap,
};

use super::layout::{DashboardLayout, contains_position};
use super::metric::{TuiMetric, YAxisMode};
use super::state::*;

const MIN_WIDTH: u16 = 60;
const MIN_HEIGHT: u16 = 18;
const MIN_ONE_LINE_HEADER_TASK_WIDTH: usize = 16;
pub(super) const FULL_PHASE_LABEL_WIDTH: usize = 19;
const SHORT_PHASE_LABEL_WIDTH: usize = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum HeaderFit {
    OneLine,
    TwoLine,
}

impl HeaderFit {
    pub(super) fn for_width(width: u16, state: &DashboardState) -> Self {
        let inner_width = block_inner_width(width);
        let suffix_width = one_line_header_suffix(state).chars().count();
        if inner_width >= suffix_width + MIN_ONE_LINE_HEADER_TASK_WIDTH {
            Self::OneLine
        } else {
            Self::TwoLine
        }
    }

    pub(super) const fn height(self) -> u16 {
        match self {
            Self::OneLine => 3,
            Self::TwoLine => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct HeaderWidths {
    generation: usize,
    phase: usize,
    stagnation: usize,
}

impl HeaderWidths {
    pub(super) fn for_state(state: &DashboardState) -> Self {
        let stagnation = state
            .history
            .last()
            .map(|point| point.stagnation)
            .unwrap_or(0);
        Self {
            generation: decimal_width(state.generation_limit.max(state.generation)),
            phase: decimal_width(state.phase_total.max(state.phase_completed).max(1)),
            stagnation: decimal_width(state.generation_limit.max(stagnation)),
        }
    }
}
pub(super) fn metric_hit_spans(state: &DashboardState, area: Rect) -> Vec<(TuiMetric, u16, u16)> {
    let layout = state.click_regions.layout.unwrap_or(DashboardLayout::Full);
    metric_tab_slots(area, layout)
        .into_iter()
        .filter_map(|(metric, slot_start, slot_width)| {
            let label = fitted_metric_tab_label(state, metric, layout, slot_width);
            let label_width = label.chars().count() as u16;
            let start = slot_start.saturating_add(slot_width.saturating_sub(label_width) / 2);
            let end = start.saturating_add(label_width);
            (end > start).then_some((metric, start, end))
        })
        .collect()
}

pub(super) fn metric_labels_start(area: Rect, layout: DashboardLayout) -> u16 {
    match layout {
        DashboardLayout::Minimal => area.x.saturating_add("Metric ".len() as u16),
        DashboardLayout::Full | DashboardLayout::Compact => area.x.saturating_add(1),
    }
}

pub(super) fn metric_labels_end(area: Rect, layout: DashboardLayout) -> u16 {
    match layout {
        DashboardLayout::Minimal => area.x.saturating_add(area.width),
        DashboardLayout::Full | DashboardLayout::Compact => {
            area.x.saturating_add(area.width).saturating_sub(1)
        }
    }
}

pub(super) fn metric_tab_slots(area: Rect, layout: DashboardLayout) -> Vec<(TuiMetric, u16, u16)> {
    let start = metric_labels_start(area, layout);
    let end = metric_labels_end(area, layout);
    let total_width = end.saturating_sub(start);
    let separator_count = metric_tab_separator_count(total_width);
    let slots_width = total_width.saturating_sub(separator_count);
    let metric_count = TuiMetric::ALL.len() as u16;
    let base_width = slots_width / metric_count;
    let remainder = slots_width % metric_count;
    let mut cursor = start;
    let mut slots = Vec::with_capacity(TuiMetric::ALL.len());
    for (index, metric) in TuiMetric::ALL.iter().copied().enumerate() {
        let width = base_width + u16::from((index as u16) < remainder);
        slots.push((metric, cursor, width));
        cursor = cursor.saturating_add(width);
        if index + 1 < TuiMetric::ALL.len() {
            cursor = cursor.saturating_add(u16::from(separator_count > 0));
        }
    }
    slots
}

pub(super) fn metric_tab_separator_count(total_width: u16) -> u16 {
    let metric_count = TuiMetric::ALL.len() as u16;
    if total_width >= metric_count.saturating_mul(2).saturating_sub(1) {
        metric_count.saturating_sub(1)
    } else {
        0
    }
}

pub(super) fn render_dashboard(frame: &mut Frame<'_>, state: &mut DashboardState) {
    let area = frame.area();
    state.click_regions = ClickRegions::default();
    state.runtime_metrics = RuntimeMetrics::current();
    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        render_too_small(frame, area);
        return;
    }

    let layout = DashboardLayout::for_area(area);
    let header_fit = HeaderFit::for_width(area.width, state);
    let header_height = header_fit.height();
    let constraints = match layout {
        DashboardLayout::Full => [
            Constraint::Length(header_height),
            Constraint::Length(3),
            Constraint::Min(7),
            Constraint::Length(4),
            Constraint::Length(1),
        ],
        DashboardLayout::Compact => [
            Constraint::Length(header_height),
            Constraint::Length(3),
            Constraint::Min(6),
            Constraint::Length(4),
            Constraint::Length(1),
        ],
        DashboardLayout::Minimal => [
            Constraint::Length(header_height),
            Constraint::Length(1),
            Constraint::Min(4),
            Constraint::Length(4),
            Constraint::Length(1),
        ],
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    state.click_regions.layout = Some(layout);
    state.click_regions.metric_area = Some(chunks[1]);
    state.click_regions.plot_area = Some(chunks[2]);
    render_header(frame, chunks[0], state, layout, header_fit);
    if layout == DashboardLayout::Minimal {
        render_metric_line(frame, chunks[1], state);
    } else {
        render_metric_tabs(frame, chunks[1], state);
    }
    if layout == DashboardLayout::Minimal || chunks[2].height < 8 {
        render_sparkline_plot(frame, chunks[2], state);
    } else {
        render_plot(frame, chunks[2], state, layout);
    }
    render_status(frame, chunks[3], state, layout);
    render_footer(frame, chunks[4], state, layout);

    if state.show_help {
        let overlay = centered_rect(64, 9, area);
        state.click_regions.help_overlay = Some(overlay);
        render_help(frame, overlay);
    } else if state.show_best_smarts {
        let overlay = centered_rect(74, 12, area);
        state.click_regions.best_overlay = Some(overlay);
        render_best_smarts(frame, overlay, state);
    }
}

pub(super) fn render_too_small(frame: &mut Frame<'_>, area: Rect) {
    let text = Paragraph::new("terminal too small for smarts-evolution TUI")
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("smarts-evolution"),
        );
    frame.render_widget(text, area);
}

pub(super) fn render_header(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &DashboardState,
    layout: DashboardLayout,
    fit: HeaderFit,
) {
    let title = format!(" smarts-evolution {} ", state.status.label());
    let lines = match fit {
        HeaderFit::OneLine => vec![Line::from(one_line_header(area, state))],
        HeaderFit::TwoLine => two_line_header(area, state, layout),
    };
    let paragraph =
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(title));
    frame.render_widget(paragraph, area);
}

pub(super) fn one_line_header(area: Rect, state: &DashboardState) -> String {
    let width = block_inner_width(area.width);
    distributed_header_line(&state.task_id, one_line_header_fields(state), width)
}

pub(super) fn one_line_header_suffix(state: &DashboardState) -> String {
    let widths = HeaderWidths::for_state(state);
    let stagnation = state
        .history
        .last()
        .map(|point| point.stagnation)
        .unwrap_or(0);
    format!(
        " gen {} {} {} best MCC {} cov {} no-imp {} ETA {} elapsed {}",
        padded_pair(state.generation, state.generation_limit, widths.generation),
        padded_phase_label(state.phase.label(), FULL_PHASE_LABEL_WIDTH),
        padded_pair(
            state.phase_completed,
            state.phase_total.max(1),
            widths.phase
        ),
        fmt_padded_opt(state.best_mcc(), 6),
        fmt_padded_opt(state.best_coverage(), 5),
        padded_value(stagnation, widths.stagnation),
        padded_eta_text(state),
        fmt_padded_duration(state.elapsed())
    )
}

pub(super) fn one_line_header_fields(state: &DashboardState) -> Vec<String> {
    let widths = HeaderWidths::for_state(state);
    let stagnation = state
        .history
        .last()
        .map(|point| point.stagnation)
        .unwrap_or(0);
    vec![
        format!(
            "gen {}",
            padded_pair(state.generation, state.generation_limit, widths.generation)
        ),
        padded_phase_label(state.phase.label(), FULL_PHASE_LABEL_WIDTH),
        padded_pair(
            state.phase_completed,
            state.phase_total.max(1),
            widths.phase,
        ),
        format!("best MCC {}", fmt_padded_opt(state.best_mcc(), 6)),
        format!("cov {}", fmt_padded_opt(state.best_coverage(), 5)),
        format!("no-imp {}", padded_value(stagnation, widths.stagnation)),
        format!("ETA {}", padded_eta_text(state)),
        format!("elapsed {}", fmt_padded_duration(state.elapsed())),
    ]
}

pub(super) fn two_line_header(
    area: Rect,
    state: &DashboardState,
    layout: DashboardLayout,
) -> Vec<Line<'static>> {
    let widths = HeaderWidths::for_state(state);
    let phase = header_phase_label(state.phase, layout);
    let width = block_inner_width(area.width);
    let first_fields = vec![
        format!(
            "gen {}",
            padded_pair(state.generation, state.generation_limit, widths.generation)
        ),
        phase,
        padded_pair(
            state.phase_completed,
            state.phase_total.max(1),
            widths.phase,
        ),
        format!("ETA {}", padded_eta_text(state)),
    ];
    let stagnation = state
        .history
        .last()
        .map(|point| point.stagnation)
        .unwrap_or(0);
    let second_fields = if layout == DashboardLayout::Full {
        vec![
            format!("best MCC {}", fmt_padded_opt(state.best_mcc(), 6)),
            format!("coverage {}", fmt_padded_opt(state.best_coverage(), 5)),
            format!("no-improve {}", padded_value(stagnation, widths.stagnation)),
            format!("elapsed {}", fmt_padded_duration(state.elapsed())),
        ]
    } else {
        vec![
            format!("best MCC {}", fmt_padded_opt(state.best_mcc(), 6)),
            format!("cov {}", fmt_padded_opt(state.best_coverage(), 5)),
            format!("no-imp {}", padded_value(stagnation, widths.stagnation)),
            format!("elapsed {}", fmt_padded_duration(state.elapsed())),
        ]
    };
    vec![
        Line::from(distributed_header_line(&state.task_id, first_fields, width)),
        Line::from(distributed_text_line(&second_fields, width)),
    ]
}

pub(super) fn distributed_header_line(task_id: &str, fields: Vec<String>, width: usize) -> String {
    let required_field_width = fields
        .iter()
        .map(|field| field.chars().count())
        .sum::<usize>();
    let gap_count = fields.len();
    let task_width = width.saturating_sub(required_field_width + gap_count);
    let mut items = Vec::with_capacity(fields.len() + 1);
    items.push(fit_text_to_width(task_id, task_width));
    items.extend(fields);
    distributed_text_line(&items, width)
}

pub(super) fn render_metric_tabs(frame: &mut Frame<'_>, area: Rect, state: &DashboardState) {
    let layout = state.click_regions.layout.unwrap_or(DashboardLayout::Full);
    let block = Block::default().borders(Borders::ALL).title(" Metric ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.is_empty() {
        return;
    }
    frame.render_widget(Paragraph::new(metric_tab_line(state, area, layout)), inner);
}

pub(super) fn render_metric_line(frame: &mut Frame<'_>, area: Rect, state: &DashboardState) {
    let mut spans = vec![Span::raw("Metric ")];
    spans.extend(metric_tab_spans(state, area, DashboardLayout::Minimal));
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

pub(super) fn metric_label_style(state: &DashboardState, metric: TuiMetric) -> Style {
    let mut style = Style::default();
    if state.selected_metric == metric {
        style = style.fg(Color::Cyan).add_modifier(Modifier::BOLD);
    }
    if state.hovered_metric == Some(metric) {
        style = style.add_modifier(Modifier::UNDERLINED);
    }
    style
}

pub(super) fn metric_tab_line<'a>(
    state: &DashboardState,
    area: Rect,
    layout: DashboardLayout,
) -> Line<'a> {
    Line::from(metric_tab_spans(state, area, layout))
}

pub(super) fn metric_tab_spans<'a>(
    state: &DashboardState,
    area: Rect,
    layout: DashboardLayout,
) -> Vec<Span<'a>> {
    let mut spans = Vec::with_capacity(TuiMetric::ALL.len() * 4);
    for (index, (metric, _, slot_width)) in metric_tab_slots(area, layout).into_iter().enumerate() {
        let label = fitted_metric_tab_label(state, metric, layout, slot_width);
        let label_width = label.chars().count() as u16;
        let left_padding = slot_width.saturating_sub(label_width) / 2;
        let right_padding = slot_width
            .saturating_sub(label_width)
            .saturating_sub(left_padding);
        push_spaces(&mut spans, left_padding);
        spans.push(Span::styled(label, metric_label_style(state, metric)));
        push_spaces(&mut spans, right_padding);
        if index + 1 < TuiMetric::ALL.len()
            && metric_tab_separator_count(metric_tab_width(area, layout)) > 0
        {
            spans.push(Span::raw("│"));
        }
    }
    spans
}

pub(super) fn metric_tab_width(area: Rect, layout: DashboardLayout) -> u16 {
    metric_labels_end(area, layout).saturating_sub(metric_labels_start(area, layout))
}

pub(super) fn fitted_metric_tab_label(
    state: &DashboardState,
    metric: TuiMetric,
    layout: DashboardLayout,
    max_width: u16,
) -> String {
    let label = metric_tab_label(state, metric, layout);
    fit_text_to_width(&label, max_width as usize)
}

pub(super) fn metric_tab_label(
    state: &DashboardState,
    metric: TuiMetric,
    layout: DashboardLayout,
) -> String {
    match layout {
        DashboardLayout::Minimal if state.selected_metric == metric => {
            format!("[{}]", metric.short_tab_label())
        }
        DashboardLayout::Minimal => metric.short_tab_label().to_string(),
        DashboardLayout::Full | DashboardLayout::Compact => metric.tab_label().to_string(),
    }
}

pub(super) fn fit_text_to_width(value: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    if value.chars().count() <= max_width {
        return value.to_string();
    }
    if max_width <= 3 {
        return value.chars().take(max_width).collect();
    }
    truncate_middle(value, max_width)
}

pub(super) fn distributed_text_line(items: &[String], width: usize) -> String {
    if width == 0 || items.is_empty() {
        return String::new();
    }
    let item_widths = items
        .iter()
        .map(|item| item.chars().count() as u16)
        .collect::<Vec<_>>();
    let starts = spread_item_offsets(width as u16, &item_widths);
    let mut line = String::new();
    let mut cursor = 0usize;
    for (item, start) in items.iter().zip(starts) {
        let start = start as usize;
        if start > cursor {
            line.push_str(&" ".repeat(start - cursor));
        }
        line.push_str(item);
        cursor = start.saturating_add(item.chars().count());
    }
    fit_text_to_width(&line, width)
}

pub(super) fn spread_item_offsets(total_width: u16, item_widths: &[u16]) -> Vec<u16> {
    if item_widths.is_empty() {
        return Vec::new();
    }
    if item_widths.len() == 1 {
        return vec![total_width.saturating_sub(item_widths[0]) / 2];
    }
    let item_total = item_widths.iter().copied().sum::<u16>();
    let gap_count = item_widths.len() as u16 - 1;
    if item_total.saturating_add(gap_count) > total_width {
        let mut cursor = 0u16;
        return item_widths
            .iter()
            .enumerate()
            .map(|(index, width)| {
                let start = cursor.min(total_width);
                cursor = cursor.saturating_add(*width);
                if index + 1 < item_widths.len() {
                    cursor = cursor.saturating_add(1);
                }
                start
            })
            .collect();
    }
    let total_gap_width = total_width - item_total;
    let base_gap = total_gap_width / gap_count;
    let extra_gaps = total_gap_width % gap_count;
    let mut cursor = 0u16;
    item_widths
        .iter()
        .enumerate()
        .map(|(index, width)| {
            let start = cursor;
            cursor = cursor.saturating_add(*width);
            if index + 1 < item_widths.len() {
                cursor = cursor
                    .saturating_add(base_gap)
                    .saturating_add(u16::from((index as u16) < extra_gaps));
            }
            start
        })
        .collect()
}

pub(super) fn push_spaces<'a>(spans: &mut Vec<Span<'a>>, width: u16) {
    if width > 0 {
        spans.push(Span::raw(" ".repeat(width as usize)));
    }
}

pub(super) fn render_plot(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &mut DashboardState,
    layout: DashboardLayout,
) {
    let metric = state.selected_metric;
    if state.history.is_empty() {
        let waiting = Paragraph::new(format!(
            "Waiting for generation history ({})",
            metric.title()
        ))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL).title(metric.title()));
        frame.render_widget(waiting, area);
        return;
    }

    let points = metric_points(metric, &state.history);
    let change_points = change_point_points(metric, &state.history, &state.change_points);
    let (x_min, x_max) = x_bounds(&points);
    let (y_min, y_max) = y_bounds(metric, &points, state.y_axis_mode);
    let dataset = Dataset::default()
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Cyan))
        .data(&points);
    let change_dataset = Dataset::default()
        .marker(symbols::Marker::Dot)
        .graph_type(GraphType::Scatter)
        .style(Style::default().fg(Color::Cyan))
        .data(&change_points);
    let mut block = Block::default().borders(Borders::ALL).title(metric.title());
    if metric.supports_y_axis_mode() {
        let title = format!(" {} ", y_axis_mode_label(state.y_axis_mode));
        let button = ButtonId::YAxisMode;
        state
            .click_regions
            .set_button(button, title_button_area(area, &title));
        block = block.title(button_title_line(state, title, button, true));
    }
    let chart = Chart::new(vec![dataset, change_dataset])
        .block(block)
        .x_axis(Axis::default().bounds([x_min, x_max]).labels(vec![
            Span::raw(format!("{x_min:.0}")),
            Span::raw(format!("{x_max:.0}")),
        ]))
        .y_axis(Axis::default().bounds([y_min, y_max]).labels(vec![
            Span::raw(format_axis_value(layout, y_min)),
            Span::raw(format_axis_value(layout, y_max)),
        ]));
    frame.render_widget(chart, area);
    render_change_point_tooltip(frame, area, state);
}

pub(super) fn render_sparkline_plot(frame: &mut Frame<'_>, area: Rect, state: &DashboardState) {
    let metric = state.selected_metric;
    if state.history.is_empty() {
        let waiting = Paragraph::new(format!("Waiting for {}", metric.title()))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL).title(metric.title()));
        frame.render_widget(waiting, area);
        return;
    }

    let values = sparkline_values(metric, &state.history);
    let sparkline = Sparkline::default()
        .block(Block::default().borders(Borders::ALL).title(metric.title()))
        .data(&values)
        .max(100)
        .style(Style::default().fg(Color::Cyan));
    frame.render_widget(sparkline, area);
}

pub(super) fn y_axis_mode_label(mode: YAxisMode) -> String {
    format!("[z: {}]", mode.label())
}

pub(super) fn render_change_point_tooltip(
    frame: &mut Frame<'_>,
    plot_area: Rect,
    state: &mut DashboardState,
) {
    let Some(hovered) = state.hovered_change_point.as_ref() else {
        return;
    };
    if !contains_position(plot_area, hovered.marker_x, hovered.marker_y) {
        return;
    }
    let Some(area) = change_point_tooltip_area(plot_area, hovered.marker_x, hovered.marker_y)
    else {
        return;
    };
    let copy_title = format!(" {} ", copy_change_point_smarts_label());
    let copy_button = ButtonId::Copy(CopyControl::ChangePoint);
    state
        .click_regions
        .set_button(copy_button, title_button_area(area, &copy_title));
    let inner_width = area.width.saturating_sub(2) as usize;
    let point = &hovered.point;
    let lines = smarts_box_lines(
        SmartsBoxContent {
            generation: Some(point.generation),
            mcc: point.mcc,
            coverage_score: point.coverage_score,
            smarts_len: point.smarts_len,
            smarts: &point.smarts,
            notice: None,
        },
        inner_width,
        area.height.saturating_sub(2) as usize,
    );
    let tooltip = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" SMARTS change ")
                .title(button_title_line(state, copy_title, copy_button, true)),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(Clear, area);
    frame.render_widget(tooltip, area);
}

pub(super) struct SmartsBoxContent<'a> {
    generation: Option<u64>,
    mcc: f64,
    coverage_score: f64,
    smarts_len: usize,
    smarts: &'a str,
    notice: Option<&'a str>,
}

pub(super) fn smarts_box_lines(
    content: SmartsBoxContent<'_>,
    width: usize,
    inner_height: usize,
) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(smarts_box_metric_header(
        content.generation,
        content.mcc,
        content.coverage_score,
        content.smarts_len,
        width,
    ))];
    if let Some(notice) = content.notice.filter(|notice| !notice.is_empty()) {
        lines.push(Line::from(centered_text_line(notice, width)));
    }

    let smarts_line = centered_text_line(&truncate_smarts(content.smarts, width.max(8)), width);
    let smarts_row = if inner_height <= 2 {
        1
    } else {
        inner_height / 2
    };
    while lines.len() < smarts_row {
        lines.push(Line::from(""));
    }
    lines.push(Line::from(smarts_line));
    lines
}

pub(super) fn smarts_box_metric_header(
    generation: Option<u64>,
    mcc: f64,
    coverage_score: f64,
    smarts_len: usize,
    width: usize,
) -> String {
    let mut metrics = Vec::with_capacity(4);
    if let Some(generation) = generation {
        metrics.push(format!("gen {generation}"));
    }
    metrics.extend([
        format!("MCC {}", fmt_padded_opt(Some(mcc), 6)),
        format!("cov {}", fmt_padded_opt(Some(coverage_score), 5)),
        format!("len {smarts_len}"),
    ]);
    distributed_text_line(&metrics, width)
}

pub(super) fn centered_text_line(value: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let fitted = fit_text_to_width(value, width);
    let fitted_width = fitted.chars().count();
    let left_padding = width.saturating_sub(fitted_width) / 2;
    format!("{}{}", " ".repeat(left_padding), fitted)
}

pub(super) fn copy_change_point_smarts_label() -> &'static str {
    "[⧉ copy]"
}

pub(super) fn button_style(state: &DashboardState, button: ButtonId, enabled: bool) -> Style {
    let mut style = if state
        .button_feedback
        .is_some_and(|feedback| feedback.is_active_for(button))
    {
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD)
    } else if !enabled {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(Color::Cyan)
    };
    if enabled && state.hovered_button == Some(button) {
        style = style.add_modifier(Modifier::UNDERLINED);
    }
    style
}

pub(super) fn button_title_line(
    state: &DashboardState,
    title: String,
    button: ButtonId,
    enabled: bool,
) -> Line<'static> {
    Line::from(Span::styled(title, button_style(state, button, enabled)))
        .alignment(Alignment::Right)
}

pub(super) fn change_point_tooltip_area(
    plot_area: Rect,
    marker_x: u16,
    marker_y: u16,
) -> Option<Rect> {
    let width = 58.min(plot_area.width.saturating_sub(2));
    let height = 4.min(plot_area.height.saturating_sub(2));
    if width < 24 || height < 3 {
        return None;
    }
    let right = plot_area.right().saturating_sub(1);
    let bottom = plot_area.bottom().saturating_sub(1);
    let x = if marker_x.saturating_add(width).saturating_add(2) < right {
        marker_x.saturating_add(2)
    } else {
        right.saturating_sub(width)
    }
    .max(plot_area.x.saturating_add(1));
    let y = if marker_y > plot_area.y.saturating_add(height).saturating_add(1) {
        marker_y.saturating_sub(height)
    } else {
        marker_y.saturating_add(1)
    }
    .min(bottom.saturating_sub(height).saturating_add(1))
    .max(plot_area.y.saturating_add(1));
    Some(Rect::new(x, y, width, height))
}

pub(super) fn render_status(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &mut DashboardState,
    layout: DashboardLayout,
) {
    let block = Block::default().borders(Borders::ALL).title(" Status ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(2)])
        .split(inner);
    state.click_regions.best_area = Some(rows[1]);
    let phase_total = state.phase_total.max(1);
    let ratio = ratio(state.phase_completed, phase_total).clamp(0.0, 1.0);
    let gauge = Gauge::default()
        .gauge_style(Style::default().fg(Color::Cyan))
        .ratio(ratio)
        .label(phase_gauge_label(state, phase_total, layout));
    frame.render_widget(gauge, rows[0]);

    let best_line = state
        .best
        .as_ref()
        .map(|best| {
            format!(
                "best len {}  {}",
                best.smarts_len,
                truncate_smarts(&best.smarts, best_smarts_width(area, state, layout))
            )
        })
        .unwrap_or_else(|| "best pending".to_string());
    let metric_line = status_metric_line(state, layout, rows[1].width as usize);
    let status = Paragraph::new(vec![Line::from(metric_line), Line::from(best_line)]);
    frame.render_widget(status, rows[1]);
}

pub(super) fn status_metric_line(
    state: &DashboardState,
    layout: DashboardLayout,
    width: usize,
) -> String {
    distributed_text_line(&status_metric_items(state, layout), width)
}

pub(super) fn status_metric_items(state: &DashboardState, layout: DashboardLayout) -> Vec<String> {
    let memory = fmt_memory_opt(state.runtime_metrics.resident_memory_bytes);
    match layout {
        DashboardLayout::Full => vec![
            format!("current {}", fmt_current_mcc(state, 6)),
            format!("gen best {}", fmt_padded_opt(state.generation_best_mcc, 6)),
            format!("best MCC {}", fmt_padded_opt(state.best_mcc(), 6)),
            format!("coverage {}", fmt_padded_opt(state.best_coverage(), 5)),
            format!(
                "threads {}",
                padded_value(state.runtime_metrics.rayon_threads, 2)
            ),
            format!("rss {memory}"),
        ],
        DashboardLayout::Compact => vec![
            format!("cur {}", fmt_current_mcc(state, 6)),
            format!("gen {}", fmt_padded_opt(state.generation_best_mcc, 6)),
            format!("best {}", fmt_padded_opt(state.best_mcc(), 6)),
            format!("cov {}", fmt_padded_opt(state.best_coverage(), 5)),
            format!(
                "thr {}",
                padded_value(state.runtime_metrics.rayon_threads, 2)
            ),
            format!("rss {memory}"),
        ],
        DashboardLayout::Minimal => vec![
            format!("cur {}", fmt_current_mcc(state, 6)),
            format!("best {}", fmt_padded_opt(state.best_mcc(), 6)),
            format!(
                "thr {}",
                padded_value(state.runtime_metrics.rayon_threads, 2)
            ),
            format!("rss {memory}"),
        ],
    }
}

pub(super) fn fmt_current_mcc(state: &DashboardState, width: usize) -> String {
    if state.current_limit_exceeded {
        format!("{:>width$}", "OOT")
    } else {
        fmt_padded_opt(state.current_mcc, width)
    }
}

pub(super) fn render_footer(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &mut DashboardState,
    layout: DashboardLayout,
) {
    state.click_regions.reset_footer_buttons();

    let segments = footer_segments(state, layout);
    let mut spans = Vec::with_capacity(segments.len() * 2);
    let widths = segments
        .iter()
        .map(|segment| segment.label.chars().count() as u16)
        .collect::<Vec<_>>();
    let starts = spread_item_offsets(area.width, &widths);
    let mut cursor = 0;
    for (segment, start) in segments.iter().zip(starts) {
        push_spaces(&mut spans, start.saturating_sub(cursor));
        let width = segment.label.chars().count() as u16;
        let region = if segment.enabled {
            footer_region(area, area.x.saturating_add(start), width)
        } else {
            None
        };
        state
            .click_regions
            .set_button(ButtonId::Footer(segment.control), region);
        spans.push(Span::styled(
            segment.label,
            button_style(state, ButtonId::Footer(segment.control), segment.enabled),
        ));
        cursor = start.saturating_add(width);
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

pub(super) fn footer_segments(
    state: &DashboardState,
    layout: DashboardLayout,
) -> Vec<FooterSegment<'static>> {
    let (previous, pause, best, next, stop, help) = match layout {
        DashboardLayout::Full => (
            "[l ← metric]",
            pause_button_label(state),
            "[b ★ best SMARTS]",
            "[r metric →]",
            stop_button_label(state),
            "[? help]",
        ),
        DashboardLayout::Compact => (
            "[l ←]",
            pause_button_label(state),
            "[b ★ best]",
            "[r →]",
            compact_stop_button_label(state),
            "[?]",
        ),
        DashboardLayout::Minimal => (
            "[l ←]",
            minimal_pause_button_label(state),
            "[b ★]",
            "[r →]",
            "[q ■]",
            "[?]",
        ),
    };
    vec![
        FooterSegment::control(FooterControl::PreviousMetric, previous, true),
        FooterSegment::control(
            FooterControl::PauseResume,
            pause,
            pause_button_enabled(state),
        ),
        FooterSegment::control(FooterControl::BestSmarts, best, state.best.is_some()),
        FooterSegment::control(FooterControl::NextMetric, next, true),
        FooterSegment::control(FooterControl::Stop, stop, stop_button_enabled(state)),
        FooterSegment::control(FooterControl::Help, help, true),
    ]
}

#[derive(Clone, Copy, Debug)]
pub(super) struct FooterSegment<'a> {
    pub(super) label: &'a str,
    pub(super) control: FooterControl,
    pub(super) enabled: bool,
}

impl<'a> FooterSegment<'a> {
    pub(super) const fn control(control: FooterControl, label: &'a str, enabled: bool) -> Self {
        Self {
            label,
            control,
            enabled,
        }
    }
}

pub(super) fn footer_region(area: Rect, start: u16, width: u16) -> Option<Rect> {
    let right = area.x.saturating_add(area.width);
    let start = start.max(area.x);
    let end = start.saturating_add(width).min(right);
    (end > start).then_some(Rect::new(start, area.y, end - start, 1))
}

pub(super) fn pause_button_label(state: &DashboardState) -> &'static str {
    match state.status {
        DashboardStatus::Pausing => "[p … pausing]",
        DashboardStatus::Paused => "[p ▶ play]",
        _ => "[p ▮▮ pause]",
    }
}

pub(super) fn minimal_pause_button_label(state: &DashboardState) -> &'static str {
    match state.status {
        DashboardStatus::Pausing => "[p …]",
        DashboardStatus::Paused => "[p ▶]",
        _ => "[p ▮▮]",
    }
}

pub(super) fn pause_button_enabled(state: &DashboardState) -> bool {
    matches!(
        state.status,
        DashboardStatus::Waiting
            | DashboardStatus::Running
            | DashboardStatus::Pausing
            | DashboardStatus::Paused
    )
}

pub(super) fn stop_button_label(state: &DashboardState) -> &'static str {
    if state.status == DashboardStatus::Stopping {
        "[q ■ stopping]"
    } else {
        "[q ■ stop]"
    }
}

pub(super) fn compact_stop_button_label(state: &DashboardState) -> &'static str {
    if state.status == DashboardStatus::Stopping {
        "[q stop]"
    } else {
        "[q ■]"
    }
}

pub(super) fn stop_button_enabled(state: &DashboardState) -> bool {
    matches!(
        state.status,
        DashboardStatus::Waiting
            | DashboardStatus::Running
            | DashboardStatus::Pausing
            | DashboardStatus::Paused
    )
}

pub(super) fn render_best_smarts(frame: &mut Frame<'_>, area: Rect, state: &mut DashboardState) {
    let Some(best) = state.best.as_ref() else {
        return;
    };
    let copy_label = copy_best_smarts_label();
    let copy_title = format!(" {copy_label} ");
    let copy_button = ButtonId::Copy(CopyControl::BestSmarts);
    state
        .click_regions
        .set_button(copy_button, title_button_area(area, &copy_title));
    let clipboard_notice = state
        .last_event
        .as_ref()
        .filter(|message| message.contains("clipboard") || message.starts_with("copied"))
        .cloned()
        .unwrap_or_default();
    let inner_width = area.width.saturating_sub(2) as usize;
    let text = smarts_box_lines(
        SmartsBoxContent {
            generation: None,
            mcc: best.mcc,
            coverage_score: best.coverage_score,
            smarts_len: best.smarts_len,
            smarts: &best.smarts,
            notice: (!clipboard_notice.is_empty()).then_some(clipboard_notice.as_str()),
        },
        inner_width,
        area.height.saturating_sub(2) as usize,
    );
    let best_smarts = Paragraph::new(text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Best SMARTS ")
                .title(button_title_line(state, copy_title, copy_button, true)),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(Clear, area);
    frame.render_widget(best_smarts, area);
}

pub(super) fn title_button_area(area: Rect, title: &str) -> Option<Rect> {
    let title_area_width = area.width.saturating_sub(2);
    let width = (title.chars().count() as u16).min(title_area_width);
    if width == 0 {
        return None;
    }
    let x = area
        .x
        .saturating_add(1)
        .saturating_add(title_area_width.saturating_sub(width));
    Some(Rect::new(x, area.y, width, 1))
}

pub(super) fn copy_best_smarts_label() -> &'static str {
    "[⧉ copy SMARTS]"
}

pub(super) fn phase_gauge_label(
    state: &DashboardState,
    phase_total: usize,
    layout: DashboardLayout,
) -> String {
    match layout {
        DashboardLayout::Full => format!(
            "{} gen {} {}/{}",
            state.phase.label(),
            state.phase_generation,
            state.phase_completed,
            phase_total
        ),
        DashboardLayout::Compact | DashboardLayout::Minimal => format!(
            "{} {} {}/{}",
            state.phase.short_label(),
            state.phase_generation,
            state.phase_completed,
            phase_total
        ),
    }
}

pub(super) fn best_smarts_width(
    area: Rect,
    state: &DashboardState,
    layout: DashboardLayout,
) -> usize {
    let available = area.width.saturating_sub(match layout {
        DashboardLayout::Full => 14,
        DashboardLayout::Compact => 12,
        DashboardLayout::Minimal => 10,
    }) as usize;
    state.best_smarts_width.min(available.max(12))
}

pub(super) fn render_help(frame: &mut Frame<'_>, area: Rect) {
    let help = Paragraph::new(vec![
        Line::from("left/right or [l ← metric]/[r metric →]: switch metric"),
        Line::from("z or [z: abs]/[z: zoom]: toggle MCC/Coverage y-axis range"),
        Line::from("[p ▮▮ pause]/[p … pausing]/[p ▶ play]: pause or resume"),
        Line::from("b, enter, or [b ★ best SMARTS]: show full best SMARTS"),
        Line::from("[q ■ stop]: request stop after current generation"),
        Line::from("[? help]: toggle help"),
        Line::from("esc: close help"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" Help "))
    .wrap(Wrap { trim: true });
    frame.render_widget(Clear, area);
    frame.render_widget(help, area);
}

pub(super) fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    }
}

pub(super) fn metric_points(metric: TuiMetric, history: &[HistoryPoint]) -> Vec<(f64, f64)> {
    history
        .iter()
        .map(|point| (point.generation as f64, metric_value(metric, point)))
        .collect()
}

pub(super) fn metric_value(metric: TuiMetric, point: &HistoryPoint) -> f64 {
    match metric {
        TuiMetric::Mcc => point.mcc,
        TuiMetric::Coverage => point.coverage,
        TuiMetric::Stagnation => point.stagnation as f64,
        TuiMetric::Unique => point.unique_ratio,
        TuiMetric::Timeouts => point.timeouts as f64,
        TuiMetric::Length => point.average_len.max(point.lead_len as f64),
    }
}

pub(super) fn change_point_points(
    metric: TuiMetric,
    history: &[HistoryPoint],
    change_points: &[ChangePointSummary],
) -> Vec<(f64, f64)> {
    change_points
        .iter()
        .filter_map(|change| {
            let point = history
                .iter()
                .find(|point| point.generation == change.generation)?;
            Some((change.generation as f64, metric_value(metric, point)))
        })
        .collect()
}

pub(super) fn sparkline_values(metric: TuiMetric, history: &[HistoryPoint]) -> Vec<u64> {
    let points = metric_points(metric, history);
    let (min, max) = y_bounds(metric, &points, YAxisMode::Absolute);
    let span = (max - min).max(f64::EPSILON);
    points
        .into_iter()
        .map(|(_, value)| (((value - min) / span) * 100.0).clamp(0.0, 100.0) as u64)
        .collect()
}

pub(super) fn x_bounds(points: &[(f64, f64)]) -> (f64, f64) {
    let first = points.first().map(|point| point.0).unwrap_or(0.0);
    let last = points.last().map(|point| point.0).unwrap_or(first + 1.0);
    if (last - first).abs() < f64::EPSILON {
        (first, first + 1.0)
    } else {
        (first, last)
    }
}

pub(super) fn y_bounds(metric: TuiMetric, points: &[(f64, f64)], mode: YAxisMode) -> (f64, f64) {
    match metric {
        TuiMetric::Mcc if mode == YAxisMode::Zoomed => zoomed_y_bounds(points, -1.0, 1.0),
        TuiMetric::Mcc => (-1.0, 1.0),
        TuiMetric::Coverage if mode == YAxisMode::Zoomed => zoomed_y_bounds(points, 0.0, 1.0),
        TuiMetric::Coverage | TuiMetric::Unique => (0.0, 1.0),
        TuiMetric::Stagnation | TuiMetric::Timeouts | TuiMetric::Length => {
            let max = points
                .iter()
                .map(|point| point.1)
                .fold(0.0_f64, f64::max)
                .max(1.0);
            (0.0, max * 1.1)
        }
    }
}

pub(super) fn zoomed_y_bounds(
    points: &[(f64, f64)],
    absolute_min: f64,
    absolute_max: f64,
) -> (f64, f64) {
    let Some((min, max)) = observed_y_range(points) else {
        return (absolute_min, absolute_max);
    };
    let span = (max - min).abs();
    let fallback_padding = ((absolute_max - absolute_min) * 0.025).max(0.01);
    let padding = if span < f64::EPSILON {
        fallback_padding
    } else {
        span * 0.12
    };
    let mut lower = (min - padding).max(absolute_min);
    let mut upper = (max + padding).min(absolute_max);
    if (upper - lower).abs() < f64::EPSILON {
        lower = (min - fallback_padding).max(absolute_min);
        upper = (max + fallback_padding).min(absolute_max);
    }
    if upper > lower {
        (lower, upper)
    } else {
        (absolute_min, absolute_max)
    }
}

pub(super) fn observed_y_range(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    let mut values = points.iter().map(|point| point.1);
    let first = values.next()?;
    let (min, max) = values.fold((first, first), |(min, max), value| {
        (min.min(value), max.max(value))
    });
    Some((min, max))
}

pub(super) fn chart_graph_area(
    area: Rect,
    layout: DashboardLayout,
    x_min: f64,
    y_min: f64,
    y_max: f64,
) -> Option<Rect> {
    let chart_area = Rect::new(
        area.x.saturating_add(1),
        area.y.saturating_add(1),
        area.width.saturating_sub(2),
        area.height.saturating_sub(2),
    );
    if chart_area.is_empty() {
        return None;
    }
    let mut x = chart_area.x;
    let mut y = chart_area.bottom().saturating_sub(1);
    if y > chart_area.y {
        y = y.saturating_sub(1);
    }

    let first_x_label_width = format!("{x_min:.0}").chars().count() as u16;
    let y_label_width = format_axis_value(layout, y_min)
        .chars()
        .count()
        .max(format_axis_value(layout, y_max).chars().count()) as u16;
    let width_left_of_y_axis = first_x_label_width.saturating_sub(1);
    x = x.saturating_add(
        y_label_width
            .max(width_left_of_y_axis)
            .min(chart_area.width / 3),
    );

    if y > chart_area.y {
        y = y.saturating_sub(1);
    }
    if x.saturating_add(1) < chart_area.right() {
        x = x.saturating_add(1);
    }

    let width = chart_area.right().saturating_sub(x);
    let height = y.saturating_sub(chart_area.y).saturating_add(1);
    (width > 0 && height > 0).then_some(Rect::new(x, chart_area.y, width, height))
}

pub(super) fn data_point_screen_position(
    graph_area: Rect,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    x: f64,
    y: f64,
) -> Option<(u16, u16)> {
    if graph_area.is_empty() {
        return None;
    }
    let x_span = (x_max - x_min).max(f64::EPSILON);
    let y_span = (y_max - y_min).max(f64::EPSILON);
    let x_ratio = ((x - x_min) / x_span).clamp(0.0, 1.0);
    let y_ratio = ((y_max - y) / y_span).clamp(0.0, 1.0);
    let x_offset = (x_ratio * f64::from(graph_area.width.saturating_sub(1))).round() as u16;
    let y_offset = (y_ratio * f64::from(graph_area.height.saturating_sub(1))).round() as u16;
    Some((
        graph_area.x.saturating_add(x_offset),
        graph_area.y.saturating_add(y_offset),
    ))
}

pub(super) fn format_axis_value(layout: DashboardLayout, value: f64) -> String {
    match layout {
        DashboardLayout::Full => format!("{value:.2}"),
        DashboardLayout::Compact | DashboardLayout::Minimal => format!("{value:.1}"),
    }
}

pub(super) fn block_inner_width(width: u16) -> usize {
    width.saturating_sub(2) as usize
}

pub(super) fn decimal_width(value: impl fmt::Display) -> usize {
    value.to_string().len().max(1)
}

pub(super) fn padded_pair(
    current: impl fmt::Display,
    total: impl fmt::Display,
    width: usize,
) -> String {
    format!("{current:>width$}/{total:>width$}")
}

pub(super) fn padded_value(value: impl fmt::Display, width: usize) -> String {
    format!("{value:>width$}")
}

pub(super) fn padded_phase_label(label: &str, width: usize) -> String {
    format!("{label:<width$}")
}

pub(super) fn header_phase_label(phase: DashboardPhase, layout: DashboardLayout) -> String {
    match layout {
        DashboardLayout::Full => padded_phase_label(phase.label(), FULL_PHASE_LABEL_WIDTH),
        DashboardLayout::Compact | DashboardLayout::Minimal => {
            padded_phase_label(phase.short_label(), SHORT_PHASE_LABEL_WIDTH)
        }
    }
}

pub(super) fn ratio(completed: usize, total: usize) -> f64 {
    completed as f64 / total.max(1) as f64
}

pub(super) fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "---".to_string())
}

pub(super) fn fmt_padded_opt(value: Option<f64>, width: usize) -> String {
    format!("{:>width$}", fmt_opt(value))
}

pub(super) fn fmt_memory_opt(bytes: Option<u64>) -> String {
    bytes
        .map(fmt_memory_bytes)
        .unwrap_or_else(|| "---".to_string())
}

pub(super) fn fmt_memory_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes = bytes as f64;
    if bytes >= GIB {
        format!("{:.1}GiB", bytes / GIB)
    } else if bytes >= MIB {
        format!("{:.1}MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.1}KiB", bytes / KIB)
    } else {
        format!("{bytes:.0}B")
    }
}

pub(super) fn padded_eta_text(state: &DashboardState) -> String {
    format!("{:>8}", eta_text(state))
}

pub(super) fn eta_text(state: &DashboardState) -> String {
    if state.generation == 0 {
        return "---".into();
    }
    let elapsed = state.elapsed();
    let per_generation = elapsed.as_secs_f64() / state.generation as f64;
    let remaining = state.generation_limit.saturating_sub(state.generation);
    fmt_duration(Duration::from_secs_f64(per_generation * remaining as f64))
}

pub(super) fn fmt_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

pub(super) fn fmt_padded_duration(duration: Duration) -> String {
    format!("{:>8}", fmt_duration(duration))
}

pub(super) fn truncate_smarts(smarts: &str, max_width: usize) -> String {
    let max_width = max_width.max(3);
    if smarts.chars().count() <= max_width {
        return smarts.to_string();
    }
    let prefix: String = smarts.chars().take(max_width.saturating_sub(3)).collect();
    format!("{prefix}...")
}

pub(super) fn truncate_middle(value: &str, max_width: usize) -> String {
    let max_width = max_width.max(5);
    let len = value.chars().count();
    if len <= max_width {
        return value.to_string();
    }
    let left = (max_width - 3) / 2;
    let right = max_width - 3 - left;
    let prefix: String = value.chars().take(left).collect();
    let suffix: String = value
        .chars()
        .rev()
        .take(right)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{prefix}...{suffix}")
}
