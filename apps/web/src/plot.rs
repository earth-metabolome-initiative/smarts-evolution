use dioxus::prelude::*;
use smarts_evolution_web_protocol::RankedCandidate;

use crate::ProgressPoint;

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use std::rc::Rc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, closure::Closure};
#[cfg(target_arch = "wasm32")]
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, MouseEvent};

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

#[component]
pub(crate) fn MccGenerationImprovementPlot(history: Vec<ProgressPoint>) -> Element {
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
                span { class: "plot-tooltip-label", "SMARTS length" }
                span { class: "plot-tooltip-value", {tooltip.candidate.smarts_len().to_string()} }
            }
            p { class: "plot-tooltip-smarts", {tooltip.candidate.smarts()} }
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
        .then_with(|| left.smarts_len().cmp(&right.smarts_len()))
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

#[cfg(test)]
mod tests {
    use super::{RankedCandidate, compare_ranked_candidates};

    #[test]
    fn ranked_candidate_order_prefers_lower_smarts_len_ties() {
        let simpler = RankedCandidate::new("[#6]", 0.8124, 1);
        let longer = RankedCandidate::new("[N]", 0.8124, 2);
        assert!(compare_ranked_candidates(&simpler, &longer).is_lt());
    }

    #[test]
    fn ranked_candidate_order_prefers_higher_mcc() {
        let lower_mcc = RankedCandidate::new("[#6]", 0.8000, 1);
        let higher_mcc = RankedCandidate::new("[#6]", 0.9000, 1);

        assert!(compare_ranked_candidates(&higher_mcc, &lower_mcc).is_lt());
    }
}
