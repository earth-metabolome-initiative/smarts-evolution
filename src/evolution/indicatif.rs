//! Terminal progress bars for evolution runs.

use alloc::format;
use alloc::string::{String, ToString};

use ::indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};

use super::config::EvolutionConfig;
use super::runner::{
    EvolutionError, EvolutionEvaluationProgress, EvolutionProgress, EvolutionProgressObserver,
    EvolutionSession, EvolutionStatus, EvolutionTask, TaskResult,
};
use crate::genome::seed::SeedCorpus;

const DEFAULT_BEST_SMARTS_WIDTH: usize = 96;

/// Indicatif-backed progress bars for one evolution run.
///
/// The top bar tracks completed generations and shows ETA, current generation
/// MCC, incumbent MCC, incumbent SMARTS, and the number of generations since
/// the incumbent last improved. The second bar tracks SMARTS evaluation inside
/// the current generation.
pub struct IndicatifEvolutionProgress {
    generation_bar: ProgressBar,
    evaluation_bar: ProgressBar,
    best_smarts_width: usize,
    clear_on_finish: bool,
}

impl Default for IndicatifEvolutionProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl IndicatifEvolutionProgress {
    /// Create progress bars that render to stderr.
    pub fn new() -> Self {
        Self::with_draw_target(ProgressDrawTarget::stderr())
    }

    /// Create progress bars with a custom indicatif draw target.
    pub fn with_draw_target(draw_target: ProgressDrawTarget) -> Self {
        let multi = MultiProgress::with_draw_target(draw_target);
        Self::attach_to(&multi)
    }

    /// Create progress bars from caller-owned indicatif bars.
    pub fn from_bars(generation_bar: ProgressBar, evaluation_bar: ProgressBar) -> Self {
        generation_bar.set_style(generation_style());
        generation_bar.set_prefix("generations");
        evaluation_bar.set_style(evaluation_style());
        evaluation_bar.set_prefix("evaluating");

        Self {
            generation_bar,
            evaluation_bar,
            best_smarts_width: DEFAULT_BEST_SMARTS_WIDTH,
            clear_on_finish: false,
        }
    }

    /// Attach evolution progress bars to an existing indicatif multi-progress.
    pub fn attach_to(multi: &MultiProgress) -> Self {
        Self::from_bars(
            multi.add(ProgressBar::new(1)),
            multi.add(ProgressBar::new(1)),
        )
    }

    /// Create hidden progress bars for tests or callers that want the same code path without I/O.
    pub fn hidden() -> Self {
        Self::with_draw_target(ProgressDrawTarget::hidden())
    }

    /// Limit the SMARTS text shown in the generation bar message.
    pub fn with_best_smarts_width(mut self, width: usize) -> Self {
        self.best_smarts_width = width.max(3);
        self
    }

    /// Use a caller-provided style for the generation progress bar.
    pub fn with_generation_style(self, style: ProgressStyle) -> Self {
        self.generation_bar.set_style(style);
        self
    }

    /// Use a caller-provided style for the SMARTS evaluation progress bar.
    pub fn with_evaluation_style(self, style: ProgressStyle) -> Self {
        self.evaluation_bar.set_style(style);
        self
    }

    /// Clear both evolution bars when the run finishes or fails.
    pub fn clear_on_finish(mut self, clear: bool) -> Self {
        self.clear_on_finish = clear;
        self
    }

    /// Set custom prefixes for the generation and evaluation bars.
    pub fn with_prefixes(
        self,
        generation: impl Into<String>,
        evaluation: impl Into<String>,
    ) -> Self {
        self.generation_bar.set_prefix(generation.into());
        self.evaluation_bar.set_prefix(evaluation.into());
        self
    }

    fn start(&mut self, task_id: &str, generation_limit: u64) {
        self.generation_bar.set_length(generation_limit.max(1));
        self.generation_bar.set_position(0);
        self.generation_bar
            .set_message(format!("task={task_id} status=queued"));
        self.evaluation_bar.set_length(1);
        self.evaluation_bar.set_position(0);
        self.evaluation_bar.set_message("waiting for generation 1");
    }

    fn record_evaluation(&mut self, progress: &EvolutionEvaluationProgress) {
        if progress.completed() == 0 {
            self.evaluation_bar.reset_elapsed();
            self.evaluation_bar.reset_eta();
        }
        self.evaluation_bar
            .set_length(progress.total().max(1) as u64);
        self.evaluation_bar
            .set_position(progress.completed() as u64);
        let mut message = format!(
            "generation={}/{} SMARTS={}/{}",
            progress.generation(),
            progress.generation_limit(),
            progress.completed(),
            progress.total()
        );
        if let Some(last_mcc) = progress.last_mcc() {
            message.push_str(&format!(" current_mcc={last_mcc:.3}"));
        }
        if let Some(last_smarts) = progress.last_smarts() {
            message.push_str(&format!(
                " current={}",
                truncate_smarts(last_smarts, self.best_smarts_width)
            ));
        }
        if let Some(best_mcc) = progress.generation_best_mcc() {
            message.push_str(&format!(" gen_best_mcc={best_mcc:.3}"));
        }
        self.evaluation_bar.set_message(message);
    }

    fn record_generation(&mut self, progress: &EvolutionProgress) {
        self.generation_bar
            .set_length(progress.generation_limit().max(1));
        self.generation_bar.set_position(progress.generation());
        self.generation_bar
            .set_message(self.generation_message(progress));
    }

    fn finish(&mut self, result: &TaskResult) {
        let message = format!(
            "done generations={} best_mcc={:.3} best_smarts={}",
            result.generations(),
            result.best_mcc(),
            truncate_smarts(result.best_smarts(), self.best_smarts_width)
        );
        if self.clear_on_finish {
            self.evaluation_bar.finish_and_clear();
            self.generation_bar.finish_and_clear();
        } else {
            self.evaluation_bar.finish_with_message("done");
            self.generation_bar.finish_with_message(message);
        }
    }

    fn abandon(&mut self, error: &EvolutionError) {
        if self.clear_on_finish {
            self.evaluation_bar.finish_and_clear();
            self.generation_bar.finish_and_clear();
        } else {
            self.evaluation_bar.abandon_with_message("failed");
            self.generation_bar
                .abandon_with_message(format!("failed: {error}"));
        }
    }

    fn generation_message(&self, progress: &EvolutionProgress) -> String {
        format!(
            "task={} status={} current_mcc={:.3} best_mcc={:.3} no_improve={} best_smarts={}",
            progress.task_id(),
            status_label(progress.status()),
            progress.best().mcc(),
            progress.best_so_far().mcc(),
            progress.stagnation(),
            truncate_smarts(progress.best_so_far().smarts(), self.best_smarts_width)
        )
    }
}

impl EvolutionProgressObserver for IndicatifEvolutionProgress {
    fn on_start(&mut self, task_id: &str, generation_limit: u64) {
        self.start(task_id, generation_limit);
    }

    fn on_evaluation(&mut self, progress: &EvolutionEvaluationProgress) {
        self.record_evaluation(progress);
    }

    fn on_generation(&mut self, progress: &EvolutionProgress) {
        self.record_generation(progress);
    }

    fn on_finish(&mut self, result: &TaskResult) {
        self.finish(result);
    }

    fn on_error(&mut self, error: &EvolutionError) {
        self.abandon(error);
    }
}

impl EvolutionTask {
    /// Evolve this task with default terminal progress bars.
    ///
    /// Enable the `indicatif` cargo feature and call this instead of
    /// [`EvolutionTask::evolve`] when a terminal progress display is wanted.
    pub fn evolve_with_indicatif(
        &self,
        config: &EvolutionConfig,
        seed_corpus: &SeedCorpus,
    ) -> Result<TaskResult, EvolutionError> {
        self.evolve_with_indicatif_progress(
            config,
            seed_corpus,
            1,
            IndicatifEvolutionProgress::new(),
        )
    }

    /// Evolve this task with caller-provided indicatif progress bars.
    pub fn evolve_with_indicatif_progress(
        &self,
        config: &EvolutionConfig,
        seed_corpus: &SeedCorpus,
        leaderboard_size: usize,
        progress: IndicatifEvolutionProgress,
    ) -> Result<TaskResult, EvolutionError> {
        self.evolve_with_observer(config, seed_corpus, leaderboard_size, progress)
    }

    /// Evolve this task by moving its folds into the session with default
    /// terminal progress bars.
    pub fn evolve_owned_with_indicatif(
        self,
        config: &EvolutionConfig,
        seed_corpus: &SeedCorpus,
    ) -> Result<TaskResult, EvolutionError> {
        self.evolve_owned_with_indicatif_progress(
            config,
            seed_corpus,
            1,
            IndicatifEvolutionProgress::new(),
        )
    }

    /// Evolve this task by moving its folds into the session with
    /// caller-provided indicatif progress bars.
    pub fn evolve_owned_with_indicatif_progress(
        self,
        config: &EvolutionConfig,
        seed_corpus: &SeedCorpus,
        leaderboard_size: usize,
        progress: IndicatifEvolutionProgress,
    ) -> Result<TaskResult, EvolutionError> {
        self.evolve_owned_with_observer(config, seed_corpus, leaderboard_size, progress)
    }
}

impl EvolutionSession {
    /// Drive this session to completion with default terminal progress bars.
    pub fn evolve_with_indicatif(self) -> Result<TaskResult, EvolutionError> {
        self.evolve_with_indicatif_progress(IndicatifEvolutionProgress::new())
    }

    /// Drive this session to completion with caller-provided indicatif progress bars.
    pub fn evolve_with_indicatif_progress(
        self,
        progress: IndicatifEvolutionProgress,
    ) -> Result<TaskResult, EvolutionError> {
        self.evolve_with_observer(progress)
    }
}

fn generation_style() -> ProgressStyle {
    bar_style("{prefix} [{wide_bar}] {pos}/{len} ETA {eta_precise} {msg}")
}

fn evaluation_style() -> ProgressStyle {
    bar_style("{prefix} [{wide_bar}] {pos}/{len} {msg}")
}

fn bar_style(template: &str) -> ProgressStyle {
    ProgressStyle::with_template(template)
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("=>-")
}

fn status_label(status: EvolutionStatus) -> &'static str {
    match status {
        EvolutionStatus::Running => "running",
        EvolutionStatus::Stagnated => "stagnated",
        EvolutionStatus::Completed => "completed",
    }
}

fn truncate_smarts(smarts: &str, max_width: usize) -> String {
    let max_width = max_width.max(3);
    if smarts.chars().count() <= max_width {
        return smarts.to_string();
    }

    let prefix: String = smarts.chars().take(max_width.saturating_sub(3)).collect();
    format!("{prefix}...")
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use std::vec;

    use smarts_rs::PreparedTarget;
    use smiles_parser::Smiles;

    use super::*;
    use crate::fitness::evaluator::{FoldData, FoldSample};

    fn target(smiles: &str) -> PreparedTarget {
        PreparedTarget::new(Smiles::from_str(smiles).unwrap())
    }

    fn sample(smiles: &str, is_positive: bool) -> FoldSample {
        FoldSample::new(target(smiles), is_positive)
    }

    #[test]
    fn truncate_smarts_preserves_short_values_and_shortens_long_values() {
        assert_eq!(truncate_smarts("[#6]", 8), "[#6]");
        assert_eq!(truncate_smarts("[#6]~[#7]", 7), "[#6]...");
        assert_eq!(truncate_smarts("[#6]", 1), "...");
    }

    #[test]
    fn hidden_indicatif_runner_completes() {
        let task = EvolutionTask::new(
            "amide-indicatif",
            vec![FoldData::new(vec![
                sample("CC(=O)N", true),
                sample("NC(=O)C", true),
                sample("CCO", false),
                sample("CCCl", false),
            ])],
        );
        let config = EvolutionConfig::builder()
            .population_size(8)
            .generation_limit(1)
            .stagnation_limit(1)
            .build()
            .unwrap();
        let seed_corpus = SeedCorpus::try_from(["[#6](=[#8])[#7]", "[#6]~[#7]", "[#7]"]).unwrap();

        let result = task
            .evolve_with_indicatif_progress(
                &config,
                &seed_corpus,
                2,
                IndicatifEvolutionProgress::hidden(),
            )
            .unwrap();

        assert_eq!(result.task_id(), "amide-indicatif");
        assert_eq!(result.generations(), 1);
        assert!(result.best_mcc().is_finite());
        assert!(!result.best_smarts().is_empty());
    }

    #[test]
    fn owned_and_session_indicatif_methods_accept_caller_bars() {
        let folds = vec![FoldData::new(vec![
            sample("CC(=O)N", true),
            sample("NC(=O)C", true),
            sample("CCO", false),
            sample("CCCl", false),
        ])];
        let config = EvolutionConfig::builder()
            .population_size(8)
            .generation_limit(1)
            .stagnation_limit(1)
            .build()
            .unwrap();
        let seed_corpus = SeedCorpus::try_from(["[#6](=[#8])[#7]", "[#6]~[#7]", "[#7]"]).unwrap();

        let multi = MultiProgress::with_draw_target(ProgressDrawTarget::hidden());
        let owned_progress = IndicatifEvolutionProgress::from_bars(
            multi.add(ProgressBar::new(1)),
            multi.add(ProgressBar::new(1)),
        )
        .with_best_smarts_width(16)
        .with_generation_style(bar_style("{prefix} {pos}/{len} {msg}"))
        .with_evaluation_style(bar_style("{prefix} {pos}/{len} {msg}"))
        .with_prefixes("evolution", "SMARTS")
        .clear_on_finish(true);
        let owned_result = EvolutionTask::new("owned-indicatif", folds.clone())
            .evolve_owned_with_indicatif_progress(&config, &seed_corpus, 2, owned_progress)
            .unwrap();

        let session = EvolutionSession::from_owned_task(
            EvolutionTask::new("session-indicatif", folds),
            &config,
            &seed_corpus,
            2,
        )
        .unwrap();
        let session_result = session
            .evolve_with_indicatif_progress(IndicatifEvolutionProgress::attach_to(&multi))
            .unwrap();

        assert_eq!(owned_result.task_id(), "owned-indicatif");
        assert_eq!(session_result.task_id(), "session-indicatif");
        assert!(owned_result.best_mcc().is_finite());
        assert!(session_result.best_mcc().is_finite());
    }
}
