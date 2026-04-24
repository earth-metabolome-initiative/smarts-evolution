//! Terminal progress bars for evolution runs.

use alloc::format;
use alloc::string::{String, ToString};

use ::indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};

use super::config::EvolutionConfig;
use super::runner::{
    EvolutionError, EvolutionEvaluationProgress, EvolutionProgress, EvolutionSession,
    EvolutionStatus, EvolutionTask, TaskResult,
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
        let generation_bar = multi.add(
            ProgressBar::new(1)
                .with_style(generation_style())
                .with_prefix("generations"),
        );
        let evaluation_bar = multi.add(
            ProgressBar::new(1)
                .with_style(evaluation_style())
                .with_prefix("evaluating"),
        );

        Self {
            generation_bar,
            evaluation_bar,
            best_smarts_width: DEFAULT_BEST_SMARTS_WIDTH,
        }
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
        self.evaluation_bar.set_message(format!(
            "generation={}/{} SMARTS={}/{}",
            progress.generation(),
            progress.generation_limit(),
            progress.completed(),
            progress.total()
        ));
    }

    fn record_generation(&mut self, progress: &EvolutionProgress) {
        self.generation_bar
            .set_length(progress.generation_limit().max(1));
        self.generation_bar.set_position(progress.generation());
        self.generation_bar
            .set_message(self.generation_message(progress));
    }

    fn finish(&mut self, result: &TaskResult) {
        self.evaluation_bar.finish_and_clear();
        self.generation_bar.finish_with_message(format!(
            "done generations={} best_mcc={:.3} best_smarts={}",
            result.generations(),
            result.best_mcc(),
            truncate_smarts(result.best_smarts(), self.best_smarts_width)
        ));
    }

    fn abandon(&mut self, error: &EvolutionError) {
        self.evaluation_bar.finish_and_clear();
        self.generation_bar
            .abandon_with_message(format!("failed: {error}"));
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
        mut progress: IndicatifEvolutionProgress,
    ) -> Result<TaskResult, EvolutionError> {
        progress.start(self.task_id(), config.generation_limit());
        let mut session = match EvolutionSession::new(self, config, seed_corpus, leaderboard_size) {
            Ok(session) => session,
            Err(error) => {
                progress.abandon(&error);
                return Err(error);
            }
        };

        while let Some(snapshot) = session
            .step_with_evaluation_progress(|evaluation| progress.record_evaluation(&evaluation))
        {
            progress.record_generation(&snapshot);
            if session.is_finished() {
                let result = session.take_result().ok_or_else(|| {
                    EvolutionError::InvalidConfig(
                        "finished evolution session did not expose a terminal result".into(),
                    )
                })?;
                progress.finish(&result);
                return Ok(result);
            }
        }

        match session.take_result() {
            Some(result) => {
                progress.finish(&result);
                Ok(result)
            }
            None => {
                let error =
                    EvolutionError::InvalidConfig("evolution session ended unexpectedly".into());
                progress.abandon(&error);
                Err(error)
            }
        }
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
}
