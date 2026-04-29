//! Ratatui dashboard for native evolution runs.

mod input;
mod layout;
mod metric;
mod platform;
mod render;
mod state;
mod worker;

#[cfg(test)]
mod tests;

use core::fmt;
use core::time::Duration;
use std::io;
use std::sync::mpsc;
use std::time::Instant;

use ratatui::crossterm::{event, execute};

use self::input::{copy_pending_clipboard_request, handle_terminal_input};
use self::render::render_dashboard;
use self::state::DashboardState;
use self::worker::{WorkerControl, WorkerOutcome, spawn_worker};
use super::config::EvolutionConfig;
use super::runner::{EvolutionError, EvolutionSession, EvolutionTask, TaskResult};
use crate::genome::seed::SeedCorpus;

pub use self::metric::TuiMetric;

const DEFAULT_DRAW_INTERVAL: Duration = Duration::from_millis(100);
const DEFAULT_BEST_SMARTS_WIDTH: usize = 72;

/// Errors produced by the native TUI runner.
#[derive(Debug)]
pub enum TuiEvolutionError {
    Evolution(EvolutionError),
    Terminal(std::io::Error),
    Stopped,
    WorkerDisconnected,
    WorkerPanicked,
}

impl fmt::Display for TuiEvolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evolution(error) => write!(f, "{error}"),
            Self::Terminal(error) => write!(f, "terminal error: {error}"),
            Self::Stopped => write!(f, "evolution stopped after the current generation"),
            Self::WorkerDisconnected => write!(f, "evolution worker disconnected"),
            Self::WorkerPanicked => write!(f, "evolution worker panicked"),
        }
    }
}

impl core::error::Error for TuiEvolutionError {}

impl From<EvolutionError> for TuiEvolutionError {
    fn from(error: EvolutionError) -> Self {
        Self::Evolution(error)
    }
}

impl From<std::io::Error> for TuiEvolutionError {
    fn from(error: std::io::Error) -> Self {
        Self::Terminal(error)
    }
}

/// Native Ratatui dashboard for one evolution run.
#[derive(Clone, Debug)]
pub struct TuiEvolutionDashboard {
    draw_interval: Duration,
    best_smarts_width: usize,
}

impl Default for TuiEvolutionDashboard {
    fn default() -> Self {
        Self::new()
    }
}

impl TuiEvolutionDashboard {
    /// Create a dashboard with conservative refresh settings.
    pub const fn new() -> Self {
        Self {
            draw_interval: DEFAULT_DRAW_INTERVAL,
            best_smarts_width: DEFAULT_BEST_SMARTS_WIDTH,
        }
    }

    /// Set the minimum interval between terminal redraws.
    pub const fn with_draw_interval(mut self, interval: Duration) -> Self {
        self.draw_interval = interval;
        self
    }

    /// Set the maximum SMARTS width used in the bottom preview.
    pub const fn with_best_smarts_width(mut self, width: usize) -> Self {
        self.best_smarts_width = width;
        self
    }

    /// Drive one session to completion while rendering the dashboard.
    pub fn run_session(self, session: EvolutionSession) -> Result<TaskResult, TuiEvolutionError> {
        let mut terminal = TerminalGuard::start()?;
        let (event_tx, event_rx) = mpsc::channel();
        let (control_tx, control_rx) = mpsc::channel();
        let worker = spawn_worker(session, event_tx, control_rx);
        let mut state = DashboardState::new(self.best_smarts_width);
        let mut final_result = None;
        let mut last_draw = Instant::now() - self.draw_interval;
        let mut stop_requested = false;

        loop {
            while let Ok(event) = event_rx.try_recv() {
                match state.apply_worker_event(event) {
                    WorkerOutcome::Continue => {}
                    WorkerOutcome::Finished(result) => final_result = Some(Ok(result)),
                    WorkerOutcome::Failed(error) => final_result = Some(Err(error)),
                    WorkerOutcome::Stopped => final_result = Some(Err(TuiEvolutionError::Stopped)),
                }
            }

            if let Err(error) = handle_terminal_input(&mut state, &control_tx, &mut stop_requested)
            {
                let _ = control_tx.send(WorkerControl::Stop);
                final_result = Some(Err(error));
            }
            copy_pending_clipboard_request(&mut state);

            if (last_draw.elapsed() >= self.draw_interval || final_result.is_some())
                && !is_terminal_error(&final_result)
            {
                if let Err(error) = terminal.draw(&mut state) {
                    let _ = control_tx.send(WorkerControl::Stop);
                    final_result = Some(Err(error));
                }
                last_draw = Instant::now();
            }

            if final_result.is_some() {
                break;
            }

            match event_rx.recv_timeout(self.draw_interval) {
                Ok(event) => match state.apply_worker_event(event) {
                    WorkerOutcome::Continue => {}
                    WorkerOutcome::Finished(result) => final_result = Some(Ok(result)),
                    WorkerOutcome::Failed(error) => final_result = Some(Err(error)),
                    WorkerOutcome::Stopped => {
                        final_result = Some(Err(TuiEvolutionError::Stopped));
                    }
                },
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    final_result = Some(Err(TuiEvolutionError::WorkerDisconnected));
                }
            }
        }

        let result = final_result.unwrap_or(Err(TuiEvolutionError::WorkerDisconnected));
        terminal.restore()?;
        worker
            .join()
            .map_err(|_| TuiEvolutionError::WorkerPanicked)?;

        result
    }
}

fn is_terminal_error(result: &Option<Result<TaskResult, TuiEvolutionError>>) -> bool {
    matches!(result, Some(Err(TuiEvolutionError::Terminal(_))))
}

impl EvolutionTask {
    /// Evolve this task with the default native TUI dashboard.
    ///
    /// Enable the `tui` cargo feature to use this method.
    pub fn evolve_with_tui(
        &self,
        config: &EvolutionConfig,
        seed_corpus: &SeedCorpus,
    ) -> Result<TaskResult, TuiEvolutionError> {
        self.evolve_with_tui_dashboard(config, seed_corpus, 1, TuiEvolutionDashboard::new())
    }

    /// Evolve this task with a caller-configured native TUI dashboard.
    pub fn evolve_with_tui_dashboard(
        &self,
        config: &EvolutionConfig,
        seed_corpus: &SeedCorpus,
        leaderboard_size: usize,
        dashboard: TuiEvolutionDashboard,
    ) -> Result<TaskResult, TuiEvolutionError> {
        let session = EvolutionSession::new(self, config, seed_corpus, leaderboard_size)?;
        dashboard.run_session(session)
    }

    /// Evolve this task by moving its folds into the session with the default
    /// native TUI dashboard.
    pub fn evolve_owned_with_tui(
        self,
        config: &EvolutionConfig,
        seed_corpus: &SeedCorpus,
    ) -> Result<TaskResult, TuiEvolutionError> {
        self.evolve_owned_with_tui_dashboard(config, seed_corpus, 1, TuiEvolutionDashboard::new())
    }

    /// Evolve this task by moving its folds into the session with a
    /// caller-configured native TUI dashboard.
    pub fn evolve_owned_with_tui_dashboard(
        self,
        config: &EvolutionConfig,
        seed_corpus: &SeedCorpus,
        leaderboard_size: usize,
        dashboard: TuiEvolutionDashboard,
    ) -> Result<TaskResult, TuiEvolutionError> {
        let session =
            EvolutionSession::from_owned_task(self, config, seed_corpus, leaderboard_size)?;
        dashboard.run_session(session)
    }
}

impl EvolutionSession {
    /// Drive this session to completion with the default native TUI dashboard.
    pub fn evolve_with_tui(self) -> Result<TaskResult, TuiEvolutionError> {
        TuiEvolutionDashboard::new().run_session(self)
    }

    /// Drive this session to completion with a caller-configured native TUI dashboard.
    pub fn evolve_with_tui_dashboard(
        self,
        dashboard: TuiEvolutionDashboard,
    ) -> Result<TaskResult, TuiEvolutionError> {
        dashboard.run_session(self)
    }
}

struct TerminalGuard {
    terminal: ratatui::DefaultTerminal,
    mouse_capture_enabled: bool,
    restored: bool,
}

impl TerminalGuard {
    fn start() -> Result<Self, TuiEvolutionError> {
        let terminal = ratatui::try_init()?;
        if let Err(error) = execute!(io::stdout(), event::EnableMouseCapture) {
            let _ = ratatui::try_restore();
            return Err(TuiEvolutionError::Terminal(error));
        }
        Ok(Self {
            terminal,
            mouse_capture_enabled: true,
            restored: false,
        })
    }

    fn draw(&mut self, state: &mut DashboardState) -> Result<(), TuiEvolutionError> {
        self.terminal.draw(|frame| render_dashboard(frame, state))?;
        Ok(())
    }

    fn restore(&mut self) -> Result<(), TuiEvolutionError> {
        let mut mouse_error = None;
        if self.mouse_capture_enabled {
            if let Err(error) = execute!(io::stdout(), event::DisableMouseCapture) {
                mouse_error = Some(error);
            }
            self.mouse_capture_enabled = false;
        }
        let restore_result = ratatui::try_restore();
        self.restored = true;
        if let Some(error) = mouse_error {
            return Err(TuiEvolutionError::Terminal(error));
        }
        restore_result?;
        Ok(())
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if !self.restored {
            if self.mouse_capture_enabled {
                let _ = execute!(io::stdout(), event::DisableMouseCapture);
                self.mouse_capture_enabled = false;
            }
            let _ = ratatui::try_restore();
        }
    }
}
