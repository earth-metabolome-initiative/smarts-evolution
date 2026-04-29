use alloc::string::{String, ToString};
use std::sync::mpsc::{Receiver, Sender};
use std::thread;

use super::TuiEvolutionError;
use crate::evolution::runner::{
    EvolutionError, EvolutionEvaluationProgress, EvolutionOffspringProgress, EvolutionProgress,
    EvolutionSession, TaskResult,
};

#[derive(Debug)]
pub(super) enum WorkerEvent {
    Started {
        task_id: String,
        generation_limit: u64,
    },
    Evaluation(EvolutionEvaluationProgress),
    Offspring(EvolutionOffspringProgress),
    Generation(EvolutionProgress),
    Paused,
    Resumed,
    Finished(Result<TaskResult, EvolutionError>),
    Stopped,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum WorkerControl {
    Stop,
    Pause,
    Resume,
}

pub(super) enum WorkerOutcome {
    Continue,
    Finished(TaskResult),
    Failed(TuiEvolutionError),
    Stopped,
}

pub(super) fn spawn_worker(
    mut session: EvolutionSession,
    event_tx: Sender<WorkerEvent>,
    control_rx: Receiver<WorkerControl>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let _ = event_tx.send(WorkerEvent::Started {
            task_id: session.task_id().to_string(),
            generation_limit: session.generation_limit(),
        });
        drive_session(&mut session, &event_tx, &control_rx);
    })
}

fn drive_session(
    session: &mut EvolutionSession,
    event_tx: &Sender<WorkerEvent>,
    control_rx: &Receiver<WorkerControl>,
) {
    let mut paused = false;
    loop {
        if process_worker_controls(&mut paused, control_rx, event_tx) {
            return;
        }
        while paused {
            match control_rx.recv() {
                Ok(WorkerControl::Resume) => {
                    paused = false;
                    let _ = event_tx.send(WorkerEvent::Resumed);
                }
                Ok(WorkerControl::Stop) | Err(_) => {
                    let _ = event_tx.send(WorkerEvent::Stopped);
                    return;
                }
                Ok(WorkerControl::Pause) => {}
            }
        }

        let evaluation_tx = event_tx.clone();
        let offspring_tx = event_tx.clone();
        let progress = session.step_with_evaluation_and_offspring_progress(
            move |progress| {
                let _ = evaluation_tx.send(WorkerEvent::Evaluation(progress));
            },
            move |progress| {
                let _ = offspring_tx.send(WorkerEvent::Offspring(progress));
            },
        );

        let Some(progress) = progress else {
            let result = session.take_result().ok_or_else(|| {
                EvolutionError::InvalidConfig("evolution session ended unexpectedly".into())
            });
            let _ = event_tx.send(WorkerEvent::Finished(result));
            return;
        };

        let _ = event_tx.send(WorkerEvent::Generation(progress));

        if session.is_finished() {
            let result = session.take_result().ok_or_else(|| {
                EvolutionError::InvalidConfig(
                    "finished evolution session did not expose a terminal result".into(),
                )
            });
            let _ = event_tx.send(WorkerEvent::Finished(result));
            return;
        }
    }
}

pub(super) fn process_worker_controls(
    paused: &mut bool,
    control_rx: &Receiver<WorkerControl>,
    event_tx: &Sender<WorkerEvent>,
) -> bool {
    let mut target_pause = None;
    while let Ok(control) = control_rx.try_recv() {
        match control {
            WorkerControl::Stop => {
                let _ = event_tx.send(WorkerEvent::Stopped);
                return true;
            }
            WorkerControl::Pause => target_pause = Some(true),
            WorkerControl::Resume => target_pause = Some(false),
        }
    }

    match target_pause {
        Some(true) if !*paused => {
            *paused = true;
            let _ = event_tx.send(WorkerEvent::Paused);
        }
        Some(false) if *paused => {
            *paused = false;
            let _ = event_tx.send(WorkerEvent::Resumed);
        }
        _ => {}
    }
    false
}
