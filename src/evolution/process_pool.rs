use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, BufWriter, Write};
#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use indicatif::ProgressBar;
use log::warn;
use serde::{Deserialize, Serialize};

use crate::fitness::mcc::{ConfusionCounts, MccFitness, compute_fold_averaged_mcc};
use crate::genome::genome::SmartsGenome;
use crate::rdkit_substruct_library::{CompiledSmartsQuery, SubstructLibraryIndex};

const EVALUATION_CHUNK_SIZE: usize = 128;
const WORKER_QUERY_CACHE_CAPACITY: usize = 8192;

#[derive(Clone, Serialize, Deserialize)]
pub struct WorkerFoldPayload {
    pub smiles: Vec<String>,
    pub is_positive: Vec<bool>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct WorkerNodeContext {
    pub folds: Vec<WorkerFoldPayload>,
}

#[derive(Clone, Serialize, Deserialize)]
enum WorkerRequest {
    SetNode(WorkerNodeContext),
    EvaluateBatch { genomes: Vec<String> },
    Shutdown,
}

#[derive(Serialize, Deserialize)]
enum WorkerResponse {
    Ack,
    BatchCounts { results: Vec<WorkerGenomeResult> },
    Error { message: String },
}

#[derive(Clone, Serialize, Deserialize)]
struct WorkerGenomeResult {
    valid: bool,
    fold_counts: Vec<ConfusionCounts>,
}

enum SupervisorMessage {
    Request { job_id: u64, request: WorkerRequest },
    Shutdown,
}

struct WorkerHandle {
    sender: Sender<SupervisorMessage>,
    join_handle: Option<thread::JoinHandle<()>>,
}

struct PoolResult {
    job_id: u64,
    response: Result<WorkerResponse, String>,
}

struct PendingWorkerJob {
    worker_idx: usize,
    genome_start: usize,
    genome_count: usize,
}

struct AggregatedBatchCounts {
    fold_counts: Vec<Vec<ConfusionCounts>>,
    invalid: Vec<bool>,
}

pub struct ProcessEvaluationBackend {
    workers: Vec<WorkerHandle>,
    results: Receiver<PoolResult>,
    result_tx: Sender<PoolResult>,
    next_job_id: u64,
    worker_executable: PathBuf,
    worker_contexts: Option<Vec<WorkerNodeContext>>,
}

impl ProcessEvaluationBackend {
    pub fn spawn(worker_processes: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Self::spawn_with_progress(worker_processes, None)
    }

    pub fn spawn_with_progress(
        worker_processes: usize,
        progress_pb: Option<&ProgressBar>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let current_exe = std::env::current_exe()?;
        Self::spawn_with_executable_and_progress(&current_exe, worker_processes, progress_pb)
    }

    pub fn spawn_with_executable(
        worker_executable: &Path,
        worker_processes: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::spawn_with_executable_and_progress(worker_executable, worker_processes, None)
    }

    fn spawn_with_executable_and_progress(
        worker_executable: &Path,
        worker_processes: usize,
        progress_pb: Option<&ProgressBar>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let worker_processes = worker_processes.max(1);
        let (result_tx, result_rx) = mpsc::channel();
        let mut workers = Vec::with_capacity(worker_processes);

        if let Some(pb) = progress_pb {
            pb.set_length(worker_processes as u64);
            pb.set_position(0);
            pb.set_message("startup: spawning worker processes".to_string());
        }
        for worker_idx in 0..worker_processes {
            workers.push(spawn_worker_supervisor(
                worker_idx,
                worker_executable,
                result_tx.clone(),
            )?);
            if let Some(pb) = progress_pb {
                pb.inc(1);
            }
        }

        let backend = Self {
            workers,
            results: result_rx,
            result_tx,
            next_job_id: 0,
            worker_executable: worker_executable.to_path_buf(),
            worker_contexts: None,
        };
        Ok(backend)
    }

    pub fn set_node_context(
        &mut self,
        context: WorkerNodeContext,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.set_node_context_with_progress(context, None)
    }

    pub fn set_node_context_with_progress(
        &mut self,
        context: WorkerNodeContext,
        progress_pb: Option<&ProgressBar>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if context
            .folds
            .iter()
            .any(|fold| fold.smiles.len() != fold.is_positive.len())
        {
            return Err("worker received inconsistent fold lengths".into());
        }

        let worker_contexts = shard_node_context(context, self.workers.len());
        if let Some(pb) = progress_pb {
            pb.set_length(self.workers.len().max(1) as u64);
            pb.set_position(0);
            pb.set_message("setup: indexing worker shards".to_string());
        }
        for (worker_idx, worker_context) in worker_contexts.iter().cloned().enumerate() {
            match self.request_worker(worker_idx, WorkerRequest::SetNode(worker_context))? {
                WorkerResponse::Ack => {
                    if let Some(pb) = progress_pb {
                        pb.inc(1);
                    }
                }
                WorkerResponse::Error { message } => return Err(message.into()),
                _ => {
                    return Err("worker returned unexpected response during node setup".into());
                }
            }
        }
        self.worker_contexts = Some(worker_contexts);
        Ok(())
    }

    pub fn score_population(
        &mut self,
        population: Vec<SmartsGenome>,
        step_pb: &ProgressBar,
    ) -> Result<Vec<(SmartsGenome, i64)>, Box<dyn std::error::Error>> {
        if population.is_empty() {
            return Ok(Vec::new());
        }

        let genome_smarts: Vec<String> = population
            .iter()
            .map(|genome| genome.smarts_string.clone())
            .collect();
        let fold_count = self.fold_count()?;
        let aggregated =
            self.evaluate_counts_across_workers(&genome_smarts, fold_count, Some(step_pb))?;

        let mut scored = Vec::with_capacity(population.len());
        for (idx, genome) in population.into_iter().enumerate() {
            let score = if aggregated.invalid[idx] {
                0
            } else {
                MccFitness::from_mcc(compute_fold_averaged_mcc(&aggregated.fold_counts[idx])).score
            };
            scored.push((genome, score));
        }

        Ok(scored)
    }

    pub fn mcc_of(&mut self, genome: &SmartsGenome) -> Result<f64, Box<dyn std::error::Error>> {
        let fold_count = self.fold_count()?;
        let aggregated =
            self.evaluate_counts_across_workers(&[genome.smarts_string.clone()], fold_count, None)?;
        if aggregated.invalid[0] {
            Ok(-1.0)
        } else {
            Ok(compute_fold_averaged_mcc(&aggregated.fold_counts[0]))
        }
    }

    fn dispatch_to_worker(
        &self,
        worker_idx: usize,
        job_id: u64,
        request: WorkerRequest,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.workers[worker_idx]
            .sender
            .send(SupervisorMessage::Request { job_id, request })?;
        Ok(())
    }

    fn next_job_id(&mut self) -> u64 {
        let job_id = self.next_job_id;
        self.next_job_id += 1;
        job_id
    }

    fn recv_expected_job(&self, job_id: u64) -> Result<WorkerResponse, String> {
        let result = self.results.recv().map_err(|err| err.to_string())?;
        if result.job_id != job_id {
            return Err(format!(
                "expected response for job {job_id}, got {}",
                result.job_id
            ));
        }
        result.response
    }

    fn restart_worker(&mut self, worker_idx: usize) -> Result<(), Box<dyn std::error::Error>> {
        let replacement =
            spawn_worker_supervisor(worker_idx, &self.worker_executable, self.result_tx.clone())?;
        let mut old_worker = std::mem::replace(&mut self.workers[worker_idx], replacement);
        let _ = old_worker.sender.send(SupervisorMessage::Shutdown);
        if let Some(join_handle) = old_worker.join_handle.take() {
            let _ = join_handle.join();
        }

        self.initialize_worker(worker_idx)?;
        Ok(())
    }

    fn initialize_worker(&mut self, worker_idx: usize) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(context) = self
            .worker_contexts
            .as_ref()
            .and_then(|c| c.get(worker_idx))
        {
            match self.request_worker(worker_idx, WorkerRequest::SetNode(context.clone()))? {
                WorkerResponse::Ack => {}
                WorkerResponse::Error { message } => return Err(message.into()),
                _ => {
                    return Err("worker returned unexpected response during node reinit".into());
                }
            }
        }

        Ok(())
    }

    fn fold_count(&self) -> Result<usize, Box<dyn std::error::Error>> {
        self.worker_contexts
            .as_ref()
            .and_then(|contexts| contexts.first())
            .map(|context| context.folds.len())
            .ok_or_else(|| "worker evaluator is not initialized for a node".into())
    }

    fn evaluate_counts_across_workers(
        &mut self,
        genomes: &[String],
        fold_count: usize,
        progress_pb: Option<&ProgressBar>,
    ) -> Result<AggregatedBatchCounts, Box<dyn std::error::Error>> {
        let mut aggregated = AggregatedBatchCounts {
            fold_counts: vec![vec![ConfusionCounts::default(); fold_count]; genomes.len()],
            invalid: vec![false; genomes.len()],
        };
        let chunk_size = EVALUATION_CHUNK_SIZE.max(1);
        let chunk_count = genomes.len().div_ceil(chunk_size);
        if let Some(pb) = progress_pb {
            pb.set_length((chunk_count * self.workers.len()).max(1) as u64);
            pb.set_position(0);
        }
        let mut pending: HashMap<u64, PendingWorkerJob> =
            HashMap::with_capacity(self.workers.len());
        let mut next_chunk_start = vec![0usize; self.workers.len()];

        for worker_idx in 0..self.workers.len() {
            schedule_next_worker_chunk(
                self,
                worker_idx,
                genomes,
                &mut next_chunk_start,
                &mut pending,
            )?;
        }

        while !pending.is_empty() {
            let result = self.results.recv()?;
            let pending_job = pending
                .remove(&result.job_id)
                .ok_or_else(|| format!("unknown batch job id {}", result.job_id))?;
            match result.response {
                Ok(WorkerResponse::BatchCounts { results }) => {
                    merge_worker_batch_results(
                        &mut aggregated,
                        results,
                        fold_count,
                        pending_job.genome_start,
                        pending_job.genome_count,
                    )?;
                }
                Ok(WorkerResponse::Error { message }) => return Err(message.into()),
                Ok(_) => {
                    return Err("worker returned unexpected response for batch evaluation".into());
                }
                Err(message) => {
                    warn!(
                        "worker {} died while scoring shard chunk [{}..{}): {}",
                        pending_job.worker_idx,
                        pending_job.genome_start,
                        pending_job.genome_start + pending_job.genome_count,
                        message
                    );
                    self.restart_worker(pending_job.worker_idx)?;
                    let rescued = self.evaluate_worker_batch_resilient(
                        pending_job.worker_idx,
                        &genomes[pending_job.genome_start
                            ..pending_job.genome_start + pending_job.genome_count],
                        fold_count,
                    )?;
                    merge_worker_batch_results(
                        &mut aggregated,
                        rescued,
                        fold_count,
                        pending_job.genome_start,
                        pending_job.genome_count,
                    )?;
                }
            }
            if let Some(pb) = progress_pb {
                pb.inc(1);
            }
            schedule_next_worker_chunk(
                self,
                pending_job.worker_idx,
                genomes,
                &mut next_chunk_start,
                &mut pending,
            )?;
        }

        Ok(aggregated)
    }

    fn evaluate_worker_batch_resilient(
        &mut self,
        worker_idx: usize,
        genomes: &[String],
        fold_count: usize,
    ) -> Result<Vec<WorkerGenomeResult>, Box<dyn std::error::Error>> {
        match self.request_worker(
            worker_idx,
            WorkerRequest::EvaluateBatch {
                genomes: genomes.to_vec(),
            },
        ) {
            Ok(WorkerResponse::BatchCounts { results }) => {
                validate_worker_batch_results(&results, genomes.len(), fold_count)?;
                return Ok(results);
            }
            Ok(WorkerResponse::Error { message }) => return Err(message.into()),
            Ok(_) => {
                return Err("worker returned unexpected response for shard batch rescue".into());
            }
            Err(err) => {
                warn!(
                    "worker {} failed full shard batch retry: {}. Falling back to per-genome rescue.",
                    worker_idx, err
                );
                self.restart_worker(worker_idx)?;
            }
        }

        let mut rescued = Vec::with_capacity(genomes.len());
        for genome in genomes {
            match self.request_worker(
                worker_idx,
                WorkerRequest::EvaluateBatch {
                    genomes: vec![genome.clone()],
                },
            ) {
                Ok(WorkerResponse::BatchCounts { mut results }) if results.len() == 1 => {
                    let result = results.pop().unwrap();
                    validate_worker_batch_results(std::slice::from_ref(&result), 1, fold_count)?;
                    rescued.push(result);
                }
                Ok(WorkerResponse::BatchCounts { results }) => {
                    return Err(format!(
                        "worker returned {} results for a single-genome shard rescue batch",
                        results.len()
                    )
                    .into());
                }
                Ok(WorkerResponse::Error { message }) => return Err(message.into()),
                Ok(_) => {
                    return Err(
                        "worker returned unexpected response for single-genome shard rescue batch"
                            .into(),
                    );
                }
                Err(err) => {
                    warn!(
                        "genome '{}' crashed worker {} during shard rescue: {}; assigning zero fitness",
                        genome, worker_idx, err
                    );
                    self.restart_worker(worker_idx)?;
                    rescued.push(WorkerGenomeResult {
                        valid: false,
                        fold_counts: vec![ConfusionCounts::default(); fold_count],
                    });
                }
            }
        }

        Ok(rescued)
    }

    fn request_worker(
        &mut self,
        worker_idx: usize,
        request: WorkerRequest,
    ) -> Result<WorkerResponse, Box<dyn std::error::Error>> {
        let job_id = self.next_job_id();
        self.dispatch_to_worker(worker_idx, job_id, request)?;
        Ok(self
            .recv_expected_job(job_id)
            .map_err(|message| -> Box<dyn std::error::Error> { message.into() })?)
    }
}

fn validate_worker_batch_results(
    results: &[WorkerGenomeResult],
    genome_count: usize,
    fold_count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if results.len() != genome_count {
        return Err(format!(
            "worker returned {} genome results for {} requested genomes",
            results.len(),
            genome_count
        )
        .into());
    }

    for result in results {
        if result.fold_counts.len() != fold_count {
            return Err(format!(
                "worker returned {} fold counts for {fold_count} expected folds",
                result.fold_counts.len()
            )
            .into());
        }
    }

    Ok(())
}

fn merge_worker_batch_results(
    aggregated: &mut AggregatedBatchCounts,
    results: Vec<WorkerGenomeResult>,
    fold_count: usize,
    genome_start: usize,
    genome_count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_worker_batch_results(&results, genome_count, fold_count)?;
    if genome_start + genome_count > aggregated.fold_counts.len() {
        return Err(format!(
            "worker returned chunk [{}..{}) outside aggregate genome range {}",
            genome_start,
            genome_start + genome_count,
            aggregated.fold_counts.len()
        )
        .into());
    }

    for (genome_idx, result) in results.into_iter().enumerate() {
        let aggregate_idx = genome_start + genome_idx;
        if !result.valid {
            aggregated.invalid[aggregate_idx] = true;
            continue;
        }

        if aggregated.invalid[aggregate_idx] {
            continue;
        }

        for (fold_idx, counts) in result.fold_counts.into_iter().enumerate() {
            aggregated.fold_counts[aggregate_idx][fold_idx] += counts;
        }
    }

    Ok(())
}

fn schedule_next_worker_chunk(
    backend: &mut ProcessEvaluationBackend,
    worker_idx: usize,
    genomes: &[String],
    next_chunk_start: &mut [usize],
    pending: &mut HashMap<u64, PendingWorkerJob>,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = next_chunk_start[worker_idx];
    if start >= genomes.len() {
        return Ok(());
    }
    let end = (start + EVALUATION_CHUNK_SIZE).min(genomes.len());
    let job_id = backend.next_job_id();
    backend.dispatch_to_worker(
        worker_idx,
        job_id,
        WorkerRequest::EvaluateBatch {
            genomes: genomes[start..end].to_vec(),
        },
    )?;
    pending.insert(
        job_id,
        PendingWorkerJob {
            worker_idx,
            genome_start: start,
            genome_count: end - start,
        },
    );
    next_chunk_start[worker_idx] = end;
    Ok(())
}

fn shard_node_context(context: WorkerNodeContext, worker_count: usize) -> Vec<WorkerNodeContext> {
    let worker_count = worker_count.max(1);
    let mut worker_contexts = vec![
        WorkerNodeContext {
            folds: context
                .folds
                .iter()
                .map(|_| WorkerFoldPayload {
                    smiles: Vec::new(),
                    is_positive: Vec::new(),
                })
                .collect(),
        };
        worker_count
    ];

    for (fold_idx, fold) in context.folds.into_iter().enumerate() {
        debug_assert_eq!(fold.smiles.len(), fold.is_positive.len());
        for (item_idx, (smiles, is_positive)) in
            fold.smiles.into_iter().zip(fold.is_positive).enumerate()
        {
            let worker_idx = item_idx % worker_count;
            worker_contexts[worker_idx].folds[fold_idx]
                .smiles
                .push(smiles);
            worker_contexts[worker_idx].folds[fold_idx]
                .is_positive
                .push(is_positive);
        }
    }

    worker_contexts
}

impl Drop for ProcessEvaluationBackend {
    fn drop(&mut self) {
        for worker in &self.workers {
            let _ = worker.sender.send(SupervisorMessage::Shutdown);
        }
        for worker in &mut self.workers {
            if let Some(join_handle) = worker.join_handle.take() {
                let _ = join_handle.join();
            }
        }
    }
}

fn spawn_worker_supervisor(
    worker_idx: usize,
    current_exe: &std::path::Path,
    result_tx: Sender<PoolResult>,
) -> Result<WorkerHandle, Box<dyn std::error::Error>> {
    let mut child = Command::new(current_exe)
        .arg("worker")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdin = child
        .stdin
        .take()
        .ok_or("worker process did not provide stdin")?;
    let stdout = child
        .stdout
        .take()
        .ok_or("worker process did not provide stdout")?;
    let (sender, receiver) = mpsc::channel();
    let join_handle = thread::Builder::new()
        .name(format!("rdkit-worker-supervisor-{worker_idx}"))
        .spawn(move || supervisor_loop(child, stdin, stdout, receiver, result_tx))?;

    Ok(WorkerHandle {
        sender,
        join_handle: Some(join_handle),
    })
}

fn supervisor_loop(
    mut child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    receiver: Receiver<SupervisorMessage>,
    result_tx: Sender<PoolResult>,
) {
    let mut writer = BufWriter::new(stdin);
    let mut reader = BufReader::new(stdout);

    for message in receiver {
        match message {
            SupervisorMessage::Request { job_id, request } => {
                let (response, worker_exited) =
                    match send_request(&mut writer, &mut reader, &request) {
                        Ok(response) => (Ok(response), false),
                        Err(err) => describe_worker_failure(&mut child, &request, err.to_string()),
                    };
                let _ = result_tx.send(PoolResult { job_id, response });
                if worker_exited {
                    break;
                }
            }
            SupervisorMessage::Shutdown => {
                let _ = send_request(&mut writer, &mut reader, &WorkerRequest::Shutdown);
                break;
            }
        }
    }

    let _ = child.wait();
}

fn send_request(
    writer: &mut BufWriter<ChildStdin>,
    reader: &mut BufReader<ChildStdout>,
    request: &WorkerRequest,
) -> Result<WorkerResponse, Box<dyn std::error::Error>> {
    serde_json::to_writer(&mut *writer, request)?;
    writer.write_all(b"\n")?;
    writer.flush()?;

    let mut line = String::new();
    let read = reader.read_line(&mut line)?;
    if read == 0 {
        return Err("worker process closed its stdout".into());
    }
    Ok(serde_json::from_str(&line)?)
}

fn describe_worker_failure(
    child: &mut Child,
    request: &WorkerRequest,
    error: String,
) -> (Result<WorkerResponse, String>, bool) {
    let request_desc = describe_worker_request(request);
    match child.try_wait() {
        Ok(Some(status)) => (
            Err(format!(
                "worker request failed during {request_desc}: {error}; worker exited with {}",
                format_exit_status(status)
            )),
            true,
        ),
        Ok(None) => (
            Err(format!(
                "worker request failed during {request_desc}: {error}"
            )),
            false,
        ),
        Err(wait_err) => (
            Err(format!(
                "worker request failed during {request_desc}: {error}; failed to inspect worker status: {wait_err}"
            )),
            false,
        ),
    }
}

fn describe_worker_request(request: &WorkerRequest) -> String {
    match request {
        WorkerRequest::SetNode(context) => {
            format!("node context setup for {} folds", context.folds.len())
        }
        WorkerRequest::EvaluateBatch { genomes } => match genomes.as_slice() {
            [] => "batch evaluation for 0 genomes".to_string(),
            [genome] => format!("batch evaluation for genome '{genome}'"),
            [first, ..] => {
                format!(
                    "batch evaluation for {} genomes (first '{first}')",
                    genomes.len()
                )
            }
        },
        WorkerRequest::Shutdown => "worker shutdown".to_string(),
    }
}

fn format_exit_status(status: ExitStatus) -> String {
    if let Some(code) = status.code() {
        return format!("exit code {code}");
    }

    #[cfg(unix)]
    if let Some(signal) = status.signal() {
        let core = if status.core_dumped() {
            " (core dumped)"
        } else {
            ""
        };
        return format!("signal {signal}{core}");
    }

    status.to_string()
}

struct WorkerState {
    libraries: Option<Vec<SubstructLibraryIndex>>,
    query_cache: HashMap<String, Option<Arc<CompiledSmartsQuery>>>,
    query_cache_order: VecDeque<String>,
}

impl WorkerState {
    fn new() -> Self {
        Self {
            libraries: None,
            query_cache: HashMap::new(),
            query_cache_order: VecDeque::new(),
        }
    }

    fn set_node_context(
        &mut self,
        context: WorkerNodeContext,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut folds = Vec::with_capacity(context.folds.len());
        for fold in context.folds {
            if fold.smiles.len() != fold.is_positive.len() {
                return Err("worker received inconsistent fold lengths".into());
            }

            let mut library = SubstructLibraryIndex::new()?;

            for (smiles, is_positive) in fold.smiles.into_iter().zip(fold.is_positive) {
                library.add_smiles(&smiles, is_positive)?;
            }

            folds.push(library);
        }

        self.libraries = Some(folds);
        Ok(())
    }

    fn evaluate_batch(
        &mut self,
        genomes: Vec<String>,
    ) -> Result<Vec<WorkerGenomeResult>, Box<dyn std::error::Error>> {
        let fold_count = self
            .libraries
            .as_ref()
            .ok_or("worker evaluator is not initialized for a node")?
            .len();
        let mut results = Vec::with_capacity(genomes.len());
        for smarts in genomes {
            let Some(query) = self.cached_query(&smarts) else {
                results.push(WorkerGenomeResult {
                    valid: false,
                    fold_counts: vec![ConfusionCounts::default(); fold_count],
                });
                continue;
            };
            let libraries = self
                .libraries
                .as_ref()
                .ok_or("worker evaluator is not initialized for a node")?;
            results.push(evaluate_query_counts(libraries, query.as_ref()));
        }
        Ok(results)
    }

    fn cached_query(&mut self, smarts: &str) -> Option<Arc<CompiledSmartsQuery>> {
        if !self.query_cache.contains_key(smarts) {
            if self.query_cache.len() >= WORKER_QUERY_CACHE_CAPACITY {
                if let Some(oldest) = self.query_cache_order.pop_front() {
                    self.query_cache.remove(&oldest);
                }
            }
            self.query_cache_order.push_back(smarts.to_string());
            self.query_cache.insert(
                smarts.to_string(),
                CompiledSmartsQuery::new(smarts).map(Arc::new),
            );
        }

        self.query_cache.get(smarts).cloned().flatten()
    }
}

fn evaluate_query_counts(
    libraries: &[SubstructLibraryIndex],
    query: &CompiledSmartsQuery,
) -> WorkerGenomeResult {
    let mut fold_counts = Vec::with_capacity(libraries.len());
    for library in libraries {
        let Some(counts) = library.count_matches_compiled(query, 1) else {
            return WorkerGenomeResult {
                valid: false,
                fold_counts: vec![ConfusionCounts::default(); libraries.len()],
            };
        };
        fold_counts.push(counts);
    }

    WorkerGenomeResult {
        valid: true,
        fold_counts,
    }
}

pub fn run_worker_stdio() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = BufWriter::new(stdout.lock());
    let mut state = WorkerState::new();

    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let request: WorkerRequest = serde_json::from_str(&line)?;
        let response = match request {
            WorkerRequest::SetNode(context) => match state.set_node_context(context) {
                Ok(()) => WorkerResponse::Ack,
                Err(err) => WorkerResponse::Error {
                    message: err.to_string(),
                },
            },
            WorkerRequest::EvaluateBatch { genomes } => match state.evaluate_batch(genomes) {
                Ok(results) => WorkerResponse::BatchCounts { results },
                Err(err) => WorkerResponse::Error {
                    message: err.to_string(),
                },
            },
            WorkerRequest::Shutdown => {
                serde_json::to_writer(&mut writer, &WorkerResponse::Ack)?;
                writer.write_all(b"\n")?;
                writer.flush()?;
                break;
            }
        };

        serde_json::to_writer(&mut writer, &response)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_node_context_distributes_entries_across_workers() {
        let context = WorkerNodeContext {
            folds: vec![
                WorkerFoldPayload {
                    smiles: vec!["a".into(), "b".into(), "c".into(), "d".into()],
                    is_positive: vec![true, false, true, false],
                },
                WorkerFoldPayload {
                    smiles: vec!["e".into(), "f".into()],
                    is_positive: vec![false, true],
                },
            ],
        };

        let shards = shard_node_context(context, 2);

        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].folds[0].smiles, vec!["a", "c"]);
        assert_eq!(shards[0].folds[0].is_positive, vec![true, true]);
        assert_eq!(shards[1].folds[0].smiles, vec!["b", "d"]);
        assert_eq!(shards[1].folds[0].is_positive, vec![false, false]);

        assert_eq!(shards[0].folds[1].smiles, vec!["e"]);
        assert_eq!(shards[0].folds[1].is_positive, vec![false]);
        assert_eq!(shards[1].folds[1].smiles, vec!["f"]);
        assert_eq!(shards[1].folds[1].is_positive, vec![true]);
    }
}
