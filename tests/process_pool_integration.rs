use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Path as AxumPath, Query, State};
use axum::http::{StatusCode, header::CONTENT_TYPE};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use flate2::Compression;
use flate2::write::GzEncoder;
use genevo::genetic::FitnessFunction;
use indicatif::ProgressBar;
use serde_json::{Value, json};
use smarts_evolution::data::compound::Compound;
use smarts_evolution::data::molecule_cache::warmup_rdkit;
use smarts_evolution::data::rdkit_lock::with_rdkit_lock;
use smarts_evolution::evolution::process_pool::{
    ProcessEvaluationBackend, WorkerFoldPayload, WorkerNodeContext,
};
use smarts_evolution::fitness::evaluator::{FoldData, SmartsEvaluator};
use smarts_evolution::genome::genome::SmartsGenome;

struct TestFixture {
    local_evaluator: SmartsEvaluator,
    worker_context: WorkerNodeContext,
}

fn make_compound(cid: u64, smiles: &str, labels: &[&[&str]]) -> Compound {
    let parsed = with_rdkit_lock(|| rdkit::ROMol::from_smiles(smiles).is_ok());
    let labels = labels
        .iter()
        .map(|level| level.iter().map(|label| (*label).to_string()).collect())
        .collect();
    Compound {
        cid,
        smiles: smiles.to_string(),
        parsed,
        labels,
    }
}

fn worker_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_smarts-evolution"))
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{}", std::process::id(), nanos));
    fs::create_dir_all(&path).unwrap();
    path
}

const MOCK_NPC_DOI: &str = "10.5281/zenodo.9991001";
const MOCK_CLASSYFIRE_DOI: &str = "10.5281/zenodo.9991002";
const MOCK_NPC_CONCEPT_RECORD_ID: u64 = 1001;
const MOCK_NPC_LATEST_RECORD_ID: u64 = 1002;
const MOCK_CLASSYFIRE_CONCEPT_RECORD_ID: u64 = 2001;
const MOCK_CLASSYFIRE_LATEST_RECORD_ID: u64 = 2002;

#[derive(Clone)]
struct MockSyncState {
    npc_concept_record: Value,
    npc_latest_record: Value,
    classyfire_concept_record: Value,
    classyfire_latest_record: Value,
    npc_download: Vec<u8>,
    classyfire_download: Vec<u8>,
    pubchem_download: Vec<u8>,
}

struct MockSyncServer {
    base_url: String,
}

impl MockSyncServer {
    fn pubchem_url(&self) -> String {
        format!("{}/pubchem/CID-SMILES.gz", self.base_url)
    }
}

fn zstd_lines(lines: &[String]) -> Vec<u8> {
    let mut encoder = zstd::Encoder::new(Vec::new(), 0).unwrap();
    for line in lines {
        writeln!(encoder, "{line}").unwrap();
    }
    encoder.finish().unwrap()
}

fn gzip_lines(lines: &[String]) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    for line in lines {
        writeln!(encoder, "{line}").unwrap();
    }
    encoder.finish().unwrap()
}

fn read_zstd_lines(path: &Path) -> Vec<String> {
    let file = File::open(path).unwrap();
    let decoder = zstd::Decoder::new(file).unwrap();
    BufReader::new(decoder)
        .lines()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

fn npc_release_lines() -> Vec<String> {
    let mut lines = Vec::new();
    for cid in 1..=20u64 {
        lines.push(format!(
            "{{\"cid\":{cid},\"smiles\":\"CC\",\"pathway_results\":[\"PathA\"],\"superclass_results\":[\"SuperA\"],\"class_results\":[\"ClassA\"]}}"
        ));
    }
    for cid in 21..=40u64 {
        lines.push(format!(
            "{{\"cid\":{cid},\"smiles\":\"N\",\"pathway_results\":[\"PathB\"],\"superclass_results\":[\"SuperB\"],\"class_results\":[\"ClassB\"]}}"
        ));
    }
    lines.push(
        "{\"cid\":41,\"smiles\":\"O\",\"pathway_results\":[\"PathC\"],\"superclass_results\":[],\"class_results\":[\"ClassC\"]}"
            .to_string(),
    );
    lines
}

fn classyfire_release_lines() -> Vec<String> {
    let mut lines = Vec::new();
    for cid in 1..=20u64 {
        lines.push(format!(
            "{{\"cid\":{cid},\"classyfire\":{{\"kingdom\":{{\"name\":\"Organic compounds\"}},\"superclass\":{{\"name\":\"Lipids and lipid-like molecules\"}},\"class\":{{\"name\":\"Fatty Acyls\"}},\"subclass\":{{\"name\":\"Fatty acids\"}},\"direct_parent\":{{\"name\":\"Saturated fatty acids\"}}}}}}"
        ));
    }
    for cid in 21..=40u64 {
        lines.push(format!(
            "{{\"cid\":{cid},\"classyfire\":{{\"kingdom\":{{\"name\":\"Organic compounds\"}},\"superclass\":{{\"name\":\"Benzenoids\"}},\"class\":{{\"name\":\"Benzene and substituted derivatives\"}},\"subclass\":{{\"name\":\"Anilides\"}},\"direct_parent\":{{\"name\":\"Acetanilides\"}}}}}}"
        ));
    }
    lines
}

fn pubchem_smiles_lines() -> Vec<String> {
    let mut lines = Vec::new();
    for cid in 1..=20u64 {
        lines.push(format!("{cid}\tCC"));
    }
    for cid in 21..=40u64 {
        lines.push(format!("{cid}\tN"));
    }
    lines
}

fn concept_record_json(base_url: &str, id: u64, doi: &str, latest_id: u64) -> Value {
    json!({
        "id": id,
        "recid": id.to_string(),
        "doi": doi,
        "conceptdoi": doi,
        "metadata": { "title": format!("Record {id}") },
        "files": [],
        "links": {
            "self": format!("{base_url}/api/records/{id}"),
            "latest": format!("{base_url}/api/records/{latest_id}")
        }
    })
}

fn latest_record_json(
    base_url: &str,
    id: u64,
    doi: &str,
    key: &str,
    download_path: &str,
    size: usize,
) -> Value {
    json!({
        "id": id,
        "recid": id.to_string(),
        "doi": doi,
        "conceptdoi": doi,
        "metadata": { "title": format!("Record {id}") },
        "files": [{
            "id": format!("file-{id}"),
            "key": key,
            "size": size,
            "links": {
                "content": format!("{base_url}{download_path}")
            }
        }],
        "links": {
            "self": format!("{base_url}/api/records/{id}")
        }
    })
}

fn start_mock_sync_server() -> MockSyncServer {
    let npc_download = zstd_lines(&npc_release_lines());
    let classyfire_download = zstd_lines(&classyfire_release_lines());
    let pubchem_download = gzip_lines(&pubchem_smiles_lines());

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    let state = Arc::new(MockSyncState {
        npc_concept_record: concept_record_json(
            &base_url,
            MOCK_NPC_CONCEPT_RECORD_ID,
            MOCK_NPC_DOI,
            MOCK_NPC_LATEST_RECORD_ID,
        ),
        npc_latest_record: latest_record_json(
            &base_url,
            MOCK_NPC_LATEST_RECORD_ID,
            MOCK_NPC_DOI,
            "completed.jsonl.zst",
            "/api/files/npc/completed.jsonl.zst",
            npc_download.len(),
        ),
        classyfire_concept_record: concept_record_json(
            &base_url,
            MOCK_CLASSYFIRE_CONCEPT_RECORD_ID,
            MOCK_CLASSYFIRE_DOI,
            MOCK_CLASSYFIRE_LATEST_RECORD_ID,
        ),
        classyfire_latest_record: latest_record_json(
            &base_url,
            MOCK_CLASSYFIRE_LATEST_RECORD_ID,
            MOCK_CLASSYFIRE_DOI,
            "classyfire-labels.jsonl.zst",
            "/api/files/classyfire/classyfire-labels.jsonl.zst",
            classyfire_download.len(),
        ),
        npc_download,
        classyfire_download,
        pubchem_download,
    });
    let app = Router::new()
        .route("/api/records", get(mock_records))
        .route("/api/records/{id}", get(mock_record))
        .route("/api/files/{dataset}/{filename}", get(mock_file_download))
        .route("/pubchem/CID-SMILES.gz", get(mock_pubchem_download))
        .with_state(state);

    listener.set_nonblocking(true).unwrap();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            axum::serve(listener, app).await.unwrap();
        });
    });
    std::thread::sleep(Duration::from_millis(50));

    MockSyncServer { base_url }
}

async fn mock_records(
    Query(query): Query<HashMap<String, String>>,
    State(state): State<Arc<MockSyncState>>,
) -> Json<Value> {
    let q = query.get("q").cloned().unwrap_or_default();
    let hits = if q.contains(MOCK_NPC_DOI) {
        vec![state.npc_concept_record.clone()]
    } else if q.contains(MOCK_CLASSYFIRE_DOI) {
        vec![state.classyfire_concept_record.clone()]
    } else {
        Vec::new()
    };
    let total = hits.len();

    Json(json!({
        "hits": {
            "hits": hits,
            "total": total
        },
        "links": {}
    }))
}

async fn mock_record(
    AxumPath(id): AxumPath<u64>,
    State(state): State<Arc<MockSyncState>>,
) -> Response {
    match id {
        MOCK_NPC_CONCEPT_RECORD_ID => Json(state.npc_concept_record.clone()).into_response(),
        MOCK_NPC_LATEST_RECORD_ID => Json(state.npc_latest_record.clone()).into_response(),
        MOCK_CLASSYFIRE_CONCEPT_RECORD_ID => {
            Json(state.classyfire_concept_record.clone()).into_response()
        }
        MOCK_CLASSYFIRE_LATEST_RECORD_ID => {
            Json(state.classyfire_latest_record.clone()).into_response()
        }
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn mock_file_download(
    AxumPath((dataset, _filename)): AxumPath<(String, String)>,
    State(state): State<Arc<MockSyncState>>,
) -> Response {
    let body = match dataset.as_str() {
        "npc" => state.npc_download.clone(),
        "classyfire" => state.classyfire_download.clone(),
        _ => return StatusCode::NOT_FOUND.into_response(),
    };

    (
        StatusCode::OK,
        [(CONTENT_TYPE, "application/octet-stream")],
        body,
    )
        .into_response()
}

async fn mock_pubchem_download(State(state): State<Arc<MockSyncState>>) -> Response {
    (
        StatusCode::OK,
        [(CONTENT_TYPE, "application/gzip")],
        state.pubchem_download.clone(),
    )
        .into_response()
}

fn managed_sync_command(server: &MockSyncServer, data_root: &Path) -> Command {
    let mut command = Command::new(worker_bin());
    command
        .env("SMARTS_EVOLUTION_DATA_ROOT", data_root)
        .env("SMARTS_EVOLUTION_ZENODO_ENDPOINT", &server.base_url)
        .env("SMARTS_EVOLUTION_NPC_CONCEPT_DOI", MOCK_NPC_DOI)
        .env(
            "SMARTS_EVOLUTION_CLASSYFIRE_CONCEPT_DOI",
            MOCK_CLASSYFIRE_DOI,
        )
        .env(
            "SMARTS_EVOLUTION_PUBCHEM_CID_SMILES_URL",
            server.pubchem_url(),
        );
    command
}

fn build_fixture() -> TestFixture {
    warmup_rdkit();

    let compounds = vec![
        make_compound(1, "CC", &[&["A"], &["A1"]]),
        make_compound(2, "CN", &[&["A"], &["A2"]]),
        make_compound(3, "O", &[&["B"], &["B1"]]),
        make_compound(4, "N", &[&["B"], &["B2"]]),
    ];
    let local_folds = vec![FoldData {
        smiles: compounds
            .iter()
            .map(|compound| compound.smiles.clone())
            .collect(),
        is_positive: vec![true, false, false, false],
    }];
    let local_evaluator = SmartsEvaluator::new(local_folds).unwrap();

    let worker_context = WorkerNodeContext {
        folds: vec![WorkerFoldPayload {
            smiles: compounds
                .iter()
                .map(|compound| compound.smiles.clone())
                .collect(),
            is_positive: vec![true, false, false, false],
        }],
    };

    TestFixture {
        local_evaluator,
        worker_context,
    }
}

#[test]
fn process_pool_matches_local_evaluator() {
    let fixture = build_fixture();
    let mut pool = ProcessEvaluationBackend::spawn_with_executable(&worker_bin(), 2).unwrap();
    pool.set_node_context(fixture.worker_context.clone())
        .unwrap();

    let genomes = vec![
        SmartsGenome::from_smarts("[#6]").unwrap(),
        SmartsGenome::from_smarts("[#7]").unwrap(),
        SmartsGenome::from_smarts("[#8]").unwrap(),
        SmartsGenome::from_smarts("[#6][#7]").unwrap(),
    ];
    let local_scores: HashMap<String, i64> = genomes
        .iter()
        .map(|genome| {
            (
                genome.smarts_string.clone(),
                fixture.local_evaluator.fitness_of(genome).score,
            )
        })
        .collect();

    let pooled_scores = pool
        .score_population(genomes.clone(), &ProgressBar::hidden())
        .unwrap();
    let pooled_score_map: HashMap<String, i64> = pooled_scores
        .into_iter()
        .map(|(genome, score)| (genome.smarts_string, score))
        .collect();

    assert_eq!(pooled_score_map, local_scores);

    let local_mcc = fixture.local_evaluator.mcc_of(&genomes[0]);
    let pooled_mcc = pool.mcc_of(&genomes[0]).unwrap();
    assert!((pooled_mcc - local_mcc).abs() < 1e-12);
}

#[test]
fn process_pool_rejects_inconsistent_fold_payloads() {
    let mut pool = ProcessEvaluationBackend::spawn_with_executable(&worker_bin(), 1).unwrap();

    let err = pool
        .set_node_context(WorkerNodeContext {
            folds: vec![WorkerFoldPayload {
                smiles: vec!["CC".to_string(), "CN".to_string()],
                is_positive: vec![true],
            }],
        })
        .unwrap_err()
        .to_string();

    assert!(err.contains("inconsistent fold lengths"));
}

#[test]
fn process_pool_requires_node_context_before_mcc_evaluation() {
    let mut pool = ProcessEvaluationBackend::spawn_with_executable(&worker_bin(), 1).unwrap();

    let err = pool
        .mcc_of(&SmartsGenome::from_smarts("[#6]").unwrap())
        .unwrap_err()
        .to_string();

    assert!(err.contains("not initialized for a node"));
}

#[test]
fn evolve_cli_runs_with_worker_processes_and_writes_checkpoint() {
    let temp_dir = unique_temp_dir("smarts-evolution-cli-test");
    let dataset_path = temp_dir.join("npc.jsonl");
    let checkpoint_dir = temp_dir.join("checkpoints");

    let mut dataset = String::new();
    for cid in 1..=20u64 {
        dataset.push_str(&format!(
            "{{\"cid\":{cid},\"smiles\":\"CC\",\"pathway_results\":[\"PathA\"],\"superclass_results\":[\"SuperA\"],\"class_results\":[\"ClassA\"]}}\n"
        ));
    }
    for cid in 21..=40u64 {
        dataset.push_str(&format!(
            "{{\"cid\":{cid},\"smiles\":\"N\",\"pathway_results\":[\"PathB\"],\"superclass_results\":[\"SuperB\"],\"class_results\":[\"ClassB\"]}}\n"
        ));
    }
    fs::write(&dataset_path, dataset).unwrap();

    let output = Command::new(worker_bin())
        .args([
            "evolve",
            "--dataset",
            "npc",
            "--path",
            dataset_path.to_str().unwrap(),
            "--population-size",
            "4",
            "--generation-limit",
            "1",
            "--stagnation-limit",
            "1",
            "--folds",
            "2",
            "--worker-processes",
            "2",
            "--checkpoint-dir",
            checkpoint_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Evolution Complete"));
    assert!(stdout.contains("Evolved "));

    let checkpoint = checkpoint_dir.join("checkpoint.json");
    assert!(
        checkpoint.exists(),
        "missing checkpoint at {}",
        checkpoint.display()
    );
    let checkpoint_data = fs::read_to_string(&checkpoint).unwrap();
    assert!(checkpoint_data.contains("\"nodes\""));
}

#[test]
fn evolve_cli_runs_with_managed_npc_sync_and_writes_checkpoint() {
    let temp_dir = unique_temp_dir("smarts-evolution-managed-npc-cli-test");
    let checkpoint_dir = temp_dir.join("checkpoints");
    let server = start_mock_sync_server();

    let output = managed_sync_command(&server, &temp_dir)
        .args([
            "evolve",
            "--dataset",
            "npc",
            "--population-size",
            "4",
            "--generation-limit",
            "1",
            "--stagnation-limit",
            "1",
            "--folds",
            "2",
            "--worker-processes",
            "2",
            "--checkpoint-dir",
            checkpoint_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let prepared_path = temp_dir.join("npc.fully_labeled.jsonl.zst");
    let raw_path = temp_dir.join(".tmp/datasets/raw/npc/completed.jsonl.zst");
    let state_path = temp_dir.join(".tmp/datasets/state/npc.json");
    let checkpoint = checkpoint_dir.join("checkpoint.json");
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains(&format!("Dataset path: {}", prepared_path.display())));
    assert!(stdout.contains("Compounds loaded: 40"));
    assert!(stdout.contains("Evolution Complete"));
    assert!(prepared_path.exists());
    assert!(raw_path.exists());
    assert!(state_path.exists());
    assert!(checkpoint.exists());

    let prepared_lines = read_zstd_lines(&prepared_path);
    assert_eq!(prepared_lines.len(), 40);
    assert!(
        prepared_lines
            .iter()
            .all(|line| !line.contains("\"cid\":41"))
    );

    let state = fs::read_to_string(state_path).unwrap();
    assert!(state.contains("\"source_record_id\": 1002"));
}

#[test]
fn load_cli_runs_with_managed_classyfire_sync() {
    let temp_dir = unique_temp_dir("smarts-evolution-managed-classyfire-cli-test");
    let server = start_mock_sync_server();

    let output = managed_sync_command(&server, &temp_dir)
        .args(["load", "--dataset", "classyfire"])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let prepared_path = temp_dir.join("classyfire.jsonl.zst");
    let raw_path = temp_dir.join(".tmp/datasets/raw/classyfire/classyfire-labels.jsonl.zst");
    let pubchem_path = temp_dir.join(".tmp/datasets/raw/pubchem/CID-SMILES.gz");
    let state_path = temp_dir.join(".tmp/datasets/state/classyfire.json");
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains(&format!("Dataset path: {}", prepared_path.display())));
    assert!(stdout.contains("Compounds loaded: 40"));
    assert!(stdout.contains("DAG:"));
    assert!(prepared_path.exists());
    assert!(raw_path.exists());
    assert!(pubchem_path.exists());
    assert!(state_path.exists());

    let prepared_lines = read_zstd_lines(&prepared_path);
    assert_eq!(prepared_lines.len(), 40);
    assert!(prepared_lines[0].contains("\"smiles\":\"CC\""));
    assert!(prepared_lines[20].contains("\"smiles\":\"N\""));

    let state = fs::read_to_string(state_path).unwrap();
    assert!(state.contains("\"source_record_id\": 2002"));
}
