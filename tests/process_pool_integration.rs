use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use genevo::genetic::FitnessFunction;
use indicatif::ProgressBar;
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
