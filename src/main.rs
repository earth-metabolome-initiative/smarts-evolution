use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use rand::SeedableRng;
use rand::rngs::SmallRng;

use smarts_evolution::data::{classyfire, molecule_cache, npc, sync};
use smarts_evolution::evolution::checkpoint::FullCheckpoint;
use smarts_evolution::evolution::config::EvolutionConfig;
use smarts_evolution::evolution::ntfy::RunNotifier;
use smarts_evolution::evolution::process_pool;
use smarts_evolution::evolution::runner;
use smarts_evolution::taxonomy::builder;
use smarts_evolution::validation::splits::{FoldSplits, MIN_POSITIVE_EXAMPLES};

#[derive(Parser)]
#[command(name = "smarts-evolution")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Load and validate a dataset, printing label statistics.
    Load {
        #[arg(long)]
        dataset: Dataset,
        /// Optional explicit dataset path. If omitted, the latest prepared snapshot is synced automatically.
        #[arg(long)]
        path: Option<PathBuf>,
    },
    /// Run SMARTS evolution on a dataset.
    Evolve {
        #[arg(long)]
        dataset: Dataset,
        /// Optional explicit dataset path. If omitted, the latest prepared snapshot is synced automatically.
        #[arg(long)]
        path: Option<PathBuf>,
        #[arg(long, default_value = "200")]
        population_size: usize,
        #[arg(long, default_value = "500")]
        generation_limit: u64,
        #[arg(long, default_value = "50")]
        stagnation_limit: u64,
        /// Number of cross-validation folds
        #[arg(long, default_value = "5")]
        folds: usize,
        #[arg(long, default_value = "checkpoints")]
        checkpoint_dir: PathBuf,
        #[arg(long)]
        worker_processes: Option<usize>,
        /// Resume from existing checkpoint
        #[arg(long)]
        resume: bool,
    },
    #[command(hide = true)]
    Worker,
}

#[derive(Clone, ValueEnum)]
enum Dataset {
    Classyfire,
    Npc,
}

fn load_dataset(
    dataset: &Dataset,
    path: Option<PathBuf>,
) -> Result<
    (
        Vec<smarts_evolution::data::compound::Compound>,
        &'static [&'static str],
        PathBuf,
    ),
    Box<dyn std::error::Error>,
> {
    let resolved_path = resolve_dataset_path(dataset, path)?;
    println!("Dataset path: {}", resolved_path.display());

    let (mut compounds, levels) = match dataset {
        Dataset::Classyfire => {
            let c = classyfire::load(&resolved_path)?;
            (c, classyfire::LEVELS)
        }
        Dataset::Npc => {
            let c = npc::load(&resolved_path)?;
            (c, npc::LEVELS)
        }
    };

    println!("Compounds loaded: {}", compounds.len());
    molecule_cache::parse_molecules(&mut compounds);
    let parsed = compounds.iter().filter(|c| c.parsed).count();
    println!("Molecules parsed: {parsed}/{}", compounds.len());

    Ok((compounds, levels, resolved_path))
}

fn resolve_dataset_path(
    dataset: &Dataset,
    path: Option<PathBuf>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    match path {
        Some(path) => Ok(path),
        None => match dataset {
            Dataset::Classyfire => sync::ensure_latest_classyfire_dataset(),
            Dataset::Npc => sync::ensure_latest_npc_dataset(),
        },
    }
}

fn step_spinner(message: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg}")
            .unwrap()
            .tick_strings(&["-", "\\", "|", "/"]),
    );
    pb.enable_steady_tick(Duration::from_millis(120));
    pb.set_message(message.to_string());
    pb
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    molecule_cache::warmup_rdkit();
    let cli = Cli::parse();

    match cli.command {
        Command::Load { dataset, path } => {
            let (compounds, levels, _) = load_dataset(&dataset, path)?;

            let dag_pb = step_spinner("Building taxonomy DAG");
            let dag = builder::build_dag(&compounds, levels)?;
            dag_pb.finish_with_message(format!(
                "Building taxonomy DAG complete ({} nodes)",
                dag.num_nodes()
            ));
            println!(
                "DAG: {} nodes, topological order length: {}",
                dag.num_nodes(),
                dag.topological_order.len()
            );

            for (level_idx, level_name) in levels.iter().enumerate() {
                let node_ids = dag.nodes_at_level(level_idx);
                let mut entries: Vec<(&str, usize)> = node_ids
                    .iter()
                    .map(|&id| {
                        (
                            dag.nodes[id].name.as_str(),
                            dag.nodes[id].compound_indices.len(),
                        )
                    })
                    .collect();
                entries.sort_by(|a, b| b.1.cmp(&a.1));

                println!("\n=== {level_name} ({} unique) ===", entries.len());
                for (name, count) in entries.iter().take(15) {
                    println!("  {count:>6}  {name}");
                }
                if entries.len() > 15 {
                    println!("  ... and {} more", entries.len() - 15);
                }
            }

            let occ = dag.occurrences();
            let non_zero = occ.iter().filter(|&&c| c > 0).count();
            println!(
                "\nOccurrences: {non_zero}/{} nodes with >0 compounds",
                dag.num_nodes()
            );

            let below_threshold = dag
                .nodes
                .iter()
                .filter(|n| n.compound_indices.len() < MIN_POSITIVE_EXAMPLES)
                .count();
            println!(
                "Nodes below {MIN_POSITIVE_EXAMPLES}-compound threshold: {below_threshold}/{}",
                dag.num_nodes()
            );

            let splits_pb = step_spinner("Building 5-fold stratified splits");
            let mut rng = SmallRng::seed_from_u64(42);
            let splits = FoldSplits::build(&dag, 5, &mut rng);
            splits_pb.finish_with_message("Building 5-fold stratified splits complete");
            for (fi, fold) in splits.fold_indices.iter().enumerate() {
                print!("  Fold {fi}: {} samples", fold.len());
                if fi < splits.k - 1 {
                    print!(",  ");
                }
            }
            println!();

            let eligible: Vec<_> = dag
                .nodes
                .iter()
                .filter(|n| FoldSplits::is_eligible(n.compound_indices.len()))
                .collect();
            println!(
                "\nEligible nodes for evolution: {}/{}",
                eligible.len(),
                dag.num_nodes()
            );
        }

        Command::Evolve {
            dataset,
            path,
            population_size,
            generation_limit,
            stagnation_limit,
            folds,
            checkpoint_dir,
            worker_processes,
            resume,
        } => {
            let notifier = RunNotifier::new();
            eprintln!("NTFY URL: {}", notifier.topic_url());
            eprintln!(
                "Open that URL in ntfy or a browser to receive per-class completion messages.\n"
            );

            let (compounds, levels, _) = load_dataset(&dataset, path)?;

            let dag_pb = step_spinner("Building taxonomy DAG");
            let dag = builder::build_dag(&compounds, levels)?;
            dag_pb.finish_with_message(format!(
                "Building taxonomy DAG complete ({} nodes)",
                dag.num_nodes()
            ));
            let eligible = dag
                .nodes
                .iter()
                .filter(|n| n.compound_indices.len() >= MIN_POSITIVE_EXAMPLES)
                .count();
            println!(
                "DAG: {} nodes, {} eligible for evolution",
                dag.num_nodes(),
                eligible
            );

            let splits_pb = step_spinner(&format!("Building {folds}-fold stratified splits"));
            let mut rng = SmallRng::seed_from_u64(42);
            let splits = FoldSplits::build(&dag, folds, &mut rng);
            splits_pb
                .finish_with_message(format!("Building {folds}-fold stratified splits complete"));

            let config = EvolutionConfig {
                population_size,
                generation_limit,
                stagnation_limit,
                checkpoint_dir: checkpoint_dir.clone(),
                worker_processes: worker_processes
                    .unwrap_or_else(|| EvolutionConfig::default().worker_processes),
                ..Default::default()
            };

            let mut checkpoint = if resume {
                let ckpt_path = checkpoint_dir.join("checkpoint.json");
                if ckpt_path.exists() {
                    println!("Resuming from {}", ckpt_path.display());
                    FullCheckpoint::load(&ckpt_path)?
                } else {
                    println!("No checkpoint found, starting fresh");
                    FullCheckpoint::default()
                }
            } else {
                FullCheckpoint::default()
            };

            println!("\nStarting evolution...\n");
            let results = runner::run_evolution(
                &compounds,
                &dag,
                &splits,
                &config,
                &mut checkpoint,
                &notifier,
            )?;

            println!("\n=== Evolution Complete ===");
            println!("Evolved {} nodes", results.len());
            println!("{:<40} {:>6}  SMARTS", "Node", "MCC");
            println!("{}", "-".repeat(80));
            for r in &results {
                let node = &dag.nodes[r.node_id];
                println!("{:<40} {:>6.3}  {}", node.name, r.best_mcc, r.best_smarts);
            }
        }
        Command::Worker => {
            process_pool::run_worker_stdio()?;
        }
    }

    Ok(())
}
