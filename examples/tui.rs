//! Sample evolution run wired to the native TUI dashboard, for screenshots and
//! manual exploration.
//!
//! Run it with the `tui` feature enabled:
//!
//! ```sh
//! cargo run --release --example tui --features tui
//! ```
//!
//! The task is an "amide vs the rest" classification over a few thousand small
//! molecules. The seed patterns are deliberately generic, so best-so-far MCC
//! starts well below 1.0 and climbs over roughly 80 generations as the search
//! discovers that the amide carbonyl must carry a carbon neighbour (which sets
//! true amides apart from the urea and carbamate decoys). At about 150 ms per
//! generation the climb takes around 15 seconds.
//!
//! For a screenshot: let it run until the MCC curve has risen, then press `p`
//! (or click `[p pause]`) to freeze it and capture the frame. By default the run
//! does not stop on its own, so press `q` (or `[q stop]`) to quit. Set the
//! `TUI_DEMO_GENERATIONS` environment variable to make it auto-exit after that
//! many generations, which is convenient when recording a fixed-length clip.
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(feature = "tui")]
fn main() {
    use smarts_evolution::{EvolutionConfig, EvolutionTask, FoldData, FoldSample, SeedCorpus};
    use smarts_rs::PreparedTarget;
    use smiles_rs::Smiles;

    fn prepared(smiles: &str) -> PreparedTarget {
        PreparedTarget::new(Smiles::from_str(smiles).unwrap())
    }

    fn chain(n: usize) -> String {
        "C".repeat(n)
    }

    // Aromatic and ring "caps" attached to a functional group enlarge the
    // molecules without changing the amide / not-amide label.
    const CAPS: [&str; 6] = [
        "",
        "c1ccccc1",
        "Cc1ccccc1",
        "c1ccc(cc1)",
        "C1CCCCC1",
        "c1ccncc1",
    ];

    let mut positives: Vec<String> = Vec::new();
    let mut negatives: Vec<String> = Vec::new();

    // Positives: true amides R-C(=O)-N, where the carbonyl carbon always has a
    // carbon neighbour (never an oxygen or a second nitrogen).
    for cap in CAPS {
        for acyl in 4..=12 {
            for amine in 0..=8 {
                positives.push(format!("{}{}C(=O)N{}", cap, chain(acyl), chain(amine)));
            }
            for left in 2..=5 {
                for right in 2..=5 {
                    positives.push(format!(
                        "{}{}C(=O)N({}){}",
                        cap,
                        chain(acyl),
                        chain(left),
                        chain(right)
                    ));
                }
            }
        }
    }

    // Hard negatives: ureas and carbamates. Their carbonyl has a nitrogen
    // neighbour but no carbon neighbour, so a naive [#6](=[#8])[#7] wrongly
    // accepts them and only the refined pattern can tell them apart.
    for cap in CAPS {
        for left in 1..=6 {
            for right in 0..=6 {
                negatives.push(format!("{}{}NC(=O)N{}", cap, chain(left), chain(right)));
            }
        }
        for left in 2..=7 {
            for right in 0..=6 {
                negatives.push(format!("{}{}OC(=O)N{}", cap, chain(left), chain(right)));
            }
        }
    }

    // Carbonyls without nitrogen: acids, esters, ketones.
    for cap in CAPS {
        for acyl in 3..=9 {
            negatives.push(format!("{}{}C(=O)O", cap, chain(acyl)));
            for alkyl in 2..=7 {
                negatives.push(format!("{}{}C(=O)O{}", cap, chain(acyl), chain(alkyl)));
                negatives.push(format!("{}{}C(=O){}", cap, chain(acyl), chain(alkyl)));
            }
        }
    }

    // Nitrogen without an amide bond, plus inert decoys.
    for cap in CAPS {
        for n in 3..=11 {
            negatives.push(format!("{}{}N", cap, chain(n)));
            negatives.push(format!("{}{}C#N", cap, chain(n)));
            negatives.push(format!("{}{}O", cap, chain(n)));
            negatives.push(format!("{}{}OCC", cap, chain(n)));
        }
        for n in 5..=13 {
            negatives.push(format!("{}{}", cap, chain(n)));
        }
    }

    let mut samples = Vec::with_capacity(positives.len() + negatives.len());
    for smiles in &positives {
        samples.push(FoldSample::positive(prepared(smiles)));
    }
    for smiles in &negatives {
        samples.push(FoldSample::negative(prepared(smiles)));
    }

    let task = EvolutionTask::new("class:amide-vs-rest", vec![FoldData::new(samples)]);

    // By default the run does not stop on its own (press `q` to quit). Set
    // TUI_DEMO_GENERATIONS to a small number to make it auto-exit after that
    // many generations, which is handy when recording a fixed-length clip.
    let generation_limit = std::env::var("TUI_DEMO_GENERATIONS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1_000_000);

    let config = EvolutionConfig::builder()
        .population_size(24)
        .generation_limit(generation_limit)
        .stagnation_limit(generation_limit)
        .rng_seed(42)
        .build()
        .unwrap();

    // Generic seeds only: none of these separate the classes on their own, so
    // the run has to discover the amide pattern rather than start from it.
    let seed_corpus =
        SeedCorpus::try_from(["[#6]", "[#7]", "[#8]", "[#6]=[#8]", "[#6]~[#7]"]).unwrap();

    let result = task
        .evolve_with_tui(&config, &seed_corpus)
        .expect("run the TUI evolution");

    println!("best SMARTS: {}", result.best_smarts());
    println!("best MCC:    {}", result.best_mcc());
}

#[cfg(not(feature = "tui"))]
fn main() {
    eprintln!("This example requires the `tui` feature. Run it with:");
    eprintln!("    cargo run --release --example tui --features tui");
}
