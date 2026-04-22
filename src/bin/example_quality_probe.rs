#[path = "../../experiments/example_support.rs"]
mod example_support;

use example_support::{EXAMPLE_DATASETS, QUALITY_SEEDS, SeedStrategy, run_example_strategy};

fn main() {
    let dataset_filters = std::env::args().skip(1).collect::<Vec<_>>();
    println!("dataset\tstrategy\tseed\tbest_mcc\tbest_size\tbest_smarts");

    for dataset in EXAMPLE_DATASETS {
        if !dataset_filters.is_empty()
            && !dataset_filters
                .iter()
                .any(|filter| dataset.name.contains(filter.as_str()))
        {
            continue;
        }
        for strategy in [SeedStrategy::Builtin, SeedStrategy::DiscriminativePaths] {
            let mut scores = Vec::new();
            for &seed in QUALITY_SEEDS {
                let result = run_example_strategy(dataset, strategy, seed);
                scores.push(result.best_mcc());
                println!(
                    "{}\t{}\t{}\t{:.6}\t{}\t{}",
                    dataset.name,
                    strategy.name(),
                    seed,
                    result.best_mcc(),
                    result.best_smarts().len(),
                    result.best_smarts()
                );
            }

            scores.sort_by(|left, right| left.partial_cmp(right).unwrap());
            let median = scores[scores.len() / 2];
            let mean = scores.iter().copied().sum::<f64>() / scores.len() as f64;
            println!(
                "{}\t{}\tmedian\t{:.6}\t-\t-",
                dataset.name,
                strategy.name(),
                median
            );
            println!(
                "{}\t{}\tmean\t{:.6}\t-\t-",
                dataset.name,
                strategy.name(),
                mean
            );
        }
    }
}
