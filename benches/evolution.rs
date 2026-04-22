use std::hint::black_box;
use std::str::FromStr;
use std::time::Duration;

#[path = "../experiments/example_support.rs"]
mod example_support;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use smarts_evolution::genome::seed::SmartsGenomeBuilder;
use smarts_evolution::operators::crossover::SmartsCrossover;
use smarts_evolution::operators::mutation::SmartsMutation;
use smarts_evolution::{
    EvolutionConfig, EvolutionTask, FoldData, FoldSample, SeedCorpus, SmartsEvaluator,
    SmartsGenome, evolve_task,
};
use smarts_validator::{CompiledQuery, PreparedTarget, matches_compiled};
use smiles_parser::Smiles;

use example_support::{
    EXAMPLE_BENCH_GENERATION_LIMIT, EXAMPLE_BENCH_POPULATION_SIZE, EXAMPLE_BENCH_SEED,
    EXAMPLE_DATASETS, EXAMPLE_EVALUATOR_QUERY, SeedStrategy, build_example_config,
    build_example_task, run_example_strategy,
};

const POSITIVE_SMILES: &[&str] = &[
    "CC(=O)N",
    "NC(=O)C",
    "CCC(=O)N",
    "CC(=O)NC",
    "CC(=O)NCC",
    "CC(=O)Nc1ccccc1",
    "O=C(N)C1CCCCC1",
    "CC(C)C(=O)N",
];

const NEGATIVE_SMILES: &[&str] = &[
    "CCO", "CCCO", "CCN", "CCOC", "c1ccccc1", "CCCl", "CCS", "CC(=O)O",
];

const SEED_SMARTS: &[&str] = &[
    "[#6](=[#8])[#7]",
    "[#6]~[#7]",
    "[#7]~[#6](=[#8])",
    "[#6](=[#8])[#6]",
];

fn prepared(smiles: &str) -> PreparedTarget {
    PreparedTarget::new(Smiles::from_str(smiles).unwrap())
}

fn build_fold(sample_count_per_class: usize) -> FoldData {
    let positives = (0..sample_count_per_class).map(|index| {
        FoldSample::positive(prepared(POSITIVE_SMILES[index % POSITIVE_SMILES.len()]))
    });
    let negatives = (0..sample_count_per_class).map(|index| {
        FoldSample::negative(prepared(NEGATIVE_SMILES[index % NEGATIVE_SMILES.len()]))
    });

    FoldData::new(positives.chain(negatives).collect())
}

fn build_task(sample_count_per_class: usize) -> EvolutionTask {
    EvolutionTask::new(
        format!("amide-bench-{sample_count_per_class}"),
        vec![build_fold(sample_count_per_class)],
    )
}

fn build_seed_corpus() -> SeedCorpus {
    SeedCorpus::from_smarts(
        SEED_SMARTS
            .iter()
            .map(|smarts| (*smarts).to_string())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

fn evaluator_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluator");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));
    group.sample_size(20);

    let genome = SmartsGenome::from_smarts("[#6](=[#8])[#7]").unwrap();
    for &sample_count_per_class in &[16usize, 64, 256, 1_024] {
        let evaluator = SmartsEvaluator::new(vec![build_fold(sample_count_per_class)]);
        group.bench_with_input(
            BenchmarkId::new("objective_of", sample_count_per_class * 2),
            &sample_count_per_class,
            |b, _| {
                b.iter(|| evaluator.objective_of(black_box(&genome)));
            },
        );
    }

    group.finish();
}

fn evaluator_component_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluator_components");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));
    group.sample_size(20);

    let genome = SmartsGenome::from_smarts("[#6](=[#8])[#7]").unwrap();
    group.bench_function("compile_query", |b| {
        b.iter(|| CompiledQuery::new(black_box(genome.query().clone())).unwrap());
    });

    for &sample_count_per_class in &[16usize, 64, 256, 1_024] {
        let fold = build_fold(sample_count_per_class);
        let compiled = CompiledQuery::new(genome.query().clone()).unwrap();
        group.bench_with_input(
            BenchmarkId::new("match_compiled", sample_count_per_class * 2),
            &sample_count_per_class,
            |b, _| {
                b.iter(|| {
                    let matched = fold
                        .samples()
                        .iter()
                        .filter(|sample| matches_compiled(&compiled, sample.target()))
                        .count();
                    black_box(matched);
                });
            },
        );
    }

    group.finish();
}

fn evolution_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("evolution");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(6));
    group.sample_size(10);

    let seed_corpus = build_seed_corpus();
    for &(population_size, generation_limit, sample_count_per_class) in
        &[(32usize, 3u64, 16usize), (64, 5, 32)]
    {
        let task = build_task(sample_count_per_class);
        let config = EvolutionConfig::builder()
            .population_size(population_size)
            .generation_limit(generation_limit)
            .stagnation_limit(generation_limit)
            .build()
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new(
                "evolve_task",
                format!(
                    "pop{}_gen{}_samples{}",
                    population_size,
                    generation_limit,
                    sample_count_per_class * 2
                ),
            ),
            &(population_size, generation_limit, sample_count_per_class),
            |b, _| {
                b.iter(|| {
                    let result = evolve_task(
                        black_box(&task),
                        black_box(&config),
                        black_box(&seed_corpus),
                    )
                    .unwrap();
                    black_box(result.best_mcc());
                });
            },
        );
    }

    group.finish();
}

fn example_evaluator_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("example_evaluator");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    let genome = SmartsGenome::from_smarts(EXAMPLE_EVALUATOR_QUERY).unwrap();

    for dataset in EXAMPLE_DATASETS {
        let task = build_example_task(dataset);
        let evaluator = SmartsEvaluator::new(task.folds().to_vec());
        group.bench_with_input(
            BenchmarkId::new("objective_of", dataset.name),
            &dataset.name,
            |b, _| {
                b.iter(|| evaluator.objective_of(black_box(&genome)));
            },
        );
    }

    group.finish();
}

fn example_evolution_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("example_evolution");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(10);

    let config = build_example_config(EXAMPLE_BENCH_SEED);
    let seed_corpus = SeedCorpus::builtin();

    for dataset in EXAMPLE_DATASETS {
        let task = build_example_task(dataset);
        group.bench_with_input(
            BenchmarkId::new(
                "evolve_task",
                format!(
                    "{}_pop{}_gen{}",
                    dataset.name, EXAMPLE_BENCH_POPULATION_SIZE, EXAMPLE_BENCH_GENERATION_LIMIT
                ),
            ),
            &dataset.name,
            |b, _| {
                b.iter(|| {
                    let result = evolve_task(
                        black_box(&task),
                        black_box(&config),
                        black_box(&seed_corpus),
                    )
                    .unwrap();
                    black_box(result.best_mcc());
                });
            },
        );
    }

    group.finish();
}

fn example_strategy_total_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("example_strategy_total");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(10);

    for strategy in [SeedStrategy::Builtin, SeedStrategy::DiscriminativePaths] {
        for dataset in EXAMPLE_DATASETS {
            group.bench_with_input(
                BenchmarkId::new(
                    strategy.name(),
                    format!(
                        "{}_pop{}_gen{}",
                        dataset.name, EXAMPLE_BENCH_POPULATION_SIZE, EXAMPLE_BENCH_GENERATION_LIMIT
                    ),
                ),
                &dataset.name,
                |b, _| {
                    b.iter(|| {
                        let result = run_example_strategy(dataset, strategy, EXAMPLE_BENCH_SEED);
                        black_box(result.best_mcc());
                    });
                },
            );
        }
    }

    group.finish();
}

fn operator_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("operators");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));
    group.sample_size(20);

    let seed_corpus = build_seed_corpus();
    let builder = SmartsGenomeBuilder::new(seed_corpus.clone());
    let mut build_rng = SmallRng::seed_from_u64(17);
    group.bench_function("build_genome", |b| {
        let mut index = 0usize;
        b.iter(|| {
            let genome = builder.build_genome(index, &mut build_rng);
            index = index.wrapping_add(1);
            black_box(genome);
        });
    });

    let parent_a = SmartsGenome::from_smarts("[#6](=[#8])[#7]~[#6]").unwrap();
    let parent_b = SmartsGenome::from_smarts("[#6;R]~[#7;R]~[#6;R]").unwrap();
    let crossover = SmartsCrossover::new(1.0);
    let mut crossover_rng = SmallRng::seed_from_u64(23);
    group.bench_function("crossover_pair", |b| {
        b.iter(|| {
            let children = crossover.crossover_pair(&parent_a, &parent_b, &mut crossover_rng);
            black_box(children);
        });
    });

    let genome = SmartsGenome::from_smarts("[#6](=[#8])[#7]~[#6]~[#8]").unwrap();
    let mutator = SmartsMutation::with_reset_pool(1.0, seed_corpus.entries().to_vec());
    let mut mutation_rng = SmallRng::seed_from_u64(41);
    group.bench_function("mutate", |b| {
        b.iter_batched(
            || genome.clone(),
            |genome| black_box(mutator.mutate(genome, &mut mutation_rng)),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    evaluator_benches,
    evaluator_component_benches,
    evolution_benches,
    example_evaluator_benches,
    example_evolution_benches,
    example_strategy_total_benches,
    operator_benches
);
criterion_main!(benches);
