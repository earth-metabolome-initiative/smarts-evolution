#![allow(clippy::unwrap_used)]

use std::hint::black_box;
use std::str::FromStr;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use smarts_evolution::fitness::mcc::{ConfusionCounts, compute_fold_averaged_mcc};
use smarts_evolution::fitness::objective::ObjectiveFitness;
use smarts_evolution::genome::seed::SmartsGenomeBuilder;
use smarts_evolution::operators::crossover::SmartsCrossover;
use smarts_evolution::operators::mutation::SmartsMutation;
use smarts_evolution::{
    EvolutionConfig, EvolutionTask, FoldData, FoldSample, SeedCorpus, SmartsEvaluator, SmartsGenome,
};
use smarts_rs::{CompiledQuery, PreparedTarget};
use smiles_parser::Smiles;

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

const EXAMPLE_BENCH_POPULATION_SIZE: usize = 64;
const EXAMPLE_BENCH_GENERATION_LIMIT: u64 = 20;
const EXAMPLE_BENCH_STAGNATION_LIMIT: u64 = 10;
const EXAMPLE_BENCH_SEED: u64 = 17;
const EXAMPLE_EVALUATOR_QUERY: &str = "[#6](~[#6])~[#6]";
const EXAMPLE_EVALUATOR_BATCH_SIZE: usize = 64;

#[derive(Clone, Copy)]
struct ExampleDataset {
    name: &'static str,
    positive_smiles: &'static str,
    negative_smiles: &'static str,
}

const EXAMPLE_DATASETS: [ExampleDataset; 5] = [
    ExampleDataset {
        name: "amphetamines",
        positive_smiles: include_str!(
            "../apps/web/examples/amphetamines_and_derivatives_positive.smiles"
        ),
        negative_smiles: include_str!(
            "../apps/web/examples/amphetamines_and_derivatives_negative.smiles"
        ),
    },
    ExampleDataset {
        name: "flavonoids",
        positive_smiles: include_str!("../apps/web/examples/flavonoids_positive.smiles"),
        negative_smiles: include_str!("../apps/web/examples/flavonoids_negative.smiles"),
    },
    ExampleDataset {
        name: "fatty-acids",
        positive_smiles: include_str!(
            "../apps/web/examples/fatty_acids_and_conjugates_positive.smiles"
        ),
        negative_smiles: include_str!(
            "../apps/web/examples/fatty_acids_and_conjugates_negative.smiles"
        ),
    },
    ExampleDataset {
        name: "penicillins",
        positive_smiles: include_str!("../apps/web/examples/penicillins_positive.smiles"),
        negative_smiles: include_str!("../apps/web/examples/penicillins_negative.smiles"),
    },
    ExampleDataset {
        name: "steroids",
        positive_smiles: include_str!(
            "../apps/web/examples/steroids_and_steroid_derivatives_positive.smiles"
        ),
        negative_smiles: include_str!(
            "../apps/web/examples/steroids_and_steroid_derivatives_negative.smiles"
        ),
    },
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
    SeedCorpus::try_from(SEED_SMARTS.to_vec()).unwrap()
}

fn parse_smiles_block(input: &str) -> Vec<PreparedTarget> {
    input
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(prepared)
        .collect()
}

fn build_example_task(dataset: &ExampleDataset) -> EvolutionTask {
    let positives = parse_smiles_block(dataset.positive_smiles)
        .into_iter()
        .map(FoldSample::positive);
    let negatives = parse_smiles_block(dataset.negative_smiles)
        .into_iter()
        .map(FoldSample::negative);
    EvolutionTask::new(
        format!("example-{}", dataset.name),
        vec![FoldData::new(positives.chain(negatives).collect())],
    )
}

fn build_example_config(seed: u64) -> EvolutionConfig {
    EvolutionConfig::builder()
        .population_size(EXAMPLE_BENCH_POPULATION_SIZE)
        .generation_limit(EXAMPLE_BENCH_GENERATION_LIMIT)
        .stagnation_limit(EXAMPLE_BENCH_STAGNATION_LIMIT)
        .rng_seed(seed)
        .build()
        .unwrap()
}

fn build_example_genome_batch(size: usize) -> Vec<SmartsGenome> {
    let builder = SmartsGenomeBuilder::new(SeedCorpus::builtin());
    let mut rng = SmallRng::seed_from_u64(EXAMPLE_BENCH_SEED);
    (0..size)
        .map(|index| builder.build_genome(index, &mut rng))
        .collect()
}

fn scalar_objective_of(genome: &SmartsGenome, folds: &[FoldData]) -> ObjectiveFitness {
    let Ok(compiled) = CompiledQuery::new(genome.query().clone()) else {
        return ObjectiveFitness::invalid();
    };
    let fold_counts = folds
        .iter()
        .map(|fold| {
            let mut counts = ConfusionCounts::default();
            for sample in fold.samples() {
                counts.record_match(compiled.matches(sample.target()), sample.is_positive());
            }
            counts
        })
        .collect::<Vec<_>>();
    ObjectiveFitness::from_mcc(compute_fold_averaged_mcc(&fold_counts))
}

fn scalar_objectives_of(genomes: &[SmartsGenome], folds: &[FoldData]) -> Vec<ObjectiveFitness> {
    genomes
        .iter()
        .map(|genome| scalar_objective_of(genome, folds))
        .collect()
}

fn indexed_objectives_of(
    genomes: &[SmartsGenome],
    evaluator: &SmartsEvaluator,
) -> Vec<ObjectiveFitness> {
    genomes
        .iter()
        .map(|genome| evaluator.objective_of(genome))
        .collect()
}

fn indexed_batch_objectives_of(
    genomes: &[SmartsGenome],
    evaluator: &SmartsEvaluator,
) -> Vec<ObjectiveFitness> {
    evaluator
        .evaluate_many(genomes.to_vec())
        .into_iter()
        .map(|(_, evaluation)| evaluation.fitness())
        .collect()
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

fn example_evaluator_batch_mode_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("example_evaluator_batch_modes");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(6));
    group.sample_size(10);

    let genomes = build_example_genome_batch(EXAMPLE_EVALUATOR_BATCH_SIZE);

    for dataset in EXAMPLE_DATASETS {
        let task = build_example_task(&dataset);
        let evaluator = SmartsEvaluator::new(task.folds().to_vec());
        let scalar = scalar_objectives_of(&genomes, task.folds());
        let indexed = indexed_objectives_of(&genomes, &evaluator);
        let indexed_batch = indexed_batch_objectives_of(&genomes, &evaluator);
        assert_eq!(
            scalar, indexed,
            "scalar/indexed drifted for {}",
            dataset.name
        );
        assert_eq!(
            scalar, indexed_batch,
            "scalar/indexed-batch drifted for {}",
            dataset.name
        );

        group.bench_with_input(
            BenchmarkId::new("scalar_batch64", dataset.name),
            &dataset.name,
            |b, _| {
                b.iter(|| {
                    let fitness =
                        scalar_objectives_of(black_box(&genomes), black_box(task.folds()));
                    black_box(fitness);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("indexed_scalar_batch64", dataset.name),
            &dataset.name,
            |b, _| {
                b.iter(|| {
                    let fitness = indexed_objectives_of(black_box(&genomes), black_box(&evaluator));
                    black_box(fitness);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("indexed_batch_batch64", dataset.name),
            &dataset.name,
            |b, _| {
                b.iter(|| {
                    let fitness =
                        indexed_batch_objectives_of(black_box(&genomes), black_box(&evaluator));
                    black_box(fitness);
                });
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
                        .filter(|sample| compiled.matches(sample.target()))
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
                "task_evolve",
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
                    let result = black_box(&task)
                        .evolve(black_box(&config), black_box(&seed_corpus))
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
        let task = build_example_task(&dataset);
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
        let task = build_example_task(&dataset);
        group.bench_with_input(
            BenchmarkId::new(
                "task_evolve",
                format!(
                    "{}_pop{}_gen{}",
                    dataset.name, EXAMPLE_BENCH_POPULATION_SIZE, EXAMPLE_BENCH_GENERATION_LIMIT
                ),
            ),
            &dataset.name,
            |b, _| {
                b.iter(|| {
                    let result = black_box(&task)
                        .evolve(black_box(&config), black_box(&seed_corpus))
                        .unwrap();
                    black_box(result.best_mcc());
                });
            },
        );
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
    example_evaluator_batch_mode_benches,
    evolution_benches,
    example_evaluator_benches,
    example_evolution_benches,
    operator_benches
);
criterion_main!(benches);
