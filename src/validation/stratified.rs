use std::collections::HashMap;

use rand::Rng;
use rand::seq::SliceRandom;

/// Assign compound indices to k folds using stratified sampling.
///
/// For single-label data, stratifies on the provided labels directly.
/// For multi-label data, uses iterative stratification (Sechidis et al., 2011):
/// assigns samples one at a time based on the currently least-represented label,
/// placing each sample in the fold that most needs examples of its rarest label.
///
/// Returns a Vec of length k, where each element is a Vec of compound indices in that fold.
pub fn stratified_k_fold<R: Rng>(
    labels_per_sample: &[Vec<String>],
    k: usize,
    rng: &mut R,
) -> Vec<Vec<usize>> {
    assert!(k >= 2, "Need at least 2 folds");
    assert!(!labels_per_sample.is_empty(), "No samples to split");

    let n = labels_per_sample.len();

    // Check if any sample is multi-label
    let is_multi_label = labels_per_sample.iter().any(|ls| ls.len() > 1);

    if is_multi_label {
        iterative_stratification(labels_per_sample, k, rng)
    } else {
        simple_stratification(labels_per_sample, k, rng, n)
    }
}

/// Simple stratified split: group by single label, round-robin assign within each group.
fn simple_stratification<R: Rng>(
    labels_per_sample: &[Vec<String>],
    k: usize,
    rng: &mut R,
    n: usize,
) -> Vec<Vec<usize>> {
    // Group sample indices by their label
    let mut groups: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, labels) in labels_per_sample.iter().enumerate() {
        let label = labels.first().map(|s| s.as_str()).unwrap_or("");
        groups.entry(label).or_default().push(idx);
    }

    let mut folds: Vec<Vec<usize>> = vec![Vec::with_capacity(n / k + 1); k];

    // Shuffle within each group and distribute round-robin
    for (_label, mut indices) in groups {
        indices.shuffle(rng);
        for (i, idx) in indices.into_iter().enumerate() {
            folds[i % k].push(idx);
        }
    }

    // Shuffle each fold for good measure
    for fold in &mut folds {
        fold.shuffle(rng);
    }

    folds
}

/// Iterative stratification for multi-label data (Sechidis et al., 2011).
///
/// Assigns samples one at a time. At each step, finds the rarest label overall,
/// then among unassigned samples with that label, assigns each to the fold
/// that is most underrepresented for that label.
fn iterative_stratification<R: Rng>(
    labels_per_sample: &[Vec<String>],
    k: usize,
    rng: &mut R,
) -> Vec<Vec<usize>> {
    let n = labels_per_sample.len();

    // Collect all unique labels and count desired proportion per fold
    let mut label_to_id: HashMap<&str, usize> = HashMap::new();
    let mut label_names: Vec<&str> = Vec::new();
    for labels in labels_per_sample {
        for label in labels {
            if !label_to_id.contains_key(label.as_str()) {
                label_to_id.insert(label.as_str(), label_names.len());
                label_names.push(label.as_str());
            }
        }
    }
    let num_labels = label_names.len();

    // Count total occurrences of each label
    let mut label_counts = vec![0usize; num_labels];
    for labels in labels_per_sample {
        for label in labels {
            label_counts[label_to_id[label.as_str()]] += 1;
        }
    }

    // Desired count per fold for each label = total * fold_size / n
    let desired_per_fold: Vec<Vec<f64>> = (0..num_labels)
        .map(|lid| {
            (0..k)
                .map(|_| label_counts[lid] as f64 / k as f64)
                .collect()
        })
        .collect();

    // Track actual counts per label per fold
    let mut actual: Vec<Vec<f64>> = vec![vec![0.0; k]; num_labels];

    let mut folds: Vec<Vec<usize>> = vec![Vec::new(); k];
    let mut assigned = vec![false; n];

    // Create shuffled order of sample indices
    let mut sample_order: Vec<usize> = (0..n).collect();
    sample_order.shuffle(rng);

    // Process labels from rarest to most common
    let mut label_order: Vec<usize> = (0..num_labels).collect();
    label_order.sort_by_key(|&lid| label_counts[lid]);

    for &lid in &label_order {
        // Collect unassigned samples that have this label
        let mut candidates: Vec<usize> = sample_order
            .iter()
            .copied()
            .filter(|&idx| {
                !assigned[idx]
                    && labels_per_sample[idx]
                        .iter()
                        .any(|l| label_to_id[l.as_str()] == lid)
            })
            .collect();
        candidates.shuffle(rng);

        for idx in candidates {
            if assigned[idx] {
                continue;
            }

            // Find the fold most in need of this label (largest deficit)
            let best_fold = (0..k)
                .min_by(|&a, &b| {
                    let deficit_a = desired_per_fold[lid][a] - actual[lid][a];
                    let deficit_b = desired_per_fold[lid][b] - actual[lid][b];
                    deficit_b
                        .partial_cmp(&deficit_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();

            folds[best_fold].push(idx);
            assigned[idx] = true;

            // Update actual counts for all labels of this sample
            for label in &labels_per_sample[idx] {
                let l = label_to_id[label.as_str()];
                actual[l][best_fold] += 1.0;
            }
        }
    }

    // Assign any remaining unassigned samples (those with empty label lists, shouldn't happen)
    for idx in 0..n {
        if !assigned[idx] {
            let smallest_fold = (0..k).min_by_key(|&f| folds[f].len()).unwrap();
            folds[smallest_fold].push(idx);
        }
    }

    folds
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn test_simple_stratification_covers_all_samples() {
        let labels: Vec<Vec<String>> = (0..100).map(|i| vec![format!("class_{}", i % 5)]).collect();
        let mut rng = SmallRng::seed_from_u64(42);
        let folds = stratified_k_fold(&labels, 5, &mut rng);

        // All samples assigned exactly once
        let mut all: Vec<usize> = folds.iter().flat_map(|f| f.iter().copied()).collect();
        all.sort();
        assert_eq!(all, (0..100).collect::<Vec<_>>());

        // Each fold has 20 samples
        for fold in &folds {
            assert_eq!(fold.len(), 20);
        }
    }

    #[test]
    fn test_simple_stratification_balanced_labels() {
        let labels: Vec<Vec<String>> = (0..100).map(|i| vec![format!("class_{}", i % 4)]).collect();
        let mut rng = SmallRng::seed_from_u64(42);
        let folds = stratified_k_fold(&labels, 5, &mut rng);

        // Each fold should have roughly 5 samples of each of the 4 classes
        for fold in &folds {
            let mut counts = HashMap::new();
            for &idx in fold {
                let label = &labels[idx][0];
                *counts.entry(label.as_str()).or_insert(0) += 1;
            }
            for &count in counts.values() {
                assert!(count >= 4 && count <= 6, "Imbalanced fold: {counts:?}");
            }
        }
    }

    #[test]
    fn test_iterative_stratification_multi_label() {
        let labels: Vec<Vec<String>> = vec![
            vec!["A".into(), "B".into()],
            vec!["A".into()],
            vec!["B".into(), "C".into()],
            vec!["C".into()],
            vec!["A".into(), "C".into()],
            vec!["B".into()],
            vec!["A".into(), "B".into()],
            vec!["C".into()],
            vec!["B".into(), "C".into()],
            vec!["A".into()],
        ];
        let mut rng = SmallRng::seed_from_u64(42);
        let folds = stratified_k_fold(&labels, 2, &mut rng);

        // All samples assigned
        let total: usize = folds.iter().map(|f| f.len()).sum();
        assert_eq!(total, 10);

        // Each fold should have roughly 5 samples (±1 for multi-label imbalance)
        for fold in &folds {
            assert!(
                fold.len() >= 4 && fold.len() <= 6,
                "Fold size {} out of range [4, 6]",
                fold.len()
            );
        }
    }
}
