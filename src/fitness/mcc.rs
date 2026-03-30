use genevo::genetic::Fitness;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfusionCounts {
    pub tp: u64,
    pub fp: u64,
    pub tn: u64,
    pub fn_: u64,
}

impl ConfusionCounts {
    pub fn total(self) -> u64 {
        self.tp + self.fp + self.tn + self.fn_
    }
}

impl std::ops::AddAssign for ConfusionCounts {
    fn add_assign(&mut self, rhs: Self) {
        self.tp += rhs.tp;
        self.fp += rhs.fp;
        self.tn += rhs.tn;
        self.fn_ += rhs.fn_;
    }
}

/// Fitness score derived only from MCC.
///
/// MCC is scaled from [-1, 1] into [0, 10000] so it can be ordered and used
/// by the GA while preserving the same ranking as raw MCC.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct MccFitness {
    pub score: i64,
}

impl MccFitness {
    #[inline]
    pub fn from_mcc(mcc: f64) -> Self {
        let score = (((mcc + 1.0) / 2.0) * 10000.0).clamp(0.0, 10000.0) as i64;
        Self { score }
    }

    #[inline]
    pub fn to_mcc(score: i64) -> f64 {
        let bounded = score.clamp(0, Self::max_score()) as f64;
        ((bounded / Self::max_score() as f64) * 2.0) - 1.0
    }

    pub fn max_score() -> i64 {
        10000
    }
}

impl Fitness for MccFitness {
    fn zero() -> Self {
        Self { score: 0 }
    }

    fn abs_diff(&self, other: &Self) -> Self {
        Self {
            score: (self.score - other.score).abs(),
        }
    }
}

pub fn compute_fold_averaged_mcc(fold_counts: &[ConfusionCounts]) -> f64 {
    let mut total_mcc = 0.0f64;
    let mut evaluated_folds = 0usize;

    for counts in fold_counts {
        if counts.total() == 0 {
            continue;
        }
        evaluated_folds += 1;
        total_mcc += compute_mcc_from_counts(counts.tp, counts.fp, counts.tn, counts.fn_);
    }

    if evaluated_folds == 0 {
        0.0
    } else {
        total_mcc / evaluated_folds as f64
    }
}

/// Compute MCC from confusion matrix counts.
#[inline]
pub fn compute_mcc_from_counts(tp: u64, fp: u64, tn: u64, fn_: u64) -> f64 {
    let numerator = (tp as f64) * (tn as f64) - (fp as f64) * (fn_ as f64);
    let denom_sq = (tp + fp) as f64 * (tp + fn_) as f64 * (tn + fp) as f64 * (tn + fn_) as f64;

    if denom_sq == 0.0 {
        0.0
    } else {
        numerator / denom_sq.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcc_perfect() {
        // Perfect classifier
        let mcc = compute_mcc_from_counts(50, 0, 50, 0);
        assert!((mcc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mcc_inverse() {
        // Perfectly wrong classifier
        let mcc = compute_mcc_from_counts(0, 50, 0, 50);
        assert!((mcc - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_mcc_random() {
        // Random classifier: TP≈FP, TN≈FN → MCC≈0
        let mcc = compute_mcc_from_counts(25, 25, 25, 25);
        assert!(mcc.abs() < 1e-10);
    }

    #[test]
    fn test_mcc_degenerate() {
        // All positive predictions, no negatives → denominator has zero factor
        let mcc = compute_mcc_from_counts(50, 50, 0, 0);
        assert_eq!(mcc, 0.0);
    }

    #[test]
    fn test_mcc_known_value() {
        // Known case: TP=20, FP=5, TN=40, FN=10
        // MCC = (20*40 - 5*10) / sqrt(25 * 30 * 45 * 50)
        //     = (800 - 50) / sqrt(1687500)
        //     = 750 / 1299.038...
        //     ≈ 0.5774
        let mcc = compute_mcc_from_counts(20, 5, 40, 10);
        assert!((mcc - 0.5774).abs() < 0.001, "MCC = {mcc}");
    }

    #[test]
    fn mcc_fitness_scales_bounds() {
        assert_eq!(MccFitness::from_mcc(-1.0).score, 0);
        assert_eq!(MccFitness::from_mcc(1.0).score, MccFitness::max_score());
    }

    #[test]
    fn mcc_fitness_round_trip_is_close() {
        let original = 0.5774;
        let recovered = MccFitness::to_mcc(MccFitness::from_mcc(original).score);
        assert!((recovered - original).abs() < 0.001, "MCC = {recovered}");
    }
}
