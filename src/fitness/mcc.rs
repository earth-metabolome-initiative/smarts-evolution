#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConfusionCounts {
    tp: u64,
    fp: u64,
    tn: u64,
    fn_: u64,
}

impl ConfusionCounts {
    pub fn new(tp: u64, fp: u64, tn: u64, fn_: u64) -> Self {
        Self { tp, fp, tn, fn_ }
    }

    pub fn total(self) -> u64 {
        self.tp + self.fp + self.tn + self.fn_
    }

    pub fn record_match(&mut self, matched: bool, is_positive: bool) {
        match (matched, is_positive) {
            (true, true) => self.tp += 1,
            (true, false) => self.fp += 1,
            (false, true) => self.fn_ += 1,
            (false, false) => self.tn += 1,
        }
    }
}

impl core::ops::AddAssign for ConfusionCounts {
    fn add_assign(&mut self, rhs: Self) {
        self.tp += rhs.tp;
        self.fp += rhs.fp;
        self.tn += rhs.tn;
        self.fn_ += rhs.fn_;
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
    fn confusion_counts_add_assign_and_fold_average_skip_empty_folds() {
        let mut total = ConfusionCounts::new(1, 2, 3, 4);
        total += ConfusionCounts::new(5, 6, 7, 8);

        assert_eq!(total, ConfusionCounts::new(6, 8, 10, 12));

        let mcc = compute_fold_averaged_mcc(&[
            ConfusionCounts::default(),
            ConfusionCounts::new(5, 1, 7, 1),
        ]);
        let expected = compute_mcc_from_counts(5, 1, 7, 1);
        assert!((mcc - expected).abs() < 1e-12);
        assert_eq!(
            compute_fold_averaged_mcc(&[ConfusionCounts::default()]),
            0.0
        );
    }
}
