use std::cmp::Ordering;
use std::time::Duration;

use genevo::genetic::Fitness;

const MCC_SCORE_SCALE: i64 = 10_000;
const OBJECTIVE_MCC_WEIGHT: i64 = 256;
const TIME_PENALTY_LOG_SCALE: f64 = 64.0;

/// Fitness used by the GA.
///
/// The objective is intentionally two-part:
/// 1. maximize MCC
/// 2. prefer SMARTS that compile and match faster
///
/// Time is log-scaled in microseconds so the signal is strong enough to matter
/// but not so noisy that tiny runtime jitter dominates selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectiveFitness {
    score: i64,
    mcc_score: i64,
    elapsed_micros: u64,
}

impl ObjectiveFitness {
    #[inline]
    pub fn from_metrics(mcc: f64, elapsed: Duration) -> Self {
        let mcc_score = scale_mcc(mcc);
        let elapsed_micros = elapsed
            .as_micros()
            .min(u128::from(u64::MAX))
            .try_into()
            .unwrap_or(u64::MAX);
        let time_penalty = time_penalty_from_micros(elapsed_micros);
        let score = mcc_score * OBJECTIVE_MCC_WEIGHT - time_penalty;

        Self {
            score,
            mcc_score,
            elapsed_micros,
        }
    }

    #[inline]
    pub fn invalid(elapsed: Duration) -> Self {
        Self::from_metrics(-1.0, elapsed)
    }

    #[inline]
    pub fn mcc(self) -> f64 {
        unscale_mcc(self.mcc_score)
    }

    #[inline]
    pub fn elapsed(self) -> Duration {
        Duration::from_micros(self.elapsed_micros)
    }

    #[inline]
    pub fn time_penalty(self) -> i64 {
        time_penalty_from_micros(self.elapsed_micros)
    }

    #[inline]
    pub fn max_score() -> i64 {
        scale_mcc(1.0) * OBJECTIVE_MCC_WEIGHT
    }

    #[inline]
    pub fn score(self) -> i64 {
        self.score
    }

    #[inline]
    pub fn mcc_score(self) -> i64 {
        self.mcc_score
    }

    #[inline]
    pub fn elapsed_micros(self) -> u64 {
        self.elapsed_micros
    }
}

impl Ord for ObjectiveFitness {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .cmp(&other.score)
            .then_with(|| self.mcc_score.cmp(&other.mcc_score))
            .then_with(|| other.elapsed_micros.cmp(&self.elapsed_micros))
    }
}

impl PartialOrd for ObjectiveFitness {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Fitness for ObjectiveFitness {
    fn zero() -> Self {
        Self::from_metrics(-1.0, Duration::ZERO)
    }

    fn abs_diff(&self, other: &Self) -> Self {
        Self {
            score: (self.score - other.score).abs(),
            mcc_score: (self.mcc_score - other.mcc_score).abs(),
            elapsed_micros: self.elapsed_micros.abs_diff(other.elapsed_micros),
        }
    }
}

#[inline]
fn scale_mcc(mcc: f64) -> i64 {
    (((mcc + 1.0) / 2.0) * MCC_SCORE_SCALE as f64).clamp(0.0, MCC_SCORE_SCALE as f64) as i64
}

#[inline]
fn unscale_mcc(score: i64) -> f64 {
    let bounded = score.clamp(0, MCC_SCORE_SCALE) as f64;
    ((bounded / MCC_SCORE_SCALE as f64) * 2.0) - 1.0
}

#[inline]
fn time_penalty_from_micros(elapsed_micros: u64) -> i64 {
    if elapsed_micros == 0 {
        return 0;
    }

    ((elapsed_micros as f64).log2() * TIME_PENALTY_LOG_SCALE).round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn higher_mcc_beats_faster_runtime() {
        let slower_better = ObjectiveFitness::from_metrics(0.8000, Duration::from_millis(5));
        let faster_worse = ObjectiveFitness::from_metrics(0.7800, Duration::from_micros(50));

        assert!(slower_better > faster_worse);
    }

    #[test]
    fn faster_runtime_breaks_close_mcc_ties() {
        let slow = ObjectiveFitness::from_metrics(0.8123, Duration::from_millis(9));
        let fast = ObjectiveFitness::from_metrics(0.8123, Duration::from_micros(200));

        assert!(fast > slow);
    }

    #[test]
    fn objective_round_trips_mcc() {
        let fitness = ObjectiveFitness::from_metrics(0.5774, Duration::from_micros(750));
        assert!((fitness.mcc() - 0.5774).abs() < 0.001);
    }

    #[test]
    fn invalid_and_zero_objectives_use_negative_unit_mcc() {
        let invalid = ObjectiveFitness::invalid(Duration::from_micros(25));
        let zero = ObjectiveFitness::zero();

        assert_eq!(invalid.mcc_score(), 0);
        assert_eq!(invalid.elapsed(), Duration::from_micros(25));
        assert_eq!(zero, ObjectiveFitness::from_metrics(-1.0, Duration::ZERO));
    }

    #[test]
    fn time_penalty_and_max_score_are_exposed() {
        let zero_elapsed = ObjectiveFitness::from_metrics(1.0, Duration::ZERO);
        let nonzero_elapsed = ObjectiveFitness::from_metrics(1.0, Duration::from_micros(256));

        assert_eq!(zero_elapsed.time_penalty(), 0);
        assert!(nonzero_elapsed.time_penalty() > 0);
        assert_eq!(ObjectiveFitness::max_score(), zero_elapsed.score());
    }

    #[test]
    fn abs_diff_tracks_score_mcc_and_runtime_distance() {
        let left = ObjectiveFitness::from_metrics(0.9, Duration::from_micros(900));
        let right = ObjectiveFitness::from_metrics(0.4, Duration::from_micros(100));
        let diff = left.abs_diff(&right);

        assert_eq!(diff.score(), (left.score() - right.score()).abs());
        assert_eq!(
            diff.mcc_score(),
            (left.mcc_score() - right.mcc_score()).abs()
        );
        assert_eq!(diff.elapsed_micros(), 800);
    }
}
