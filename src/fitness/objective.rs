use core::cmp::Ordering;

const MCC_SCORE_SCALE: i64 = 10_000;

/// Fitness used by the GA.
///
/// The objective is just fold-averaged MCC. SMARTS length is handled outside
/// this type as a secondary ranking rule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectiveFitness {
    mcc_score: i64,
}

impl ObjectiveFitness {
    #[inline]
    pub fn from_mcc(mcc: f64) -> Self {
        Self {
            mcc_score: scale_mcc(mcc),
        }
    }

    #[inline]
    pub fn invalid() -> Self {
        Self::from_mcc(-1.0)
    }

    #[inline]
    pub fn mcc(self) -> f64 {
        unscale_mcc(self.mcc_score)
    }

    pub fn zero() -> Self {
        Self::invalid()
    }
}

impl Ord for ObjectiveFitness {
    fn cmp(&self, other: &Self) -> Ordering {
        self.mcc_score.cmp(&other.mcc_score)
    }
}

impl PartialOrd for ObjectiveFitness {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn higher_mcc_is_better() {
        let stronger = ObjectiveFitness::from_mcc(0.8000);
        let weaker = ObjectiveFitness::from_mcc(0.7800);

        assert!(stronger > weaker);
    }

    #[test]
    fn objective_round_trips_mcc() {
        let fitness = ObjectiveFitness::from_mcc(0.5774);
        assert!((fitness.mcc() - 0.5774).abs() < 0.001);
    }

    #[test]
    fn invalid_and_zero_objectives_use_negative_unit_mcc() {
        let invalid = ObjectiveFitness::invalid();
        let zero = ObjectiveFitness::zero();

        assert_eq!(invalid, ObjectiveFitness::from_mcc(-1.0));
        assert_eq!(zero, ObjectiveFitness::from_mcc(-1.0));
    }

    #[test]
    fn perfect_mcc_uses_max_scaled_score() {
        let perfect = ObjectiveFitness::from_mcc(1.0);

        assert_eq!(perfect, ObjectiveFitness::from_mcc(1.0));
    }
}
