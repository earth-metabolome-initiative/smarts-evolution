/// Metric shown in the single primary TUI plot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TuiMetric {
    Mcc,
    Coverage,
    Stagnation,
    Unique,
    Timeouts,
    Length,
}

impl TuiMetric {
    pub(super) const ALL: [Self; 6] = [
        Self::Mcc,
        Self::Coverage,
        Self::Stagnation,
        Self::Unique,
        Self::Timeouts,
        Self::Length,
    ];

    pub(super) fn tab_label(self) -> &'static str {
        match self {
            Self::Mcc => "◆ MCC",
            Self::Coverage => "▣ Coverage",
            Self::Stagnation => "↻ Stagnation",
            Self::Unique => "◇ Unique",
            Self::Timeouts => "◷ Timeouts",
            Self::Length => "↔ Length",
        }
    }

    pub(super) fn short_tab_label(self) -> &'static str {
        match self {
            Self::Mcc => "◆ MCC",
            Self::Coverage => "▣ Cov",
            Self::Stagnation => "↻ Stag",
            Self::Unique => "◇ Uniq",
            Self::Timeouts => "◷ TO",
            Self::Length => "↔ Len",
        }
    }

    pub(super) fn title(self) -> &'static str {
        match self {
            Self::Mcc => "MCC best-so-far",
            Self::Coverage => "Coverage best-so-far",
            Self::Stagnation => "Generations since improvement",
            Self::Unique => "Unique population ratio",
            Self::Timeouts => "Match timeouts per generation",
            Self::Length => "SMARTS length",
        }
    }

    pub(super) fn next(self) -> Self {
        let index = Self::ALL
            .iter()
            .position(|metric| *metric == self)
            .unwrap_or(0);
        Self::ALL[(index + 1) % Self::ALL.len()]
    }

    pub(super) fn previous(self) -> Self {
        let index = Self::ALL
            .iter()
            .position(|metric| *metric == self)
            .unwrap_or(0);
        Self::ALL[(index + Self::ALL.len() - 1) % Self::ALL.len()]
    }

    pub(super) const fn supports_y_axis_mode(self) -> bool {
        matches!(self, Self::Mcc | Self::Coverage)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum YAxisMode {
    Absolute,
    Zoomed,
}

impl YAxisMode {
    pub(super) const fn toggle(self) -> Self {
        match self {
            Self::Absolute => Self::Zoomed,
            Self::Zoomed => Self::Absolute,
        }
    }

    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Absolute => "abs",
            Self::Zoomed => "zoom",
        }
    }
}
