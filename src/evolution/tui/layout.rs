use ratatui::prelude::Rect;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DashboardLayout {
    Full,
    Compact,
    Minimal,
}

impl DashboardLayout {
    pub(super) fn for_area(area: Rect) -> Self {
        if area.width >= 100 && area.height >= 28 {
            Self::Full
        } else if area.width >= 80 && area.height >= 22 {
            Self::Compact
        } else {
            Self::Minimal
        }
    }
}

pub(super) fn contains_position(area: Rect, column: u16, row: u16) -> bool {
    column >= area.x
        && column < area.x.saturating_add(area.width)
        && row >= area.y
        && row < area.y.saturating_add(area.height)
}
