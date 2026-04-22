use dioxus::prelude::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AppIcon {
    Bolt,
    Capsules,
    ChartLine,
    ChevronLeft,
    ChevronRight,
    CodeBranch,
    Crown,
    Droplet,
    Dumbbell,
    FaceFrown,
    FaceSmile,
    Filter,
    HourglassHalf,
    Leaf,
    ListCheck,
    PeopleGroup,
    Play,
    Seedling,
    Shuffle,
    Sliders,
    Stop,
    Timeline,
    Trophy,
    WandMagicSparkles,
}

pub(crate) fn app_icon(icon: AppIcon) -> Element {
    let svg = match icon {
        AppIcon::Bolt => include_str!("../assets/icons/solid/bolt.svg"),
        AppIcon::Capsules => include_str!("../assets/icons/solid/capsules.svg"),
        AppIcon::ChartLine => include_str!("../assets/icons/solid/chart-line.svg"),
        AppIcon::ChevronLeft => include_str!("../assets/icons/solid/chevron-left.svg"),
        AppIcon::ChevronRight => include_str!("../assets/icons/solid/chevron-right.svg"),
        AppIcon::CodeBranch => include_str!("../assets/icons/solid/code-branch.svg"),
        AppIcon::Crown => include_str!("../assets/icons/solid/crown.svg"),
        AppIcon::Droplet => include_str!("../assets/icons/solid/droplet.svg"),
        AppIcon::Dumbbell => include_str!("../assets/icons/solid/dumbbell.svg"),
        AppIcon::FaceFrown => include_str!("../assets/icons/regular/face-frown.svg"),
        AppIcon::FaceSmile => include_str!("../assets/icons/regular/face-smile.svg"),
        AppIcon::Filter => include_str!("../assets/icons/solid/filter.svg"),
        AppIcon::HourglassHalf => include_str!("../assets/icons/solid/hourglass-half.svg"),
        AppIcon::Leaf => include_str!("../assets/icons/solid/leaf.svg"),
        AppIcon::ListCheck => include_str!("../assets/icons/solid/list-check.svg"),
        AppIcon::PeopleGroup => include_str!("../assets/icons/solid/people-group.svg"),
        AppIcon::Play => include_str!("../assets/icons/solid/play.svg"),
        AppIcon::Seedling => include_str!("../assets/icons/solid/seedling.svg"),
        AppIcon::Shuffle => include_str!("../assets/icons/solid/shuffle.svg"),
        AppIcon::Sliders => include_str!("../assets/icons/solid/sliders.svg"),
        AppIcon::Stop => include_str!("../assets/icons/solid/stop.svg"),
        AppIcon::Timeline => include_str!("../assets/icons/solid/timeline.svg"),
        AppIcon::Trophy => include_str!("../assets/icons/solid/trophy.svg"),
        AppIcon::WandMagicSparkles => {
            include_str!("../assets/icons/solid/wand-magic-sparkles.svg")
        }
    };

    rsx! {
        span {
            class: "app-icon",
            aria_hidden: "true",
            dangerous_inner_html: "{svg}",
        }
    }
}
