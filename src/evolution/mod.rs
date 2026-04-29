//! Evolution entry points and configuration.

pub mod config;
#[cfg(feature = "indicatif")]
pub mod indicatif;
pub mod runner;
#[cfg(all(feature = "tui", not(target_arch = "wasm32")))]
pub mod tui;
