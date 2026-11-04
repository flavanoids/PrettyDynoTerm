pub mod benchmark;
pub mod cli;
pub mod cpu;
pub mod gpu;
pub mod self_test;
pub mod shaders;
pub mod telemetry;
pub mod terminal;

pub use cli::{parse_resolution, Args};
