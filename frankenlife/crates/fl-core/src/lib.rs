//! # fl-core — FieldEcologyV1
//!
//! The authoritative model of the FrankenLife design plan's first vertical
//! slice: a periodic 2-D energy/moment field fed by moving Gaussian sources;
//! point organisms with independent neural genomes that sense the field,
//! absorb it under a two-pass demand/grant protocol, react to its composition,
//! move by chemotaxis, emit paid packets, split into mutated descendants and
//! die when exhausted. Interaction is field-only and every unit of energy is
//! accounted for in a global ledger.
//!
//! The tick is computed into a candidate state and published atomically after
//! invariant checks; a rejected candidate (for example a capacity pause) leaves
//! the last committed world authoritative.

#![forbid(unsafe_code)]

pub mod checkpoint;
pub mod chemistry;
pub mod config;
pub mod field;
pub mod genome;
pub mod nn;
pub mod profile;
pub mod rng;
pub mod sensors;
pub mod sources;
pub mod world;

/// Number of signed chemical moment channels.
pub const CHANNELS: usize = 32;

/// Model identity string. Changing any frozen rule changes this.
pub const MODEL_VERSION: &str = "field-ecology-v1/fl-core-0.1";

pub use config::Config;
pub use profile::{CpuStage, GpuCounter, ProfileMeta, StageProfiler};
pub use world::{Command, ControllerBackend, DeviceBackend, Disposal, FieldAccounting, FieldBackend, FieldDeposit, FinishReport, Inference, ReferenceDevice, ReferenceResident, ResidentBackend, UptakeJob, UptakeOut, InferenceJob, ReferenceBackend, ReferenceSenseInfer, SenseInferBackend, SenseJob, SenseOutput, TickError, TickReport, World};
