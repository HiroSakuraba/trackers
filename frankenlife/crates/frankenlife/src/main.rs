//! `frankenlife` — run, render and inspect FieldEcologyV1 worlds headlessly.

#![forbid(unsafe_code)]

mod painter;
mod profile_cli;
mod render;
mod viewer;

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, Subcommand};
use fl_core::profile::CpuStage;
use fl_core::world::{Command, Disposal};
use fl_core::{checkpoint, Config, ProfileMeta, TickError, World, MODEL_VERSION};

use profile_cli::GpuRollup;
use serde_json::json;

use render::{render, write_png, RenderOpts, View};

#[derive(Parser)]
#[command(name = "frankenlife", version, about = "FrankenLife FieldEcologyV1: an evolving, field-mediated neural ecology")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Validate a configuration and print the derived capacity/cost report.
    Validate {
        /// TOML configuration (defaults to the embedded reference preset).
        config: Option<PathBuf>,
    },
    /// Run a world headlessly, writing metrics, frames and inspections.
    Run(RunArgs),
    /// Inspect organisms in a saved checkpoint.
    Inspect {
        /// Checkpoint file (`.flck`).
        checkpoint: PathBuf,
        /// Stable organism IDs to inspect (default: the most derived ones).
        #[arg(long)]
        id: Vec<u64>,
        /// How many organisms to show when no ID is given.
        #[arg(long, default_value_t = 3)]
        top: usize,
    },
    /// Open a desktop window and watch the world live.
    View(viewer::ViewArgs),
    /// Open the GPU, check it against the CPU reference, and benchmark inference.
    GpuCheck {
        /// Organisms (independent networks) per batch.
        #[arg(long, default_value_t = 4096)]
        organisms: usize,
        /// Hidden layers: 8 | 32 | 64,32 | 96,64 | 128,96.
        #[arg(long, default_value = "64,32")]
        hidden: String,
        /// Timed batches per backend.
        #[arg(long, default_value_t = 20)]
        iters: usize,
        /// Idle milliseconds between GPU batches, to imitate a run where the CPU
        /// works between dispatches (exposes GPU power-state downclocking).
        #[arg(long, default_value_t = 0)]
        gap_ms: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Backend {
    /// fl-core's scalar reference evaluator (multithreaded, bitwise deterministic).
    Cpu,
    /// wgpu compute on the GPU (tolerance-equivalent to the reference).
    Gpu,
    /// Sensing pyramid, sampling and inference fused on the GPU (inputs never leave the GPU).
    GpuFused,
    /// `gpu-fused` plus field transport, sources and tiny clearing on the GPU.
    GpuField,
    /// The field stays on the GPU across steps; uptake exchange and deposits run there too.
    GpuResident,
    /// The resident path with the whole front half of a tick (field step,
    /// sensing, inference, uptake) queued together and read with one wait.
    /// Same rules; the uptake request is evaluated in f32 on the device.
    GpuRealtime,
}

#[derive(Parser)]
struct RunArgs {
    /// TOML configuration (defaults to the embedded reference preset).
    #[arg(long)]
    config: Option<PathBuf>,
    /// Override the seed.
    #[arg(long)]
    seed: Option<u64>,
    /// Simulated seconds to run (converted to integer ticks).
    #[arg(long, default_value_t = 30.0)]
    seconds: f64,
    /// Output directory.
    #[arg(long, default_value = "runs/latest")]
    out: PathBuf,
    /// Render a frame every K ticks (0 disables frames).
    #[arg(long, default_value_t = 32)]
    frame_every: u64,
    /// Square frame size in pixels.
    #[arg(long, default_value_t = 1024)]
    size: usize,
    #[arg(long, value_enum, default_value_t = View::Composite)]
    view: View,
    /// Worker threads for sensing/inference (execution only; science is unchanged).
    #[arg(long)]
    workers: Option<usize>,
    /// Use the exhibition capacity policy (stable-order birth rejection) instead of pausing.
    #[arg(long)]
    exhibition: bool,
    /// Schedule an erase intervention: TICK:X:Y:RADIUS (repeatable).
    #[arg(long, value_parser = parse_erase)]
    erase: Vec<(u64, f64, f64, f64)>,
    /// Number of organisms to dump into inspection.json at the end.
    #[arg(long, default_value_t = 5)]
    inspect: usize,
    /// Resume from a checkpoint instead of a fresh world (`--config`/`--seed` are then ignored);
    /// `--seconds` counts additional simulated time.
    #[arg(long)]
    resume: Option<PathBuf>,
    /// Write an exact checkpoint every S simulated seconds (plus one at the end).
    #[arg(long)]
    checkpoint_every: Option<f64>,
    /// Where controller networks are evaluated.
    #[arg(long, value_enum, default_value_t = Backend::Cpu)]
    backend: Backend,
    /// Opt-in per-stage profile. Off by default (no extra stage timers, no extra
    /// GPU synchronization). Writes `profile.json` and a short summary.
    /// GPU submit/wait figures are not GPU kernel time.
    #[arg(long)]
    profile: bool,
}

fn parse_erase(s: &str) -> Result<(u64, f64, f64, f64), String> {
    let p: Vec<&str> = s.split(':').collect();
    if p.len() != 4 {
        return Err("expected TICK:X:Y:RADIUS".into());
    }
    let f = |v: &str| v.parse::<f64>().map_err(|e| e.to_string());
    Ok((p[0].parse::<u64>().map_err(|e| e.to_string())?, f(p[1])?, f(p[2])?, f(p[3])?))
}

fn load(path: &Option<PathBuf>) -> Result<Config, String> {
    match path {
        Some(p) => Config::from_toml(&std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?),
        None => Ok(Config::reference()),
    }
}

fn main() {
    let cli = Cli::parse();
    let code = match cli.cmd {
        Cmd::Validate { config } => validate(&config),
        Cmd::Run(a) => run(a),
        Cmd::View(a) => viewer::view(a),
        Cmd::Inspect { checkpoint, id, top } => inspect_checkpoint(&checkpoint, &id, top),
        Cmd::GpuCheck { organisms, hidden, iters, gap_ms } => gpu_check(organisms, &hidden, iters, gap_ms),
    };
    std::process::exit(code);
}

fn validate(path: &Option<PathBuf>) -> i32 {
    let cfg = match load(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    match cfg.validate() {
        Ok(r) => {
            println!("{}", serde_json::to_string_pretty(&r).unwrap());
            0
        }
        Err(errs) => {
            for e in errs {
                eprintln!("invalid: {e}");
            }
            1
        }
    }
}

#[cfg(feature = "gpu")]
fn gpu_check(organisms: usize, hidden: &str, iters: usize, gap_ms: u64) -> i32 {
    use fl_core::nn::{INPUT_WIDTH, OUTPUT_WIDTH};
    use fl_core::{ControllerBackend, InferenceJob, ReferenceBackend};
    let hidden: Vec<u32> = match hidden.split(',').map(|v| v.trim().parse::<u32>()).collect() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("--hidden: {e}");
            return 2;
        }
    };
    let mut cfg = Config::reference();
    cfg.controller.hidden_widths = hidden.clone();
    if let Err(e) = cfg.validate() {
        eprintln!("invalid topology {hidden:?}: {e:?}");
        return 2;
    }
    let mut gpu = match fl_gpu::GpuBackend::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("GPU unavailable: {e}");
            return 1;
        }
    };
    println!("adapter      : {} ({}, {})", gpu.info.name, gpu.info.backend, gpu.info.device_type);
    println!("driver       : {}", gpu.info.driver);
    println!("binding limit: {} MiB per storage buffer", gpu.info.max_storage_binding_bytes >> 20);
    let rng = fl_core::rng::SemanticRng::new(1);
    let genomes: Vec<_> = (0..organisms as u64).map(|i| fl_core::genome::founder(&cfg, &rng, i, i)).collect();
    let params = genomes[0].param_count();
    let inputs: Vec<[f32; INPUT_WIDTH]> = (0..organisms).map(|g| std::array::from_fn(|i| ((g * 37 + i * 11) % 200) as f32 / 100.0 - 1.0)).collect();
    let jobs: Vec<InferenceJob<'_>> = genomes.iter().zip(&inputs).map(|(g, x)| InferenceJob { organism: g.id, genome: g, input: x }).collect();
    let mut a = vec![[0.0f32; OUTPUT_WIDTH]; jobs.len()];
    let mut b = vec![[0.0f32; OUTPUT_WIDTH]; jobs.len()];
    let t = Instant::now();
    if let Err(e) = gpu.evaluate(&jobs, &mut a) {
        eprintln!("GPU evaluation failed: {e}");
        return 1;
    }
    let upload_s = t.elapsed().as_secs_f64();
    let _ = ReferenceBackend.evaluate(&jobs, &mut b);
    let diff = a.iter().zip(&b).flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs())).fold(0.0f32, f32::max);
    let (mut prep_ms, mut round_ms, mut kernel_ms, mut kernel_n) = (0.0, 0.0, 0.0, 0usize);
    let mut busy_s = 0.0;
    for _ in 0..iters {
        if gap_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(gap_ms));
        }
        let t = Instant::now();
        let _ = gpu.evaluate(&jobs, &mut a);
        busy_s += t.elapsed().as_secs_f64();
        let l = gpu.stats.last;
        prep_ms += l.host_prep_ms;
        round_ms += l.gpu_roundtrip_ms;
        if let Some(k) = l.kernel_ms {
            kernel_ms += k;
            kernel_n += 1;
        }
    }
    let gpu_s = busy_s / iters.max(1) as f64;
    let it = iters.max(1) as f64;
    if gap_ms > 0 {
        println!("idle gap     : {gap_ms} ms before each timed batch (imitating CPU work between dispatches)");
    }
    let t = Instant::now();
    for _ in 0..iters.clamp(1, 3) {
        let _ = ReferenceBackend.evaluate(&jobs, &mut b);
    }
    let cpu_s = t.elapsed().as_secs_f64() / iters.clamp(1, 3) as f64;
    let weights_mib = (organisms * params * 4) as f64 / (1024.0 * 1024.0);
    println!("networks     : {organisms} independent x {params} params ({weights_mib:.0} MiB of weights)");
    println!("first batch  : {:.1} ms (includes uploading every genome)", upload_s * 1e3);
    println!("max |dlogit| : {diff:.2e} vs CPU reference ({})", if diff < 1e-4 { "OK" } else { "TOO LARGE" });
    println!("GPU batch    : {:.2} ms  ->  {:.0} network evals/s, {:.1} GB/s of weights", gpu_s * 1e3, organisms as f64 / gpu_s, weights_mib / 1024.0 / gpu_s);
    println!("  host prep  : {:.2} ms (slot lookup, input packing, buffer writes)", prep_ms / it);
    println!("  round trip : {:.2} ms (submit, wait, read back logits; includes the kernel)", round_ms / it);
    if kernel_n > 0 {
        let k = kernel_ms / kernel_n as f64;
        println!("  kernel     : {:.2} ms (GPU timestamps)  ->  {:.1} GB/s of weights in the kernel alone", k, weights_mib / 1024.0 / (k / 1e3));
    } else {
        println!("  kernel     : n/a (this adapter does not support timestamp queries)");
    }
    println!("weights bound: {:.2} ms if the kernel streamed weights at 320 GB/s (GTX 1080 peak)", weights_mib * 1024.0 * 1024.0 / 320e9 * 1e3);
    println!("budget       : {:.2} ms per whole simulation step at 128 steps/s", 1000.0 / 128.0);
    println!("CPU batch    : {:.2} ms (scalar reference, one thread)  ->  GPU is {:.1}x faster", cpu_s * 1e3, cpu_s / gpu_s);
    if diff < 1e-4 { 0 } else { 1 }
}

#[cfg(not(feature = "gpu"))]
fn gpu_check(_: usize, _: &str, _: usize, _: u64) -> i32 {
    eprintln!("this build has no GPU support (rebuild with the default `gpu` feature)");
    2
}

fn most_derived(world: &World, k: usize) -> Vec<u64> {
    let mut idx: Vec<usize> = (0..world.orgs.len()).collect();
    idx.sort_by(|&i, &j| world.orgs.generation[j].cmp(&world.orgs.generation[i]).then(world.orgs.energy[j].total_cmp(&world.orgs.energy[i])));
    idx.into_iter().take(k).map(|i| world.orgs.id[i]).collect()
}

fn inspect_checkpoint(path: &Path, ids: &[u64], top: usize) -> i32 {
    let world = match std::fs::read(path).map_err(|e| e.to_string()).and_then(|b| checkpoint::load(&b)) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("{}: {e}", path.display());
            return 2;
        }
    };
    let ids: Vec<u64> = if ids.is_empty() { most_derived(&world, top) } else { ids.to_vec() };
    let mut out = Vec::new();
    for id in ids {
        match world.inspect(id) {
            Some(i) => out.push(serde_json::to_value(i).unwrap()),
            None => out.push(json!({"id": id, "error": "not alive at this tick"})),
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "checkpoint": path.display().to_string(),
            "tick": world.tick,
            "simulation_seconds": world.simulation_seconds(),
            "population": world.orgs.len(),
            "state_hash": format!("{:016x}", world.state_hash()),
            "organisms": out,
        }))
        .unwrap()
    );
    0
}

fn write_checkpoint(world: &World, dir: &Path, retain: usize) {
    let path = dir.join(format!("tick_{:010}.flck", world.tick));
    if let Err(e) = std::fs::write(&path, checkpoint::save(world)) {
        eprintln!("checkpoint {}: {e}", path.display());
        return;
    }
    // Retain only the most recent `retain` checkpoints written by this run.
    if let Ok(rd) = std::fs::read_dir(dir) {
        let mut all: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "flck")).collect();
        all.sort();
        let n = all.len();
        for p in all.into_iter().take(n.saturating_sub(retain.max(1))) {
            std::fs::remove_file(p).ok();
        }
    }
}

pub(crate) struct Launch {
    pub config: Option<PathBuf>,
    pub seed: Option<u64>,
    pub exhibition: bool,
    pub workers: Option<usize>,
    pub resume: Option<PathBuf>,
    pub erase: Vec<(u64, f64, f64, f64)>,
    pub profile: bool,
    pub backend: Backend,
}

pub(crate) struct Prepared {
    pub world: World,
    #[cfg(feature = "gpu")]
    pub gpu: Option<fl_gpu::GpuBackend>,
    pub ts_supported: Option<bool>,
    pub report: fl_core::config::ValidationReport,
    pub start_tick: u64,
}

pub(crate) fn launch_of(a: &RunArgs) -> Launch {
    Launch {
        config: a.config.clone(),
        seed: a.seed,
        exhibition: a.exhibition,
        workers: a.workers,
        resume: a.resume.clone(),
        erase: a.erase.clone(),
        profile: a.profile,
        backend: a.backend,
    }
}

pub(crate) fn prepare(a: &Launch) -> Result<Prepared, i32> {
    let mut world = if let Some(path) = &a.resume {
        match std::fs::read(path).map_err(|e| e.to_string()).and_then(|b| checkpoint::load(&b)) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                return Err(2);
            }
        }
    } else {
        let mut cfg = match load(&a.config) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("{e}");
                return Err(2);
            }
        };
        if let Some(s) = a.seed {
            cfg.seed = s;
        }
        if a.exhibition {
            cfg.population.capacity_policy = "reject-births-stable-order".into();
        }
        match World::new(cfg) {
            Ok(w) => w,
            Err(errs) => {
                for e in errs {
                    eprintln!("invalid: {e}");
                }
                return Err(1);
            }
        }
    };
    let cfg = world.cfg.clone();
    let report = cfg.validate().expect("world config was validated");
    let start_tick = world.tick;
    if let Some(w) = a.workers {
        world.workers = w.max(1);
    } else {
        world.workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    }
    world.set_profiling(a.profile);
    #[cfg(feature = "gpu")]
    let gpu = match a.backend {
        Backend::Gpu | Backend::GpuFused | Backend::GpuField | Backend::GpuResident | Backend::GpuRealtime => match fl_gpu::GpuBackend::new() {
            Ok(mut g) => {
                g.set_profiling(a.profile);
                g.set_single_wait(a.backend == Backend::GpuRealtime);
                eprintln!("GPU: {}", g.capability());
                if a.profile {
                    eprintln!(
                        "stage profile: on | TIMESTAMP_QUERY {} | resident timestamp spans only on uptake/finish, which already wait",
                        if g.timestamps_supported() { "supported" } else { "not supported; spans omitted" }
                    );
                }
                Some(g)
            }
            Err(e) => {
                eprintln!("cannot use the GPU backend: {e}");
                return Err(2);
            }
        },
        Backend::Cpu => None,
    };
    #[cfg(feature = "gpu")]
    let ts_supported = gpu.as_ref().map(|g| g.timestamps_supported());
    #[cfg(not(feature = "gpu"))]
    let ts_supported: Option<bool> = None;
    #[cfg(not(feature = "gpu"))]
    if a.backend != Backend::Cpu {
        eprintln!("this build has no GPU support (rebuild with the default `gpu` feature)");
        return Err(2);
    }
    for &(t, x, y, r) in &a.erase {
        if let Err(e) = world.schedule(t, Command::EraseDisk { x, y, radius: r, disposal: Disposal::Recycle }) {
            eprintln!("cannot schedule erase: {e}");
            return Err(2);
        }
    }
    Ok(Prepared {
        world,
        #[cfg(feature = "gpu")]
        gpu,
        ts_supported,
        report,
        start_tick,
    })
}

fn run(a: RunArgs) -> i32 {
    let prep = match prepare(&launch_of(&a)) {
        Ok(p) => p,
        Err(code) => return code,
    };
    let mut world = prep.world;
    #[cfg(feature = "gpu")]
    let mut gpu = prep.gpu;
    let report = prep.report;
    let start_tick = prep.start_tick;
    let ts_supported = prep.ts_supported;
    let cfg = world.cfg.clone();
    let frames_dir = a.out.join("frames");
    if let Err(e) = std::fs::create_dir_all(&frames_dir) {
        eprintln!("{}: {e}", frames_dir.display());
        return 2;
    }
    let ticks = (a.seconds / world.dt()).round() as u64;
    let ckpt_dir = a.out.join("checkpoints");
    let ckpt_every = a.checkpoint_every.map(|s| ((s / world.dt()).round() as u64).max(1));
    if ckpt_every.is_some() {
        if let Err(e) = std::fs::create_dir_all(&ckpt_dir) {
            eprintln!("{}: {e}", ckpt_dir.display());
            return 2;
        }
    }
    let retain = cfg.persistence.retain_recent_checkpoints as usize;
    let metrics_every = cfg.persistence.metrics_interval_ticks as u64;
    let mut metrics = std::io::BufWriter::new(std::fs::File::create(a.out.join("metrics.jsonl")).expect("metrics file"));
    let opts_base = RenderOpts {
        size: a.size,
        view: a.view,
        bloom_strength: if cfg.presentation.bloom_enabled { cfg.presentation.bloom_strength as f32 } else { 0.0 },
        point_radius: cfg.presentation.point_radius_pixels as f32 * 1.6,
        background: render::parse_hex(&cfg.presentation.background),
        telemetry: vec![],
    };

    eprintln!(
        "frankenlife {MODEL_VERSION} | {} | seed {} | {} organisms, {}x{} field, dt = 1/{} s | {} ticks | workers {}",
        cfg.name, cfg.seed, cfg.population.initial_count, cfg.world.grid_width, cfg.world.grid_height, cfg.world.tick_denominator, ticks, world.workers
    );
    let wall0 = Instant::now();
    let mut interval_wall = Instant::now();
    let mut interval_ticks = 0u64;
    let mut interval = (0u32, 0u32);
    let mut frame_no = 0usize;
    let mut stop_reason = "completed".to_string();
    let mut steps_per_s = 0.0;
    let mut phases = Phases::default();
    let mut run_phases = Phases::default();
    let mut gpu_rollup = GpuRollup::default();
    let mut gpu_mark = GpuRollup::default();
    let mut profile_mark = world.stage_profile().totals();
    if a.frame_every > 0 {
        let stamp = world.stage_profile().start();
        save_frame(&world, &opts_base, &frames_dir, frame_no, steps_per_s);
        world.stage_profile_mut().stop(CpuStage::Render, stamp);
        frame_no += 1;
    }
    for _ in 0..ticks {
        let stepped = step_world(&mut world, a.backend, #[cfg(feature = "gpu")] &mut gpu);
        let r = match stepped {
            Ok(r) => r,
            Err(e) => {
                stop_reason = match &e {
                    TickError::CapacityPaused { .. } => format!("paused before commit: {e}"),
                    _ => format!("stopped: {e}"),
                };
                eprintln!("{stop_reason}");
                break;
            }
        };
        phases.add(&r.timing);
        run_phases.add(&r.timing);
        #[cfg(feature = "gpu")]
        if let Some(g) = gpu.as_ref() {
            let l = g.stats.last;
            let fl = g.stats.field_last;
            let rl = g.stats.resident_last;
            let v = [
                l.host_prep_ms, l.gpu_roundtrip_ms, l.kernel_ms.unwrap_or(0.0), l.bookkeeping_ms, l.total_ms, l.pyramid_ms.unwrap_or(0.0),
                fl.host_ms, fl.roundtrip_ms, fl.kernel_ms.unwrap_or(0.0),
                rl.begin_ms, rl.uptake_ms, rl.deposit_ms, rl.finish_ms, rl.commit_ms,
            ];
            for k in 0..14 {
                phases.gpu[k] += v[k];
                run_phases.gpu[k] += v[k];
            }
            if a.profile {
                gpu_rollup.add_backend(g);
            }
        }
        interval.0 += r.counts.births;
        interval.1 += r.counts.deaths_exhaustion + r.counts.deaths_user;
        interval_ticks += 1;
        if r.population == 0 {
            stop_reason = format!("extinction at tick {}", r.tick);
            eprintln!("{stop_reason}");
            write_metrics(&mut metrics, &world, interval, steps_per_s, &phases, interval_profile(&world, &a, &profile_mark, &gpu_mark, &gpu_rollup, phases.step_ms).as_ref().map(|p| &p.1));
            break;
        }
        if world.tick % metrics_every == 0 {
            let el = interval_wall.elapsed().as_secs_f64();
            steps_per_s = interval_ticks as f64 / el.max(1e-9);
            let stage_line = interval_profile(&world, &a, &profile_mark, &gpu_mark, &gpu_rollup, phases.step_ms);
            write_metrics(&mut metrics, &world, interval, steps_per_s, &phases, stage_line.as_ref().map(|p| &p.1));
            let l = &world.ledger;
            let n = phases.ticks.max(1) as f64;
            eprintln!(
                "tick {:>6}  t={:>7.2}s  pop {:>6}  births {:>5}  deaths {:>5}  resid {:>9.2e}  {:>6.1} steps/s | step {:>6.2} ms = field {:.2} + sense/infer {:.2} (backend {:.2}) + exchange {:.2} + lifecycle {:.2}",
                world.tick, world.simulation_seconds(), world.orgs.len(), interval.0, interval.1, l.residual, steps_per_s,
                phases.step_ms / n, phases.transport_ms / n, phases.sense_and_infer_ms / n, phases.backend_inference_ms / n, phases.exchange_ms / n, phases.lifecycle_ms / n
            );
            if phases.gpu[4] > 0.0 {
                eprintln!(
                    "             gpu backend {:.2} ms = host prep {:.2} + submit/wait {:.2} (timestamp span {:.2}, pyramid span {:.2}) + bookkeeping {:.2}; outside evaluate {:.2}",
                    phases.gpu[4] / n, phases.gpu[0] / n, phases.gpu[1] / n, phases.gpu[2] / n, phases.gpu[5] / n, phases.gpu[3] / n,
                    (phases.backend_inference_ms - phases.gpu[4]) / n
                );
            }
            if phases.gpu[10] > 0.0 {
                eprintln!(
                    "             resident: begin {:.2} | uptake {:.2} | deposits {:.2} | finish {:.2} | commit {:.2}",
                    phases.gpu[9] / n, phases.gpu[10] / n, phases.gpu[11] / n, phases.gpu[12] / n, phases.gpu[13] / n
                );
            }
            if phases.gpu[7] > 0.0 {
                eprintln!(
                    "             gpu field step {:.2} ms = host {:.2} + submit/wait {:.2} (timestamp span {:.2}, not kernel-exclusive) | setup {:.2} + validate {:.2} | other {:.2}",
                    phases.transport_ms / n, phases.gpu[6] / n, phases.gpu[7] / n, phases.gpu[8] / n, phases.setup_ms / n, phases.validate_ms / n,
                    (phases.step_ms - phases.transport_ms - phases.sense_and_infer_ms - phases.exchange_ms - phases.lifecycle_ms - phases.setup_ms - phases.validate_ms) / n
                );
            } else {
                eprintln!(
                    "             setup {:.2} + validate {:.2} | other {:.2}",
                    phases.setup_ms / n, phases.validate_ms / n,
                    (phases.step_ms - phases.transport_ms - phases.sense_and_infer_ms - phases.exchange_ms - phases.lifecycle_ms - phases.setup_ms - phases.validate_ms) / n
                );
            }
            if let Some((line, _)) = stage_line {
                eprintln!("             {line}");
            }
            phases = Phases::default();
            interval = (0, 0);
            interval_ticks = 0;
            interval_wall = Instant::now();
            profile_mark = world.stage_profile().totals();
            gpu_mark = gpu_rollup;
        }
        let wants_ckpt = ckpt_every.is_some_and(|k| world.tick % k == 0);
        let wants_frame = a.frame_every > 0 && world.tick % a.frame_every == 0;
        if wants_ckpt || wants_frame {
            #[cfg(feature = "gpu")]
            if let Some(g) = gpu.as_mut() {
                download_field(&mut world, g);
            }
        }
        if wants_ckpt {
            let t = Instant::now();
            write_checkpoint(&world, &ckpt_dir, retain);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if a.profile {
                world.stage_profile_mut().add_sample(CpuStage::Checkpoint, (ms * 1e6).round() as u64);
            }
            phases.checkpoint_ms += ms;
            run_phases.checkpoint_ms += ms;
        }
        if wants_frame {
            let t = Instant::now();
            save_frame(&world, &opts_base, &frames_dir, frame_no, steps_per_s);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if a.profile {
                world.stage_profile_mut().add_sample(CpuStage::Render, (ms * 1e6).round() as u64);
            }
            phases.frame_ms += ms;
            phases.frames += 1;
            run_phases.frame_ms += ms;
            run_phases.frames += 1;
            frame_no += 1;
        }
    }
    metrics.flush().ok();
    let wall = wall0.elapsed().as_secs_f64();
    #[cfg(feature = "gpu")]
    if let Some(g) = gpu.as_mut() {
        download_field(&mut world, g);
    }
    if ckpt_every.is_some() && !world.orgs.is_empty() {
        let t = Instant::now();
        write_checkpoint(&world, &ckpt_dir, retain);
        if a.profile {
            world.stage_profile_mut().add_sample(CpuStage::Checkpoint, t.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64);
        }
    }

    // Inspection of the oldest lineages' most derived descendants.
    let inspections: Vec<_> = most_derived(&world, a.inspect).into_iter().filter_map(|id| world.inspect(id)).collect();
    std::fs::write(a.out.join("inspection.json"), serde_json::to_string_pretty(&inspections).unwrap()).ok();

    let mut manifest = json!({
        "schema": "frankenlife.run-manifest.v0",
        "model_version": MODEL_VERSION,
        "stop_reason": stop_reason,
        "resumed_from": a.resume.as_ref().map(|p| p.display().to_string()),
        "start_tick": start_tick,
        "committed_tick": world.tick,
        "simulation_seconds": world.simulation_seconds(),
        "wall_seconds": wall,
        "workers": world.workers,
        "inference_backend": format!("{:?}", a.backend),
        "timing": run_phases.json(),
        "final_state_hash": format!("{:016x}", world.state_hash()),
        "population": world.orgs.len(),
        "totals": world.totals,
        "ledger": world.ledger,
        "validation": report,
        "interventions": a.erase.iter().map(|e| json!({"tick": e.0, "x": e.1, "y": e.2, "radius": e.3, "disposal": "recycle"})).collect::<Vec<_>>(),
        "frames": frame_no,
        "view": format!("{:?}", a.view),
        "config": cfg,
    });
    if a.profile {
        let doc = stage_document(&world, &a, &gpu_rollup, run_phases.step_ms, wall, ts_supported);
        let path = a.out.join("profile.json");
        std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap()).ok();
        eprintln!("{}", stage_summary(&world, &a, &gpu_rollup, run_phases.step_ms, wall, ts_supported));
        eprintln!("stage profile written to {}", path.display());
        manifest["stage_profile"] = doc;
    }
    std::fs::write(a.out.join("manifest.json"), serde_json::to_string_pretty(&manifest).unwrap()).ok();
    eprintln!(
        "{stop_reason}: tick {} ({:.1} sim s) in {:.1} wall s; population {}; births {}; deaths {}; frames {} -> {}",
        world.tick,
        world.simulation_seconds(),
        wall,
        world.orgs.len(),
        world.totals.births,
        world.totals.deaths_exhaustion + world.totals.deaths_user,
        frame_no,
        a.out.display()
    );
    if stop_reason == "completed" || stop_reason.starts_with("paused") {
        0
    } else {
        3
    }
}

pub(crate) fn step_world(world: &mut World, backend: Backend, #[cfg(feature = "gpu")] gpu: &mut Option<fl_gpu::GpuBackend>) -> Result<fl_core::world::TickReport, TickError> {
    #[cfg(feature = "gpu")]
    {
        return match gpu.as_mut() {
            Some(g) if backend == Backend::GpuFused => world.step_inference(fl_core::Inference::Fused(g)),
            Some(g) if backend == Backend::GpuField => world.step_inference(fl_core::Inference::Device(g)),
            Some(g) if matches!(backend, Backend::GpuResident | Backend::GpuRealtime) => world.step_inference(fl_core::Inference::Resident(g)),
            Some(g) => world.step_with(Some(g as &mut dyn fl_core::ControllerBackend)),
            None => world.step(),
        };
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = backend;
        world.step()
    }
}

fn save_frame(world: &World, base: &RenderOpts, dir: &Path, frame: usize, steps_per_s: f64) {
    let mut o = base.clone();
    o.telemetry = vec![
        format!("FRANKENLIFE  FIELD-ECOLOGY-V1  SEED {}", world.cfg.seed),
        format!("TICK {}  SIM {:.2} S  POP {}", world.tick, world.simulation_seconds(), world.orgs.len()),
        format!("GEN MAX {}  BIRTHS {}  DEATHS {}", world.orgs.generation.iter().max().copied().unwrap_or(0), world.totals.births, world.totals.deaths_exhaustion + world.totals.deaths_user),
        format!("ACHIEVED {:.1} STEPS/S (WALL)  REQUESTED NONE (HEADLESS)", steps_per_s),
    ];
    let rgb = render(world, &o);
    let path = dir.join(format!("frame_{frame:05}.png"));
    if let Err(e) = write_png(&path, &rgb, o.size, o.size) {
        eprintln!("frame {}: {e}", path.display());
    }
}

/// Wall-clock cost per phase, summed over one metrics interval (execution telemetry only).
#[derive(Default, Clone, Copy)]
struct Phases {
    ticks: u64,
    step_ms: f64,
    transport_ms: f64,
    sense_and_infer_ms: f64,
    backend_inference_ms: f64,
    exchange_ms: f64,
    lifecycle_ms: f64,
    setup_ms: f64,
    validate_ms: f64,
    frame_ms: f64,
    frames: u64,
    checkpoint_ms: f64,
    /// GPU backend breakdown (summed over ticks): host prep, round trip, kernel, bookkeeping, evaluate total, pyramid,
    /// field host, field round trip, field kernels, resident begin, uptake, deposit, finish, commit.
    gpu: [f64; 14],
}

impl Phases {
    fn add(&mut self, t: &fl_core::world::TickTiming) {
        self.ticks += 1;
        self.step_ms += t.total_ms;
        self.transport_ms += t.transport_ms;
        self.sense_and_infer_ms += t.sense_and_infer_ms;
        self.backend_inference_ms += t.backend_inference_ms;
        self.exchange_ms += t.exchange_ms;
        self.lifecycle_ms += t.lifecycle_ms;
        self.setup_ms += t.setup_ms;
        self.validate_ms += t.validate_publish_ms;
    }
    fn json(&self) -> serde_json::Value {
        let n = self.ticks.max(1) as f64;
        let step = self.step_ms / n;
        json!({
            "ticks": self.ticks,
            "mean_step_ms": step,
            "simulation_only_steps_per_second": if step > 0.0 { 1000.0 / step } else { 0.0 },
            "mean_ms": {
                "field_transport_and_sources": self.transport_ms / n,
                "sensing_and_inference": self.sense_and_infer_ms / n,
                "of_which_backend_inference": self.backend_inference_ms / n,
                "uptake_exchange_and_settlement": self.exchange_ms / n,
                "motion_emission_lifecycle_deposits": self.lifecycle_ms / n,
                "setup_clone_commands": self.setup_ms / n,
                "validate_and_publish": self.validate_ms / n,
                "other_in_step": (self.step_ms - self.transport_ms - self.sense_and_infer_ms - self.exchange_ms - self.lifecycle_ms - self.setup_ms - self.validate_ms) / n,
            },
            "frames_written": self.frames,
            "mean_frame_ms": if self.frames > 0 { self.frame_ms / self.frames as f64 } else { 0.0 },
            "checkpoint_ms": self.checkpoint_ms,
            "budget_ms_at_128_steps_per_second": 1000.0 / 128.0,
            "gpu_backend_mean_ms": {
                "host_prep": self.gpu[0] / n,
                "round_trip_including_kernel": self.gpu[1] / n,
                "kernel_gpu_timestamps": self.gpu[2] / n,
                "bookkeeping": self.gpu[3] / n,
                "evaluate_total": self.gpu[4] / n,
                "pyramid_gpu_timestamps": self.gpu[5] / n,
                "field_host": self.gpu[6] / n,
                "field_round_trip_including_kernels": self.gpu[7] / n,
                "field_kernels_gpu_timestamps": self.gpu[8] / n,
                "resident_begin": self.gpu[9] / n,
                "resident_uptake": self.gpu[10] / n,
                "resident_deposit": self.gpu[11] / n,
                "resident_finish": self.gpu[12] / n,
                "resident_commit": self.gpu[13] / n,
            },
        })
    }
}

fn device_call(backend: Backend) -> bool {
    !matches!(backend, Backend::Cpu)
}

fn profile_meta(world: &World, backend: Backend, step_wall_ms: f64, wall_seconds: f64, ts_supported: Option<bool>) -> ProfileMeta {
    ProfileMeta {
        backend: format!("{backend:?}"),
        workers: world.workers,
        simulation_seconds: world.simulation_seconds(),
        wall_seconds,
        step_wall_ms,
        population_final: world.orgs.len(),
        controller_evaluations: world.totals.controller_evaluations as u64,
        timestamp_queries_supported: ts_supported,
        external_evaluate_is_device_call: device_call(backend),
    }
}

fn stage_document(world: &World, a: &RunArgs, gpu: &GpuRollup, step_wall_ms: f64, wall_seconds: f64, ts_supported: Option<bool>) -> serde_json::Value {
    let meta = profile_meta(world, a.backend, step_wall_ms, wall_seconds, ts_supported);
    world.stage_profile().document(&meta, &gpu.counters())
}

fn stage_summary(world: &World, a: &RunArgs, gpu: &GpuRollup, step_wall_ms: f64, wall_seconds: f64, ts_supported: Option<bool>) -> String {
    let meta = profile_meta(world, a.backend, step_wall_ms, wall_seconds, ts_supported);
    world.stage_profile().summary(&meta, &gpu.counters())
}

fn interval_profile(world: &World, a: &RunArgs, profile_mark: &fl_core::profile::ProfileTotals, gpu_mark: &GpuRollup, gpu_rollup: &GpuRollup, step_ms: f64) -> Option<(String, serde_json::Value)> {
    if !a.profile {
        return None;
    }
    let delta = fl_core::StageProfiler::from_totals(true, world.stage_profile().totals().delta(*profile_mark));
    let ticks = delta.ticks().max(1) as f64;
    let buckets = delta.buckets_ns(device_call(a.backend));
    let ms = |ns: u64| ns as f64 / 1e6 / ticks;
    let unaccounted = step_ms / ticks - (delta.in_step_ns() as f64 / 1e6) / ticks;
    let value = json!({
        "time_unit": "milliseconds",
        "ticks": delta.ticks(),
        "sensing": ms(buckets.sensing),
        "chemistry_field": ms(buckets.chemistry_field),
        "networks": ms(buckets.networks),
        "transfers_sync": ms(buckets.transfers_sync),
        "other": ms(buckets.other),
        "unaccounted_step": unaccounted,
        "note": "Per-tick CPU wall for this metrics interval. transfers_sync is submit/wait and copies, not GPU kernel time. unaccounted_step is the gap inside the step wall.",
        "gpu_clocks_not_added_to_cpu_wall": gpu_rollup.delta(*gpu_mark).counters().iter().map(|c| json!({
            "id": c.id,
            "clock": c.clock,
            "total_ms": c.total_ms,
            "samples": c.samples,
        })).collect::<Vec<_>>(),
    });
    let line = format!(
        "profile cpu ms/tick: sensing {:.2} | chemistry/field {:.2} | networks {:.2} | transfers/sync {:.2} | other {:.2} | unaccounted {:.2}",
        ms(buckets.sensing),
        ms(buckets.chemistry_field),
        ms(buckets.networks),
        ms(buckets.transfers_sync),
        ms(buckets.other),
        unaccounted
    );
    Some((line, value))
}

#[cfg(feature = "gpu")]
fn download_field(world: &mut World, gpu: &mut fl_gpu::GpuBackend) {
    if !world.field_on_device() {
        return;
    }
    let stamp = world.stage_profile().start();
    if let Err(e) = world.sync_field(gpu) {
        eprintln!("cannot copy the field back from the GPU: {e}");
    }
    world.stage_profile_mut().stop(CpuStage::FieldDownload, stamp);
}

fn write_metrics(out: &mut impl Write, w: &World, interval: (u32, u32), steps_per_s: f64, phases: &Phases, stage_profile: Option<&serde_json::Value>) {
    let o = &w.orgs;
    let n = o.len().max(1) as f64;
    let mut topo: BTreeMap<String, usize> = BTreeMap::new();
    for g in &o.genome {
        *topo.entry(format!("{:?}", g.hidden())).or_default() += 1;
    }
    let lineages: BTreeSet<u64> = o.lineage.iter().copied().collect();
    let mean = |v: &dyn Fn(usize) -> f64| (0..o.len()).map(v).sum::<f64>() / n;
    let mut rec = json!({
        "tick": w.tick,
        "simulation_seconds": w.simulation_seconds(),
        "population": o.len(),
        "births_in_interval": interval.0,
        "deaths_in_interval": interval.1,
        "distinct_founder_lineages": lineages.len(),
        "mean_generation": mean(&|i| o.generation[i] as f64),
        "max_generation": o.generation.iter().max().copied().unwrap_or(0),
        "mean_energy": mean(&|i| o.energy[i]),
        "mean_parameters": mean(&|i| o.genome[i].param_count() as f64),
        "mean_sensing_scale": mean(&|i| o.genome[i].scale()),
        "mean_speed": mean(&|i| (o.vx[i] * o.vx[i] + o.vy[i] * o.vy[i]).sqrt()),
        "topologies": topo,
        "field_energy": w.last_report.map_or_else(|| w.field.total_energy(), |r| r.ledger.field_energy_after),
        "organism_energy": o.energy.iter().sum::<f64>(),
        "ledger": w.ledger,
        "achieved_steps_per_wall_second": steps_per_s,
        "timing": phases.json(),
    });
    if let Some(stage) = stage_profile {
        rec["stage_profile_ms_per_tick"] = stage.clone();
    }
    writeln!(out, "{rec}").ok();
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn profile_flag_defaults_off_and_parses_the_hardware_command() {
        let cli = Cli::try_parse_from(["frankenlife", "run", "--seconds", "6", "--frame-every", "0"]).unwrap();
        match cli.cmd {
            Cmd::Run(a) => {
                assert!(!a.profile);
                assert_eq!(a.backend, Backend::Cpu);
            }
            _ => panic!("expected run"),
        }
        let cli = Cli::try_parse_from([
            "frankenlife",
            "run",
            "--config",
            r"presets\target-20k-field-ecology.toml",
            "--backend",
            "gpu-resident",
            "--seconds",
            "6",
            "--frame-every",
            "0",
            "--profile",
            "--out",
            r"runs\t20r-profile",
        ])
        .unwrap();
        match cli.cmd {
            Cmd::Run(a) => {
                assert!(a.profile);
                assert_eq!(a.backend, Backend::GpuResident);
                assert_eq!(a.seconds, 6.0);
                assert_eq!(a.frame_every, 0);
                assert!(a.out.to_string_lossy().contains("t20r-profile"));
            }
            _ => panic!("expected run"),
        }
        let cli = Cli::try_parse_from([
            "frankenlife",
            "view",
            "--config",
            r"presets\target-5k-field-ecology.toml",
            "--backend",
            "gpu-realtime",
            "--seconds",
            "86400",
            "--out",
            r"runs\rt5-live",
        ])
        .unwrap();
        match cli.cmd {
            Cmd::View(a) => {
                assert_eq!(a.backend, Backend::GpuRealtime);
                assert_eq!(a.seconds, 86400.0);
                assert!(a.screenshot.is_none());
            }
            _ => panic!("expected view"),
        }
    }
}
