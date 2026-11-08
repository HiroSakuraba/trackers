//! `frankenlife` — run, render and inspect FieldEcologyV1 worlds headlessly.

#![forbid(unsafe_code)]

mod render;

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, Subcommand};
use fl_core::world::{Command, Disposal};
use fl_core::{checkpoint, Config, TickError, World, MODEL_VERSION};
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
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Backend {
    /// fl-core's scalar reference evaluator (multithreaded, bitwise deterministic).
    Cpu,
    /// wgpu compute on the GPU (tolerance-equivalent to the reference).
    Gpu,
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
        Cmd::Inspect { checkpoint, id, top } => inspect_checkpoint(&checkpoint, &id, top),
        Cmd::GpuCheck { organisms, hidden, iters } => gpu_check(organisms, &hidden, iters),
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
fn gpu_check(organisms: usize, hidden: &str, iters: usize) -> i32 {
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
    let t = Instant::now();
    for _ in 0..iters {
        let _ = gpu.evaluate(&jobs, &mut a);
    }
    let gpu_s = t.elapsed().as_secs_f64() / iters.max(1) as f64;
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
    println!("CPU batch    : {:.2} ms (scalar reference, one thread)  ->  GPU is {:.1}x faster", cpu_s * 1e3, cpu_s / gpu_s);
    if diff < 1e-4 { 0 } else { 1 }
}

#[cfg(not(feature = "gpu"))]
fn gpu_check(_: usize, _: &str, _: usize) -> i32 {
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

fn run(a: RunArgs) -> i32 {
    let mut world = if let Some(path) = &a.resume {
        match std::fs::read(path).map_err(|e| e.to_string()).and_then(|b| checkpoint::load(&b)) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                return 2;
            }
        }
    } else {
        let mut cfg = match load(&a.config) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("{e}");
                return 2;
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
                return 1;
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
    #[cfg(feature = "gpu")]
    let mut gpu = match a.backend {
        Backend::Gpu => match fl_gpu::GpuBackend::new() {
            Ok(g) => {
                eprintln!("GPU: {}", g.capability());
                Some(g)
            }
            Err(e) => {
                eprintln!("cannot use the GPU backend: {e}");
                return 2;
            }
        },
        Backend::Cpu => None,
    };
    #[cfg(not(feature = "gpu"))]
    if a.backend == Backend::Gpu {
        eprintln!("this build has no GPU support (rebuild with the default `gpu` feature)");
        return 2;
    }
    for &(t, x, y, r) in &a.erase {
        if let Err(e) = world.schedule(t, Command::EraseDisk { x, y, radius: r, disposal: Disposal::Recycle }) {
            eprintln!("cannot schedule erase: {e}");
            return 2;
        }
    }
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
    if a.frame_every > 0 {
        save_frame(&world, &opts_base, &frames_dir, frame_no, steps_per_s);
        frame_no += 1;
    }
    for _ in 0..ticks {
        #[cfg(feature = "gpu")]
        let stepped = match gpu.as_mut() {
            Some(g) => world.step_with(Some(g as &mut dyn fl_core::ControllerBackend)),
            None => world.step(),
        };
        #[cfg(not(feature = "gpu"))]
        let stepped = world.step();
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
        interval.0 += r.counts.births;
        interval.1 += r.counts.deaths_exhaustion + r.counts.deaths_user;
        interval_ticks += 1;
        if r.population == 0 {
            stop_reason = format!("extinction at tick {}", r.tick);
            eprintln!("{stop_reason}");
            write_metrics(&mut metrics, &world, interval, steps_per_s);
            break;
        }
        if world.tick % metrics_every == 0 {
            let el = interval_wall.elapsed().as_secs_f64();
            steps_per_s = interval_ticks as f64 / el.max(1e-9);
            write_metrics(&mut metrics, &world, interval, steps_per_s);
            let l = &world.ledger;
            eprintln!(
                "tick {:>6}  t={:>7.2}s  pop {:>6}  births {:>5}  deaths {:>5}  E_field {:>9.2}  E_org {:>9.2}  resid {:>9.2e}  {:>6.1} steps/s",
                world.tick, world.simulation_seconds(), world.orgs.len(), interval.0, interval.1, r.ledger.field_energy_after, r.ledger.organism_energy_after, l.residual, steps_per_s
            );
            interval = (0, 0);
            interval_ticks = 0;
            interval_wall = Instant::now();
        }
        if ckpt_every.is_some_and(|k| world.tick % k == 0) {
            write_checkpoint(&world, &ckpt_dir, retain);
        }
        if a.frame_every > 0 && world.tick % a.frame_every == 0 {
            save_frame(&world, &opts_base, &frames_dir, frame_no, steps_per_s);
            frame_no += 1;
        }
    }
    metrics.flush().ok();
    let wall = wall0.elapsed().as_secs_f64();
    if ckpt_every.is_some() && !world.orgs.is_empty() {
        write_checkpoint(&world, &ckpt_dir, retain);
    }

    // Inspection of the oldest lineages' most derived descendants.
    let inspections: Vec<_> = most_derived(&world, a.inspect).into_iter().filter_map(|id| world.inspect(id)).collect();
    std::fs::write(a.out.join("inspection.json"), serde_json::to_string_pretty(&inspections).unwrap()).ok();

    let manifest = json!({
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
    if let Err(e) = write_png(&path, &rgb, o.size) {
        eprintln!("frame {}: {e}", path.display());
    }
}

fn write_metrics(out: &mut impl Write, w: &World, interval: (u32, u32), steps_per_s: f64) {
    let o = &w.orgs;
    let n = o.len().max(1) as f64;
    let mut topo: BTreeMap<String, usize> = BTreeMap::new();
    for g in &o.genome {
        *topo.entry(format!("{:?}", g.hidden())).or_default() += 1;
    }
    let lineages: BTreeSet<u64> = o.lineage.iter().copied().collect();
    let mean = |v: &dyn Fn(usize) -> f64| (0..o.len()).map(v).sum::<f64>() / n;
    let rec = json!({
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
        "field_energy": w.field.total_energy(),
        "organism_energy": o.energy.iter().sum::<f64>(),
        "ledger": w.ledger,
        "achieved_steps_per_wall_second": steps_per_s,
    });
    writeln!(out, "{rec}").ok();
}
