//! `frankenlife-m0` — the plan's M0 "compatible foundation" program (§22.1):
//! one native process that runs an Asupersync session, typed FrankenNumPy
//! array operations, a FrankenTorch independent-network batch and
//! FrankenManim frames, around the authoritative fl-core world.
//!
//! Ownership (plan §12.1): the Asupersync runtime owns every task. A region
//! scope holds three region-owned tasks — the **world owner** (the only writer
//! of scientific state), the **renderer** (an observer), and the **driver**
//! (issues control commands). Commands travel over a bounded `mpsc` control
//! channel; committed render epochs are published on a `watch` channel that
//! keeps only the latest value, so a slow renderer skips epochs and never
//! back-pressures the simulation (plan §12.4, C.14: observation cannot change
//! which births are admitted). The owner checkpoints cancellation between
//! ticks; a tick is itself transactional, so cancellation never leaves a
//! half-applied tick. When the scope closes, all three tasks are complete.

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use asupersync::prelude::*;
use fl_core::nn::OUTPUT_WIDTH;
use fl_core::{Config, ControllerBackend, InferenceJob, ReferenceBackend, World};
use fl_manim::RenderEpoch;
use fl_torch::TorchBackend;
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendChoice {
    Reference,
    Torch,
    /// Run both; the reference's logits drive the world, FrankenTorch's are compared.
    Compare,
}

struct Args {
    config: Option<PathBuf>,
    ticks: u64,
    frame_every: u64,
    pixels: u32,
    out: PathBuf,
    population: Option<u32>,
    backend: BackendChoice,
    verify_control: bool,
}

fn parse() -> Result<Args, String> {
    let mut a = Args { config: None, ticks: 256, frame_every: 64, pixels: 720, out: "runs/m0".into(), population: None, backend: BackendChoice::Compare, verify_control: true };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or(format!("{k} needs a value"));
        match k.as_str() {
            "--config" => a.config = Some(v()?.into()),
            "--ticks" => a.ticks = v()?.parse().map_err(|e| format!("{e}"))?,
            "--frame-every" => a.frame_every = v()?.parse().map_err(|e| format!("{e}"))?,
            "--pixels" => a.pixels = v()?.parse().map_err(|e| format!("{e}"))?,
            "--out" => a.out = v()?.into(),
            "--population" => a.population = Some(v()?.parse().map_err(|e| format!("{e}"))?),
            "--backend" => {
                a.backend = match v()?.as_str() {
                    "reference" => BackendChoice::Reference,
                    "torch" => BackendChoice::Torch,
                    "compare" => BackendChoice::Compare,
                    o => return Err(format!("unknown backend {o:?} (reference|torch|compare)")),
                }
            }
            "--no-verify-control" => a.verify_control = false,
            "-h" | "--help" => {
                println!("frankenlife-m0 [--config FILE] [--ticks N] [--frame-every K] [--pixels P] [--out DIR] [--population N] [--backend reference|torch|compare] [--no-verify-control]");
                std::process::exit(0);
            }
            o => return Err(format!("unknown argument {o:?}")),
        }
    }
    Ok(a)
}

/// Runs the scalar reference and FrankenTorch on the same jobs and tracks the discrepancy.
struct CompareBackend {
    torch: TorchBackend,
    max_abs_diff: f32,
    scratch: Vec<[f32; OUTPUT_WIDTH]>,
}

impl ControllerBackend for CompareBackend {
    fn name(&self) -> &str {
        "compare(reference drives, frankentorch checked)"
    }
    fn evaluate(&mut self, jobs: &[InferenceJob<'_>], out: &mut [[f32; OUTPUT_WIDTH]]) -> Result<(), String> {
        ReferenceBackend.evaluate(jobs, out)?;
        self.scratch.resize(jobs.len(), [0.0; OUTPUT_WIDTH]);
        self.torch.evaluate(jobs, &mut self.scratch)?;
        for (a, b) in out.iter().zip(&self.scratch) {
            for (p, q) in a.iter().zip(b) {
                self.max_abs_diff = self.max_abs_diff.max((p - q).abs());
            }
        }
        Ok(())
    }
}

enum Control {
    Advance(u64),
    Shutdown,
}

#[derive(Debug, Default)]
struct OwnerSummary {
    ticks: u64,
    epochs_published: u64,
    final_hash: u64,
    population: usize,
    births: u32,
    deaths: u32,
    ledger_residual: f64,
    step_wall_s: f64,
    torch_max_abs_diff: Option<f32>,
    torch_stats: Option<String>,
    torch_resident_bytes: u64,
    npz_bytes: usize,
    stop: String,
}

#[derive(Debug, Default)]
struct RendererSummary {
    frames: u64,
    skipped_epochs: u64,
    render_wall_s: f64,
    last_report: String,
}

enum Done {
    Owner(OwnerSummary),
    Renderer(RendererSummary),
    Driver,
}

fn main() {
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let mut cfg = match &args.config {
        Some(p) => std::fs::read_to_string(p).map_err(|e| e.to_string()).and_then(|t| Config::from_toml(&t)),
        None => Ok(Config::reference()),
    }
    .unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    });
    if let Some(n) = args.population {
        cfg.population.initial_count = n;
        cfg.population.maximum_count = cfg.population.maximum_count.max(4 * n);
    }
    cfg.population.capacity_policy = "reject-births-stable-order".into();
    std::fs::create_dir_all(args.out.join("frames")).expect("output directory");

    let runtime = RuntimeBuilder::new().worker_threads(2).build().expect("asupersync runtime");
    let wall = Instant::now();
    let (owner, renderer) = runtime.block_on(async {
        let cx = Cx::current().expect("block_on installs a capability context");
        session(&cx, &args, cfg.clone()).await
    });
    let wall_s = wall.elapsed().as_secs_f64();

    // Control: the same world advanced directly, with no runtime, no observers, reference inference.
    let control = if args.verify_control && args.backend != BackendChoice::Torch {
        let mut w = World::new(cfg.clone()).expect("valid config");
        for _ in 0..owner.ticks {
            w.step().expect("control tick");
        }
        Some(w.state_hash())
    } else {
        None
    };

    let report = json!({
        "program": "frankenlife-m0",
        "model_version": fl_core::MODEL_VERSION,
        "stack": {
            "asupersync": "460306de2aa4169510873716018ade44fdb01534",
            "frankentorch": "287e14d335dd79d4e0e704c0e9d4231e1d64fcc5",
            "franken_numpy": "20159e89c4ba2d6822b395eeff538f8c47ed8b6f",
            "franken_manim": "a12661bc59cef63526538afb7bd885bd5bf5ca04",
        },
        "capabilities": {
            "asupersync": "RuntimeBuilder (2 workers); one region scope owning world-owner, renderer and driver tasks; bounded mpsc control channel; watch channel of committed render epochs (latest only); cx.checkpoint() between ticks",
            "frankentorch": TorchBackend::capability(),
            "franken_numpy": "NdLayout descriptors for the field checked against fl-core buffers; ArrayStorage::F32 planes; fnp-io .npz export of the final epoch",
            "franken_manim": "fmn render() native PNG per published epoch: one ImageMobject (field) + up to 12 DotClouds (color buckets)",
            "inference_backend": format!("{:?}", args.backend),
            "known_gaps": [
                "world.step() runs on the owner task rather than Asupersync's blocking pool",
                "FrankenTorch CPU bmm uses its own Rayon pool (executor boundary, plan §3.4)",
                "DotCloud has one color per cloud: organism colors are quantized to 12 buckets",
                "no Metal path on this Linux host; no durable command journal"
            ],
        },
        "result": {
            "stop": owner.stop,
            "ticks": owner.ticks,
            "population": owner.population,
            "births": owner.births,
            "deaths": owner.deaths,
            "ledger_cumulative_residual": owner.ledger_residual,
            "final_state_hash": format!("{:016x}", owner.final_hash),
            "control_state_hash": control.map(|h| format!("{h:016x}")),
            "runtime_and_observers_are_view_neutral": control.map(|h| h == owner.final_hash),
            "torch_max_abs_logit_diff_vs_reference": owner.torch_max_abs_diff,
            "torch_stats": owner.torch_stats,
            "torch_resident_weight_bytes": owner.torch_resident_bytes,
            "epochs_published": owner.epochs_published,
            "frames_rendered": renderer.frames,
            "epochs_skipped_by_renderer": renderer.skipped_epochs,
            "last_render": renderer.last_report,
            "npz_bytes": owner.npz_bytes,
            "wall_seconds": wall_s,
            "owner_step_wall_seconds": owner.step_wall_s,
            "renderer_wall_seconds": renderer.render_wall_s,
            "achieved_ticks_per_wall_second": owner.ticks as f64 / owner.step_wall_s.max(1e-9),
        }
    });
    let text = serde_json::to_string_pretty(&report).unwrap();
    std::fs::write(args.out.join("m0_report.json"), &text).ok();
    println!("{text}");
    let ok = owner.stop == "completed" && renderer.frames > 0 && control.is_none_or(|h| h == owner.final_hash);
    std::process::exit(if ok { 0 } else { 1 });
}

async fn session(cx: &Cx, args: &Args, cfg: Config) -> (OwnerSummary, RendererSummary) {
    let control_capacity = cfg.execution.control_queue_capacity as usize;
    let (ctl_tx, mut ctl_rx) = mpsc::channel::<Control>(control_capacity);
    let (epoch_tx, mut epoch_rx) = watch::channel::<Option<Arc<RenderEpoch>>>(None);
    let scope = cx.scope_with_budget(Budget::new());
    let mut tasks: JoinSet<'_, Done, String, _> = JoinSet::new(&scope);

    // ── world owner: the only writer of scientific state ──
    let (ticks, frame_every, backend_choice, out) = (args.ticks, args.frame_every.max(1), args.backend, args.out.clone());
    tasks
        .spawn(cx, move |cx| async move {
            let mut world = World::new(cfg).map_err(|e| e.join("; "))?;
            world.workers = 1; // Asupersync owns scheduling; no extra fl-core threads here.
            let mut torch = TorchBackend::new();
            let mut compare = CompareBackend { torch: TorchBackend::new(), max_abs_diff: 0.0, scratch: Vec::new() };
            let mut s = OwnerSummary { stop: "completed".into(), ..Default::default() };
            let layout = fl_fnp::field_layout(&world.field)?;
            let publish = |w: &World, s: &mut OwnerSummary| {
                let _ = epoch_tx.send(Some(Arc::new(RenderEpoch::from_world(w))));
                s.epochs_published += 1;
            };
            publish(&world, &mut s);
            'outer: loop {
                match ctl_rx.recv(&cx).await {
                    Ok(Control::Advance(n)) => {
                        for _ in 0..n {
                            if cx.checkpoint().is_err() {
                                s.stop = "cancelled between ticks (last committed epoch kept)".into();
                                break 'outer;
                            }
                            let t = Instant::now();
                            let r = match backend_choice {
                                BackendChoice::Reference => world.step(),
                                BackendChoice::Torch => world.step_with(Some(&mut torch)),
                                BackendChoice::Compare => world.step_with(Some(&mut compare)),
                            };
                            s.step_wall_s += t.elapsed().as_secs_f64();
                            match r {
                                Ok(rep) => {
                                    s.ticks += 1;
                                    s.births += rep.counts.births;
                                    s.deaths += rep.counts.deaths_exhaustion + rep.counts.deaths_user;
                                }
                                Err(e) => {
                                    s.stop = format!("stopped: {e}");
                                    break 'outer;
                                }
                            }
                            fl_fnp::check_field(&world.field, &layout)?;
                            if world.tick % frame_every == 0 {
                                publish(&world, &mut s);
                            }
                            if world.orgs.is_empty() {
                                s.stop = "extinction".into();
                                break 'outer;
                            }
                            asupersync::runtime::yield_now().await;
                        }
                    }
                    Ok(Control::Shutdown) | Err(_) => break,
                }
            }
            s.final_hash = world.state_hash();
            s.population = world.orgs.len();
            s.ledger_residual = world.ledger.residual;
            let npz = fl_fnp::export_npz(&world)?;
            s.npz_bytes = npz.len();
            std::fs::write(out.join("final_epoch.npz"), npz).map_err(|e| e.to_string())?;
            match backend_choice {
                BackendChoice::Torch => {
                    s.torch_stats = Some(format!("{:?}", torch.stats));
                    s.torch_resident_bytes = torch.resident_bytes();
                }
                BackendChoice::Compare => {
                    s.torch_max_abs_diff = Some(compare.max_abs_diff);
                    s.torch_stats = Some(format!("{:?}", compare.torch.stats));
                    s.torch_resident_bytes = compare.torch.resident_bytes();
                }
                BackendChoice::Reference => {}
            }
            drop(epoch_tx); // closes the epoch stream: the renderer finishes its last frame and exits
            Ok(Done::Owner(s))
        })
        .map_err(|e| format!("spawn owner: {e:?}"))
        .expect("spawn owner");

    // ── renderer: an observer of committed epochs ──
    let (pixels, frames_dir) = (args.pixels, args.out.join("frames"));
    tasks
        .spawn(cx, move |cx| async move {
            let mut r = RendererSummary::default();
            let mut last_tick: Option<u64> = None;
            loop {
                let epoch = epoch_rx.borrow_and_update().clone();
                if let Some(e) = epoch {
                    if last_tick != Some(e.tick) {
                        if let Some(lt) = last_tick {
                            // Epochs the owner published that this observer never saw.
                            let expected = (e.tick - lt) / frame_every;
                            r.skipped_epochs += expected.saturating_sub(1);
                        }
                        last_tick = Some(e.tick);
                        let t = Instant::now();
                        let path = frames_dir.join(format!("tick_{:08}.png", e.tick));
                        let rep = fl_manim::render_png(&e, &path, pixels)?;
                        r.render_wall_s += t.elapsed().as_secs_f64();
                        r.frames += 1;
                        r.last_report = format!("{} ({} bytes)", rep.artifact.path.display(), rep.artifact.bytes);
                    }
                }
                if epoch_rx.changed(&cx).await.is_err() {
                    // Sender dropped: render the final epoch if it is new, then stop.
                    let epoch = epoch_rx.borrow_and_update().clone();
                    if let Some(e) = epoch {
                        if last_tick != Some(e.tick) {
                            let path = frames_dir.join(format!("tick_{:08}.png", e.tick));
                            fl_manim::render_png(&e, &path, pixels)?;
                            r.frames += 1;
                        }
                    }
                    break;
                }
            }
            Ok(Done::Renderer(r))
        })
        .map_err(|e| format!("spawn renderer: {e:?}"))
        .expect("spawn renderer");

    // ── driver: issues control commands in bounded batches ──
    tasks
        .spawn(cx, move |cx| async move {
            let mut left = ticks;
            while left > 0 {
                let n = left.min(32);
                ctl_tx.send(&cx, Control::Advance(n)).await.map_err(|_| "control channel closed".to_string())?;
                left -= n;
            }
            ctl_tx.send(&cx, Control::Shutdown).await.map_err(|_| "control channel closed".to_string())?;
            Ok(Done::Driver)
        })
        .map_err(|e| format!("spawn driver: {e:?}"))
        .expect("spawn driver");

    let mut owner = OwnerSummary { stop: "owner did not report".into(), ..Default::default() };
    let mut renderer = RendererSummary::default();
    for outcome in tasks.join_all(cx).await {
        match outcome {
            Outcome::Ok(Done::Owner(s)) => owner = s,
            Outcome::Ok(Done::Renderer(r)) => renderer = r,
            Outcome::Ok(Done::Driver) => {}
            Outcome::Err(e) => eprintln!("task error: {e}"),
            Outcome::Cancelled(c) => eprintln!("task cancelled: {c:?}"),
            Outcome::Panicked(p) => eprintln!("task panicked: {p:?}"),
        }
    }
    (owner, renderer)
}
