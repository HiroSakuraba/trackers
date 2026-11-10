//! Desktop viewer. The tick is `step_world`, the same function `run` uses.
//! Drawing only reads committed state. Pause, step and inspect are the only
//! ways a watched run diverges from a headless one.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use fl_view::{chem_basis, chem_color, lineage_color};

use super::painter::{self, PaintStats, Painter, Planes, Scene};
use super::render::View;
use super::{download_field, prepare, step_world, write_checkpoint, Backend, Launch, Prepared};
use fl_core::world::{Command, Disposal, Param, TickError};
use winit::keyboard::{Key, NamedKey};
use fl_core::{Config, World, MODEL_VERSION};

#[derive(clap::Parser, Debug)]
pub struct ViewArgs {
    /// TOML configuration (defaults to the embedded reference preset).
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub seed: Option<u64>,
    /// Simulated seconds. 0 runs until the window is closed.
    #[arg(long, default_value_t = 0.0)]
    pub seconds: f64,
    #[arg(long, default_value = "runs/view")]
    pub out: PathBuf,
    #[arg(long, value_enum, default_value_t = View::Composite)]
    pub view: View,
    #[arg(long)]
    pub workers: Option<usize>,
    #[arg(long)]
    pub exhibition: bool,
    #[arg(long, value_parser = super::parse_erase)]
    pub erase: Vec<(u64, f64, f64, f64)>,
    #[arg(long)]
    pub resume: Option<PathBuf>,
    #[arg(long)]
    pub checkpoint_every: Option<f64>,
    #[arg(long, default_value_t = 5)]
    pub inspect: usize,
    #[arg(long, value_enum, default_value_t = Backend::Cpu)]
    pub backend: Backend,
    /// Window width. Defaults to the preset's presentation width.
    #[arg(long)]
    pub width: Option<u32>,
    /// Window height. Defaults to the preset's presentation height.
    #[arg(long)]
    pub height: Option<u32>,
    /// Draw the viewer into a PNG and exit, without opening a window.
    /// With `--seconds 0` this is the world at tick 0.
    #[arg(long)]
    pub screenshot: Option<PathBuf>,
    /// Press keys at a tick, for scripted runs and screenshots: `TICK:KEYS`, e.g.
    /// `--press 0:www --press 600:@2k`. Any control key works; `@N` selects source N
    /// and `@` alone clears the selection. Repeatable.
    #[arg(long = "press", value_parser = parse_press)]
    pub press: Vec<(u64, String)>,
}

fn parse_press(s: &str) -> Result<(u64, String), String> {
    let (tick, keys) = s.split_once(':').ok_or("expected TICK:KEYS")?;
    Ok((tick.trim().parse().map_err(|e| format!("bad tick `{tick}`: {e}"))?, keys.to_string()))
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Speed {
    Unlimited,
    Realtime(f64),
}

#[derive(Clone, Copy)]
struct Camera {
    center_x: f64,
    center_y: f64,
    /// 1 shows the whole world (contain). Larger zooms in.
    zoom: f64,
}

impl Camera {
    fn fit(world_w: f64, world_h: f64) -> Self {
        Camera { center_x: world_w * 0.5, center_y: world_h * 0.5, zoom: 1.0 }
    }

    fn px_per_unit(self, sw: f64, sh: f64, ww: f64, wh: f64) -> f64 {
        (sw / ww).min(sh / wh) * self.zoom
    }

    fn screen_to_world(self, px: f64, py: f64, sw: f64, sh: f64, ww: f64, wh: f64) -> (f64, f64) {
        let z = self.px_per_unit(sw, sh, ww, wh).max(1e-9);
        let x = self.center_x + (px - sw * 0.5) / z;
        let y = self.center_y + (py - sh * 0.5) / z;
        (x.rem_euclid(ww), y.rem_euclid(wh))
    }

    /// Inclusive tile offsets whose copies intersect the view, plus a margin for splats.
    fn tile_span(self, sw: f64, sh: f64, ww: f64, wh: f64, margin: f64) -> (i32, i32, i32, i32) {
        let z = self.px_per_unit(sw, sh, ww, wh).max(1e-9);
        let half_w = sw * 0.5 / z + margin;
        let half_h = sh * 0.5 / z + margin;
        let x0 = ((self.center_x - half_w) / ww).floor() as i32;
        let x1 = ((self.center_x + half_w) / ww).floor() as i32;
        let y0 = ((self.center_y - half_h) / wh).floor() as i32;
        let y1 = ((self.center_y + half_h) / wh).floor() as i32;
        (x0, x1, y0, y1)
    }

    fn zoom_at(mut self, factor: f64, px: f64, py: f64, sw: f64, sh: f64, ww: f64, wh: f64) -> Self {
        let before = self.screen_to_world(px, py, sw, sh, ww, wh);
        self.zoom = (self.zoom * factor).clamp(1.0 / 3.0, 64.0);
        let after = self.screen_to_world(px, py, sw, sh, ww, wh);
        self.center_x = (self.center_x + before.0 - after.0).rem_euclid(ww);
        self.center_y = (self.center_y + before.1 - after.1).rem_euclid(wh);
        self
    }
}

struct Pace {
    speed: Speed,
    paused: bool,
    accum: f64,
    last: Instant,
    steps_per_s: f64,
    window_start: Instant,
    window_ticks: u64,
    sim_ms: f64,
    present_ms: f64,
    frames: u64,
}

impl Pace {
    fn new() -> Self {
        let now = Instant::now();
        Pace { speed: Speed::Unlimited, paused: false, accum: 0.0, last: now, steps_per_s: 0.0, window_start: now, window_ticks: 0, sim_ms: 0.0, present_ms: 0.0, frames: 0 }
    }

    fn note_step(&mut self, sim_ms: f64) {
        self.sim_ms += sim_ms;
        self.window_ticks += 1;
        let el = self.window_start.elapsed().as_secs_f64();
        if el >= 0.25 {
            self.steps_per_s = self.window_ticks as f64 / el;
            self.window_start = Instant::now();
            self.window_ticks = 0;
        }
    }

    /// How many ticks to run before yielding. Real time is one simulated second per wall second.
    fn take(&mut self, tick_hz: f64, budget: Duration, unlimited_slice: bool) -> u64 {
        let now = Instant::now();
        let dt = now.saturating_duration_since(self.last).as_secs_f64();
        self.last = now;
        if self.paused {
            return 0;
        }
        match self.speed {
            Speed::Unlimited => {
                if unlimited_slice {
                    u64::MAX
                } else {
                    // The caller stops on the wall budget. A large cap keeps the window responsive.
                    let _ = budget;
                    1_000_000
                }
            }
            Speed::Realtime(m) => {
                self.accum += dt * m.max(0.0);
                let tick = 1.0 / tick_hz;
                // A hitch longer than half a second is dropped, so the window does not fast-forward.
                if self.accum > 0.5 {
                    self.accum = 0.0;
                    return (tick_hz * m * 0.05).ceil().max(1.0) as u64;
                }
                (self.accum / tick).floor() as u64
            }
        }
    }
}

/// How much one key press scales a value, and a mutation press scales the rates.
const TUNE_STEP: f64 = 1.25;
const MUTATION_STEP: f64 = 1.5;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Group {
    Energy,
    Speed,
    Diffusion,
    Decay,
    Mutation,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Control {
    Tune(Group, f64),
    Drift(f64),
    Reset,
}

/// Keys that change the running world. Lower and upper neighbours on the keyboard are
/// the down and up pair for each quantity.
fn control_for(key: &str) -> Option<Control> {
    Some(match key.to_ascii_lowercase().as_str() {
        "q" => Control::Tune(Group::Energy, 1.0 / TUNE_STEP),
        "w" => Control::Tune(Group::Energy, TUNE_STEP),
        "a" => Control::Tune(Group::Speed, 1.0 / TUNE_STEP),
        "s" => Control::Tune(Group::Speed, TUNE_STEP),
        "d" => Control::Tune(Group::Diffusion, 1.0 / TUNE_STEP),
        "f" => Control::Tune(Group::Diffusion, TUNE_STEP),
        "c" => Control::Tune(Group::Decay, 1.0 / TUNE_STEP),
        "v" => Control::Tune(Group::Decay, TUNE_STEP),
        "z" => Control::Tune(Group::Mutation, 1.0 / MUTATION_STEP),
        "x" => Control::Tune(Group::Mutation, MUTATION_STEP),
        "j" => Control::Drift(5.0),
        "k" => Control::Drift(25.0),
        "r" => Control::Reset,
        _ => return None,
    })
}

const MUTATION_PARAMS: [Param; 5] =
    [Param::MutationWeightProbability, Param::MutationWeightSd, Param::MutationReplacementProbability, Param::MutationChemistrySd, Param::MutationSensingSd];

/// What the viewer has asked of the world so far. The world holds the truth; this is
/// the request side, shown on the HUD and used to compute the next step.
struct Tuner {
    /// Values at launch, in `Param::ALL` order.
    base: [f64; 9],
    /// Latest requested values, same order.
    want: [f64; 9],
    mutation_scale: f64,
    /// The source a drift key acts on; `None` is every source.
    source: Option<u32>,
    drift_degrees: f64,
    log: Vec<serde_json::Value>,
    log_file: Option<PathBuf>,
}

impl Tuner {
    fn new(cfg: &Config, log_file: Option<PathBuf>) -> Self {
        let base: [f64; 9] = std::array::from_fn(|i| Param::ALL[i].get(cfg));
        Tuner { base, want: base, mutation_scale: 1.0, source: None, drift_degrees: 0.0, log: Vec::new(), log_file }
    }

    fn record(&mut self, tick: u64, entry: serde_json::Value) {
        let line = serde_json::json!({ "tick": tick, "change": entry });
        eprintln!("live change at tick {tick}: {entry}");
        if let Some(path) = &self.log_file {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let _ = writeln!(f, "{line}");
            }
        }
        self.log.push(line);
    }

    fn set(&mut self, world: &mut World, param: Param, value: f64) -> bool {
        match world.schedule(world.tick, Command::SetParam { param, value }) {
            Ok(_) => {
                self.want[param.tag() as usize] = value;
                self.record(world.tick, serde_json::json!({ "param": param.label(), "value": value }));
                true
            }
            Err(e) => {
                eprintln!("{}: {e}", param.label());
                false
            }
        }
    }

    fn scale(&mut self, world: &mut World, param: Param, factor: f64, floor: f64) {
        let i = param.tag() as usize;
        let (cur, base) = (self.want[i], self.base[i]);
        let lo = if base > 0.0 { base / 256.0 } else { 0.0 };
        let hi = if base > 0.0 { base * 256.0 } else { floor * 1000.0 };
        let next = if cur <= 0.0 { if factor > 1.0 { floor } else { 0.0 } } else { cur * factor };
        self.set(world, param, next.clamp(lo, hi));
    }

    /// Apply one control key to `world`. Returns whether the key was a control.
    fn press(&mut self, world: &mut World, key: &str) -> bool {
        let Some(control) = control_for(key) else { return false };
        match control {
            Control::Tune(Group::Energy, f) => self.scale(world, Param::SourcePower, f, 1.0),
            Control::Tune(Group::Speed, f) => self.scale(world, Param::SourceSpeed, f, 0.25),
            Control::Tune(Group::Diffusion, f) => self.scale(world, Param::Diffusion, f, 0.01),
            Control::Tune(Group::Decay, f) => self.scale(world, Param::Decay, f, 0.005),
            Control::Tune(Group::Mutation, f) => {
                let s = (self.mutation_scale * f).clamp(1.0 / 64.0, 64.0);
                for p in MUTATION_PARAMS {
                    let mut v = self.base[p.tag() as usize] * s;
                    if matches!(p, Param::MutationWeightProbability | Param::MutationReplacementProbability) {
                        v = v.min(1.0);
                    }
                    self.set(world, p, v);
                }
                self.mutation_scale = s;
            }
            Control::Drift(degrees) => {
                let source = self.source;
                if world.schedule(world.tick, Command::DriftChemistry { source, degrees }).is_ok() {
                    self.drift_degrees += degrees;
                    self.record(world.tick, serde_json::json!({ "chemistry_drift_degrees": degrees, "source": source }));
                }
            }
            Control::Reset => {
                for p in [Param::SourcePower, Param::SourceSpeed, Param::Diffusion, Param::Decay] {
                    let b = self.base[p.tag() as usize];
                    self.set(world, p, b);
                }
                for p in MUTATION_PARAMS {
                    let b = self.base[p.tag() as usize];
                    self.set(world, p, b);
                }
                self.mutation_scale = 1.0;
            }
        }
        true
    }

    /// `@N` selects source N, `@` clears; anything else is a control key.
    fn press_script(&mut self, world: &mut World, keys: &str) {
        let mut chars = keys.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '@' {
                let mut n = String::new();
                while let Some(d) = chars.peek().filter(|d| d.is_ascii_digit()) {
                    n.push(*d);
                    chars.next();
                }
                self.source = n.parse::<u32>().ok().filter(|&k| k < world.cfg.sources.count);
            } else {
                self.press(world, &c.to_string());
            }
        }
    }

    fn hud_line(&self) -> String {
        let g = |p: Param| self.want[p.tag() as usize];
        let chem = match self.source {
            Some(k) => format!("SRC {k}"),
            None => "ALL".into(),
        };
        format!(
            "ENERGY {:.1}/S  SPEED {:.2}  DIFF {:.3}  DECAY {:.3}  MUT X{:.2}  CHEM {chem} DRIFT {:.0} DEG",
            g(Param::SourcePower),
            g(Param::SourceSpeed),
            g(Param::Diffusion),
            g(Param::Decay),
            self.mutation_scale,
            self.drift_degrees
        )
    }
}

/// Where each source is drawn: its center for the tick the next step will take.
fn source_centers(world: &World) -> Vec<[f64; 2]> {
    world
        .sources
        .iter()
        .map(|s| {
            let mut s = s.clone();
            s.advance_to(world.tick, &world.cfg, &world.rng);
            s.center_mid(world.tick, &world.cfg)
        })
        .collect()
}

/// The source whose center lies within `reach` world units of `(wx, wy)`, nearest first.
fn source_at(world: &World, wx: f64, wy: f64, reach: f64) -> Option<u32> {
    let (ww, wh) = (world.field.width(), world.field.height());
    source_centers(world)
        .iter()
        .enumerate()
        .map(|(k, c)| {
            let (dx, dy) = (min_image(c[0] - wx, ww), min_image(c[1] - wy, wh));
            (dx * dx + dy * dy, k as u32)
        })
        .filter(|&(d, _)| d <= reach * reach)
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, k)| k)
}

struct Session {
    args: ViewArgs,
    world: World,
    #[cfg(feature = "gpu")]
    gpu: Option<fl_gpu::GpuBackend>,
    backend: Backend,
    start_tick: u64,
    tick_limit: Option<u64>,
    ckpt_every: Option<u64>,
    ckpt_dir: PathBuf,
    retain: usize,
    stop_reason: String,
    done: bool,
    view_mode: View,
    camera: Camera,
    pace: Pace,
    inspect: Option<String>,
    tune: Tuner,
    presses: Vec<(u64, String, bool)>,
    wall0: Instant,
    /// The config the run started from; a restart with a new organism count builds from it.
    launch_cfg: Config,
    /// Digits typed so far while the organism count is being entered (`None` when it is not).
    count_entry: Option<String>,
    /// How many times the run has been restarted with a typed count.
    restarts: u32,
    /// Last message about a count that could not be applied, shown on the HUD until the next key.
    notice: Option<String>,
    #[cfg(feature = "gpu")]
    device_planes: Option<fl_gpu::FieldPlanes>,
    cpu_link: Option<painter::GpuLink>,
}

pub fn view(args: ViewArgs) -> i32 {
    let launch = Launch {
        config: args.config.clone(),
        seed: args.seed,
        exhibition: args.exhibition,
        workers: args.workers,
        resume: args.resume.clone(),
        erase: args.erase.clone(),
        profile: false,
        backend: args.backend,
    };
    let prepared = match prepare(&launch) {
        Ok(p) => p,
        Err(code) => return code,
    };
    if let Err(e) = std::fs::create_dir_all(&args.out) {
        eprintln!("{}: {e}", args.out.display());
        return 2;
    }
    let mut session = Session::from_prepared(args, prepared);
    if session.args.screenshot.is_some() {
        return session.run_offscreen();
    }
    session.run_window()
}

impl Session {
    fn from_prepared(args: ViewArgs, prepared: Prepared) -> Self {
        let world = prepared.world;
        let dt = world.dt();
        let limit = if args.seconds > 0.0 { Some(prepared.start_tick + (args.seconds / dt).round() as u64) } else { None };
        let ckpt_every = args.checkpoint_every.map(|s| ((s / dt).round() as u64).max(1));
        let ckpt_dir = args.out.join("checkpoints");
        if ckpt_every.is_some() {
            std::fs::create_dir_all(&ckpt_dir).ok();
        }
        let camera = Camera::fit(world.field.width(), world.field.height());
        let view_mode = args.view;
        let backend = args.backend;
        let retain = world.cfg.persistence.retain_recent_checkpoints as usize;
        eprintln!(
            "frankenlife view {MODEL_VERSION} | {} | seed {} | backend {:?} | dt = 1/{} s | close the window to stop",
            world.cfg.name, world.cfg.seed, args.backend, world.cfg.world.tick_denominator
        );
        eprintln!("controls: space pause | . step | 1 realtime | 2 2x | 4 4x | 0 unlimited | tab view | drag pan | wheel zoom | click inspect | esc quit");
        eprintln!("live tuning: q/w energy | a/s source speed | d/f diffusion | c/v decay | z/x mutation | j/k chemistry drift | click a source to pick it | r reset");
        eprintln!("organism count: p, type the number, enter (restarts the run with that many; esc cancels)");
        let tune_cfg = world.cfg.clone();
        let launch_cfg = tune_cfg.clone();
        let tune_log = args.out.join("controls.jsonl");
        let presses: Vec<(u64, String, bool)> = args.press.iter().map(|(t, k)| (*t, k.clone(), false)).collect();
        Session {
            args,
            world,
            #[cfg(feature = "gpu")]
            gpu: prepared.gpu,
            backend,
            start_tick: prepared.start_tick,
            tick_limit: limit,
            ckpt_every,
            ckpt_dir,
            retain,
            stop_reason: "window closed".into(),
            done: false,
            view_mode,
            camera,
            pace: Pace::new(),
            inspect: None,
            tune: Tuner::new(&tune_cfg, Some(tune_log)),
            presses,
            wall0: Instant::now(),
            launch_cfg,
            count_entry: None,
            restarts: 0,
            notice: None,
            #[cfg(feature = "gpu")]
            device_planes: None,
            cpu_link: None,
        }
    }

    /// Start a fresh world from the launch settings with `n` organisms. The window, camera and pace stay;
    /// the GPU backend is rebuilt on the same device so no organism from the old world lingers on it.
    fn restart_with_count(&mut self, n: u32) -> Result<(), String> {
        let mut cfg = self.launch_cfg.clone();
        cfg.population.initial_count = n;
        cfg.population.maximum_count = cfg.population.maximum_count.max(n);
        let mut world = World::new(cfg).map_err(|errs| errs.join("; "))?;
        world.workers = self.world.workers;
        #[cfg(feature = "gpu")]
        let fresh = match self.gpu.as_ref() {
            Some(g) => {
                let mut f = g.fresh()?;
                f.set_single_wait(self.backend == Backend::GpuRealtime);
                Some(f)
            }
            None => None,
        };
        for &(t, x, y, r) in &self.args.erase {
            world.schedule(t, Command::EraseDisk { x, y, radius: r, disposal: Disposal::Recycle }).map_err(|e| e.to_string())?;
        }
        #[cfg(feature = "gpu")]
        {
            self.device_planes = None;
            self.gpu = fresh;
        }
        let dt = world.dt();
        self.tick_limit = if self.args.seconds > 0.0 { Some((self.args.seconds / dt).round() as u64) } else { None };
        self.start_tick = 0;
        self.world = world;
        let mut tune = Tuner::new(&self.launch_cfg, Some(self.args.out.join("controls.jsonl")));
        tune.log = std::mem::take(&mut self.tune.log);
        self.tune = tune;
        self.tune.record(self.world.tick, serde_json::json!({ "restart_with_organisms": n }));
        // Timing, checkpoints and pacing belong to the new run: the manifest's rates must not mix in the old
        // one, and checkpoints at low ticks must not be pruned in favour of the old run's higher ones.
        self.restarts += 1;
        if self.ckpt_every.is_some() {
            self.ckpt_dir = self.args.out.join("checkpoints").join(format!("restart-{}", self.restarts));
            std::fs::create_dir_all(&self.ckpt_dir).ok();
        }
        let (speed, paused) = (self.pace.speed, if self.done { false } else { self.pace.paused });
        self.pace = Pace::new();
        self.pace.speed = speed;
        self.pace.paused = paused;
        self.wall0 = Instant::now();
        self.done = false;
        self.stop_reason = "window closed".into();
        self.inspect = None;
        Ok(())
    }

    /// One key while the organism count is being typed. Returns true if the key was consumed.
    fn count_key(&mut self, key: &Key) -> bool {
        let Some(entry) = self.count_entry.as_mut() else { return false };
        match key {
            Key::Character(s) if s.chars().all(|c| c.is_ascii_digit()) && !s.is_empty() => {
                if entry.len() < 7 {
                    entry.push_str(s.as_str());
                }
            }
            Key::Named(NamedKey::Backspace) => {
                entry.pop();
            }
            Key::Named(NamedKey::Escape) => self.count_entry = None,
            Key::Named(NamedKey::Enter) => {
                let typed = std::mem::take(entry);
                self.count_entry = None;
                match typed.parse::<u32>() {
                    Ok(n) if n >= 1 => {
                        if let Err(e) = self.restart_with_count(n) {
                            eprintln!("organism count {n}: {e}");
                            self.notice = Some(format!("CANNOT START WITH {n}: {}", e.chars().take(70).collect::<String>()));
                        }
                    }
                    _ => self.notice = Some("TYPE A NUMBER OF AT LEAST 1".into()),
                }
            }
            _ => {}
        }
        true
    }

    fn tick_hz(&self) -> f64 {
        1.0 / self.world.dt()
    }

    fn step_one(&mut self) -> Result<(), String> {
        if self.done {
            return Ok(());
        }
        if self.tick_limit.is_some_and(|n| self.world.tick >= n) {
            self.done = true;
            self.pace.paused = true;
            self.stop_reason = "completed".into();
            return Ok(());
        }
        for (tick, keys, applied) in self.presses.iter_mut() {
            if !*applied && *tick <= self.world.tick {
                *applied = true;
                self.tune.press_script(&mut self.world, keys);
            }
        }
        let t = Instant::now();
        let stepped = step_world(&mut self.world, self.backend, #[cfg(feature = "gpu")] &mut self.gpu);
        let report = match stepped {
            Ok(r) => r,
            Err(e) => {
                self.done = true;
                self.pace.paused = true;
                self.stop_reason = match &e {
                    TickError::CapacityPaused { .. } => format!("paused before commit: {e}"),
                    _ => format!("stopped: {e}"),
                };
                eprintln!("{}", self.stop_reason);
                return Ok(());
            }
        };
        self.pace.note_step(report.timing.total_ms);
        if self.ckpt_every.is_some_and(|k| self.world.tick % k == 0) {
            self.sync_field();
            write_checkpoint(&self.world, &self.ckpt_dir, self.retain);
        }
        if report.population == 0 {
            self.done = true;
            self.pace.paused = true;
            self.stop_reason = format!("extinction at tick {}", report.tick);
            eprintln!("{}", self.stop_reason);
        } else if self.tick_limit.is_some_and(|n| self.world.tick >= n) {
            self.done = true;
            self.pace.paused = true;
            self.stop_reason = "completed".into();
        }
        let _ = t;
        Ok(())
    }

    fn step_budget(&mut self, budget: Duration) -> Result<(), String> {
        let hz = self.tick_hz();
        let n = self.pace.take(hz, budget, false);
        if n == 0 {
            return Ok(());
        }
        let start = Instant::now();
        let mut i = 0u64;
        let tick = 1.0 / hz;
        while i < n && start.elapsed() < budget && !self.done && !self.pace.paused {
            self.step_one()?;
            if matches!(self.pace.speed, Speed::Realtime(_)) {
                self.pace.accum = (self.pace.accum - tick).max(0.0);
            }
            i += 1;
        }
        Ok(())
    }

    fn sync_field(&mut self) {
        #[cfg(feature = "gpu")]
        if let Some(g) = self.gpu.as_mut() {
            download_field(&mut self.world, g);
        }
    }

    fn finish(&mut self) -> i32 {
        self.sync_field();
        if self.ckpt_every.is_some() && !self.world.orgs.is_empty() {
            write_checkpoint(&self.world, &self.ckpt_dir, self.retain);
        }
        let inspections: Vec<_> = super::most_derived(&self.world, self.args.inspect).into_iter().filter_map(|id| self.world.inspect(id)).collect();
        std::fs::write(self.args.out.join("inspection.json"), serde_json::to_string_pretty(&inspections).unwrap()).ok();
        let wall = self.wall0.elapsed().as_secs_f64();
        let steps = self.world.tick.saturating_sub(self.start_tick);
        let wall_rate = steps as f64 / wall.max(1e-9);
        let sim_ms = if steps > 0 { self.pace.sim_ms / steps as f64 } else { 0.0 };
        let present_ms = if self.pace.frames > 0 { self.pace.present_ms / self.pace.frames as f64 } else { 0.0 };
        let present_per_step = if steps > 0 { self.pace.present_ms / steps as f64 } else { 0.0 };
        let manifest = serde_json::json!({
            "schema": "frankenlife.run-manifest.v0",
            "model_version": MODEL_VERSION,
            "stop_reason": self.stop_reason,
            "viewer": true,
            "start_tick": self.start_tick,
            "committed_tick": self.world.tick,
            "simulation_seconds": self.world.simulation_seconds(),
            "wall_seconds": wall,
            "workers": self.world.workers,
            "inference_backend": format!("{:?}", self.backend),
            "final_state_hash": format!("{:016x}", self.world.state_hash()),
            "population": self.world.orgs.len(),
            "totals": self.world.totals,
            "ledger": self.world.ledger,
            "frames_presented": self.pace.frames,
            "view": format!("{:?}", self.view_mode),
            "present_cpu_ms_mean": present_ms,
            "present_cpu_ms_per_step": present_per_step,
            "simulation_mean_step_ms": sim_ms,
            "wall_steps_per_second": wall_rate,
            "live_changes": self.tune.log,
            "config": self.world.cfg,
        });
        std::fs::write(self.args.out.join("manifest.json"), serde_json::to_string_pretty(&manifest).unwrap()).ok();
        let realtime = wall_rate * self.world.dt();
        eprintln!(
            "{reason}: tick {tick} ({sim:.1} sim s) in {wall:.1} wall s; population {pop}; hash {hash:016x}",
            reason = self.stop_reason,
            tick = self.world.tick,
            sim = self.world.simulation_seconds(),
            pop = self.world.orgs.len(),
            hash = self.world.state_hash(),
        );
        eprintln!(
            "viewer overhead: simulation {sim_ms:.2} ms/step ({sim_rate:.0} steps/s if drawing were free), wall {wall_rate:.0} steps/s ({realtime:.2}x real time), present {present_ms:.2} ms/frame CPU over {frames} frames ({present_per_step:.3} ms/step)",
            sim_rate = if sim_ms > 0.0 { 1000.0 / sim_ms } else { 0.0 },
            frames = self.pace.frames,
        );
        if self.stop_reason == "completed" || self.stop_reason == "window closed" || self.stop_reason.starts_with("paused") {
            0
        } else {
            3
        }
    }

    fn run_offscreen(&mut self) -> i32 {
        let Some(path) = self.args.screenshot.clone() else { return 2 };
        let (painter_device, painter_queue, planes_gpu) = match self.open_painter_device() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                return 2;
            }
        };
        let mut painter = match Painter::new(painter_device, painter_queue) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{e}");
                return 2;
            }
        };
        let _ = planes_gpu;
        let size = (self.args.width.unwrap_or(1280).max(2), self.args.height.unwrap_or(720).max(2));
        let frame_gap = Duration::from_secs_f64(1.0 / 60.0);
        let mut next_frame = Instant::now();
        let mut last_stats = PaintStats { cpu_ms: 0.0 };
        self.pace.speed = Speed::Unlimited;
        while self.tick_limit.is_some() && !self.done {
            if let Err(e) = self.step_budget(Duration::from_millis(12)) {
                eprintln!("{e}");
                return 2;
            }
            let now = Instant::now();
            if now >= next_frame || self.done {
                if let Err(e) = self.present_offscreen(&mut painter, size, &mut last_stats) {
                    eprintln!("{e}");
                    return 2;
                }
                next_frame = now + frame_gap;
            }
            if self.pace.paused && !self.done {
                break;
            }
        }
        if self.pace.frames == 0 {
            if let Err(e) = self.present_offscreen(&mut painter, size, &mut last_stats) {
                eprintln!("{e}");
                return 2;
            }
        }
        // The last present already wrote nothing to disk. Render once more so the PNG is the final state.
        match self.render_png(&mut painter, size) {
            Ok(rgb) => {
                if let Err(e) = super::render::write_png(&path, &rgb, size.0 as usize, size.1 as usize) {
                    eprintln!("{}: {e}", path.display());
                    return 2;
                }
                eprintln!("wrote {}", path.display());
            }
            Err(e) => {
                eprintln!("{e}");
                return 2;
            }
        }
        self.finish()
    }

    fn render_png(&mut self, painter: &mut Painter, size: (u32, u32)) -> Result<Vec<u8>, String> {
        self.paint(painter, size, true)?.1.ok_or_else(|| "frame was not read back".into())
    }

    fn paint(&mut self, painter: &mut Painter, size: (u32, u32), readback: bool) -> Result<(PaintStats, Option<Vec<u8>>), String> {
        let scene_orgs = self.organism_instances(size);
        let (hud, hw, hh) = self.hud_image(size);
        let raw_field = self.view_mode == View::Field;
        let show_field = self.view_mode != View::Raw;
        let show_organisms = self.view_mode != View::Field;
        let filmic = matches!(self.view_mode, View::Composite | View::Chemistry);
        let cfg = &self.world.cfg.presentation;
        let bloom = if cfg.bloom_enabled && filmic { cfg.bloom_strength as f32 } else { 0.0 };
        let background = super::render::parse_hex(&cfg.background);
        let ww = self.world.field.width() as f32;
        let wh = self.world.field.height() as f32;
        let zoom = self.camera.px_per_unit(size.0 as f64, size.1 as f64, ww as f64, wh as f64) as f32;
        let reach = point_reach(cfg.point_radius_pixels as f32, size, zoom, self.camera.zoom as f32);
        let center = [self.camera.center_x as f32, self.camera.center_y as f32];
        let point = self.view_mode == View::Raw;
        let planes = self.planes()?;
        let scene = Scene {
            planes,
            raw_field,
            show_field,
            show_organisms,
            filmic,
            bloom,
            background,
            world: [ww, wh],
            center,
            zoom,
            reach,
            point,
            organisms: &scene_orgs,
            hud: &hud,
            hud_size: (hw, hh),
        };
        let (stats, rgb) = if readback {
            let (rgb, stats) = painter.render_rgba(&scene, size)?;
            (stats, Some(rgb))
        } else {
            (painter.draw_offscreen(&scene, size)?, None)
        };
        self.pace.present_ms += stats.cpu_ms;
        self.pace.frames += 1;
        Ok((stats, rgb))
    }

    fn present_offscreen(&mut self, painter: &mut Painter, size: (u32, u32), last: &mut PaintStats) -> Result<(), String> {
        let (stats, _) = self.paint(painter, size, false)?;
        *last = stats;
        Ok(())
    }

    fn planes(&mut self) -> Result<Planes<'_>, String> {
        #[cfg(feature = "gpu")]
        {
            if self.world.field_on_device() {
                let gpu = self.gpu.as_mut().ok_or("resident field without a GPU backend")?;
                let planes = gpu.field_planes(&self.world.field, true)?;
                // FieldPlanes is owned. Scene wants a borrow of the buffers that live in it.
                // Leak the owned planes into the session? They contain Buffer clones, so we can
                // return Device with references if we store them. Store on self.
                self.device_planes = Some(planes);
                let p = self.device_planes.as_ref().expect("planes");
                return Ok(Planes::Device { energy: &p.energy, moment: &p.moment, width: p.width, height: p.height, cell_area: p.cell_area, generation: p.generation });
            }
        }
        let f = &self.world.field;
        Ok(Planes::Host { energy: &f.e, moment: &f.m, width: f.gx as u32, height: f.gy as u32, cell_area: (f.hx * f.hy) as f32 })
    }

    fn organism_instances(&self, screen: (u32, u32)) -> Vec<f32> {
        let ww = self.world.field.width();
        let wh = self.world.field.height();
        let sw = screen.0 as f64;
        let sh = screen.1 as f64;
        let cfg = &self.world.cfg.presentation;
        let zoom = self.camera.px_per_unit(sw, sh, ww, wh) as f32;
        let reach = point_reach(cfg.point_radius_pixels as f32, screen, zoom, self.camera.zoom as f32);
        let margin = reach as f64 / zoom.max(1e-6) as f64;
        let (x0, x1, y0, y1) = self.camera.tile_span(sw, sh, ww, wh, margin);
        let basis = chem_basis();
        let orgs = &self.world.orgs;
        let mut out = Vec::with_capacity(orgs.len() * 6);
        for i in 0..orgs.len() {
            let col = match self.view_mode {
                View::Chemistry => chem_color(&orgs.genome[i].affinity, &basis),
                _ => lineage_color(orgs.lineage[i]),
            };
            let e = orgs.energy[i];
            let bright = if self.view_mode == View::Raw { 1.0 } else { (0.35 + 1.4 * e / (e + 3.0)) as f32 };
            for ty in y0..=y1 {
                for tx in x0..=x1 {
                    out.extend_from_slice(&[(orgs.x[i] + tx as f64 * ww) as f32, (orgs.y[i] + ty as f64 * wh) as f32, col[0], col[1], col[2], bright]);
                }
            }
        }
        // Source markers: a ring of half a sigma around each center, a full-sigma white
        // ring around the selected one.
        let sigma = self.world.cfg.sources.sigma_world_units;
        for (k, c) in source_centers(&self.world).iter().enumerate() {
            let selected = self.tune.source == Some(k as u32);
            let (n, radius, col, bright) = if selected { (72, sigma, [1.0f32, 1.0, 1.0], 1.8f32) } else { (28, 0.5 * sigma, [0.65f32, 0.8, 0.95], 0.8f32) };
            for s in 0..n {
                let a = std::f64::consts::TAU * s as f64 / n as f64;
                let (px, py) = (c[0] + radius * a.cos(), c[1] + radius * a.sin());
                for ty in y0..=y1 {
                    for tx in x0..=x1 {
                        out.extend_from_slice(&[(px + tx as f64 * ww) as f32, (py + ty as f64 * wh) as f32, col[0], col[1], col[2], bright]);
                    }
                }
            }
        }
        out
    }

    fn hud_image(&self, screen: (u32, u32)) -> (Vec<u8>, u32, u32) {
        let rate = self.pace.steps_per_s;
        let realtime = rate * self.world.dt();
        let speed = match self.pace.speed {
            Speed::Unlimited => "unlimited".to_string(),
            Speed::Realtime(m) => format!("{m:.2}x"),
        };
        let state = if self.done {
            self.stop_reason.clone()
        } else if self.pace.paused {
            "paused".into()
        } else {
            "running".into()
        };
        let mut lines = vec![
            format!("TICK {}  SIM {:.2} S  POP {}", self.world.tick, self.world.simulation_seconds(), self.world.orgs.len()),
            format!("{:.0} STEPS/S  {:.2}X REALTIME  SPEED {speed}  {state}", rate, realtime),
            format!("VIEW {:?}  ZOOM {:.2}X", self.view_mode, self.camera.zoom),
            "SPACE PAUSE  . STEP  1 REALTIME  2 2X  4 4X  0 MAX  TAB VIEW  ESC QUIT".into(),
            self.tune.hud_line(),
            "Q/W ENERGY  A/S SOURCE SPEED  D/F DIFFUSION  C/V DECAY  Z/X MUTATION  J/K CHEM DRIFT  CLICK SOURCE  R RESET".into(),
        ];
        lines.push("P ORGANISM COUNT (TYPE A NUMBER, ENTER RESTARTS THE RUN)".into());
        if let Some(entry) = &self.count_entry {
            lines.push(format!("ORGANISMS: {entry}_   ENTER START  ESC CANCEL  (NOW {})", self.world.orgs.len()));
        } else if let Some(n) = &self.notice {
            lines.push(n.clone());
        }
        if let Some(s) = &self.inspect {
            lines.push(s.clone());
        }
        let scale = if screen.0 >= 1600 { 3 } else { 2 };
        blit_hud(&lines, scale)
    }

    /// A click selects the source under the cursor (click it again to clear), else
    /// inspects the organism there. Clicking empty space clears the source selection.
    fn click(&mut self, px: f64, py: f64, screen: (u32, u32)) {
        let ww = self.world.field.width();
        let wh = self.world.field.height();
        let (wx, wy) = self.camera.screen_to_world(px, py, screen.0 as f64, screen.1 as f64, ww, wh);
        let z = self.camera.px_per_unit(screen.0 as f64, screen.1 as f64, ww, wh).max(1e-9);
        // Close to the center, so organisms feeding on a source stay clickable.
        let reach = (14.0 / z).max(0.5 * self.world.cfg.sources.sigma_world_units);
        if let Some(k) = source_at(&self.world, wx, wy, reach) {
            self.tune.source = if self.tune.source == Some(k) { None } else { Some(k) };
            self.inspect = None;
            return;
        }
        self.inspect = self.pick(px, py, screen);
        if self.inspect.is_none() {
            self.tune.source = None;
        }
    }

    fn pick(&self, px: f64, py: f64, screen: (u32, u32)) -> Option<String> {
        let ww = self.world.field.width();
        let wh = self.world.field.height();
        let (wx, wy) = self.camera.screen_to_world(px, py, screen.0 as f64, screen.1 as f64, ww, wh);
        let z = self.camera.px_per_unit(screen.0 as f64, screen.1 as f64, ww, wh).max(1e-9);
        let r = (10.0 / z).max(ww / 256.0);
        let orgs = &self.world.orgs;
        let mut best: Option<(f64, usize)> = None;
        for i in 0..orgs.len() {
            let dx = min_image(orgs.x[i] - wx, ww);
            let dy = min_image(orgs.y[i] - wy, wh);
            let d = dx * dx + dy * dy;
            if d <= r * r && best.is_none_or(|b| d < b.0) {
                best = Some((d, i));
            }
        }
        best.map(|(_, i)| format!("ID {}  LIN {}  GEN {}  E {:.3}  AT {:.1},{:.1}", orgs.id[i], orgs.lineage[i], orgs.generation[i], orgs.energy[i], orgs.x[i], orgs.y[i]))
    }

    fn open_painter_device(&mut self) -> Result<(wgpu::Device, wgpu::Queue, bool), String> {
        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            let g = self.gpu.as_ref().expect("gpu");
            return Ok((g.device().clone(), g.queue().clone(), true));
        }
        let link = Painter::open_device()?;
        self.cpu_link = Some(link);
        let link = self.cpu_link.as_ref().expect("link");
        Ok((link.device.clone(), link.queue.clone(), false))
    }

    fn run_window(&mut self) -> i32 {
        let event_loop = match winit::event_loop::EventLoop::new() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("cannot open a window: {e}");
                return 2;
            }
        };
        let mut app = WindowApp { session: self, painter: None, swap: None, cursor: (0.0, 0.0), drag: None, drag_moved: false, wrote: false, showed_final: false };
        if let Err(e) = event_loop.run_app(&mut app) {
            eprintln!("window: {e}");
            if !app.wrote {
                return app.session.finish();
            }
            return 2;
        }
        if !app.wrote {
            app.session.finish()
        } else {
            0
        }
    }
}

fn point_reach(point_radius_pixels: f32, screen: (u32, u32), px_per_unit: f32, zoom: f32) -> f32 {
    let n = screen.0.min(screen.1) as f32;
    let base = (point_radius_pixels * 1.6 * n / 1024.0).max(0.6);
    // Zoom 1 matches the headless rasterizer. Zooming in grows the splat with the world.
    (base * zoom.max(1.0)).min(px_per_unit * 4.0).max(0.6) * 2.5
}

fn min_image(d: f64, span: f64) -> f64 {
    let mut x = d % span;
    if x > span * 0.5 {
        x -= span;
    } else if x < -span * 0.5 {
        x += span;
    }
    x
}

fn blit_hud(lines: &[String], scale: usize) -> (Vec<u8>, u32, u32) {
    let scale = scale.max(1);
    let pad = 4 * scale;
    let line_h = 7 * scale;
    let width = lines.iter().map(|l| l.chars().count()).max().unwrap_or(1) * 4 * scale + pad * 2;
    let height = lines.len() * line_h + pad * 2;
    let mut rgba = vec![0u8; width * height * 4];
    for px in rgba.chunks_exact_mut(4) {
        px[0] = 4;
        px[1] = 8;
        px[2] = 14;
        px[3] = 190;
    }
    for (i, line) in lines.iter().enumerate() {
        blit_text(&mut rgba, width, pad, pad + i * line_h, scale, line, [226, 234, 242, 255]);
    }
    (rgba, width as u32, height as u32)
}

fn blit_text(rgba: &mut [u8], width: usize, x0: usize, y0: usize, s: usize, text: &str, col: [u8; 4]) {
    for (ci, ch) in text.chars().enumerate() {
        let g = glyph(ch).as_bytes();
        for row in 0..5 {
            for colm in 0..3 {
                if g[row * 3 + colm] != b'1' {
                    continue;
                }
                for dy in 0..s {
                    for dx in 0..s {
                        let x = x0 + (ci * 4 + colm) * s + dx;
                        let y = y0 + row * s + dy;
                        if x < width {
                            let p = (y * width + x) * 4;
                            if p + 3 < rgba.len() {
                                rgba[p..p + 4].copy_from_slice(&col);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn glyph(c: char) -> &'static str {
    match c.to_ascii_uppercase() {
        '0' => "111101101101111", '1' => "010110010010111", '2' => "111001111100111", '3' => "111001111001111",
        '4' => "101101111001001", '5' => "111100111001111", '6' => "111100111101111", '7' => "111001001001001",
        '8' => "111101111101111", '9' => "111101111001111", 'A' => "010101111101101", 'B' => "110101110101110",
        'C' => "011100100100011", 'D' => "110101101101110", 'E' => "111100110100111", 'F' => "111100110100100",
        'G' => "011100101101011", 'H' => "101101111101101", 'I' => "111010010010111", 'J' => "001001001101010",
        'K' => "101101110101101", 'L' => "100100100100111", 'M' => "101111111101101", 'N' => "110101101101101",
        'O' => "010101101101010", 'P' => "110101110100100", 'Q' => "010101101110011", 'R' => "110101110101101",
        'S' => "011100010001110", 'T' => "111010010010010", 'U' => "101101101101111", 'V' => "101101101101010",
        'W' => "101101111111101", 'X' => "101101010101101", 'Y' => "101101010010010", 'Z' => "111001010100111",
        '.' => "000000000000010", ':' => "000010000010000", '/' => "001001010100100", '=' => "000111000111000",
        '-' => "000000111000000", '_' => "000000000000111", '%' => "101001010100101", '(' => "010100100100010", ')' => "010001001001010",
        _ => "000000000000000",
    }
}

struct WindowApp<'a> {
    session: &'a mut Session,
    painter: Option<Painter>,
    swap: Option<Swap>,
    cursor: (f64, f64),
    drag: Option<(f64, f64, f64, f64)>,
    drag_moved: bool,
    wrote: bool,
    showed_final: bool,
}

struct Swap {
    window: Arc<winit::window::Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    same: bool,
    device: wgpu::Device,
    queue: wgpu::Queue,
    tonemap: wgpu::RenderPipeline,
    hud: wgpu::RenderPipeline,
    encode_srgb: bool,
}

impl Session {
    fn open_link(&mut self) -> Result<painter::GpuLink, String> {
        #[cfg(feature = "gpu")]
        if let Some(g) = self.gpu.as_ref() {
            return Ok(painter::GpuLink { instance: g.instance().clone(), adapter: g.adapter().clone(), device: g.device().clone(), queue: g.queue().clone() });
        }
        Painter::open_device()
    }
}

impl WindowApp<'_> {
    fn ensure(&mut self, window: Arc<winit::window::Window>) -> Result<(), String> {
        if self.painter.is_some() && self.swap.is_some() {
            return Ok(());
        }
        let link = self.session.open_link()?;
        let painter = Painter::new(link.device.clone(), link.queue.clone())?;
        let size = window.inner_size();
        let (surface, adapter, device, queue, same) = attach_surface(&link, window.clone())?;
        let config = surface_config(&surface, &adapter, size.width, size.height)?;
        surface.configure(&device, &config);
        let encode_srgb = !is_srgb(config.format);
        eprintln!(
            "viewer surface: format {:?} | color space {:?} | present {:?} | alpha {:?} | {}x{} px | scale {:.2} | {}",
            config.format,
            config.color_space,
            config.present_mode,
            config.alpha_mode,
            config.width,
            config.height,
            window.scale_factor(),
            if same { "simulation device presents" } else { "separate present device" }
        );
        let (tonemap, hud) = painter.pipelines_for(config.format);
        // pipelines_for uses painter.device. If the surface device differs, those pipelines
        // cannot draw to the surface. Rebuild them on the surface device when needed.
        let (tonemap, hud) = if same {
            (tonemap, hud)
        } else {
            let layout_holder = Painter::new(device.clone(), queue.clone())?;
            layout_holder.pipelines_for(config.format)
        };
        self.painter = Some(painter);
        self.swap = Some(Swap { window, surface, config, same, device, queue, tonemap, hud, encode_srgb });
        Ok(())
    }

    fn draw(&mut self) -> Result<(), String> {
        let Some(swap) = self.swap.as_mut() else { return Ok(()) };
        let size = swap.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        if swap.config.width != size.width || swap.config.height != size.height {
            swap.config.width = size.width;
            swap.config.height = size.height;
            swap.surface.configure(&swap.device, &swap.config);
        }
        let (frame, suboptimal) = match swap.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => (t, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => (t, true),
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return Ok(()),
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                swap.surface.configure(&swap.device, &swap.config);
                swap.window.request_redraw();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => return Err("surface validation error".into()),
        };
        let frame_size = frame.texture.size();
        if (frame_size.width != size.width || frame_size.height != size.height) && (swap.config.width != size.width || swap.config.height != size.height) {
            drop(frame);
            swap.config.width = size.width;
            swap.config.height = size.height;
            swap.surface.configure(&swap.device, &swap.config);
            swap.window.request_redraw();
            return Ok(());
        }
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let screen = (size.width, size.height);
        let orgs = self.session.organism_instances(screen);
        let (hud_px, hw, hh) = self.session.hud_image(screen);
        let cfg = &self.session.world.cfg.presentation;
        let ww = self.session.world.field.width() as f32;
        let wh = self.session.world.field.height() as f32;
        let zoom = self.session.camera.px_per_unit(screen.0 as f64, screen.1 as f64, ww as f64, wh as f64) as f32;
        let reach = point_reach(cfg.point_radius_pixels as f32, screen, zoom, self.session.camera.zoom as f32);
        let mode = self.session.view_mode;
        let filmic = matches!(mode, View::Composite | View::Chemistry);
        let bloom = if cfg.bloom_enabled && filmic { cfg.bloom_strength as f32 } else { 0.0 };
        let background = super::render::parse_hex(&cfg.background);
        let center = [self.session.camera.center_x as f32, self.session.camera.center_y as f32];
        let planes = self.session.planes()?;
        let scene = Scene {
            planes,
            raw_field: mode == View::Field,
            show_field: mode != View::Raw,
            show_organisms: mode != View::Field,
            filmic,
            bloom,
            background,
            world: [ww, wh],
            center,
            zoom,
            reach,
            point: mode == View::Raw,
            organisms: &orgs,
            hud: &hud_px,
            hud_size: (hw, hh),
        };
        let same = swap.same;
        let encode = swap.encode_srgb;
        let tonemap = swap.tonemap.clone();
        let hud = swap.hud.clone();
        let queue = swap.queue.clone();
        if same {
            let painter = self.painter.as_mut().expect("painter");
            let stats = painter.draw(&scene, &view, screen, encode, &tonemap, &hud)?;
            // The view holds the swapchain image. Drop it before present or the
            // image stays in use and Windows shows the untouched window.
            drop(view);
            let _ = painter.device.poll(wgpu::PollType::Poll);
            if let Some(swap) = self.swap.as_ref() {
                swap.window.pre_present_notify();
            }
            queue.present(frame);
            self.session.pace.present_ms += stats.cpu_ms;
            self.session.pace.frames += 1;
            if suboptimal {
                if let Some(swap) = self.swap.as_mut() {
                    let size = swap.window.inner_size();
                    swap.config.width = size.width.max(1);
                    swap.config.height = size.height.max(1);
                    swap.surface.configure(&swap.device, &swap.config);
                }
            }
        } else {
            drop(view);
            drop(frame);
            let painter = self.painter.as_mut().expect("painter");
            let (rgb, stats) = painter.render_rgba(&scene, screen)?;
            self.session.pace.present_ms += stats.cpu_ms;
            self.session.pace.frames += 1;
            // A foreign present device is reported; the window still shows the frame via the
            // same-device path on Vulkan and DX12. Reaching here means the sim adapter could
            // not present, so keep the readback cost visible in the overhead line.
            let _ = rgb;
            eprintln!("viewer: the simulation GPU cannot present to this window; the frame was rendered offscreen");
        }
        let title = format!(
            "FrankenLife  tick {}  t={:.2}s  pop {}  {:.0} steps/s  {:.2}x  {:?}",
            self.session.world.tick,
            self.session.world.simulation_seconds(),
            self.session.world.orgs.len(),
            self.session.pace.steps_per_s,
            self.session.pace.steps_per_s * self.session.world.dt(),
            self.session.view_mode
        );
        if let Some(swap) = self.swap.as_ref() {
            swap.window.set_title(&title);
        }
        Ok(())
    }
}

fn is_srgb(format: wgpu::TextureFormat) -> bool {
    matches!(format, wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Bgra8UnormSrgb)
}

fn choose_format(formats: &[wgpu::TextureFormat]) -> wgpu::TextureFormat {
    const PREFERRED: [wgpu::TextureFormat; 4] = [
        wgpu::TextureFormat::Bgra8UnormSrgb,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Rgba8Unorm,
    ];
    for format in PREFERRED {
        if formats.contains(&format) {
            return format;
        }
    }
    formats.first().copied().unwrap_or(wgpu::TextureFormat::Bgra8Unorm)
}

/// Mailbox if the adapter has it, otherwise FIFO.
///
/// `PresentMode::AutoNoVsync` resolves to immediate when the driver offers it.
/// On Windows, Vulkan immediate present does not go through DWM, so the client
/// area stays the window's white background while the title bar (drawn by DWM)
/// keeps updating.
fn choose_present(modes: &[wgpu::PresentMode]) -> wgpu::PresentMode {
    if modes.contains(&wgpu::PresentMode::Mailbox) {
        wgpu::PresentMode::Mailbox
    } else {
        wgpu::PresentMode::Fifo
    }
}

fn surface_config(surface: &wgpu::Surface<'_>, adapter: &wgpu::Adapter, width: u32, height: u32) -> Result<wgpu::SurfaceConfiguration, String> {
    let caps = surface.get_capabilities(adapter);
    if caps.formats.is_empty() {
        return Err("this adapter cannot present to the window".into());
    }
    let format = choose_format(&caps.formats);
    let color_space = match wgpu::SurfaceColorSpace::Srgb.to_color_spaces() {
        Some(bit) if caps.color_spaces(format).contains(bit) => wgpu::SurfaceColorSpace::Srgb,
        _ => wgpu::SurfaceColorSpace::Auto,
    };
    let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
        wgpu::CompositeAlphaMode::Opaque
    } else {
        wgpu::CompositeAlphaMode::Auto
    };
    Ok(wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        color_space,
        width: width.max(1),
        height: height.max(1),
        present_mode: choose_present(&caps.present_modes),
        desired_maximum_frame_latency: 2,
        alpha_mode,
        view_formats: vec![],
    })
}

fn attach_surface(link: &painter::GpuLink, window: Arc<winit::window::Window>) -> Result<(wgpu::Surface<'static>, wgpu::Adapter, wgpu::Device, wgpu::Queue, bool), String> {
    if let Ok(surface) = link.instance.create_surface(window.clone()) {
        if link.adapter.is_surface_supported(&surface) {
            return Ok((surface, link.adapter.clone(), link.device.clone(), link.queue.clone(), true));
        }
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let surface = instance.create_surface(window).map_err(|e| format!("cannot create a window surface: {e}"))?;
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: Some(&surface),
        apply_limit_buckets: false,
    }))
    .map_err(|e| format!("no adapter can present to the window: {e}"))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("frankenlife-present"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .map_err(|e| format!("present device: {e}"))?;
    Ok((surface, adapter, device, queue, false))
}

impl winit::application::ApplicationHandler for WindowApp<'_> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.swap.is_some() {
            return;
        }
        let (w, h) = (
            self.session.args.width.unwrap_or(self.session.world.cfg.presentation.width).max(320),
            self.session.args.height.unwrap_or(self.session.world.cfg.presentation.height).max(240),
        );
        let attrs = winit::window::WindowAttributes::default().with_title("FrankenLife").with_inner_size(winit::dpi::LogicalSize::new(w, h));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("cannot create the window: {e}");
                event_loop.exit();
                return;
            }
        };
        if let Err(e) = self.ensure(window.clone()) {
            eprintln!("{e}");
            event_loop.exit();
            return;
        }
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, _id: winit::window::WindowId, event: winit::event::WindowEvent) {
        use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};

        match event {
            WindowEvent::Resized(size) => {
                if let Some(swap) = self.swap.as_mut() {
                    if size.width > 0 && size.height > 0 && (swap.config.width != size.width || swap.config.height != size.height) {
                        swap.config.width = size.width;
                        swap.config.height = size.height;
                        swap.surface.configure(&swap.device, &swap.config);
                    }
                    swap.window.request_redraw();
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                eprintln!("viewer surface: scale factor is now {scale_factor:.2}");
                if let Some(swap) = self.swap.as_ref() {
                    swap.window.request_redraw();
                }
            }
            WindowEvent::CloseRequested => {
                self.wrote = true;
                let code_unused = self.session.finish();
                let _ = code_unused;
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.draw() {
                    eprintln!("{e}");
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed || event.repeat {
                    return;
                }
                let session = &mut *self.session;
                session.notice = None;
                if session.count_key(&event.logical_key) {
                    // A run that ended and was restarted shows its final frame again when it ends.
                    self.showed_final = false;
                    if let Some(w) = self.swap.as_ref() {
                        w.window.request_redraw();
                    }
                    return;
                }
                match event.logical_key {
                    Key::Character(ref s) if s.eq_ignore_ascii_case("p") => {
                        session.count_entry = Some(String::new());
                    }
                    Key::Named(NamedKey::Escape) => {
                        self.wrote = true;
                        session.finish();
                        event_loop.exit();
                    }
                    Key::Named(NamedKey::Space) => {
                        session.pace.paused = !session.pace.paused;
                        session.pace.accum = 0.0;
                        session.pace.last = Instant::now();
                    }
                    Key::Named(NamedKey::Tab) => {
                        session.view_mode = match session.view_mode {
                            View::Composite => View::Chemistry,
                            View::Chemistry => View::Field,
                            View::Field => View::Raw,
                            View::Raw => View::Composite,
                        };
                    }
                    Key::Character(s) if s.as_str() == "." || s.eq_ignore_ascii_case("n") => {
                        session.pace.paused = true;
                        session.done = false;
                        if session.stop_reason == "completed" {
                            // A finished run does not step past its limit.
                        } else {
                            session.step_one().ok();
                        }
                    }
                    Key::Character(s) if s.as_str() == "1" => {
                        session.pace.speed = Speed::Realtime(1.0);
                        session.pace.accum = 0.0;
                        session.pace.last = Instant::now();
                    }
                    Key::Character(s) if s.as_str() == "2" => {
                        session.pace.speed = Speed::Realtime(2.0);
                        session.pace.accum = 0.0;
                        session.pace.last = Instant::now();
                    }
                    Key::Character(s) if s.as_str() == "4" => {
                        session.pace.speed = Speed::Realtime(4.0);
                        session.pace.accum = 0.0;
                        session.pace.last = Instant::now();
                    }
                    Key::Character(s) if s.as_str() == "0" => {
                        session.pace.speed = Speed::Unlimited;
                    }
                    Key::Character(s) if control_for(s.as_str()).is_some() => {
                        session.tune.press(&mut session.world, s.as_str());
                    }
                    _ => {}
                }
                if let Some(w) = self.swap.as_ref() {
                    w.window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = (position.x, position.y);
                if let Some((x0, y0, cx, cy)) = self.drag {
                    let dx = position.x - x0;
                    let dy = position.y - y0;
                    if dx * dx + dy * dy > 16.0 {
                        self.drag_moved = true;
                        let ww = self.session.world.field.width();
                        let wh = self.session.world.field.height();
                        let screen = self.swap.as_ref().map(|s| s.window.inner_size()).unwrap_or_default();
                        let z = self.session.camera.px_per_unit(screen.width.max(1) as f64, screen.height.max(1) as f64, ww, wh).max(1e-9);
                        self.session.camera.center_x = (cx - dx / z).rem_euclid(ww);
                        self.session.camera.center_y = (cy - dy / z).rem_euclid(wh);
                        if let Some(w) = self.swap.as_ref() {
                            w.window.request_redraw();
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                if state == ElementState::Pressed {
                    let c = self.session.camera;
                    self.drag = Some((self.cursor.0, self.cursor.1, c.center_x, c.center_y));
                    self.drag_moved = false;
                } else if state == ElementState::Released {
                    if !self.drag_moved {
                        let screen = self.swap.as_ref().map(|s| {
                            let z = s.window.inner_size();
                            (z.width, z.height)
                        });
                        if let Some(screen) = screen {
                            self.session.click(self.cursor.0, self.cursor.1, screen);
                            if let Some(w) = self.swap.as_ref() {
                                w.window.request_redraw();
                            }
                        }
                    }
                    self.drag = None;
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 60.0,
                };
                if dy.abs() > 0.0 {
                    let screen = self.swap.as_ref().map(|s| s.window.inner_size());
                    if let Some(screen) = screen {
                        let ww = self.session.world.field.width();
                        let wh = self.session.world.field.height();
                        let factor = (1.1f64).powf(dy);
                        self.session.camera = self.session.camera.zoom_at(factor, self.cursor.0, self.cursor.1, screen.width.max(1) as f64, screen.height.max(1) as f64, ww, wh);
                        if let Some(w) = self.swap.as_ref() {
                            w.window.request_redraw();
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.painter.is_none() {
            return;
        }
        let hz = self.session.tick_hz();
        if let Err(e) = self.session.step_budget(Duration::from_millis(12)) {
            eprintln!("{e}");
        }
        // Present near the display rate without blocking the step loop on vsync.
        // A paused or finished window waits for input; requesting a redraw here
        // would spin, because the redraw wakes the loop again.
        let present_due = !self.session.pace.paused && !self.session.done && {
            let frames = self.session.pace.frames.max(1) as f64;
            let wall = self.session.wall0.elapsed().as_secs_f64().max(1e-6);
            wall * 60.0 >= frames
        };
        let first = self.session.pace.frames == 0;
        let just_finished = self.session.done && !self.showed_final;
        if just_finished {
            self.showed_final = true;
        }
        if (first || present_due || just_finished) && self.swap.as_ref().is_some_and(|s| s.window.is_visible().unwrap_or(true)) {
            if let Some(s) = self.swap.as_ref() {
                s.window.request_redraw();
            }
        }
        if self.session.pace.paused || self.session.done {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
        } else if matches!(self.session.pace.speed, Speed::Realtime(m) if m <= 2.0) {
            let tick = Duration::from_secs_f64((1.0 / hz / self.session.pace.speed_factor()).max(0.001));
            event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(Instant::now() + tick));
        } else {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
        }
    }
}

impl Speed {
    fn factor(self) -> f64 {
        match self {
            Speed::Unlimited => 1.0,
            Speed::Realtime(m) => m,
        }
    }
}

impl Pace {
    fn speed_factor(&self) -> f64 {
        self.speed.factor()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn present_mode_skips_immediate_even_when_the_driver_offers_it() {
        use wgpu::PresentMode::{Fifo, Immediate, Mailbox};
        assert_eq!(choose_present(&[Immediate, Mailbox, Fifo]), Mailbox);
        assert_eq!(choose_present(&[Immediate, Fifo]), Fifo);
        assert_eq!(choose_present(&[Fifo]), Fifo);
        assert_eq!(choose_format(&[wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureFormat::Bgra8UnormSrgb]), wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn camera_roundtrips_the_center_and_wraps() {
        let cam = Camera { center_x: 40.0, center_y: 10.0, zoom: 2.0 };
        let (x, y) = cam.screen_to_world(200.0, 100.0, 400.0, 200.0, 128.0, 128.0);
        let z = cam.px_per_unit(400.0, 200.0, 128.0, 128.0);
        let px = (x - cam.center_x) * z + 200.0;
        let py = (y - cam.center_y) * z + 100.0;
        assert!((px - 200.0).abs() < 1e-6 && (py - 100.0).abs() < 1e-6);
        let zoomed = cam.zoom_at(2.0, 200.0, 100.0, 400.0, 200.0, 128.0, 128.0);
        let (x2, y2) = zoomed.screen_to_world(200.0, 100.0, 400.0, 200.0, 128.0, 128.0);
        assert!((x2 - x).abs() < 1e-6 && (y2 - y).abs() < 1e-6, "zoom keeps the cursor's world point");
        let (x0, x1, y0, y1) = Camera::fit(128.0, 128.0).tile_span(128.0, 128.0, 128.0, 128.0, 0.0);
        assert!(x1 - x0 <= 2 && y1 - y0 <= 2, "a fitted view only needs the neighboring tiles, got {x0}..={x1}, {y0}..={y1}");
    }

    #[test]
    fn realtime_schedules_one_second_of_ticks() {
        let mut pace = Pace::new();
        pace.speed = Speed::Realtime(1.0);
        pace.last = Instant::now() - Duration::from_millis(100);
        let n = pace.take(128.0, Duration::from_millis(14), false);
        assert!((12..=13).contains(&n), "0.1s at 128 Hz schedules {n} ticks");
    }

    fn tuned_world() -> World {
        let mut c = tiny_config();
        c.sources.count = 3;
        c.sources.path_speed = 2.0;
        World::new(c).unwrap()
    }

    #[test]
    fn control_keys_map_to_opposite_pairs() {
        for (down, up) in [("q", "w"), ("a", "s"), ("d", "f"), ("c", "v"), ("z", "x")] {
            let (Some(Control::Tune(g1, f1)), Some(Control::Tune(g2, f2))) = (control_for(down), control_for(up)) else { panic!("{down}/{up}") };
            assert_eq!(g1, g2);
            assert!(f1 < 1.0 && f2 > 1.0 && (f1 * f2 - 1.0).abs() < 1e-12, "{down}/{up} undo each other");
            assert_eq!(control_for(&up.to_uppercase()), control_for(up), "shift does not matter");
        }
        assert_eq!(control_for("j"), Some(Control::Drift(5.0)));
        assert_eq!(control_for("r"), Some(Control::Reset));
        for k in [" ", ".", "n", "0", "1", "2", "4", "tab", "e", "u"] {
            assert_eq!(control_for(k), None, "{k} keeps its old meaning");
        }
    }

    #[test]
    fn keys_change_the_running_world_and_reset_restores_it() {
        let mut w = tuned_world();
        let mut tune = Tuner::new(&w.cfg, None);
        let (power, speed, diff, decay, mw) = (w.cfg.sources.power_per_source, w.cfg.sources.path_speed, w.cfg.world.diffusion, w.cfg.world.decay_per_second, w.cfg.mutation.weight_standard_deviation);
        assert!(tune.press(&mut w, "w") && tune.press(&mut w, "w"));
        assert!(tune.press(&mut w, "s"));
        assert!(tune.press(&mut w, "d"));
        assert!(tune.press(&mut w, "v"));
        assert!(tune.press(&mut w, "x"));
        assert!(!tune.press(&mut w, "e"), "not a control");
        w.step().unwrap();
        assert!((w.cfg.sources.power_per_source - power * 1.25 * 1.25).abs() < 1e-9);
        assert!((w.cfg.sources.path_speed - speed * 1.25).abs() < 1e-9);
        assert!((w.cfg.world.diffusion - diff / 1.25).abs() < 1e-9);
        assert!((w.cfg.world.decay_per_second - decay * 1.25).abs() < 1e-9);
        assert!((w.cfg.mutation.weight_standard_deviation - mw * 1.5).abs() < 1e-9);
        assert!((tune.mutation_scale - 1.5).abs() < 1e-12);
        assert!(tune.hud_line().contains("MUT X1.50"));
        tune.press(&mut w, "r");
        w.step().unwrap();
        assert!((w.cfg.sources.power_per_source - power).abs() < 1e-9);
        assert!((w.cfg.sources.path_speed - speed).abs() < 1e-9);
        assert!((w.cfg.mutation.weight_standard_deviation - mw).abs() < 1e-12);
        assert_eq!(tune.log.len(), 2 + 1 + 1 + 1 + 5 + 4 + 5);
    }

    #[test]
    fn drift_keys_act_on_the_selected_source_only() {
        let mut w = tuned_world();
        let mut tune = Tuner::new(&w.cfg, None);
        let before: Vec<_> = w.sources.iter().map(|s| s.chem).collect();
        tune.press_script(&mut w, "@1k");
        w.step().unwrap();
        assert_eq!(w.sources[0].chem, before[0]);
        assert_ne!(w.sources[1].chem, before[1]);
        assert_eq!(w.sources[2].chem, before[2]);
        tune.press_script(&mut w, "@j");
        assert_eq!(tune.source, None);
        w.step().unwrap();
        assert_ne!(w.sources[0].chem, before[0], "with nothing selected every source drifts");
        assert!(tune.hud_line().contains("CHEM ALL DRIFT 30 DEG"));
        tune.press_script(&mut w, "@9");
        assert_eq!(tune.source, None, "a source that does not exist is not selected");
    }

    #[test]
    fn clicking_near_a_source_center_picks_it() {
        let w = tuned_world();
        let centers = source_centers(&w);
        for (k, c) in centers.iter().enumerate() {
            assert_eq!(source_at(&w, c[0] + 0.1, c[1] - 0.1, 0.5), Some(k as u32));
            assert_eq!(source_at(&w, c[0] + 100.0, c[1], 0.5), None);
        }
    }

    #[test]
    fn hud_text_uses_only_drawable_glyphs() {
        let mut w = tuned_world();
        let mut tune = Tuner::new(&w.cfg, None);
        tune.press_script(&mut w, "@2wk");
        for line in [tune.hud_line(), "Q/W ENERGY  A/S SOURCE SPEED  D/F DIFFUSION  C/V DECAY  Z/X MUTATION  J/K CHEM DRIFT  CLICK SOURCE  R RESET".to_string()] {
            for c in line.chars().filter(|c| *c != ' ') {
                assert_ne!(glyph(c), "000000000000000", "no glyph for {c:?} in {line}");
            }
        }
    }

    #[test]
    fn viewer_steps_match_the_headless_world() {
        let cfg = tiny_config();
        let mut a = World::new(cfg.clone()).unwrap();
        let mut b = World::new(cfg).unwrap();
        for _ in 0..12 {
            step_world(&mut a, Backend::Cpu, #[cfg(feature = "gpu")] &mut None).unwrap();
            b.step().unwrap();
        }
        assert_eq!(a.state_hash(), b.state_hash());
        assert_eq!(a.orgs, b.orgs);
    }

    fn tiny_session() -> Session {
        use clap::Parser;
        let args = ViewArgs::parse_from(["view", "--out", "target/viewer-count-test"]);
        let world = World::new(tiny_config()).unwrap();
        let report = world.cfg.validate().unwrap();
        Session::from_prepared(args, Prepared { world, #[cfg(feature = "gpu")] gpu: None, ts_supported: None, report, start_tick: 0 })
    }

    fn typed(s: &mut Session, keys: &[Key]) {
        for k in keys {
            assert!(s.count_key(k), "the count entry takes the key");
        }
    }

    #[test]
    fn typing_a_count_restarts_the_run_with_that_many_organisms() {
        let mut s = tiny_session();
        for _ in 0..5 {
            s.step_one().unwrap();
        }
        assert_eq!(s.world.orgs.len(), 24);
        assert!(s.world.tick >= 5);
        s.count_entry = Some(String::new());
        typed(&mut s, &[Key::Character("3".into()), Key::Character("0".into()), Key::Character("9".into()), Key::Named(NamedKey::Backspace), Key::Character("1".into()), Key::Named(NamedKey::Enter)]);
        assert!(s.count_entry.is_none());
        assert_eq!(s.world.orgs.len(), 301, "3, 0, 9, backspace, 1 types 301");
        assert_eq!(s.world.tick, 0, "a fresh world");
        assert!(s.world.cfg.population.maximum_count >= 301, "room for the larger count");
        s.step_one().unwrap();
        assert_eq!(s.world.tick, 1);
    }

    #[test]
    fn a_restart_starts_timing_and_checkpoints_afresh() {
        use clap::Parser;
        let out = std::env::temp_dir().join(format!("fl-restart-test-{}", std::process::id()));
        let args = ViewArgs::parse_from(["view", "--out", out.to_str().unwrap(), "--checkpoint-every", "0.01"]);
        let world = World::new(tiny_config()).unwrap();
        let report = world.cfg.validate().unwrap();
        let mut s = Session::from_prepared(args, Prepared { world, #[cfg(feature = "gpu")] gpu: None, ts_supported: None, report, start_tick: 0 });
        for _ in 0..4 {
            s.step_one().unwrap();
        }
        s.pace.frames = 9;
        s.pace.present_ms = 3.0;
        s.pace.speed = Speed::Realtime(2.0);
        let before = s.ckpt_dir.clone();
        s.restart_with_count(40).unwrap();
        assert!(s.pace.sim_ms == 0.0 && s.pace.frames == 0 && s.pace.present_ms == 0.0, "timing counters start over");
        assert!(matches!(s.pace.speed, Speed::Realtime(m) if m == 2.0), "the chosen speed stays");
        assert_ne!(s.ckpt_dir, before, "the new run writes its checkpoints in its own folder");
        assert!(s.ckpt_dir.is_dir());
        std::fs::remove_dir_all(&out).ok();
    }

    #[test]
    fn escape_cancels_the_count_and_a_bad_count_is_refused() {
        let mut s = tiny_session();
        s.count_entry = Some(String::new());
        typed(&mut s, &[Key::Character("7".into()), Key::Named(NamedKey::Escape)]);
        assert!(s.count_entry.is_none());
        assert_eq!(s.world.orgs.len(), 24, "nothing changed");
        s.count_entry = Some(String::new());
        typed(&mut s, &[Key::Named(NamedKey::Enter)]);
        assert!(s.notice.is_some(), "an empty entry is refused");
        assert_eq!(s.world.orgs.len(), 24);
        assert!(!s.count_key(&Key::Character("q".into())), "keys are not taken when no count is being typed");
    }

    fn tiny_config() -> fl_core::Config {
        let mut c = fl_core::Config::reference();
        c.world.width = 16.0;
        c.world.height = 16.0;
        c.world.grid_width = 16;
        c.world.grid_height = 16;
        c.population.initial_count = 24;
        c.population.maximum_count = 48;
        c.population.initial_energy = 4.0;
        c.sources.count = 1;
        c.sources.power_per_source = 8.0;
        c.sources.sigma_world_units = 2.0;
        c.sensors.maximum_scale_world_units = 4.0;
        c.controller.hidden_widths = vec![8];
        c
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn field_preview_matches_fl_view_and_does_not_change_the_run() {
        let Ok(mut gpu_a) = fl_gpu::GpuBackend::new() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let Ok(mut gpu_b) = fl_gpu::GpuBackend::new() else {
            eprintln!("second GPU device unavailable; skipping");
            return;
        };
        gpu_a.set_single_wait(true);
        gpu_b.set_single_wait(true);
        let cfg = tiny_config();
        let mut a = World::new(cfg.clone()).unwrap();
        let mut b = World::new(cfg).unwrap();
        let mut painter = Painter::new(gpu_a.device().clone(), gpu_a.queue().clone()).unwrap();
        for _ in 0..6 {
            a.step_inference(fl_core::Inference::Resident(&mut gpu_a)).unwrap();
            b.step_inference(fl_core::Inference::Resident(&mut gpu_b)).unwrap();
            let planes = gpu_a.field_planes(&a.field, true).unwrap();
            let scene_planes = Planes::Device {
                energy: &planes.energy,
                moment: &planes.moment,
                width: planes.width,
                height: planes.height,
                cell_area: planes.cell_area,
                generation: planes.generation,
            };
            let empty = [];
            let scene = Scene {
                planes: scene_planes,
                raw_field: false,
                show_field: true,
                show_organisms: false,
                filmic: true,
                bloom: 0.0,
                background: [0.0; 3],
                world: [a.field.width() as f32, a.field.height() as f32],
                center: [a.field.width() as f32 * 0.5, a.field.height() as f32 * 0.5],
                zoom: 8.0,
                reach: 2.0,
                point: false,
                organisms: &empty,
                hud: &[],
                hud_size: (0, 0),
            };
            painter.render_rgba(&scene, (64, 64)).unwrap();
        }
        let colored = painter.read_field_rgb().unwrap();
        a.sync_field(&mut gpu_a).unwrap();
        b.sync_field(&mut gpu_b).unwrap();
        assert_eq!(a.state_hash(), b.state_hash(), "sampling the resident field for display changed the run");
        let cpu = fl_view::field_cell_rgb(&a.field, &chem_basis(), false);
        let mut worst = 0.0f32;
        for (i, (got, expect)) in colored.chunks_exact(3).zip(cpu.chunks_exact(3)).enumerate() {
            for k in 0..3 {
                worst = worst.max((got[k] - expect[k]).abs());
            }
            let _ = i;
        }
        assert!(worst < 2.0e-3, "field preview differs from fl-view by {worst}");
    }
}
