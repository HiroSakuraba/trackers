//! WebAssembly boundary for the browser Studio.
//!
//! The only non-safe-Rust construct in FrankenLife lives here: the
//! `#[no_mangle]` attribute that exports a symbol (the `unsafe_code` lint
//! counts it). No function in this crate performs an unsafe operation. Data
//! crosses the boundary through buffers owned by Rust whose addresses are
//! handed to JavaScript, which reads/writes the exported linear memory.
//!
//! The Studio only observes committed epochs and submits commands at tick
//! boundaries; it never mutates scientific state inside a tick.

#![deny(unsafe_code)]

use std::cell::RefCell;

use fl_core::world::{Command, Disposal};
use fl_core::{Config, World};
use fl_view::{chem_basis, chem_color, field_cell_rgb, lineage_color};
use serde_json::json;

const VIVID: &str = include_str!("../../../presets/vivid-field-ecology.toml");
const REFERENCE: &str = fl_core::config::REFERENCE_PRESET;

#[derive(Default)]
struct Studio {
    world: Option<World>,
    saved: Option<World>,
    input: Vec<u8>,
    out: Vec<u8>,
    image: Vec<u8>,
    particles: Vec<f32>,
    selected: Option<u64>,
    births: u64,
    deaths: u64,
}

thread_local! {
    static S: RefCell<Studio> = RefCell::new(Studio::default());
}

fn with<R>(f: impl FnOnce(&mut Studio) -> R) -> R {
    S.with(|s| f(&mut s.borrow_mut()))
}

fn set_out(s: &mut Studio, text: String) -> u32 {
    s.out = text.into_bytes();
    s.out.len() as u32
}

macro_rules! export {
    ($($item:item)*) => { $( #[allow(unsafe_code)] #[no_mangle] $item )* };
}

/// Build a configuration for a named preset with Studio-scale overrides.
pub fn preset_config(preset: u32, initial_count: u32, topology: u32, seed: u32) -> Result<Config, String> {
    let mut c = Config::from_toml(if preset == 1 { REFERENCE } else { VIVID })?;
    c.seed = seed as u64;
    if initial_count > 0 {
        c.population.initial_count = initial_count;
        c.population.maximum_count = c.population.maximum_count.max(initial_count * 4);
    }
    let hidden: &[u32] = match topology {
        0 => &[8],
        1 => &[32],
        2 => &[64, 32],
        3 => &[96, 64],
        _ => &[128, 96],
    };
    c.controller.hidden_widths = hidden.to_vec();
    // Browser runs are single-threaded and never pause on capacity.
    c.execution.cpu_workers = 1;
    c.population.capacity_policy = "reject-births-stable-order".into();
    c.validate().map_err(|e| e.join("; "))?;
    Ok(c)
}

export! {
    pub extern "C" fn input_ptr(len: u32) -> u32 {
        with(|s| {
            s.input.clear();
            s.input.resize(len as usize, 0);
            s.input.as_ptr() as u32
        })
    }

    pub extern "C" fn out_ptr() -> u32 {
        with(|s| s.out.as_ptr() as u32)
    }

    pub extern "C" fn out_len() -> u32 {
        with(|s| s.out.len() as u32)
    }

    /// Create a world from a preset. Returns 1 on success, 0 with a message in `out`.
    pub extern "C" fn world_preset(preset: u32, initial_count: u32, topology: u32, seed: u32) -> u32 {
        with(|s| match preset_config(preset, initial_count, topology, seed).and_then(|c| World::new(c).map_err(|e| e.join("; "))) {
            Ok(w) => {
                s.world = Some(w);
                s.saved = None;
                s.selected = None;
                s.births = 0;
                s.deaths = 0;
                1
            }
            Err(e) => {
                set_out(s, e);
                0
            }
        })
    }

    /// Create a world from a TOML configuration previously written to the input buffer.
    pub extern "C" fn world_from_input() -> u32 {
        with(|s| {
            let text = String::from_utf8_lossy(&s.input).to_string();
            match Config::from_toml(&text).and_then(|mut c| {
                c.execution.cpu_workers = 1;
                World::new(c).map_err(|e| e.join("; "))
            }) {
                Ok(w) => {
                    s.world = Some(w);
                    s.saved = None;
                    s.selected = None;
                    1
                }
                Err(e) => {
                    set_out(s, e);
                    0
                }
            }
        })
    }

    /// Advance up to `n` ticks. Returns ticks committed; on error the message is in `out`.
    pub extern "C" fn step(n: u32) -> u32 {
        with(|s| {
            let Some(w) = s.world.as_mut() else { return 0 };
            let mut done = 0;
            for _ in 0..n {
                if w.orgs.is_empty() {
                    break;
                }
                match w.step() {
                    Ok(r) => {
                        s.births += r.counts.births as u64;
                        s.deaths += (r.counts.deaths_exhaustion + r.counts.deaths_user) as u64;
                        done += 1;
                    }
                    Err(e) => {
                        let msg = e.to_string();
                        s.out = msg.into_bytes();
                        break;
                    }
                }
            }
            done
        })
    }

    pub extern "C" fn world_width() -> f64 {
        with(|s| s.world.as_ref().map_or(0.0, |w| w.field.width()))
    }

    pub extern "C" fn world_height() -> f64 {
        with(|s| s.world.as_ref().map_or(0.0, |w| w.field.height()))
    }

    pub extern "C" fn field_w() -> u32 {
        with(|s| s.world.as_ref().map_or(0, |w| w.field.gx as u32))
    }

    pub extern "C" fn field_h() -> u32 {
        with(|s| s.world.as_ref().map_or(0, |w| w.field.gy as u32))
    }

    /// RGBA field image (one pixel per cell, sRGB). `view`: 0 chemistry glow, 1 raw energy, 2 none.
    pub extern "C" fn field_image(view: u32) -> u32 {
        with(|s| {
            let Some(w) = s.world.as_ref() else { return 0 };
            let cells = w.field.cells();
            s.image.resize(cells * 4, 0);
            let bg = [0.008f32, 0.012, 0.031];
            if view == 2 {
                for c in 0..cells {
                    s.image[c * 4..c * 4 + 4].copy_from_slice(&[2, 3, 8, 255]);
                }
                return s.image.as_ptr() as u32;
            }
            let rgb = field_cell_rgb(&w.field, &chem_basis(), view == 1);
            for c in 0..cells {
                for k in 0..3 {
                    let v = rgb[c * 3 + k];
                    let t = if view == 1 { v.min(1.0) } else { 1.0 - (-v * 1.1).exp() };
                    let lin = bg[k] + (1.0 - bg[k]) * t;
                    s.image[c * 4 + k] = (lin.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8;
                }
                s.image[c * 4 + 3] = 255;
            }
            s.image.as_ptr() as u32
        })
    }

    /// Particle buffer: per organism `[x, y, r, g, b, energy]` (f32). `color`: 0 lineage, 1 affinity.
    pub extern "C" fn particles(color: u32) -> u32 {
        with(|s| {
            let Some(w) = s.world.as_ref() else { return 0 };
            let o = &w.orgs;
            let basis = chem_basis();
            s.particles.clear();
            s.particles.reserve(o.len() * 6);
            for i in 0..o.len() {
                let c = if color == 1 { chem_color(&o.genome[i].affinity, &basis) } else { lineage_color(o.lineage[i]) };
                s.particles.extend_from_slice(&[o.x[i] as f32, o.y[i] as f32, c[0], c[1], c[2], o.energy[i] as f32]);
            }
            s.particles.as_ptr() as u32
        })
    }

    pub extern "C" fn particle_count() -> u32 {
        with(|s| s.world.as_ref().map_or(0, |w| w.orgs.len() as u32))
    }

    /// Telemetry JSON into `out`; returns its length.
    pub extern "C" fn stats() -> u32 {
        with(|s| {
            let Some(w) = s.world.as_ref() else { return 0 };
            let o = &w.orgs;
            let mut lineages: Vec<u64> = o.lineage.clone();
            lineages.sort_unstable();
            lineages.dedup();
            let mut topo = std::collections::BTreeMap::<String, usize>::new();
            for g in &o.genome {
                *topo.entry(format!("{:?}", g.hidden())).or_default() += 1;
            }
            let n = o.len().max(1) as f64;
            let sel = s.selected.and_then(|id| o.index_of(id)).map(|i| json!({"id": o.id[i], "x": o.x[i], "y": o.y[i]}));
            let l = &w.ledger;
            let text = json!({
                "tick": w.tick,
                "sim_seconds": w.simulation_seconds(),
                "dt": w.dt(),
                "population": o.len(),
                "births": s.births,
                "deaths": s.deaths,
                "lineages": lineages.len(),
                "max_generation": o.generation.iter().max().copied().unwrap_or(0),
                "mean_generation": o.generation.iter().map(|&g| g as f64).sum::<f64>() / n,
                "mean_energy": o.energy.iter().sum::<f64>() / n,
                "mean_speed": (0..o.len()).map(|i| (o.vx[i] * o.vx[i] + o.vy[i] * o.vy[i]).sqrt()).sum::<f64>() / n,
                "mean_params": o.genome.iter().map(|g| g.param_count() as f64).sum::<f64>() / n,
                "topologies": topo,
                "field_energy": w.field.total_energy(),
                "organism_energy": o.energy.iter().sum::<f64>(),
                "injected": l.injected,
                "dissipated": l.dissipated.total(),
                "residual": l.residual,
                "pending_commands": w.pending.len(),
                "has_snapshot": s.saved.is_some(),
                "snapshot_tick": s.saved.as_ref().map(|x| x.tick),
                "selected": sel,
                "seed": w.cfg.seed,
                "name": w.cfg.name,
            })
            .to_string();
            set_out(s, text)
        })
    }

    /// Select the organism nearest to (x, y) within `radius` world units; returns 1 if found.
    pub extern "C" fn pick(x: f64, y: f64, radius: f64) -> u32 {
        with(|s| {
            let Some(w) = s.world.as_ref() else { return 0 };
            let o = &w.orgs;
            let (ww, hh) = (w.field.width(), w.field.height());
            let mut best: Option<(f64, u64)> = None;
            for i in 0..o.len() {
                let dx = fl_core::field::min_image(o.x[i] - x, ww);
                let dy = fl_core::field::min_image(o.y[i] - y, hh);
                let d = dx * dx + dy * dy;
                if d <= radius * radius && best.is_none_or(|b| d < b.0) {
                    best = Some((d, o.id[i]));
                }
            }
            s.selected = best.map(|b| b.1);
            s.selected.is_some() as u32
        })
    }

    pub extern "C" fn clear_selection() {
        with(|s| s.selected = None);
    }

    /// Inspection JSON of the selected organism into `out`; returns its length (0 if none/dead).
    pub extern "C" fn inspect_selected() -> u32 {
        with(|s| {
            let Some(w) = s.world.as_ref() else { return 0 };
            match s.selected.and_then(|id| w.inspect(id)) {
                Some(i) => {
                    let t = serde_json::to_string(&i).unwrap_or_default();
                    set_out(s, t)
                }
                None => {
                    s.out.clear();
                    0
                }
            }
        })
    }

    /// Erase organisms in a disk at the current tick boundary (energy recycled to the field).
    pub extern "C" fn erase(x: f64, y: f64, radius: f64) -> u32 {
        with(|s| {
            let Some(w) = s.world.as_mut() else { return 0 };
            w.schedule(w.tick, Command::EraseDisk { x, y, radius, disposal: Disposal::Recycle }).is_ok() as u32
        })
    }

    /// Inject `amount` neutral energy into a disk at the current tick boundary.
    pub extern "C" fn energy_brush(x: f64, y: f64, radius: f64, amount: f64) -> u32 {
        with(|s| {
            let Some(w) = s.world.as_mut() else { return 0 };
            w.schedule(w.tick, Command::EnergyBrush { x, y, radius, amount, direction: None }).is_ok() as u32
        })
    }

    /// Keep an exact in-memory copy of the committed world.
    pub extern "C" fn snapshot_save() -> u32 {
        with(|s| {
            s.saved = s.world.clone();
            s.saved.is_some() as u32
        })
    }

    /// Return to the saved world (branch from it again).
    pub extern "C" fn snapshot_restore() -> u32 {
        with(|s| match s.saved.clone() {
            Some(w) => {
                s.world = Some(w);
                1
            }
            None => 0,
        })
    }

    pub extern "C" fn state_hash_hi() -> u32 {
        with(|s| s.world.as_ref().map_or(0, |w| (w.state_hash() >> 32) as u32))
    }

    pub extern "C" fn state_hash_lo() -> u32 {
        with(|s| s.world.as_ref().map_or(0, |w| w.state_hash() as u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn studio_flow_is_consistent() {
        assert_eq!(world_preset(0, 256, 0, 7), 1);
        assert!(step(20) == 20);
        assert!(stats() > 0);
        assert!(particle_count() > 0);
        assert!(field_image(0) != 0);
        let (x, y) = with(|s| {
            let w = s.world.as_ref().unwrap();
            (w.orgs.x[0], w.orgs.y[0])
        });
        assert_eq!(pick(x, y, 1.0), 1);
        assert!(inspect_selected() > 0);
        assert_eq!(snapshot_save(), 1);
        let h = (state_hash_hi(), state_hash_lo());
        erase(x, y, 10.0);
        step(10);
        assert_eq!(snapshot_restore(), 1);
        assert_eq!((state_hash_hi(), state_hash_lo()), h, "restore returns to the exact saved epoch");
        assert_eq!(world_preset(1, 0, 2, 1), 1, "reference preset builds");
    }
}
