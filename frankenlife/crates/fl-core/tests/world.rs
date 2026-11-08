//! World-level fixtures: ledger closure, determinism, view neutrality,
//! exhaustion ordering, cadence, capacity transactions and interventions.

use fl_core::world::{Command, Disposal};
use fl_core::{Config, TickError, World};

/// A small, fast world that reproduces quickly.
fn small() -> Config {
    let mut c = Config::reference();
    c.world.width = 32.0;
    c.world.height = 32.0;
    c.world.grid_width = 32;
    c.world.grid_height = 32;
    c.population.initial_count = 256;
    c.population.maximum_count = 4096;
    c.population.initial_energy = 4.5;
    c.sources.count = 3;
    c.sources.power_per_source = 20.0;
    c.sources.sigma_world_units = 4.0;
    c.sensors.maximum_scale_world_units = 8.0;
    c.reproduction.maximum_attempt_rate_per_second = 4.0;
    c.reproduction.cooldown_seconds = 0.25;
    c.controller.hidden_widths = vec![8];
    c
}

fn run(c: Config, ticks: u64, workers: usize) -> World {
    let mut w = World::new(c).expect("valid config");
    w.workers = workers;
    for _ in 0..ticks {
        w.step().expect("tick");
    }
    w
}

#[test]
fn ledger_closes_and_ecology_turns_over() {
    let w = run(small(), 400, 1);
    let l = &w.ledger;
    let expected = l.initial_endowment + l.injected - l.removed - l.dissipated.total();
    let actual = w.total_energy();
    let rel = (actual - expected).abs() / l.transferred.max(1.0);
    assert!(rel < 1e-5, "cumulative residual {} (actual {actual}, expected {expected})", actual - expected);
    assert!((l.residual - (actual - expected)).abs() < 1e-6);
    assert!(w.totals.births > 0, "births happened");
    assert!(w.totals.deaths_exhaustion + w.totals.births > 0);
    assert!(w.totals.reproduction_attempts > 0);
    assert!(l.dissipated.uptake_inefficiency > 0.0 && l.dissipated.maintenance > 0.0);
    // Children carry their own genomes.
    let o = &w.orgs;
    if let Some(i) = (0..o.len()).find(|&i| o.generation[i] > 0) {
        let g = &o.genome[i];
        assert!(g.parent.is_some());
        assert_ne!(g.id, o.genome[0].id);
        assert!(o.energy[i] > 0.0);
    }
}

#[test]
fn deterministic_and_worker_count_neutral() {
    let a = run(small(), 120, 1);
    let b = run(small(), 120, 1);
    let c = run(small(), 120, 4);
    assert_eq!(a.state_hash(), b.state_hash());
    assert_eq!(a.state_hash(), c.state_hash(), "executor partitioning must not change science");
    let mut other = small();
    other.seed = 7;
    assert_ne!(a.state_hash(), run(other, 120, 1).state_hash());
}

#[test]
fn c7_inspection_is_view_neutral() {
    let mut a = World::new(small()).unwrap();
    let mut b = World::new(small()).unwrap();
    for t in 0..60 {
        a.step().unwrap();
        b.step().unwrap();
        if t % 7 == 0 {
            let id = b.orgs.id[b.orgs.len() / 2];
            let _ = b.inspect(id).unwrap();
        }
    }
    assert_eq!(a.state_hash(), b.state_hash());
}

fn c12_world() -> Config {
    let mut c = Config::reference();
    c.world.width = 8.0;
    c.world.height = 8.0;
    c.world.grid_width = 4;
    c.world.grid_height = 4;
    c.world.tick_numerator = 1;
    c.world.tick_denominator = 1;
    c.world.diffusion = 0.0;
    c.world.decay_per_second = 0.0;
    c.population.initial_count = 1;
    c.population.maximum_count = 4;
    c.population.initial_energy = 1.0;
    c.sources.count = 0;
    c.sensors.minimum_scale_world_units = 2.0;
    c.sensors.initial_scale_world_units = 2.0;
    c.sensors.maximum_scale_world_units = 4.0;
    c.chemistry.drag_per_second = 0.0;
    c.chemistry.maximum_speed = 1.0;
    c.costs.basal_energy_per_second = 0.0;
    c.costs.sensor_energy_per_second_at_unit_scale = 0.0;
    c.costs.neural_energy_per_parameter_evaluation = 0.0;
    c.costs.motion_energy_per_speed_squared_second = 1.0;
    c
}

#[test]
fn c12_exhaustion_after_motion() {
    let mut w = World::new(c12_world()).unwrap();
    w.orgs.vx[0] = 1.0;
    let x0 = w.orgs.x[0];
    let r = w.step().unwrap();
    assert_eq!(r.population, 0, "exhausted organism is absent from the committed population");
    assert_eq!(r.counts.deaths_exhaustion, 1);
    assert_eq!(r.counts.emitting, 0);
    assert_eq!(r.counts.births, 0);
    assert_eq!(r.ledger.dissipated.motion, 1.0);
    assert_eq!(r.ledger.dissipated.total(), 1.0);
    assert!(r.ledger.residual.abs() < 1e-12);
    // Resume with field energy available: the dead identity never reappears.
    w.schedule(1, Command::EnergyBrush { x: x0 + 1.0, y: w.field.height() / 2.0, radius: 8.0, amount: 10.0, direction: None }).unwrap();
    let r = w.step().unwrap();
    assert_eq!(r.population, 0);
    assert!(w.orgs.index_of(0).is_none());
    assert!((w.total_energy() - 10.0).abs() < 1e-5);
}

#[test]
fn c12_exhaustion_at_maintenance() {
    let mut c = c12_world();
    c.costs.basal_energy_per_second = 2.0; // more than the organism holds
    let mut w = World::new(c).unwrap();
    let r = w.step().unwrap();
    assert_eq!(r.population, 0);
    assert_eq!(r.ledger.dissipated.maintenance, 1.0, "only the actually paid cost is dissipated");
    assert!(r.ledger.residual.abs() < 1e-12);
}

#[test]
fn c13_newborn_forced_evaluation_and_stable_phase() {
    let mut c = small();
    c.controller.evaluation_interval_ticks = 4;
    let mut w = World::new(c).unwrap();
    // Founders: first advance forces evaluation for everyone.
    let r = w.step().unwrap();
    assert_eq!(r.counts.controller_evaluations, 256);
    let mut checked = false;
    for _ in 0..400 {
        let born_before: Vec<u64> = w.orgs.id.clone();
        let r = w.step().unwrap();
        let n = w.tick; // committed tick b
        for i in 0..w.orgs.len() {
            let o = &w.orgs;
            if o.birth_tick[i] == n {
                assert!(!o.held_valid[i], "newborn starts with invalid held output");
                assert_eq!(o.energy_change_rate[i], 0.0);
                assert_eq!(o.vx[i], 0.0);
                assert_eq!(o.phase[i] as u64, o.id[i] % 4);
                assert!(!born_before.contains(&o.id[i]));
            }
        }
        if r.counts.births > 0 {
            let newborn: Vec<u64> = (0..w.orgs.len()).filter(|&i| w.orgs.birth_tick[i] == n).map(|i| w.orgs.id[i]).collect();
            w.step().unwrap();
            for id in newborn {
                if let Some(i) = w.orgs.index_of(id) {
                    assert!(w.orgs.held_valid[i]);
                    assert_eq!(w.orgs.held_tick[i], n + 1, "forced first evaluation on advance to b+1");
                    checked = true;
                }
            }
        }
        // Regular cadence: every held output was evaluated at a phase tick or a forced first tick.
        for i in 0..w.orgs.len() {
            let o = &w.orgs;
            if o.held_valid[i] && o.held_tick[i] != o.birth_tick[i] + 1 {
                assert_eq!(o.held_tick[i] % 4, o.phase[i] as u64);
            }
        }
        if checked {
            break;
        }
    }
    assert!(checked, "observed at least one newborn's forced evaluation");
}

#[test]
fn capacity_pause_leaves_committed_world_untouched() {
    let mut c = small();
    c.population.maximum_count = 256; // births immediately exceed the quota
    let mut w = World::new(c).unwrap();
    loop {
        let before = w.clone();
        match w.step() {
            Ok(_) => continue,
            Err(TickError::CapacityPaused { proposed_births, .. }) => {
                assert!(proposed_births > 0);
                assert_eq!(w, before, "candidate discarded");
                break;
            }
            Err(e) => panic!("{e}"),
        }
    }
    // Exhibition policy: same quota enforced by stable-order rejection.
    let mut c = small();
    c.population.maximum_count = 256;
    c.population.capacity_policy = "reject-births-stable-order".into();
    let w = run(c, 300, 1);
    assert!(w.orgs.len() <= 256);
    assert!(w.totals.rejected_births_capacity > 0);
}

#[test]
fn erase_brush_records_its_disposal() {
    let mut w = World::new(small()).unwrap();
    for _ in 0..5 {
        w.step().unwrap();
    }
    let before = w.orgs.len();
    w.schedule(w.tick, Command::EraseDisk { x: 16.0, y: 16.0, radius: 8.0, disposal: Disposal::Remove }).unwrap();
    let r = w.step().unwrap();
    assert!(r.counts.deaths_user > 0);
    assert!(r.ledger.removed > 0.0);
    assert!(w.orgs.len() < before);
    assert!(r.ledger.residual.abs() / r.ledger.transferred.max(1.0) < 1e-5);
    for i in 0..w.orgs.len() {
        let dx = fl_core::field::min_image(w.orgs.x[i] - 16.0, 32.0);
        let dy = fl_core::field::min_image(w.orgs.y[i] - 16.0, 32.0);
        // Survivors inside the disk can only be newborns of this tick or movers.
        if dx * dx + dy * dy < 7.0 * 7.0 {
            assert_eq!(w.orgs.birth_tick[i], w.tick);
        }
    }
}

#[test]
fn c15_integer_ticks_keep_aging_past_f32_horizon() {
    let mut w = World::new(small()).unwrap();
    w.step().unwrap();
    w.tick = 16_777_216 + 1;
    for s in &mut w.sources {
        s.ordinal = 0;
    }
    // Sources roll forward lazily; organisms age by integer ticks.
    let id = w.orgs.id[0];
    let i = w.inspect(id).unwrap();
    assert!((i.age_seconds - (w.tick - i.birth_tick) as f64 / 128.0).abs() < 1e-9);
    assert!(i.age_seconds > 131_072.0, "beyond the F32 accumulation horizon");
    w.step().unwrap();
    let i2 = w.inspect(id).map(|v| v.age_seconds).unwrap_or(i.age_seconds + 1.0 / 128.0);
    assert!(i2 > i.age_seconds);
}

#[test]
fn shipped_presets_validate() {
    for p in ["reference-field-ecology.toml", "vivid-field-ecology.toml", "target-20k-field-ecology.toml", "target-80k-field-ecology.toml"] {
        let path = format!("{}/../../presets/{p}", env!("CARGO_MANIFEST_DIR"));
        let c = Config::from_toml(&std::fs::read_to_string(&path).unwrap()).unwrap();
        c.validate().unwrap_or_else(|e| panic!("{p}: {e:?}"));
    }
}

#[test]
fn reference_backend_through_the_port_matches_built_in_path() {
    let mut a = World::new(small()).unwrap();
    let mut b = World::new(small()).unwrap();
    let mut be = fl_core::ReferenceBackend;
    for _ in 0..80 {
        a.step().unwrap();
        b.step_with(Some(&mut be)).unwrap();
    }
    assert_eq!(a.state_hash(), b.state_hash(), "same arithmetic, same science");
}

#[test]
fn reference_fused_port_matches_built_in_path() {
    let mut a = World::new(small()).unwrap();
    let mut b = World::new(small()).unwrap();
    let mut be = fl_core::ReferenceSenseInfer;
    for _ in 0..80 {
        a.step().unwrap();
        b.step_inference(fl_core::Inference::Fused(&mut be)).unwrap();
    }
    assert!(a.totals.births > 0);
    assert_eq!(a.state_hash(), b.state_hash(), "fused port with reference arithmetic reproduces the built-in path");
}
