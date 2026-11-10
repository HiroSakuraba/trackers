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
fn other_cpu_parts_at_resident_population() {
    let mut c = small();
    c.population.initial_count = 20_000;
    c.population.maximum_count = 24_000;
    c.world.grid_width = 64;
    c.world.grid_height = 64;
    c.world.width = 64.0;
    c.world.height = 64.0;
    c.sensors.maximum_scale_world_units = 8.0;
    c.sources.sigma_world_units = 4.0;
    let ticks = 4u64;
    for workers in [1usize, 4] {
        let mut w = World::new(c.clone()).expect("valid");
        w.workers = workers;
        w.set_profiling(true);
        let mut be = fl_core::ReferenceResident::default();
        for _ in 0..ticks {
            w.step_inference(fl_core::Inference::Resident(&mut be)).unwrap();
        }
        let p = w.stage_profile();
        let ms = |ns: u64| ns as f64 / 1e6 / ticks as f64;
        eprintln!(
            "resident-parts workers {workers} pop {} | setup {:.3} scatter {:.3} held {:.3} uptake {:.3} motion {:.3} commit {:.3} publish {:.3} | other {:.3}",
            w.orgs.len(),
            ms(p.other_part_ns(fl_core::profile::OtherPart::Setup)),
            ms(p.other_part_ns(fl_core::profile::OtherPart::SenseScatter)),
            ms(p.other_part_ns(fl_core::profile::OtherPart::HeldStore)),
            ms(p.other_part_ns(fl_core::profile::OtherPart::UptakeScatter)),
            ms(p.other_part_ns(fl_core::profile::OtherPart::LifecycleMotion)),
            ms(p.other_part_ns(fl_core::profile::OtherPart::LifecycleCommit)),
            ms(p.other_part_ns(fl_core::profile::OtherPart::Publish)),
            ms(p.stage_ns(fl_core::CpuStage::Other)),
        );
        let parts = fl_core::profile::OtherPart::ALL.into_iter().map(|part| p.other_part_ns(part)).sum::<u64>();
        assert_eq!(parts, p.stage_ns(fl_core::CpuStage::Other), "other parts must sum to the other stage");
    }
}

#[test]
fn profiling_does_not_change_results() {
    let mut plain = World::new(small()).unwrap();
    let mut profiled = World::new(small()).unwrap();
    let mut profiled_workers = World::new(small()).unwrap();
    profiled.set_profiling(true);
    profiled_workers.set_profiling(true);
    profiled_workers.workers = 4;
    let mut resident = World::new(small()).unwrap();
    resident.set_profiling(true);
    let mut be = fl_core::ReferenceResident::default();
    for _ in 0..40 {
        plain.step().unwrap();
        profiled.step().unwrap();
        profiled_workers.step().unwrap();
        resident.step_inference(fl_core::Inference::Resident(&mut be)).unwrap();
    }
    resident.sync_field(&mut be).unwrap();
    assert_eq!(plain.state_hash(), profiled.state_hash());
    assert_eq!(plain.state_hash(), profiled_workers.state_hash());
    assert_eq!(plain.state_hash(), resident.state_hash());
    assert_eq!(plain.ledger, profiled.ledger);
    assert_eq!(plain.ledger, resident.ledger);
    assert_eq!(plain.orgs, profiled.orgs);
    assert_eq!(plain.orgs, resident.orgs);
    assert!(!plain.stage_profile().is_enabled());
    assert_eq!(plain.stage_profile().stage_ns(fl_core::CpuStage::Sensing), 0);
    assert_eq!(plain.stage_profile().ticks(), 0);
    let p = profiled.stage_profile();
    assert!(p.is_enabled());
    assert_eq!(p.ticks(), 40);
    assert!(p.stage_ns(fl_core::CpuStage::Sensing) > 0, "sensing wall");
    assert!(p.stage_ns(fl_core::CpuStage::ChemistryField) > 0, "chemistry/field wall");
    assert!(p.stage_ns(fl_core::CpuStage::Networks) > 0, "cpu network forward");
    let r = resident.stage_profile();
    assert!(r.stage_ns(fl_core::CpuStage::TransfersSync) > 0, "resident copies and submits");
    assert!(r.stage_ns(fl_core::CpuStage::ExternalEvaluate) > 0, "fused backend wall");
    assert_eq!(r.ticks(), 40);
}

#[test]
fn deterministic_and_worker_count_neutral() {
    let a = run(small(), 120, 1);
    let b = run(small(), 120, 1);
    let c = run(small(), 120, 4);
    assert_eq!(a.state_hash(), b.state_hash());
    assert_eq!(a.state_hash(), c.state_hash(), "executor partitioning must not change science");
    assert_eq!(a.ledger, c.ledger, "worker split must not change the ledger");
    assert_eq!(a.orgs, c.orgs, "worker split must not change organisms");
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
    for p in ["reference-field-ecology.toml", "vivid-field-ecology.toml", "target-10k-field-ecology.toml", "target-20k-field-ecology.toml", "target-80k-field-ecology.toml", "lively-field-ecology.toml", "lively-evolving-field-ecology.toml", "lively-recurrent-field-ecology.toml"] {
        let path = format!("{}/../../presets/{p}", env!("CARGO_MANIFEST_DIR"));
        let c = Config::from_toml(&std::fs::read_to_string(&path).unwrap()).unwrap();
        c.validate().unwrap_or_else(|e| panic!("{p}: {e:?}"));
    }
    let root = format!("{}/../../presets", env!("CARGO_MANIFEST_DIR"));
    let mut t10 = Config::from_toml(&std::fs::read_to_string(format!("{root}/target-10k-field-ecology.toml")).unwrap()).unwrap();
    let t20 = Config::from_toml(&std::fs::read_to_string(format!("{root}/target-20k-field-ecology.toml")).unwrap()).unwrap();
    assert_eq!(t10.population.initial_count, 10_000);
    assert_eq!(t10.population.maximum_count, 12_000);
    assert_eq!(t20.population.initial_count, 20_000);
    assert_eq!(t20.population.maximum_count, 24_000);
    t10.name = t20.name.clone();
    t10.population.initial_count = t20.population.initial_count;
    t10.population.maximum_count = t20.population.maximum_count;
    assert_eq!(t10, t20, "10k preset is the 20k preset with organism counts halved");
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

#[test]
fn reference_device_port_matches_built_in_path() {
    let mut a = World::new(small()).unwrap();
    let mut b = World::new(small()).unwrap();
    let mut be = fl_core::ReferenceDevice;
    for _ in 0..80 {
        a.step().unwrap();
        b.step_inference(fl_core::Inference::Device(&mut be)).unwrap();
    }
    assert!(a.totals.births > 0);
    assert_eq!(a.state_hash(), b.state_hash(), "device port with reference arithmetic reproduces the built-in path");
    assert_eq!(a.ledger, b.ledger);
}

#[test]
fn reference_resident_port_matches_built_in_path() {
    let mut a = World::new(small()).unwrap();
    let mut b = World::new(small()).unwrap();
    for w in [&mut a, &mut b] {
        w.schedule(20, Command::EraseDisk { x: 16.0, y: 16.0, radius: 5.0, disposal: Disposal::Recycle }).unwrap();
        w.schedule(30, Command::EnergyBrush { x: 5.0, y: 9.0, radius: 3.0, amount: 7.5, direction: None }).unwrap();
    }
    let mut be = fl_core::ReferenceResident::default();
    for _ in 0..120 {
        a.step().unwrap();
        b.step_inference(fl_core::Inference::Resident(&mut be)).unwrap();
    }
    assert!(b.field_on_device());
    assert!(b.step().is_err(), "a CPU step while the field is on the device is refused");
    b.sync_field(&mut be).unwrap();
    assert!(a.totals.births > 0 && a.totals.deaths_user > 0);
    assert_eq!(a.state_hash(), b.state_hash(), "resident port with reference arithmetic reproduces the built-in path");
    assert_eq!(a.ledger, b.ledger);
    b.detach_device(&mut be).unwrap();
    a.step().unwrap();
    b.step().unwrap();
    assert_eq!(a.state_hash(), b.state_hash(), "detached world continues on the CPU");
}

#[test]
fn resident_capacity_pause_keeps_committed_field() {
    let mut c = small();
    c.population.maximum_count = 256;
    let mut w = World::new(c).unwrap();
    let mut be = fl_core::ReferenceResident::default();
    loop {
        match w.step_inference(fl_core::Inference::Resident(&mut be)) {
            Ok(_) => continue,
            Err(TickError::CapacityPaused { .. }) => break,
            Err(e) => panic!("{e}"),
        }
    }
    let tick = w.tick;
    let mut f1 = w.field.clone();
    w.sync_field(&mut be).unwrap();
    f1.clone_from(&w.field);
    // Retrying the same tick after the pause starts from the same committed field.
    let mut be2 = be.clone();
    let _ = w.step_inference(fl_core::Inference::Resident(&mut be2));
    assert_eq!(w.tick, tick);
    let mut f2 = f1.clone();
    w.sync_field(&mut be2).unwrap();
    f2.clone_from(&w.field);
    assert_eq!(f1, f2);
}

// ── Live tuning commands ─────────────────────────────────────────────────

use fl_core::world::Param;

#[test]
fn live_source_power_takes_effect_on_the_next_tick_and_stays() {
    let mut w = World::new(small()).unwrap();
    for _ in 0..5 {
        w.step().unwrap();
    }
    let before = w.step().unwrap().ledger.injected_sources;
    w.schedule(w.tick, Command::SetParam { param: Param::SourcePower, value: 40.0 }).unwrap();
    let after = w.step().unwrap().ledger.injected_sources;
    assert!((after / before - 2.0).abs() < 1e-6, "20 -> 40 per source doubles the injection: {before} -> {after}");
    assert_eq!(w.cfg.sources.power_per_source, 40.0);
    let later = w.step().unwrap().ledger.injected_sources;
    assert!((later / after - 1.0).abs() < 1e-6, "the change persists");
    let l = &w.ledger;
    let expected = l.initial_endowment + l.injected - l.removed - l.dissipated.total();
    assert!((w.total_energy() - expected).abs() / l.transferred.max(1.0) < 1e-5, "the ledger still closes");
}

#[test]
fn live_tuning_does_not_change_a_run_that_has_none() {
    let a = run(small(), 60, 1);
    let mut b = World::new(small()).unwrap();
    // Setting a parameter to the value it already has is a no-op for the science.
    b.schedule(10, Command::SetParam { param: Param::Decay, value: b.cfg.world.decay_per_second }).unwrap();
    for _ in 0..60 {
        b.step().unwrap();
    }
    assert_eq!(a.state_hash(), b.state_hash());
}

#[test]
fn live_chemistry_drift_moves_only_the_chosen_source() {
    let mut a = World::new(small()).unwrap();
    let mut b = World::new(small()).unwrap();
    let chem0: Vec<_> = a.sources.iter().map(|s| s.chem).collect();
    for w in [&mut a, &mut b] {
        w.schedule(3, Command::DriftChemistry { source: Some(1), degrees: 20.0 }).unwrap();
        for _ in 0..4 {
            w.step().unwrap();
        }
    }
    assert_eq!(a.sources[0].chem, chem0[0]);
    assert_eq!(a.sources[2].chem, chem0[2]);
    let dot: f64 = (0..32).map(|k| a.sources[1].chem[k] as f64 * chem0[1][k] as f64).sum();
    assert!((dot - 20f64.to_radians().cos()).abs() < 1e-5, "rotated by 20 degrees, dot {dot}");
    assert_eq!(a.sources, b.sources, "the drift is deterministic");
    // All sources at once.
    let mut c = World::new(small()).unwrap();
    c.schedule(0, Command::DriftChemistry { source: None, degrees: 10.0 }).unwrap();
    c.step().unwrap();
    for (s, old) in c.sources.iter().zip(&chem0) {
        let dot: f64 = (0..32).map(|k| s.chem[k] as f64 * old[k] as f64).sum();
        assert!((dot - 10f64.to_radians().cos()).abs() < 1e-5);
    }
}

#[test]
fn live_source_speed_change_does_not_jump_the_sources() {
    let mut w = World::new(small()).unwrap();
    for _ in 0..40 {
        w.step().unwrap();
    }
    let centers = |w: &World, speed: f64| -> Vec<[f64; 2]> {
        let mut cfg = w.cfg.clone();
        cfg.sources.path_speed = speed;
        w.sources
            .iter()
            .map(|s| {
                let mut s = s.clone();
                s.advance_to(w.tick, &cfg, &w.rng);
                s.center_mid(w.tick, &cfg)
            })
            .collect()
    };
    let old = w.cfg.sources.path_speed;
    let before = centers(&w, old);
    w.schedule(w.tick, Command::SetParam { param: Param::SourceSpeed, value: old * 4.0 }).unwrap();
    w.step().unwrap();
    let after = centers(&w, old * 4.0);
    let (ww, wh) = (w.field.width(), w.field.height());
    for (a, b) in before.iter().zip(&after) {
        let d = (fl_core::field::min_image(a[0] - b[0], ww).powi(2) + fl_core::field::min_image(a[1] - b[1], wh).powi(2)).sqrt();
        assert!(d < old * 4.0 * w.cfg.dt() * 2.0, "source moved {d} units in one tick");
    }
}

#[test]
fn live_tuning_rejects_values_the_configuration_would_reject() {
    let mut w = World::new(small()).unwrap();
    assert!(w.schedule(0, Command::SetParam { param: Param::SourcePower, value: -1.0 }).is_err());
    assert!(w.schedule(0, Command::SetParam { param: Param::SourceSpeed, value: f64::NAN }).is_err());
    assert!(w.schedule(0, Command::SetParam { param: Param::MutationWeightProbability, value: 1.5 }).is_err());
    assert!(w.schedule(0, Command::SetParam { param: Param::Diffusion, value: 1.0e6 }).is_err());
    assert!(w.schedule(0, Command::DriftChemistry { source: Some(3), degrees: 5.0 }).is_err());
    assert!(w.schedule(0, Command::DriftChemistry { source: None, degrees: 400.0 }).is_err());
    assert!(w.pending.is_empty(), "nothing rejected is kept");
    assert!(w.schedule(0, Command::SetParam { param: Param::Diffusion, value: 0.3 }).is_ok());
}

fn recurrent() -> Config {
    let mut c = small();
    c.controller.hidden_widths = vec![32; 8];
    c.population.initial_count = 128;
    c.architecture_mutation.maximum_parameters = 40_000;
    c
}

#[test]
fn recurrent_world_keeps_memory_and_stays_deterministic() {
    let a = run(recurrent(), 80, 1);
    let b = run(recurrent(), 80, 1);
    let c = run(recurrent(), 80, 4);
    assert_eq!(a.state_hash(), b.state_hash());
    assert_eq!(a.state_hash(), c.state_hash(), "worker split must not change recurrent science");
    assert_eq!(a.orgs, c.orgs);
    let o = &a.orgs;
    assert!(o.len() > 0);
    assert!(o.memory.iter().any(|m| m.iter().any(|&v| v != 0.0)), "evaluated organisms carry memory");
    assert!(o.memory.iter().all(|m| m.iter().all(|v| v.abs() < 1.0)), "memory is squashed by tanh");
    let l = &a.ledger;
    let expected = l.initial_endowment + l.injected - l.removed - l.dissipated.total();
    assert!((a.total_energy() - expected).abs() / l.transferred.max(1.0) < 1e-5, "ledger closes with memory networks");
}

#[test]
fn recurrent_memory_changes_what_the_network_decides() {
    let mut a = run(recurrent(), 20, 1);
    let mut b = a.clone();
    for m in b.orgs.memory.iter_mut() {
        *m = [0.0; 16];
    }
    assert!(a.orgs.memory.iter().any(|m| m.iter().any(|&v| v != 0.0)));
    for _ in 0..4 {
        a.step().unwrap();
        b.step().unwrap();
    }
    assert_ne!(a.orgs.held, b.orgs.held, "memory feeds back into the logits");
}

#[test]
fn newborns_start_with_empty_memory() {
    let mut w = World::new(recurrent()).unwrap();
    let mut seen = 0;
    for _ in 0..200 {
        w.step().unwrap();
        for i in 0..w.orgs.len() {
            if w.orgs.birth_tick[i] == w.tick && w.orgs.generation[i] > 0 {
                seen += 1;
                assert_eq!(w.orgs.memory[i], [0.0; 16]);
            }
        }
    }
    assert!(seen > 0, "some births happened");
}

#[test]
fn recurrent_world_survives_a_checkpoint() {
    let mut a = run(recurrent(), 30, 1);
    let bytes = fl_core::checkpoint::save(&a);
    let mut b = fl_core::checkpoint::load(&bytes).unwrap();
    assert_eq!(a.state_hash(), b.state_hash());
    assert_eq!(a.orgs.memory, b.orgs.memory);
    for _ in 0..20 {
        a.step().unwrap();
        b.step().unwrap();
    }
    assert_eq!(a.state_hash(), b.state_hash());
}

#[test]
fn controller_only_backends_refuse_recurrent_networks() {
    let mut w = World::new(recurrent()).unwrap();
    let mut be = fl_core::ReferenceBackend;
    match w.step_with(Some(&mut be)) {
        Err(TickError::NumericalFault { detail, .. }) => assert!(detail.contains("memory"), "{detail}"),
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

fn recurrent_f16() -> Config {
    let mut c = recurrent();
    c.controller.canonical_weight_dtype = "f16".into();
    c
}

#[test]
fn half_precision_world_is_deterministic_and_stores_sixteen_bit_weights() {
    use fl_core::genome::Weights;
    let a = run(recurrent_f16(), 80, 1);
    let b = run(recurrent_f16(), 80, 4);
    assert_eq!(a.state_hash(), b.state_hash(), "worker split must not change f16 science");
    assert!(a.orgs.len() > 0);
    assert!(a.orgs.genome.iter().all(|g| matches!(g.params, Weights::F16(_))));
    // It really is a different run from the 32-bit one (rounded founders), and it still balances its books.
    assert_ne!(a.state_hash(), run(recurrent(), 80, 1).state_hash());
    let l = &a.ledger;
    let expected = l.initial_endowment + l.injected - l.removed - l.dissipated.total();
    assert!((a.total_energy() - expected).abs() / l.transferred.max(1.0) < 1e-5);
}

#[test]
fn half_precision_world_survives_a_checkpoint() {
    let mut a = run(recurrent_f16(), 30, 1);
    let bytes = fl_core::checkpoint::save(&a);
    let wide = fl_core::checkpoint::save(&run(recurrent(), 30, 1));
    assert!(bytes.len() < wide.len(), "16-bit genomes make a smaller checkpoint ({} vs {})", bytes.len(), wide.len());
    let mut b = fl_core::checkpoint::load(&bytes).unwrap();
    assert_eq!(a.state_hash(), b.state_hash());
    for _ in 0..20 {
        a.step().unwrap();
        b.step().unwrap();
    }
    assert_eq!(a.state_hash(), b.state_hash());
}

#[test]
fn half_precision_is_accepted_for_the_memoryless_networks_too() {
    let mut c = small();
    c.controller.canonical_weight_dtype = "f16".into();
    let a = run(c.clone(), 40, 1);
    assert_eq!(a.state_hash(), run(c, 40, 2).state_hash());
}

/// Big enough (more than the parallel loops' minimum chunk) to take the threaded paths.
fn crowd() -> Config {
    let mut c = small();
    c.controller.hidden_widths = vec![8];
    c.population.initial_count = 3000;
    c.population.maximum_count = 12_000;
    c.architecture_mutation.enabled = false;
    c.population.initial_energy = 6.0;
    c
}

fn crowd_recurrent() -> Config {
    let mut c = recurrent();
    c.population.initial_count = 2200;
    c.population.maximum_count = 12_000;
    c.population.initial_energy = 6.0;
    c
}

/// The threaded per-organism loops give bit-identical worlds to the serial ones.
#[test]
fn threaded_loops_match_serial_loops() {
    for (name, cfg, ticks) in [("crowd", crowd(), 60u64), ("crowd-recurrent", crowd_recurrent(), 25)] {
        let serial = run(cfg.clone(), ticks, 1);
        let threaded = run(cfg, ticks, 4);
        println!("hash {name} {:016x} pop {} births {}", serial.state_hash(), serial.orgs.len(), serial.totals.births);
        assert_eq!(serial.state_hash(), threaded.state_hash(), "{name}: threads changed the science");
        assert_eq!(serial.ledger.dissipated.total().to_bits(), threaded.ledger.dissipated.total().to_bits(), "{name}: ledger sums are order-exact");
        assert_eq!(serial.ledger.transferred.to_bits(), threaded.ledger.transferred.to_bits());
    }
}
