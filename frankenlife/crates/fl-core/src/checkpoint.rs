//! Exact checkpoints of a committed world (plan §16.3, fixture C.13).
//!
//! A checkpoint holds the *complete* authoritative state of a committed tick:
//! effective configuration, field planes, every organism's hot state
//! (including held controller outputs, their evaluation ticks and validity,
//! cadence phases, cooldown deadlines and the one-tick energy-change history),
//! the genome bank, source path state, identity counters, accepted
//! future-tick commands, and the cumulative ledger. Restoring and continuing
//! is bit-identical to an uninterrupted run; nothing is recomputed from the
//! current field on resume.
//!
//! Container (`FLCK`, version 1), all integers and floats little-endian, floats
//! stored as raw bits:
//!
//! ```text
//! magic "FLCK" | u32 version | u64 payload_len | payload | u64 FNV-1a(payload)
//! ```
//!
//! Execution settings (worker count) and wall-clock telemetry are not
//! scientific state and are not stored.

use std::collections::HashMap;
use std::sync::Arc;

use crate::config::Config;
use crate::field::Field;
use crate::config::WeightDtype;
use crate::genome::{Genome, MutationRecord, Weights, MOTION_WIDTH};
use crate::nn::{is_recurrent, param_count, CATALOG, MEMORY_WIDTH, OUTPUT_WIDTH};
use crate::rng::SemanticRng;
use crate::sources::Source;
use crate::world::{Command, CumulativeLedger, Disposal, Dissipation, Organisms, ScheduledCommand, TickCounts, World};
use crate::{CHANNELS, MODEL_VERSION};

pub const MAGIC: &[u8; 4] = b"FLCK";
pub const VERSION: u32 = 1;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[derive(Default)]
struct W(Vec<u8>);

impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.u64(v.to_bits());
    }
    fn f32s(&mut self, v: &[f32]) {
        self.u64(v.len() as u64);
        for x in v {
            self.0.extend_from_slice(&x.to_bits().to_le_bytes());
        }
    }
    fn f16s(&mut self, v: &[u16]) {
        self.u64(v.len() as u64);
        for h in v {
            self.0.extend_from_slice(&h.to_le_bytes());
        }
    }
    fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.0.extend_from_slice(s.as_bytes());
    }
    fn opt_u64(&mut self, v: Option<u64>) {
        match v {
            Some(x) => {
                self.u8(1);
                self.u64(x);
            }
            None => self.u8(0),
        }
    }
}

struct R<'a> {
    b: &'a [u8],
    p: usize,
}

type Res<T> = Result<T, String>;

impl<'a> R<'a> {
    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let end = self.p.checked_add(n).ok_or("length overflow")?;
        if end > self.b.len() {
            return Err(format!("truncated checkpoint at byte {}", self.p));
        }
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u8(&mut self) -> Res<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Res<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Res<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Res<f64> {
        Ok(f64::from_bits(self.u64()?))
    }
    fn len(&mut self, max: usize) -> Res<usize> {
        let n = self.u64()?;
        if n > max as u64 {
            return Err(format!("length {n} exceeds bound {max}"));
        }
        Ok(n as usize)
    }
    fn f32s(&mut self, max: usize) -> Res<Vec<f32>> {
        let n = self.len(max)?;
        let raw = self.take(n * 4)?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap()))).collect())
    }
    fn f16s(&mut self, max: usize) -> Res<Vec<u16>> {
        let n = self.len(max)?;
        let raw = self.take(n * 2)?;
        Ok(raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }
    fn f32_array<const N: usize>(&mut self) -> Res<[f32; N]> {
        let v = self.f32s(N)?;
        v.try_into().map_err(|_| format!("expected {N} floats"))
    }
    fn str(&mut self, max: usize) -> Res<String> {
        let n = self.len(max)?;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|e| e.to_string())
    }
    fn opt_u64(&mut self) -> Res<Option<u64>> {
        Ok(match self.u8()? {
            0 => None,
            1 => Some(self.u64()?),
            t => return Err(format!("bad option tag {t}")),
        })
    }
}

const DISSIPATION_FIELDS: usize = 9;

fn put_dissipation(w: &mut W, d: &Dissipation) {
    for v in [d.decay, d.uptake_inefficiency, d.toxicity, d.maintenance, d.motion, d.emission, d.birth, d.death, d.tiny_reservoir] {
        w.f64(v);
    }
}

fn get_dissipation(r: &mut R) -> Res<Dissipation> {
    let mut v = [0.0; DISSIPATION_FIELDS];
    for x in &mut v {
        *x = r.f64()?;
    }
    Ok(Dissipation {
        decay: v[0],
        uptake_inefficiency: v[1],
        toxicity: v[2],
        maintenance: v[3],
        motion: v[4],
        emission: v[5],
        birth: v[6],
        death: v[7],
        tiny_reservoir: v[8],
    })
}

fn put_counts(w: &mut W, c: &TickCounts) {
    for v in [c.births, c.deaths_exhaustion, c.deaths_user, c.reproduction_attempts, c.rejected_births_viability, c.rejected_births_mutation, c.rejected_births_capacity, c.topology_mutations, c.controller_evaluations] {
        w.u32(v);
    }
    w.u64(c.parameter_evaluations);
    w.u32(c.emitting);
    w.u32(c.cone_projections);
    w.u32(c.transport_subcycles);
}

fn get_counts(r: &mut R) -> Res<TickCounts> {
    let mut a = [0u32; 9];
    for x in &mut a {
        *x = r.u32()?;
    }
    Ok(TickCounts {
        births: a[0],
        deaths_exhaustion: a[1],
        deaths_user: a[2],
        reproduction_attempts: a[3],
        rejected_births_viability: a[4],
        rejected_births_mutation: a[5],
        rejected_births_capacity: a[6],
        topology_mutations: a[7],
        controller_evaluations: a[8],
        parameter_evaluations: r.u64()?,
        emitting: r.u32()?,
        cone_projections: r.u32()?,
        transport_subcycles: r.u32()?,
    })
}

fn put_command(w: &mut W, c: &ScheduledCommand) {
    w.u64(c.tick);
    w.u64(c.seq);
    match &c.command {
        Command::EraseDisk { x, y, radius, disposal } => {
            w.u8(1);
            w.f64(*x);
            w.f64(*y);
            w.f64(*radius);
            w.u8(match disposal {
                Disposal::Remove => 0,
                Disposal::Dissipate => 1,
                Disposal::Recycle => 2,
            });
        }
        Command::EnergyBrush { x, y, radius, amount, direction } => {
            w.u8(2);
            w.f64(*x);
            w.f64(*y);
            w.f64(*radius);
            w.f64(*amount);
            match direction {
                Some(d) => {
                    w.u8(1);
                    w.f32s(d);
                }
                None => w.u8(0),
            }
        }
        Command::SetParam { param, value } => {
            w.u8(3);
            w.u8(param.tag());
            w.f64(*value);
        }
        Command::DriftChemistry { source, degrees } => {
            w.u8(4);
            match source {
                Some(k) => {
                    w.u8(1);
                    w.u32(*k);
                }
                None => w.u8(0),
            }
            w.f64(*degrees);
        }
    }
}

fn get_command(r: &mut R) -> Res<ScheduledCommand> {
    let tick = r.u64()?;
    let seq = r.u64()?;
    let command = match r.u8()? {
        1 => Command::EraseDisk {
            x: r.f64()?,
            y: r.f64()?,
            radius: r.f64()?,
            disposal: match r.u8()? {
                0 => Disposal::Remove,
                1 => Disposal::Dissipate,
                2 => Disposal::Recycle,
                t => return Err(format!("bad disposal {t}")),
            },
        },
        2 => {
            let (x, y, radius, amount) = (r.f64()?, r.f64()?, r.f64()?, r.f64()?);
            let direction = match r.u8()? {
                0 => None,
                1 => Some(r.f32_array::<CHANNELS>()?),
                t => return Err(format!("bad direction tag {t}")),
            };
            Command::EnergyBrush { x, y, radius, amount, direction }
        }
        3 => {
            let tag = r.u8()?;
            let param = crate::world::Param::from_tag(tag).ok_or(format!("bad parameter tag {tag}"))?;
            Command::SetParam { param, value: r.f64()? }
        }
        4 => {
            let source = match r.u8()? {
                0 => None,
                1 => Some(r.u32()?),
                t => return Err(format!("bad source tag {t}")),
            };
            Command::DriftChemistry { source, degrees: r.f64()? }
        }
        t => return Err(format!("unknown command tag {t}")),
    };
    Ok(ScheduledCommand { tick, seq, command })
}

fn put_genome(w: &mut W, g: &Genome) {
    w.u64(g.id);
    w.opt_u64(g.parent);
    w.u8(g.topology);
    // The weight type comes from the configuration stored in the checkpoint, so no tag is written.
    match &g.params {
        Weights::F32(v) => w.f32s(v),
        Weights::F16(v) => w.f16s(v),
    }
    w.f32s(&g.affinity);
    w.f32s(&g.motion);
    w.u32(g.log_scale.to_bits());
    match &g.mutation {
        Some(m) => {
            w.u8(1);
            w.u32(m.candidate_index);
            w.u8(m.topology_changed as u8);
            w.u32(m.parameters_perturbed);
            w.u32(m.parameters_replaced);
            w.u32(m.rejected_candidates);
        }
        None => w.u8(0),
    }
}

fn get_genome(r: &mut R, dtype: WeightDtype) -> Res<Genome> {
    let id = r.u64()?;
    let parent = r.opt_u64()?;
    let topology = r.u8()?;
    let hidden = CATALOG.get(topology as usize).ok_or(format!("unknown topology {topology}"))?;
    let params = match dtype {
        WeightDtype::F32 => Weights::F32(r.f32s(param_count(hidden))?),
        WeightDtype::F16 => Weights::F16(r.f16s(param_count(hidden))?),
    };
    if params.len() != param_count(hidden) {
        return Err(format!("genome {id}: {} params for topology {hidden:?}", params.len()));
    }
    let affinity = r.f32_array::<CHANNELS>()?;
    let motion = r.f32_array::<MOTION_WIDTH>()?;
    let log_scale = f32::from_bits(r.u32()?);
    let mutation = match r.u8()? {
        0 => None,
        1 => Some(MutationRecord {
            candidate_index: r.u32()?,
            topology_changed: r.u8()? != 0,
            parameters_perturbed: r.u32()?,
            parameters_replaced: r.u32()?,
            rejected_candidates: r.u32()?,
        }),
        t => return Err(format!("bad mutation tag {t}")),
    };
    Ok(Genome { id, parent, topology, params, affinity, motion, log_scale, mutation })
}

/// Serialize the committed world.
pub fn save(world: &World) -> Vec<u8> {
    let mut w = W::default();
    w.str(MODEL_VERSION);
    w.str(&toml::to_string(&world.cfg).expect("config serializes"));
    w.u64(world.tick);
    w.u64(world.next_org_id);
    w.u64(world.next_genome_id);
    w.u64(world.next_command_seq);

    let f = &world.field;
    w.u64(f.gx as u64);
    w.u64(f.gy as u64);
    w.f32s(&f.e);
    w.f32s(&f.m);

    w.u64(world.sources.len() as u64);
    for s in &world.sources {
        w.u32(s.id);
        w.f64(s.anchor[0]);
        w.f64(s.anchor[1]);
        w.u64(s.ordinal);
        for v in [s.start_dir[0], s.start_dir[1], s.end_dir[0], s.end_dir[1]] {
            w.f64(v);
        }
        w.f32s(&s.chem);
    }

    // Genome bank: each distinct genome once, in first-reference order.
    let o = &world.orgs;
    let mut index: HashMap<u64, u64> = HashMap::new();
    let mut bank: Vec<&Genome> = Vec::new();
    for g in &o.genome {
        index.entry(g.id).or_insert_with(|| {
            bank.push(g);
            (bank.len() - 1) as u64
        });
    }
    w.u64(bank.len() as u64);
    for g in &bank {
        put_genome(&mut w, g);
    }

    w.u64(o.len() as u64);
    for i in 0..o.len() {
        w.u64(o.id[i]);
        w.u64(o.lineage[i]);
        w.u32(o.generation[i]);
        for v in [o.x[i], o.y[i], o.vx[i], o.vy[i], o.energy[i], o.energy_change_rate[i]] {
            w.f64(v);
        }
        w.u64(o.birth_tick[i]);
        w.u64(o.cooldown_until[i]);
        w.u32(o.phase[i]);
        w.f32s(&o.held[i]);
        w.u64(o.held_tick[i]);
        w.u8(o.held_valid[i] as u8);
        w.u64(index[&o.genome[i].id]);
        if let Some(m) = o.memory.get(i) {
            w.f32s(m);
        }
    }

    w.u64(world.pending.len() as u64);
    for c in &world.pending {
        put_command(&mut w, c);
    }

    let l = &world.ledger;
    for v in [l.initial_endowment, l.injected, l.removed] {
        w.f64(v);
    }
    put_dissipation(&mut w, &l.dissipated);
    for v in [l.transferred, l.residual, l.max_abs_tick_residual] {
        w.f64(v);
    }
    put_counts(&mut w, &world.totals);

    let payload = w.0;
    let mut out = Vec::with_capacity(payload.len() + 24);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&payload);
    out.extend_from_slice(&fnv1a(&payload).to_le_bytes());
    out
}

/// Restore a committed world. The configuration is re-validated and the
/// checkpoint is refused if its model version differs from this build's.
pub fn load(bytes: &[u8]) -> Result<World, String> {
    if bytes.len() < 24 || &bytes[..4] != MAGIC {
        return Err("not a FrankenLife checkpoint".into());
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(format!("unsupported checkpoint version {version}"));
    }
    let len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    if bytes.len() != 16 + len + 8 {
        return Err(format!("checkpoint length mismatch: header says {len} payload bytes"));
    }
    let payload = &bytes[16..16 + len];
    let sum = u64::from_le_bytes(bytes[16 + len..].try_into().unwrap());
    if fnv1a(payload) != sum {
        return Err("checkpoint checksum mismatch".into());
    }
    let mut r = R { b: payload, p: 0 };
    let model = r.str(256)?;
    if model != MODEL_VERSION {
        return Err(format!("checkpoint model {model:?} differs from this build's {MODEL_VERSION:?}"));
    }
    let cfg = Config::from_toml(&r.str(1 << 20)?)?;
    let cells = cfg.world.grid_width as usize * cfg.world.grid_height as usize;
    // World::new validates and gives us the derived, non-stored members.
    let mut world = World::new(cfg.clone()).map_err(|e| format!("checkpoint config invalid: {e:?}"))?;
    world.rng = SemanticRng::new(cfg.seed);
    world.tick = r.u64()?;
    world.next_org_id = r.u64()?;
    world.next_genome_id = r.u64()?;
    world.next_command_seq = r.u64()?;

    let (gx, gy) = (r.u64()? as usize, r.u64()? as usize);
    if gx * gy != cells {
        return Err("field shape differs from configuration".into());
    }
    let mut field = Field::new(gx, gy, cfg.world.width, cfg.world.height);
    field.e = r.f32s(cells)?;
    field.m = r.f32s(cells * CHANNELS)?;
    if field.e.len() != cells || field.m.len() != cells * CHANNELS {
        return Err("field plane length mismatch".into());
    }
    world.field = field;

    let ns = r.len(1 << 16)?;
    let mut sources = Vec::with_capacity(ns);
    for _ in 0..ns {
        let id = r.u32()?;
        let anchor = [r.f64()?, r.f64()?];
        let ordinal = r.u64()?;
        let start_dir = [r.f64()?, r.f64()?];
        let end_dir = [r.f64()?, r.f64()?];
        let chem = r.f32_array::<CHANNELS>()?;
        sources.push(Source { id, anchor, ordinal, start_dir, end_dir, chem });
    }
    world.sources = sources;

    let max_orgs = cfg.population.maximum_count as usize;
    let nb = r.len(max_orgs)?;
    let mut bank: Vec<Arc<Genome>> = Vec::with_capacity(nb);
    for _ in 0..nb {
        bank.push(Arc::new(get_genome(&mut r, cfg.controller.weight_dtype())?));
    }
    let no = r.len(max_orgs)?;
    let mut o = Organisms::default();
    for _ in 0..no {
        o.id.push(r.u64()?);
        o.lineage.push(r.u64()?);
        o.generation.push(r.u32()?);
        o.x.push(r.f64()?);
        o.y.push(r.f64()?);
        o.vx.push(r.f64()?);
        o.vy.push(r.f64()?);
        o.energy.push(r.f64()?);
        o.energy_change_rate.push(r.f64()?);
        o.birth_tick.push(r.u64()?);
        o.cooldown_until.push(r.u64()?);
        o.phase.push(r.u32()?);
        o.held.push(r.f32_array::<OUTPUT_WIDTH>()?);
        o.held_tick.push(r.u64()?);
        o.held_valid.push(r.u8()? != 0);
        let gi = r.u64()? as usize;
        let genome = bank.get(gi).ok_or(format!("genome index {gi} out of range"))?.clone();
        if is_recurrent(genome.hidden()) {
            o.memory.push(r.f32_array::<MEMORY_WIDTH>()?);
        }
        o.genome.push(genome);
    }
    if o.id.windows(2).any(|p| p[0] >= p[1]) {
        return Err("organisms are not in strictly increasing stable-ID order".into());
    }
    if o.energy.iter().any(|&e| !(e > 0.0) || !e.is_finite()) {
        return Err("checkpoint contains a non-positive or non-finite organism energy".into());
    }
    world.orgs = o;

    let nc = r.len(cfg.execution.command_queue_capacity as usize)?;
    world.pending = (0..nc).map(|_| get_command(&mut r)).collect::<Res<_>>()?;

    let (initial_endowment, injected, removed) = (r.f64()?, r.f64()?, r.f64()?);
    let dissipated = get_dissipation(&mut r)?;
    let (transferred, residual, max_abs_tick_residual) = (r.f64()?, r.f64()?, r.f64()?);
    world.ledger = CumulativeLedger { initial_endowment, injected, removed, dissipated, transferred, residual, max_abs_tick_residual };
    world.totals = get_counts(&mut r)?;
    world.last_report = None;
    if r.p != payload.len() {
        return Err(format!("{} trailing bytes in checkpoint", payload.len() - r.p));
    }
    Ok(world)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Config {
        let mut c = Config::reference();
        c.world.width = 32.0;
        c.world.height = 32.0;
        c.world.grid_width = 32;
        c.world.grid_height = 32;
        c.population.initial_count = 128;
        c.population.initial_energy = 4.5;
        c.sources.count = 3;
        c.sources.power_per_source = 20.0;
        c.sources.sigma_world_units = 4.0;
        c.sensors.maximum_scale_world_units = 8.0;
        c.reproduction.maximum_attempt_rate_per_second = 4.0;
        c.reproduction.cooldown_seconds = 0.25;
        c.controller.hidden_widths = vec![8];
        c.controller.evaluation_interval_ticks = 4;
        c
    }

    fn same_science(a: &World, b: &World) {
        assert_eq!(a.state_hash(), b.state_hash());
        assert_eq!(a.ledger, b.ledger);
        assert_eq!(a.totals, b.totals);
        assert_eq!(a.pending, b.pending);
        assert_eq!(a.sources, b.sources);
        assert_eq!(a.orgs, b.orgs);
    }

    #[test]
    fn c13_off_phase_resume_is_bit_identical() {
        let mut a = World::new(small()).unwrap();
        // Accepted future-tick commands must survive the checkpoint, including
        // one far beyond it and one scheduled out of tick order.
        a.schedule(1000, Command::EnergyBrush { x: 3.0, y: 4.0, radius: 2.0, amount: 5.0, direction: None }).unwrap();
        a.schedule(101, Command::EraseDisk { x: 16.0, y: 16.0, radius: 3.0, disposal: Disposal::Recycle }).unwrap();
        for _ in 0..5 {
            a.step().unwrap();
        }
        // Off-phase checkpoint: tick 5 with a cadence of 4.
        assert_eq!(a.tick, 5);
        let bytes = save(&a);
        let mut b = load(&bytes).unwrap();
        same_science(&a, &b);
        assert_eq!(save(&b), bytes, "save(load(x)) == x");
        b.workers = 3; // execution setting only
        for _ in 0..200 {
            a.step().unwrap();
            b.step().unwrap();
        }
        same_science(&a, &b);
        assert!(a.totals.births > 0, "the comparison covered births");
        assert_eq!(a.pending.len(), 1, "the tick-1000 command is still pending");
    }

    #[test]
    fn live_tuning_survives_a_checkpoint() {
        use crate::world::Param;
        let mut a = World::new(small()).unwrap();
        a.schedule(2, Command::SetParam { param: Param::SourcePower, value: 55.0 }).unwrap();
        a.schedule(2, Command::DriftChemistry { source: Some(1), degrees: 30.0 }).unwrap();
        a.schedule(500, Command::SetParam { param: Param::MutationWeightSd, value: 0.3 }).unwrap();
        a.schedule(501, Command::DriftChemistry { source: None, degrees: 10.0 }).unwrap();
        for _ in 0..5 {
            a.step().unwrap();
        }
        assert_eq!(a.cfg.sources.power_per_source, 55.0, "the applied change is part of the configuration");
        let bytes = save(&a);
        let mut b = load(&bytes).unwrap();
        same_science(&a, &b);
        assert_eq!(b.cfg.sources.power_per_source, 55.0);
        assert_eq!(save(&b), bytes);
        for _ in 0..600 {
            a.step().unwrap();
            b.step().unwrap();
        }
        same_science(&a, &b);
        assert_eq!(b.cfg.mutation.weight_standard_deviation, 0.3);
        assert!(a.pending.is_empty());
    }

    #[test]
    fn corrupt_or_foreign_checkpoints_are_refused() {
        let mut w = World::new(small()).unwrap();
        w.step().unwrap();
        let bytes = save(&w);
        let mut bad = bytes.clone();
        let k = bad.len() / 2;
        bad[k] ^= 0x40;
        assert!(load(&bad).unwrap_err().contains("checksum"));
        assert!(load(&bytes[..bytes.len() - 3]).is_err());
        assert!(load(b"PNG\0not a checkpoint at all......").is_err());
    }
}
