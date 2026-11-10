//! Single-wait front half of a resident tick.
//!
//! The classic resident path waits on the GPU four times per tick (field
//! accounting, sense/MLP, uptake, finish). Here the field step, the sensor
//! pyramid, every sense/MLP group, the uptake request (computed on the device
//! from the sensed density) and the uptake exchange are queued back to back and
//! read with one wait. Placement of genomes into weight slots is remembered in
//! job order from the previous tick, so the steady state does no hash lookups.
//!
//! The arithmetic of every kernel is unchanged. The only numerical difference
//! from the classic path is the uptake request, which is evaluated in f32 on
//! the device instead of f64 on the host.

use super::*;

const DEMAND_SHADER: &str = include_str!("demand.wgsl");
const HDR_IN_STRIDE: usize = 4;

#[derive(Default)]
pub(crate) struct FrontState {
    pipe: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout, wgpu::Buffer)>,
    hdr_in: Option<wgpu::Buffer>,
    hdr_cap: usize,
    bind: Option<((usize, usize), wgpu::BindGroup)>,
    /// Previous call's placement in job order, keyed by organism id (an organism
    /// keeps its genome for life, so a hit needs no look at the genome).
    prev: Vec<Placed>,
    /// `GpuBackend::call` of the call that filled `prev` and `statics_prev`.
    prev_call: u64,
    /// Static job words (slot, scale per row) uploaded by the previous call, and the job capacity they went to.
    statics_prev: Vec<f32>,
    statics_cap: usize,
    /// Per (topology, page): its own sense uniform and bind group, so every group
    /// of a tick can sit in one submission.
    groups: HashMap<(usize, usize), GroupBind>,
    /// The sensor pyramid of this tick already rode with the field step.
    pub(crate) pyr_done: bool,
    /// `begin_tick`'s encoder, left open so it joins the first field-step submission.
    pub(crate) open_enc: Option<wgpu::CommandEncoder>,
    /// The last tick's commit copy has not been queued yet.
    pub(crate) commit_pending: bool,
    /// Deposits packed by `deposit` and queued by `finish` (count, then items).
    pub(crate) deps_n: Option<usize>,
    pub(crate) deps_items: Vec<f32>,
    order: Vec<u32>,
    hdr: Vec<f32>,
    dynamic: Vec<f32>,
    statics: Vec<f32>,
}

#[derive(Clone, Copy)]
struct Placed {
    organism: u64,
    topology: u32,
    slot: u32,
    affinity: u32,
    scale: f32,
}

struct GroupBind {
    key: (usize, u64, u64),
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
    words: Option<[u32; 76]>,
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && bytemuck::cast_slice::<f32, u32>(a) == bytemuck::cast_slice::<f32, u32>(b)
}

impl GpuBackend {
    /// Queue transport, sensing, inference and uptake together and wait once
    /// (`--backend gpu-realtime`). Off by default: the classic resident path and
    /// its results are unchanged.
    pub fn set_single_wait(&mut self, enabled: bool) {
        self.single_wait = enabled;
    }

    pub fn single_wait(&self) -> bool {
        self.single_wait
    }

    fn ensure_front(&mut self, n: usize) -> Result<(), String> {
        if self.front.pipe.is_none() {
            let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("demand"), source: wgpu::ShaderSource::Wgsl(DEMAND_SHADER.into()) });
            let uentry = wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            };
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("demand"),
                entries: &[uentry, storage_entry(1, true), storage_entry(2, true), storage_entry(3, false)],
            });
            let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("demand"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
            let pipe = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("make_headers"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("make_headers"),
                compilation_options: Default::default(),
                cache: None,
            });
            let uniform = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("demand-dims"), size: 32, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            self.front.pipe = Some((pipe, layout, uniform));
        }
        self.ensure_sense_io(n);
        self.ensure_res_work(n)?;
        if self.front.hdr_cap < n {
            let cap = n.next_power_of_two().max(1024);
            self.front.hdr_in = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("uptake-hdr-in"),
                size: (cap * HDR_IN_STRIDE * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.front.hdr_cap = cap;
            self.front.bind = None;
        }
        let st = self.resident.as_ref().ok_or("resident state not initialized")?;
        let key = (self.job_cap, st.items_cap);
        if self.front.bind.as_ref().is_none_or(|(k, _)| *k != key) {
            let io = self.sense_io.as_ref().expect("sense io");
            let (_, layout, uniform) = self.front.pipe.as_ref().expect("demand pipeline");
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("demand"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.front.hdr_in.as_ref().expect("hdr in").as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: io.outputs.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: st.headers.as_entire_binding() },
                ],
            });
            self.front.bind = Some((key, bg));
        }
        Ok(())
    }

    /// Queue a commit copy that `commit` deferred, if there is one.
    pub(crate) fn flush_commit(&mut self) {
        if !std::mem::take(&mut self.front.commit_pending) {
            return;
        }
        let (Some(st), Some(fg)) = (self.resident.as_ref(), self.field_gpu.as_ref()) else { return };
        let cells = st.cells as u64;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("commit") });
        enc.copy_buffer_to_buffer(&fg.fe, 0, &st.ce, 0, cells * 4);
        enc.copy_buffer_to_buffer(&fg.fm, 0, &st.cm, 0, cells * 32 * 4);
        self.queue.submit([enc.finish()]);
    }

    /// Record the sensor-pyramid passes (level 0, then each coarser level) for the
    /// field currently in the working buffers.
    pub(crate) fn encode_pyramid(&self, enc: &mut wgpu::CommandEncoder, ts: Option<&Timestamps>) {
        let fg = self.field_gpu.as_ref().expect("field resources");
        let nl = fg.binds.len();
        for l in 0..nl {
            let tw = ts.filter(|_| l == 0 || l == nl - 1).map(|ts| wgpu::ComputePassTimestampWrites {
                query_set: &ts.set,
                beginning_of_pass_write_index: (l == 0).then_some(0),
                end_of_pass_write_index: (l == nl - 1).then_some(1),
            });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pyramid"), timestamp_writes: tw });
            pass.set_pipeline(if l == 0 { &self.sense.pyr_level0 } else { &self.sense.pyr_down });
            pass.set_bind_group(0, &fg.binds[l], &[]);
            pass.dispatch_workgroups(fg.grids[l].0, fg.grids[l].1, 1);
        }
    }

    /// Weight slot, affinity slot and sensing scale of every job, reusing the
    /// previous call's answers while the job list lines up with it.
    fn place_front(&mut self, jobs: &[SenseJob<'_>]) -> Result<Vec<Placed>, String> {
        let prev = std::mem::take(&mut self.front.prev);
        let trust = self.call == self.front.prev_call + 1;
        let mut cur = Vec::with_capacity(jobs.len());
        let mut c = 0usize;
        for j in jobs {
            let mut hit = None;
            if trust {
                // Births append and deaths remove, so a miss is usually a short skip ahead.
                for d in 0..16 {
                    match prev.get(c + d) {
                        Some(e) if e.organism == j.organism => {
                            hit = Some(*e);
                            c += d + 1;
                            break;
                        }
                        Some(_) => {}
                        None => break,
                    }
                }
            }
            let e = match hit {
                Some(e) => {
                    self.slabs[e.topology as usize].last_used[e.slot as usize] = self.call;
                    e
                }
                None => {
                    let placed = self.place(std::iter::once(j.genome))?;
                    let affinity = self.affinity_slot(j.genome.id, &j.genome.affinity)?;
                    Placed { organism: j.organism, topology: placed[0].0 as u32, slot: placed[0].1 as u32, affinity, scale: j.genome.scale() as f32 }
                }
            };
            cur.push(e);
        }
        Ok(cur)
    }

    /// Uniform and bind group of one (topology, page) group, with its dims written when they changed.
    fn ensure_group(&mut self, t: usize, page: usize, n: usize, base: usize, sensors: &fl_core::config::SensorsCfg) {
        let io = self.sense_io.as_ref().expect("sense io");
        let fg = self.field_gpu.as_ref().expect("field resources");
        let slab = &self.slabs[t];
        let key = (io.cap, self.field_gen, slab.epoch);
        if self.front.groups.get(&(t, page)).is_none_or(|g| g.key != key) {
            let uniform = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("sense-dims-group"), size: 76 * 4, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let wb = |l: usize| slab.pages[page].get(l).unwrap_or(&self.dummy).as_entire_binding();
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sense-group"),
                layout: &self.sense.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: io.inputs.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: fg.pyr.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: slab.traits[page].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: wb(0) },
                    wgpu::BindGroupEntry { binding: 5, resource: wb(1) },
                    wgpu::BindGroupEntry { binding: 6, resource: if slab.dims.len() == MAX_LAYERS { wb(2) } else { self.dummy.as_entire_binding() } },
                    wgpu::BindGroupEntry { binding: 7, resource: io.outputs.as_entire_binding() },
                ],
            });
            self.front.groups.insert((t, page), GroupBind { key, uniform, bind, words: None });
        }
        let (grid_x, _) = grid_for(n as u32);
        let mut w = [0u32; 76];
        w[0] = n as u32;
        w[1] = slab.dims.len() as u32;
        w[2] = slab.dims[0].0 as u32;
        w[3] = slab.dims[0].1 as u32;
        w[4] = slab.dims[1].1 as u32;
        w[5] = slab.dims.get(2).map_or(0, |d| d.1 as u32);
        w[6] = grid_x;
        w[7] = fg.levels.len() as u32;
        w[8] = fg.s0.to_bits();
        w[9] = (sensors.composition_energy_floor_density as f32).to_bits();
        w[10] = (sensors.energy_density_scale as f32).to_bits();
        w[11] = (sensors.gradient_length_scale_world_units as f32).to_bits();
        for (l, &(gx, gy, off, hx, hy)) in fg.levels.iter().enumerate() {
            w[12 + 4 * l] = gx;
            w[12 + 4 * l + 1] = gy;
            w[12 + 4 * l + 2] = off;
            w[44 + 4 * l] = hx.to_bits();
            w[44 + 4 * l + 1] = hy.to_bits();
        }
        // lu[0].w: first row of this group in the shared job and output buffers.
        w[15] = base as u32;
        let st = self.strides();
        w[19] = st.job as u32;
        w[23] = st.out as u32;
        let g = self.front.groups.get_mut(&(t, page)).expect("group");
        if g.words != Some(w) {
            self.queue.write_buffer(&g.uniform, 0, bytemuck::cast_slice(&w));
            g.words = Some(w);
        }
    }

    pub(crate) fn sense_uptake_impl(&mut self, cfg: &Config, extra_levels: usize, jobs: &[SenseJob<'_>], sense_out: &mut [SenseOutput], uptake_out: &mut [UptakeOut]) -> Result<(FieldAccounting, f64), String> {
        let pending = self.pending_field.take().ok_or("transport was not submitted")?;
        let mut discard = Field { gx: 1, gy: 1, hx: 1.0, hy: 1.0, e: Vec::new(), m: Vec::new() };
        self.stats.last = EvalTiming::default();
        let n = jobs.len();
        if n == 0 {
            let acct = self.finish_field_read(pending, &mut discard)?;
            return Ok((acct, 0.0));
        }
        let total = std::time::Instant::now();
        self.call += 1;
        self.stats.calls += 1;
        self.stats.evaluations += jobs.iter().filter(|j| j.due).count() as u64;
        self.mem_mode = jobs.first().is_some_and(|j| is_recurrent(j.genome.hidden()));
        let sensors = &cfg.sensors;
        let ch = &cfg.chemistry;
        let workers = cfg.execution.cpu_workers as usize;

        // ---- pyramid of the observation field the queued field step leaves on the device ----
        // Queued first: it needs nothing from the jobs, so the GPU works through
        // the field step and the pyramid while the host places and packs.
        if self.field_gpu.as_ref().is_none_or(|f| f.levels.len() != extra_levels + 1) {
            return Err("resident field pyramid does not match the requested levels".into());
        }
        self.field_resident = false;
        let stamp = self.profiling;
        let pyr_rx = if std::mem::take(&mut self.front.pyr_done) {
            None
        } else {
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pyramid") });
            let ts = self.pyramid_timestamps.as_ref().filter(|_| stamp);
            self.encode_pyramid(&mut enc, ts);
            if let Some(ts) = ts {
                enc.resolve_query_set(&ts.set, 0..2, &ts.resolve, 0);
                enc.copy_buffer_to_buffer(&ts.resolve, 0, &ts.readback, 0, 16);
            }
            self.queue.submit([enc.finish()]);
            ts.map(|ts| {
                let (tx, rx) = mpsc::channel();
                ts.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                rx
            })
        };

        // ---- placement (may upload new genomes and grow slabs) ----
        let placed = self.place_front(jobs)?;
        self.ensure_front(n)?;
        self.ensure_expand_bind()?;
        self.ensure_job_expand_bind();
        let (gx, gy, hx, hy) = self.res_geom()?;

        // ---- item order: jobs grouped by (topology, weight page) ----
        let grp: Vec<(usize, usize)> = placed.iter().map(|e| (e.topology as usize, e.slot as usize / self.slabs[e.topology as usize].page_slots)).collect();
        let mut order = std::mem::take(&mut self.front.order);
        order.clear();
        order.extend(0..n as u32);
        let single = grp.iter().all(|g| *g == grp[0]);
        if !single {
            order.sort_by_key(|&k| grp[k as usize]);
        }

        // ---- per-item host words, in item order (stencils are host f64, as before) ----
        let mut hdr = std::mem::take(&mut self.front.hdr);
        let mut dynamic = std::mem::take(&mut self.front.dynamic);
        let mut statics = std::mem::take(&mut self.front.statics);
        let (dyn_stride, mem_mode) = (self.strides().dynamic, self.mem_mode);
        hdr.clear();
        hdr.resize(n * HDR_IN_STRIDE, 0.0);
        dynamic.clear();
        dynamic.resize(n * dyn_stride, 0.0);
        statics.clear();
        statics.resize(n * STATIC_STRIDE, 0.0);
        let page_slots: Vec<usize> = self.slabs.iter().map(|s| s.page_slots).collect();
        fl_core::par::for_each_rows3(cfg.execution.cpu_workers as usize, n, &mut hdr, HDR_IN_STRIDE, &mut dynamic, dyn_stride, &mut statics, STATIC_STRIDE, |row, h, d, st| {
            let k = order[row] as usize;
            let j = &jobs[k];
            let e = &placed[k];
            let (cell, tx, ty) = stencil_base(gx, gy, hx, hy, j.x, j.y);
            h.copy_from_slice(&[f32::from_bits(cell), tx, ty, f32::from_bits(e.affinity)]);
            d[..3].copy_from_slice(&[j.x as f32, j.y as f32, if j.due { 1.0 } else { 0.0 }]);
            d[3..8].copy_from_slice(&j.intrinsic);
            if mem_mode {
                d[8..8 + MEMORY_WIDTH].copy_from_slice(j.memory);
            }
            st.copy_from_slice(&[f32::from_bits((e.slot as usize % page_slots[e.topology as usize]) as u32), e.scale]);
        });
        self.queue.write_buffer(self.front.hdr_in.as_ref().expect("hdr in"), 0, f32_bytes(&hdr));
        self.queue.write_buffer(self.job_dyn.as_ref().expect("dynamic jobs"), 0, f32_bytes(&dynamic));
        let trust = self.call == self.front.prev_call + 1 && self.front.statics_cap == self.job_cap;
        if !(trust && same_bits(&statics, &self.front.statics_prev)) {
            self.queue.write_buffer(self.job_static.as_ref().expect("static jobs"), 0, f32_bytes(&statics));
        }
        // The classic path's signature of the static buffer no longer describes it.
        self.sense_static_sig.clear();
        std::mem::swap(&mut self.front.statics_prev, &mut statics);
        self.front.statics_cap = self.job_cap;
        self.front.hdr = hdr;
        self.front.dynamic = dynamic;
        self.front.statics = statics;

        // ---- every sense group, then the uptake headers, in one submission ----
        let mut bounds: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut base = 0usize;
        while base < n {
            let (t, page) = grp[order[base] as usize];
            let mut end = base + 1;
            while end < n && grp[order[end] as usize] == (t, page) {
                end += 1;
            }
            self.ensure_group(t, page, end - base, base, sensors);
            bounds.push((t, page, base, end - base));
            base = end;
        }
        let items_grid = grid_for((n as u32).div_ceil(64));
        {
            let st = self.strides();
            let dims = [n as u32, items_grid.0, st.dynamic as u32, st.job as u32];
            self.queue.write_buffer(self.job_expand_uniform.as_ref().expect("job expand uniform"), 0, bytemuck::cast_slice(&dims));
            let (_, _, uniform) = self.front.pipe.as_ref().expect("demand pipeline");
            let dt_umax = (cfg.dt() * ch.uptake_capacity_per_second) as f32;
            let half = ch.uptake_half_saturation_density as f32;
            let w: [u32; 8] = [n as u32, items_grid.0, 0, st.out as u32, dt_umax.to_bits(), half.to_bits(), 0, 0];
            self.queue.write_buffer(uniform, 0, bytemuck::cast_slice(&w));
        }
        let sense_ts = self.timestamps.as_ref().filter(|_| stamp);
        if stamp && self.pass_timer.is_none() && self.timestamps.is_some() {
            self.pass_timer = Some(PassTimer::new(&self.device, &self.queue));
        }
        if let Some(pt) = &self.pass_timer {
            pt.begin(stamp);
        }
        let pt = |name| self.pass_timer.as_ref().and_then(|t| t.tw(name));
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("front-sense") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("expand-jobs"), timestamp_writes: pt("expand_jobs") });
            pass.set_pipeline(self.job_expand_pipe.as_ref().expect("job expand"));
            pass.set_bind_group(0, self.job_expand_bind.as_ref().expect("job expand bind"), &[]);
            pass.dispatch_workgroups(items_grid.0, items_grid.1, 1);
        }
        for (gi, &(t, page, _, ng)) in bounds.iter().enumerate() {
            let tw = sense_ts.filter(|_| gi == 0 || gi + 1 == bounds.len()).map(|ts| wgpu::ComputePassTimestampWrites {
                query_set: &ts.set,
                beginning_of_pass_write_index: (gi == 0).then_some(0),
                end_of_pass_write_index: (gi + 1 == bounds.len()).then_some(1),
            });
            let (grid_x, grid_y) = grid_for(ng as u32);
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sense"), timestamp_writes: tw });
            pass.set_pipeline(if self.slabs[t].half { &self.sense.deep_half } else if self.slabs[t].packed { &self.sense.deep } else { &self.sense.pipeline });
            pass.set_bind_group(0, &self.front.groups[&(t, page)].bind, &[]);
            pass.dispatch_workgroups(grid_x, grid_y, 1);
        }
        {
            let io = self.sense_io.as_ref().expect("sense io");
            enc.copy_buffer_to_buffer(&io.outputs, 0, &io.staging, 0, (n * self.strides().out * 4) as u64);
            if let Some(ts) = sense_ts {
                enc.resolve_query_set(&ts.set, 0..2, &ts.resolve, 0);
                enc.copy_buffer_to_buffer(&ts.resolve, 0, &ts.readback, 0, 16);
            }
            let (pipe, _, _) = self.front.pipe.as_ref().expect("demand pipeline");
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("make-headers"), timestamp_writes: pt("make_headers") });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &self.front.bind.as_ref().expect("demand bind").1, &[]);
            pass.dispatch_workgroups(items_grid.0, items_grid.1, 1);
        }
        self.stats.dispatches += bounds.len() as u64;
        self.stats.last.dispatches += bounds.len() as u32;

        // ---- uptake exchange over all items, same submission ----
        let sense_ts_rx = {
            self.write_res_uniform(n, Some(ch), 0.0);
            let dims = [n as u32, items_grid.0, 0, 0];
            self.queue.write_buffer(&self.res_pipes.expand_uniform, 0, bytemuck::cast_slice(&dims));
            {
                let st = self.resident.as_ref().expect("resident");
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("expand-items"), timestamp_writes: pt("expand_items") });
                pass.set_pipeline(&self.res_pipes.expand);
                pass.set_bind_group(0, st.expand_bind.as_ref().expect("expand bind"), &[]);
                pass.dispatch_workgroups(items_grid.0, items_grid.1, 1);
            }
            self.encode_binning(&mut enc, n, BinStamp::Off, false);
            let st = self.resident.as_ref().expect("resident");
            let p = &self.res_pipes;
            let cells_grid = grid_for((st.cells as u32).div_ceil(64));
            dispatch_resident(st, &mut enc, &p.uptake_demand, cells_grid, pt("uptake_demand"));
            dispatch_resident(st, &mut enc, &p.uptake_org, items_grid, pt("uptake_org"));
            dispatch_resident(st, &mut enc, &p.uptake_debit, cells_grid, pt("uptake_debit"));
            enc.copy_buffer_to_buffer(&st.outv, 0, &st.staging, 0, (n * 16) as u64);
            if let Some(t) = &self.pass_timer {
                t.resolve(&mut enc);
            }
            self.queue.submit([enc.finish()]);
            sense_ts.map(|ts| {
                let (ttx, trx) = mpsc::channel();
                ts.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    let _ = ttx.send(r);
                });
                trx
            })
        };
        self.stats.last.host_prep_ms = total.elapsed().as_secs_f64() * 1e3;

        // ---- the one wait ----
        let out_stride = self.strides().out;
        let round = std::time::Instant::now();
        let io = self.sense_io.as_ref().expect("sense io");
        let st = self.resident.as_ref().expect("resident");
        let sense_slice = io.staging.slice(0..(n * out_stride * 4) as u64);
        let up_slice = st.staging.slice(0..(n * 16) as u64);
        let (stx, srx) = mpsc::channel();
        sense_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = stx.send(r);
        });
        let (utx, urx) = mpsc::channel();
        up_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = utx.send(r);
        });
        let acct = self.finish_field_read(pending, &mut discard)?;
        self.field_resident = false;
        let io = self.sense_io.as_ref().expect("sense io");
        let st = self.resident.as_ref().expect("resident");
        let sense_slice = io.staging.slice(0..(n * out_stride * 4) as u64);
        let up_slice = st.staging.slice(0..(n * 16) as u64);
        srx.recv().map_err(|e| format!("GPU map callback: {e}"))?.map_err(|e| format!("GPU sense readback: {e}"))?;
        urx.recv().map_err(|e| format!("GPU map callback: {e}"))?.map_err(|e| format!("GPU uptake readback: {e}"))?;
        if let (Some(ts), Some(rx)) = (&self.timestamps, sense_ts_rx) {
            self.stats.last.kernel_ms = read_timestamps(&self.device, ts, rx);
        }
        if let (Some(ts), Some(rx)) = (&self.pyramid_timestamps, pyr_rx) {
            self.stats.last.pyramid_ms = read_timestamps(&self.device, ts, rx);
        }
        if let Some(t) = &self.pass_timer {
            t.add("sense", self.stats.last.kernel_ms.unwrap_or(0.0));
            t.add("pyramid", self.stats.last.pyramid_ms.unwrap_or(0.0));
            t.collect(&self.device);
        }
        self.stats.last.gpu_roundtrip_ms = round.elapsed().as_secs_f64() * 1e3;
        let bk = std::time::Instant::now();
        {
            let view = sense_slice.get_mapped_range().map_err(|e| format!("GPU map range: {e:?}"))?;
            let rows: &[f32] = bytemuck::try_cast_slice(&view).map_err(|e| format!("sense readback alignment: {e:?}"))?;
            let mem_mode = self.mem_mode;
            let copy_row = |g: usize, o: &mut SenseOutput| {
                let row = &rows[g * out_stride..(g + 1) * out_stride];
                o.logits.copy_from_slice(&row[..OUTPUT_WIDTH]);
                o.motion_drive = [row[36] as f64, row[37] as f64];
                o.density0 = row[38] as f64;
                if mem_mode {
                    o.memory.copy_from_slice(&row[40..40 + MEMORY_WIDTH]);
                }
            };
            if single {
                // One weight group: item order is job order, so the rows can be copied on several threads.
                fl_core::par::for_each_mut(workers, sense_out, |g, o| copy_row(g, o));
            } else {
                for (g, &k) in order.iter().enumerate() {
                    copy_row(g, &mut sense_out[k as usize]);
                }
            }
        }
        io.staging.unmap();
        {
            let view = up_slice.get_mapped_range().map_err(|e| format!("GPU map range: {e:?}"))?;
            let rows: &[f32] = bytemuck::try_cast_slice(&view).map_err(|e| format!("uptake readback alignment: {e:?}"))?;
            let to_out = |g: usize| UptakeOut { absorbed: rows[g * 4] as f64, useful_gain: rows[g * 4 + 1] as f64, toxicity_requested: rows[g * 4 + 2] as f64 };
            if single {
                fl_core::par::for_each_mut(workers, uptake_out, |g, u| *u = to_out(g));
            } else {
                for (g, &k) in order.iter().enumerate() {
                    uptake_out[k as usize] = to_out(g);
                }
            }
        }
        st.staging.unmap();
        let granted = uptake_out.iter().map(|o| o.absorbed).sum();
        self.front.order = order;
        self.front.prev = placed;
        self.front.prev_call = self.call;
        self.evict_stale();
        self.stats.last.bookkeeping_ms = bk.elapsed().as_secs_f64() * 1e3;
        self.stats.last.total_ms = total.elapsed().as_secs_f64() * 1e3;
        Ok((acct, granted))
    }
}
