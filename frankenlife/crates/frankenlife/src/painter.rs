//! GPU drawer for the live viewer.
//!
//! Field planes are read. The tick's buffers are never written. A resident field
//! stays on the device: the coloring dispatch samples it and the result is a
//! display buffer, thrown away with the frame.

use std::sync::{Arc, Mutex};

use fl_view::chem_basis;

const SHADER: &str = include_str!("viewer.wgsl");

#[derive(Clone, Copy)]
pub struct PaintStats {
    pub cpu_ms: f64,
}

pub struct GpuLink {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

pub enum Planes<'a> {
    /// Resident working field. Read-only.
    Device {
        energy: &'a wgpu::Buffer,
        moment: &'a wgpu::Buffer,
        width: u32,
        height: u32,
        cell_area: f32,
        generation: u64,
    },
    /// Authoritative CPU field (every backend except the resident ones).
    Host {
        energy: &'a [f32],
        moment: &'a [f32],
        width: u32,
        height: u32,
        cell_area: f32,
    },
}

pub struct Scene<'a> {
    pub planes: Planes<'a>,
    pub raw_field: bool,
    pub show_field: bool,
    pub show_organisms: bool,
    pub filmic: bool,
    pub bloom: f32,
    pub background: [f32; 3],
    pub world: [f32; 2],
    pub center: [f32; 2],
    pub zoom: f32,
    pub reach: f32,
    pub point: bool,
    /// 6 floats per organism: x, y, r, g, b, bright. Already wrapped into view tiles.
    pub organisms: &'a [f32],
    pub hud: &'a [u8],
    pub hud_size: (u32, u32),
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FieldU {
    gx: u32,
    gy: u32,
    raw: u32,
    stride: u32,
    area: f32,
    pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CamU {
    center: [f32; 2],
    world: [f32; 2],
    screen: [f32; 2],
    zoom: f32,
    reach: f32,
    inv2s2: f32,
    point: f32,
    grid: [f32; 2],
    bg: [f32; 4],
    flags: [f32; 4],
    hud: [f32; 2],
    pad1: [f32; 2],
}

struct FieldBind {
    key: (u64, u32, u32, bool),
    group: wgpu::BindGroup,
}

pub struct Painter {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    errors: Arc<Mutex<Vec<String>>>,
    shader: wgpu::ShaderModule,
    field_layout: wgpu::BindGroupLayout,
    field_pipe: wgpu::ComputePipeline,
    field_uniform: wgpu::Buffer,
    basis: wgpu::Buffer,
    world_layout: wgpu::BindGroupLayout,
    present_layout: wgpu::BindGroupLayout,
    hud_layout: wgpu::BindGroupLayout,
    field_draw: wgpu::RenderPipeline,
    org_draw: wgpu::RenderPipeline,
    tonemap: wgpu::RenderPipeline,
    hud_pipe: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    cam: wgpu::Buffer,
    scratch_e: Option<wgpu::Buffer>,
    scratch_m: Option<wgpu::Buffer>,
    scratch_cells: usize,
    rgb: Option<wgpu::Buffer>,
    rgb_cells: usize,
    /// Readback for the preview test. Unused by the window.
    #[allow(dead_code)]
    rgb_read: Option<wgpu::Buffer>,
    field_bind: Option<FieldBind>,
    world_bind: Option<wgpu::BindGroup>,
    present_bind: Option<wgpu::BindGroup>,
    hud_bind: Option<wgpu::BindGroup>,
    hdr: Option<wgpu::Texture>,
    hdr_view: Option<wgpu::TextureView>,
    hdr_size: (u32, u32),
    hud_tex: wgpu::Texture,
    hud_view: wgpu::TextureView,
    hud_size: (u32, u32),
    orgs: wgpu::Buffer,
    org_cap: usize,
    offscreen: Option<wgpu::Texture>,
    offscreen_view: Option<wgpu::TextureView>,
    offscreen_size: (u32, u32),
    offscreen_read: Option<wgpu::Buffer>,
}

impl Painter {
    pub fn open_device() -> Result<GpuLink, String> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                    apply_limit_buckets: false,
                })
                .await
                .map_err(|e| format!("no GPU adapter found: {e}"))?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("frankenlife-view"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                })
                .await
                .map_err(|e| format!("could not open a device for the viewer: {e}"))?;
            Ok(GpuLink { instance, adapter, device, queue })
        })
    }

    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self, String> {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let log = errors.clone();
        device.on_uncaptured_error(Arc::new(move |e| {
            let msg = e.to_string();
            eprintln!("viewer wgpu: {msg}");
            log.lock().expect("error log").push(msg);
        }));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("viewer"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let storage = |binding, read_only, stages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: stages,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let uniform = |binding, stages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: stages,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let field_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer-field"),
            entries: &[
                uniform(0, wgpu::ShaderStages::COMPUTE),
                storage(1, true, wgpu::ShaderStages::COMPUTE),
                storage(2, true, wgpu::ShaderStages::COMPUTE),
                storage(3, true, wgpu::ShaderStages::COMPUTE),
                storage(4, false, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let field_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("viewer-field"), bind_group_layouts: &[Some(&field_layout)], immediate_size: 0 });
        let field_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("color-field"),
            layout: Some(&field_pl),
            module: &shader,
            entry_point: Some("color_field"),
            compilation_options: Default::default(),
            cache: None,
        });
        let world_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer-world"),
            entries: &[uniform(0, wgpu::ShaderStages::VERTEX_FRAGMENT), storage(1, true, wgpu::ShaderStages::FRAGMENT)],
        });
        let present_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer-present"),
            entries: &[
                uniform(0, wgpu::ShaderStages::FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
            ],
        });
        let hud_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer-hud"),
            entries: &[
                uniform(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let world_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("viewer-world"), bind_group_layouts: &[Some(&world_layout)], immediate_size: 0 });
        let present_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("viewer-present"), bind_group_layouts: &[Some(&present_layout)], immediate_size: 0 });
        let hud_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("viewer-hud"), bind_group_layouts: &[Some(&hud_layout)], immediate_size: 0 });
        let field_draw = render_pipe(&device, &world_pl, &shader, "vs_field", "fs_field", &[], wgpu::TextureFormat::Rgba16Float, Some(wgpu::BlendState::REPLACE));
        let org_layout = wgpu::VertexBufferLayout {
            array_stride: 24,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 8, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 20, shader_location: 2 },
            ],
        };
        let org_draw = render_pipe(&device, &world_pl, &shader, "vs_org", "fs_org", &[Some(org_layout)], wgpu::TextureFormat::Rgba16Float, Some(wgpu::BlendState::ADDITIVE));
        let tonemap = render_pipe(&device, &present_pl, &shader, "vs_quad", "fs_tonemap", &[], wgpu::TextureFormat::Rgba8Unorm, None);
        let hud_pipe = render_pipe(&device, &hud_pl, &shader, "vs_hud", "fs_hud", &[], wgpu::TextureFormat::Rgba8Unorm, Some(wgpu::BlendState::ALPHA_BLENDING));
        let field_uniform = device.create_buffer(&wgpu::BufferDescriptor { label: Some("field-u"), size: 32, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut basis_f = [0.0f32; 96];
        let b = chem_basis();
        for r in 0..3 {
            basis_f[r * 32..r * 32 + 32].copy_from_slice(&b[r]);
        }
        let basis = device.create_buffer(&wgpu::BufferDescriptor { label: Some("chem-basis"), size: 96 * 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        queue.write_buffer(&basis, 0, bytemuck::cast_slice(&basis_f));
        let cam = device.create_buffer(&wgpu::BufferDescriptor { label: Some("cam"), size: std::mem::size_of::<CamU>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("hud"), mag_filter: wgpu::FilterMode::Nearest, min_filter: wgpu::FilterMode::Nearest, ..Default::default() });
        let (hud_tex, hud_view) = rgba_tex(&device, 1, 1, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let orgs = device.create_buffer(&wgpu::BufferDescriptor { label: Some("orgs"), size: 24, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        Ok(Painter {
            device,
            queue,
            errors,
            shader,
            field_layout,
            field_pipe,
            field_uniform,
            basis,
            world_layout,
            present_layout,
            hud_layout,
            field_draw,
            org_draw,
            tonemap,
            hud_pipe,
            sampler,
            cam,
            scratch_e: None,
            scratch_m: None,
            scratch_cells: 0,
            rgb: None,
            rgb_cells: 0,
            rgb_read: None,
            field_bind: None,
            world_bind: None,
            present_bind: None,
            hud_bind: None,
            hdr: None,
            hdr_view: None,
            hdr_size: (0, 0),
            hud_tex,
            hud_view,
            hud_size: (1, 1),
            orgs,
            org_cap: 1,
            offscreen: None,
            offscreen_view: None,
            offscreen_size: (0, 0),
            offscreen_read: None,
        })
    }

    /// Draw into an offscreen target and return. Does not wait for the GPU.
    pub fn draw_offscreen(&mut self, scene: &Scene<'_>, size: (u32, u32)) -> Result<PaintStats, String> {
        self.ensure_offscreen(size);
        let view = self.offscreen_view.as_ref().expect("offscreen").clone();
        let tonemap = self.tonemap.clone();
        let hud = self.hud_pipe.clone();
        self.draw(scene, &view, size, true, &tonemap, &hud)
    }

    pub fn render_rgba(&mut self, scene: &Scene<'_>, size: (u32, u32)) -> Result<(Vec<u8>, PaintStats), String> {
        let stats = self.draw_offscreen(scene, size)?;
        let tex = self.offscreen.as_ref().expect("offscreen").clone();
        let bytes = read_rgba(&self.device, &self.queue, &tex, size, &mut self.offscreen_read)?;
        self.check_errors()?;
        Ok((bytes, stats))
    }

    /// Draw into `target`. `encode_srgb` when the target format does not encode sRGB itself.
    /// `tonemap` and `hud` must target `target`'s format.
    pub fn draw(&mut self, scene: &Scene<'_>, target: &wgpu::TextureView, size: (u32, u32), encode_srgb: bool, tonemap: &wgpu::RenderPipeline, hud: &wgpu::RenderPipeline) -> Result<PaintStats, String> {
        let t0 = std::time::Instant::now();
        let (w, h, area, generation, host) = match scene.planes {
            Planes::Device { width, height, cell_area, generation, .. } => (width, height, cell_area, generation, false),
            Planes::Host { width, height, cell_area, .. } => (width, height, cell_area, 0, true),
        };
        if host {
            if let Planes::Host { energy, moment, .. } = scene.planes {
                self.upload_host(energy, moment, w, h);
            }
        }
        self.ensure_rgb((w * h) as usize);
        let groups = (w * h).div_ceil(64);
        let gx = groups.min(65535);
        let gy = groups.div_ceil(gx).max(1);
        let stride = gx * 64;
        let fu = FieldU { gx: w, gy: h, raw: u32::from(scene.raw_field), stride, area, pad: [0.0; 3] };
        self.queue.write_buffer(&self.field_uniform, 0, bytemuck::bytes_of(&fu));
        self.ensure_field_bind(scene, generation, host);
        let radius = if scene.point { 0.5 } else { (scene.reach / 2.5).max(0.6) };
        let cam = CamU {
            center: scene.center,
            world: scene.world,
            screen: [size.0 as f32, size.1 as f32],
            zoom: scene.zoom,
            reach: scene.reach.max(0.6),
            inv2s2: 1.0 / (2.0 * radius * radius),
            point: if scene.point { 1.0 } else { 0.0 },
            grid: [w as f32, h as f32],
            bg: [scene.background[0], scene.background[1], scene.background[2], 1.0],
            flags: [if scene.filmic { 1.0 } else { 0.0 }, if encode_srgb { 1.0 } else { 0.0 }, if scene.filmic { scene.bloom } else { 0.0 }, 0.0],
            hud: [scene.hud_size.0 as f32, scene.hud_size.1 as f32],
            pad1: [0.0; 2],
        };
        self.queue.write_buffer(&self.cam, 0, bytemuck::bytes_of(&cam));
        self.ensure_hdr(size);
        self.ensure_hud(scene);
        self.ensure_draw_bind();
        let n_org = if scene.show_organisms { scene.organisms.len() / 6 } else { 0 };
        if n_org > 0 {
            self.ensure_orgs(n_org);
            self.queue.write_buffer(&self.orgs, 0, bytemuck::cast_slice(&scene.organisms[..n_org * 6]));
        }
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("viewer") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("color-field"), timestamp_writes: None });
            pass.set_pipeline(&self.field_pipe);
            pass.set_bind_group(0, &self.field_bind.as_ref().expect("field bind").group, &[]);
            if scene.show_field {
                pass.dispatch_workgroups(gx, gy, 1);
            }
        }
        let hdr_view = self.hdr_view.as_ref().expect("hdr").clone();
        let world_bind = self.world_bind.as_ref().expect("world bind").clone();
        let present_bind = self.present_bind.as_ref().expect("present bind").clone();
        let hud_bind = self.hud_bind.as_ref().expect("hud bind").clone();
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("world"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &hdr_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &world_bind, &[]);
            if scene.show_field {
                pass.set_pipeline(&self.field_draw);
                pass.draw(0..3, 0..1);
            }
            if n_org > 0 {
                pass.set_pipeline(&self.org_draw);
                pass.set_vertex_buffer(0, self.orgs.slice(..(n_org * 24) as u64));
                pass.draw(0..6, 0..n_org as u32);
            }
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tonemap"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &present_bind, &[]);
            pass.set_pipeline(tonemap);
            pass.draw(0..3, 0..1);
            if scene.hud_size.0 > 0 && scene.hud_size.1 > 0 && !scene.hud.is_empty() {
                pass.set_bind_group(0, &hud_bind, &[]);
                pass.set_pipeline(hud);
                pass.draw(0..6, 0..1);
            }
        }
        self.queue.submit([enc.finish()]);
        self.check_errors()?;
        Ok(PaintStats { cpu_ms: t0.elapsed().as_secs_f64() * 1e3 })
    }

    /// RGB floats of the colored field, one cell after the most recent draw.
    #[allow(dead_code)]
    pub fn read_field_rgb(&mut self) -> Result<Vec<f32>, String> {
        let rgb = self.rgb.as_ref().ok_or("field has not been colored")?.clone();
        let cells = self.rgb_cells;
        let bytes = (cells * 3 * 4) as u64;
        if self.rgb_read.as_ref().is_none_or(|b| b.size() < bytes) {
            self.rgb_read = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rgb-read"),
                size: bytes.max(4),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let read = self.rgb_read.as_ref().expect("read").clone();
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rgb-read") });
        enc.copy_buffer_to_buffer(&rgb, 0, &read, 0, bytes);
        self.queue.submit([enc.finish()]);
        let slice = read.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll: {e}"))?;
        rx.recv().map_err(|e| format!("map callback: {e}"))?.map_err(|e| format!("read field: {e}"))?;
        let out = {
            let view = slice.get_mapped_range().map_err(|e| format!("map range: {e:?}"))?;
            bytemuck::cast_slice(&view).to_vec()
        };
        read.unmap();
        Ok(out)
    }

    /// Pipelines whose color target is `format` (the swapchain).
    pub fn pipelines_for(&self, format: wgpu::TextureFormat) -> (wgpu::RenderPipeline, wgpu::RenderPipeline) {
        let present_pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("viewer-present"), bind_group_layouts: &[Some(&self.present_layout)], immediate_size: 0 });
        let hud_pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("viewer-hud"), bind_group_layouts: &[Some(&self.hud_layout)], immediate_size: 0 });
        let tonemap = render_pipe(&self.device, &present_pl, &self.shader, "vs_quad", "fs_tonemap", &[], format, None);
        let hud = render_pipe(&self.device, &hud_pl, &self.shader, "vs_hud", "fs_hud", &[], format, Some(wgpu::BlendState::ALPHA_BLENDING));
        (tonemap, hud)
    }

    fn upload_host(&mut self, energy: &[f32], moment: &[f32], w: u32, h: u32) {
        let cells = (w * h) as usize;
        if self.scratch_cells < cells {
            let mk = |label, floats| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: (floats * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            self.scratch_e = Some(mk("host-e", cells));
            self.scratch_m = Some(mk("host-m", cells * 32));
            self.scratch_cells = cells;
            self.field_bind = None;
        }
        self.queue.write_buffer(self.scratch_e.as_ref().expect("e"), 0, bytemuck::cast_slice(energy));
        self.queue.write_buffer(self.scratch_m.as_ref().expect("m"), 0, bytemuck::cast_slice(moment));
    }

    fn ensure_rgb(&mut self, cells: usize) {
        if self.rgb_cells >= cells && self.rgb.is_some() {
            return;
        }
        let cap = cells.next_power_of_two().max(64);
        self.rgb = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field-rgb"),
            size: (cap * 3 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.rgb_cells = cap;
        self.field_bind = None;
        self.world_bind = None;
        self.present_bind = None;
        self.hud_bind = None;
    }

    fn ensure_field_bind(&mut self, scene: &Scene<'_>, generation: u64, host: bool) {
        let (ew, mw) = match scene.planes {
            Planes::Device { energy, moment, .. } => (energy, moment),
            Planes::Host { .. } => (self.scratch_e.as_ref().expect("host energy"), self.scratch_m.as_ref().expect("host moment")),
        };
        let key = (generation, scene_wh(&scene.planes).0, scene_wh(&scene.planes).1, host);
        if self.field_bind.as_ref().is_some_and(|b| b.key == key) {
            return;
        }
        let rgb = self.rgb.as_ref().expect("rgb");
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer-field"),
            layout: &self.field_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.field_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ew.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: mw.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.basis.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: rgb.as_entire_binding() },
            ],
        });
        self.field_bind = Some(FieldBind { key, group });
        self.world_bind = None;
        self.present_bind = None;
        self.hud_bind = None;
    }

    fn ensure_hdr(&mut self, size: (u32, u32)) {
        let size = (size.0.max(1), size.1.max(1));
        if self.hdr_size == size && self.hdr.is_some() {
            return;
        }
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr"),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.hdr_view = Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
        self.hdr = Some(tex);
        self.hdr_size = size;
        self.world_bind = None;
        self.present_bind = None;
        self.hud_bind = None;
    }

    fn ensure_hud(&mut self, scene: &Scene<'_>) {
        let (w, h) = if scene.hud.is_empty() { (1, 1) } else { scene.hud_size };
        if self.hud_size != (w, h) {
            let (tex, view) = rgba_tex(&self.device, w.max(1), h.max(1), wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
            self.hud_tex = tex;
            self.hud_view = view;
            self.hud_size = (w.max(1), h.max(1));
            self.world_bind = None;
        self.present_bind = None;
        self.hud_bind = None;
        }
        if !scene.hud.is_empty() {
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let row = (w * 4).div_ceil(align) * align;
            let packed = if row == w * 4 {
                scene.hud.to_vec()
            } else {
                let mut out = vec![0u8; (row * h) as usize];
                for y in 0..h as usize {
                    let s = y * w as usize * 4;
                    let d = y * row as usize;
                    out[d..d + w as usize * 4].copy_from_slice(&scene.hud[s..s + w as usize * 4]);
                }
                out
            };
            self.queue.write_texture(
                self.hud_tex.as_image_copy(),
                &packed,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }
    }

    fn ensure_draw_bind(&mut self) {
        if self.world_bind.is_some() && self.present_bind.is_some() && self.hud_bind.is_some() {
            return;
        }
        let rgb = self.rgb.as_ref().expect("rgb");
        let hdr = self.hdr_view.as_ref().expect("hdr view");
        self.world_bind = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer-world"),
            layout: &self.world_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.cam.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: rgb.as_entire_binding() },
            ],
        }));
        self.present_bind = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer-present"),
            layout: &self.present_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.cam.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(hdr) },
            ],
        }));
        self.hud_bind = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer-hud"),
            layout: &self.hud_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.cam.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.hud_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        }));
    }

    fn ensure_orgs(&mut self, n: usize) {
        if self.org_cap >= n {
            return;
        }
        let cap = n.next_power_of_two();
        self.orgs = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orgs"),
            size: (cap * 24) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.org_cap = cap;
    }

    fn ensure_offscreen(&mut self, size: (u32, u32)) {
        if self.offscreen_size == size && self.offscreen.is_some() {
            return;
        }
        let (tex, view) = rgba_tex(&self.device, size.0.max(1), size.1.max(1), wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        self.offscreen = Some(tex);
        self.offscreen_view = Some(view);
        self.offscreen_size = size;
    }

    fn check_errors(&self) -> Result<(), String> {
        let mut log = self.errors.lock().expect("error log");
        if log.is_empty() {
            return Ok(());
        }
        let msg = log.join("; ");
        log.clear();
        Err(msg)
    }
}

fn scene_wh(p: &Planes<'_>) -> (u32, u32) {
    match *p {
        Planes::Device { width, height, .. } | Planes::Host { width, height, .. } => (width, height),
    }
}

fn render_pipe(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    vs: &str,
    fs: &str,
    buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(fs),
        layout: Some(layout),
        vertex: wgpu::VertexState { module: shader, entry_point: Some(vs), compilation_options: Default::default(), buffers: buffers },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fs),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn rgba_tex(device: &wgpu::Device, w: u32, h: u32, usage: wgpu::TextureUsages) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rgba"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn read_rgba(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture, size: (u32, u32), cache: &mut Option<wgpu::Buffer>) -> Result<Vec<u8>, String> {
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let row = (size.0 * 4).div_ceil(align) * align;
    let bytes = row as u64 * size.1 as u64;
    if cache.as_ref().is_none_or(|b| b.size() < bytes) {
        *cache = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame-read"),
            size: bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }
    let buf = cache.as_ref().expect("read").clone();
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame-read") });
    enc.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(size.1) } },
        wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    let slice = buf.slice(..bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll: {e}"))?;
    rx.recv().map_err(|e| format!("map callback: {e}"))?.map_err(|e| format!("read frame: {e}"))?;
    let mut rgb = vec![0u8; (size.0 * size.1 * 3) as usize];
    {
        let view = slice.get_mapped_range().map_err(|e| format!("map range: {e:?}"))?;
        for y in 0..size.1 as usize {
            for x in 0..size.0 as usize {
                let s = y * row as usize + x * 4;
                let d = (y * size.0 as usize + x) * 3;
                rgb[d..d + 3].copy_from_slice(&view[s..s + 3]);
            }
        }
    }
    buf.unmap();
    Ok(rgb)
}

