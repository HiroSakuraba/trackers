//! FrankenManim rendering of committed FrankenLife epochs (plan §13).
//!
//! A [`RenderEpoch`] is the bounded, immutable display projection of one
//! committed tick (plan §9.5, §13.3): organism positions, a quantized display
//! color and energy, plus the field as an RGBA image. It is built from a
//! committed `World` and never touches scientific state.
//!
//! The world is lowered to a *small number of batched objects* (plan §13.2):
//! one `ImageMobject` for the whole field and one `DotCloud` per display-color
//! bucket, never one mobject per organism. FrankenManim then owns composition,
//! camera framing and the native PNG/GIF/Y4M encoders.
//!
//! Gap recorded for upstream (plan §22.4): `DotCloud` takes a single color, so
//! per-organism colors are quantized into `BUCKETS` palette buckets here; a
//! per-point color/radius bulk primitive would remove that quantization.

#![forbid(unsafe_code)]

use std::path::Path;

use fl_core::World;
use fl_view::{chem_basis, field_cell_rgb, lineage_color, PALETTE};
use fmn::prelude::*;
use fmn_library::{DotCloud, ImageMobject};

/// Display-color buckets: six palette hues × two energy brightness levels.
pub const BUCKETS: usize = 12;

/// Immutable display projection of one committed tick.
#[derive(Debug, Clone)]
pub struct RenderEpoch {
    pub tick: u64,
    pub simulation_seconds: f64,
    pub population: usize,
    pub world_size: (f64, f64),
    /// Organism positions grouped by display bucket.
    pub buckets: Vec<Vec<[f64; 2]>>,
    /// Field image (one pixel per cell, sRGB RGBA8, row 0 = y 0).
    pub field_px: (u32, u32),
    pub field_rgba: Vec<u8>,
}

fn nearest_palette(c: [f32; 3]) -> usize {
    (0..PALETTE.len())
        .min_by(|&a, &b| {
            let d = |p: [f32; 3]| (0..3).map(|k| (p[k] - c[k]).powi(2)).sum::<f32>();
            d(PALETTE[a]).total_cmp(&d(PALETTE[b]))
        })
        .unwrap_or(0)
}

impl RenderEpoch {
    pub fn from_world(w: &World) -> RenderEpoch {
        let o = &w.orgs;
        let mut buckets = vec![Vec::new(); BUCKETS];
        for i in 0..o.len() {
            let hue = nearest_palette(lineage_color(o.lineage[i]));
            let bright = (o.energy[i] >= 1.5) as usize;
            buckets[hue * 2 + bright].push([o.x[i], o.y[i]]);
        }
        let f = &w.field;
        let rgb = field_cell_rgb(f, &chem_basis(), false);
        let bg = [0.008f32, 0.012, 0.031];
        let mut rgba = vec![0u8; f.cells() * 4];
        for c in 0..f.cells() {
            for k in 0..3 {
                let t = 1.0 - (-rgb[c * 3 + k] * 1.1).exp();
                let lin = bg[k] + (1.0 - bg[k]) * t;
                rgba[c * 4 + k] = (lin.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8;
            }
            rgba[c * 4 + 3] = 255;
        }
        RenderEpoch {
            tick: w.tick,
            simulation_seconds: w.simulation_seconds(),
            population: o.len(),
            world_size: (f.width(), f.height()),
            buckets,
            field_px: (f.gx as u32, f.gy as u32),
            field_rgba: rgba,
        }
    }
}

/// FrankenManim scene that lays out one epoch.
struct EpochScene<'a> {
    epoch: &'a RenderEpoch,
    frame_height: f64,
}

impl SceneConstruct for EpochScene<'_> {
    fn construct(&mut self, stage: &mut Stage<'_>) -> fmn::Result<()> {
        let e = self.epoch;
        let side = self.frame_height;
        // Manim's y axis points up; flip rows so world y=0 is at the top like the CLI frames.
        let (gw, gh) = e.field_px;
        let mut flipped = vec![0u8; e.field_rgba.len()];
        let row = gw as usize * 4;
        for y in 0..gh as usize {
            flipped[y * row..(y + 1) * row].copy_from_slice(&e.field_rgba[(gh as usize - 1 - y) * row..(gh as usize - y) * row]);
        }
        let field = ImageMobject::from_rgba8(gw, gh, flipped).map_err(|err| fmn::Error::from(fmn::LibraryError::from(err)))?.with_height(side).with_z_index(0);
        stage.add(field)?;
        let (ww, wh) = e.world_size;
        let to_scene = |p: [f64; 2]| -> Vec3 { [(p[0] / ww - 0.5) * side, (0.5 - p[1] / wh) * side, 0.0] };
        let radius = side / ww * 0.28;
        for (b, pts) in e.buckets.iter().enumerate() {
            if pts.is_empty() {
                continue;
            }
            let c = PALETTE[b / 2];
            let gain = if b % 2 == 1 { 1.0 } else { 0.62 };
            let color = Srgb { r: (c[0] * gain) as f64, g: (c[1] * gain) as f64, b: (c[2] * gain) as f64 };
            let cloud = DotCloud::new(pts.iter().map(|&p| to_scene(p))).with_radius(radius).with_glow_factor(1.0).colored(color, 1.0);
            stage.add(cloud)?;
        }
        stage.wait(1.0 / 30.0)?;
        Ok(())
    }
}

/// Render one epoch to a still PNG through FrankenManim's native encoder.
pub fn render_png(epoch: &RenderEpoch, path: &Path, pixels: u32) -> Result<RenderReport, String> {
    let mut opts = RenderOptions::with_format(path, RenderFormat::Png).map_err(|e| format!("{e}"))?;
    opts.config.camera.resolution = (pixels, pixels);
    opts.config.camera.background_color = "#020308".into();
    opts.config.camera.fps = 30;
    // Image content (the field) is drawn by FrankenManim's camera route.
    opts.camera = Some(opts.camera_config().map_err(|e| format!("{e}"))?);
    let frame_height = opts.config.sizes.frame_height;
    let mut scene = EpochScene { epoch, frame_height };
    render(&mut scene, opts).map_err(|e| format!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fl_core::Config;

    #[test]
    fn epoch_projection_counts_every_organism() {
        let mut c = Config::reference();
        c.population.initial_count = 300;
        let w = World::new(c).unwrap();
        let e = RenderEpoch::from_world(&w);
        assert_eq!(e.buckets.iter().map(Vec::len).sum::<usize>(), 300);
        assert_eq!(e.field_rgba.len(), 128 * 128 * 4);
    }
}
