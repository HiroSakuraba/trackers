//! Headless CPU rasterizer for committed world epochs.
//!
//! This is a stand-in for the plan's FrankenManim bulk-particle primitive: it
//! only *reads* a committed `World` and never feeds anything back. Colors come
//! from declared, stable mappings (founder lineage, genetic affinity, field
//! composition), never fresh per-frame randomness.

use fl_core::field::Field;
use fl_core::World;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum View {
    /// Field glow + lineage-colored organisms + bloom.
    Composite,
    /// Organisms colored by genetic chemical affinity.
    Chemistry,
    /// Raw diagnostic: field energy density only, no bloom or color mixing.
    Field,
    /// Raw diagnostic: one pixel per organism, lineage color, no bloom.
    Raw,
}

#[derive(Debug, Clone)]
pub struct RenderOpts {
    pub size: usize,
    pub view: View,
    pub bloom_strength: f32,
    pub point_radius: f32,
    pub background: [f32; 3],
    pub telemetry: Vec<String>,
}

use fl_view::{chem_basis, chem_color, field_cell_rgb, lineage_color};

pub fn render(world: &World, o: &RenderOpts) -> Vec<u8> {
    let n = o.size;
    let mut hdr = vec![0.0f32; n * n * 3];
    let basis = chem_basis();
    let field = &world.field;
    let (w, h) = (field.width(), field.height());
    let px_per_unit = n as f64 / w.max(h);

    if matches!(o.view, View::Composite | View::Chemistry | View::Field) {
        draw_field(field, &mut hdr, n, &basis, o.view == View::Field);
    }

    if o.view != View::Field {
        let orgs = &world.orgs;
        let r = (o.point_radius as f64 * n as f64 / 1024.0).max(0.6);
        let reach = (r * 2.5).ceil() as i64;
        let inv2s2 = 1.0 / (2.0 * r * r);
        for i in 0..orgs.len() {
            let col = match o.view {
                View::Chemistry => chem_color(&orgs.genome[i].affinity, &basis),
                _ => lineage_color(orgs.lineage[i]),
            };
            let e = orgs.energy[i];
            let bright = (0.35 + 1.4 * e / (e + 3.0)) as f32;
            let cx = orgs.x[i] * px_per_unit;
            let cy = orgs.y[i] * px_per_unit;
            if o.view == View::Raw {
                let (x, y) = (cx as usize % n, cy as usize % n);
                let p = (y * n + x) * 3;
                for k in 0..3 {
                    hdr[p + k] += col[k];
                }
                continue;
            }
            let (ix, iy) = (cx.floor() as i64, cy.floor() as i64);
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    let (px, py) = (ix + dx, iy + dy);
                    let fx = px as f64 + 0.5 - cx;
                    let fy = py as f64 + 0.5 - cy;
                    let wgt = (-(fx * fx + fy * fy) * inv2s2).exp() as f32 * bright;
                    if wgt < 1e-3 {
                        continue;
                    }
                    // Consistent treatment of the toroidal seam: wrap splats.
                    let (x, y) = (px.rem_euclid(n as i64) as usize, py.rem_euclid(n as i64) as usize);
                    let p = (y * n + x) * 3;
                    for k in 0..3 {
                        hdr[p + k] += col[k] * wgt;
                    }
                }
            }
        }
    }

    if matches!(o.view, View::Composite | View::Chemistry) && o.bloom_strength > 0.0 {
        bloom(&mut hdr, n, o.bloom_strength);
    }

    let mut rgb = vec![0u8; n * n * 3];
    for p in 0..n * n {
        for k in 0..3 {
            let v = hdr[p * 3 + k];
            let t = if matches!(o.view, View::Raw | View::Field) { v.min(1.0) } else { 1.0 - (-v * 1.1).exp() };
            let c = o.background[k] + (1.0 - o.background[k]) * t;
            rgb[p * 3 + k] = (c.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8;
        }
    }
    let scale = (n / 340).max(1);
    for (li, line) in o.telemetry.iter().enumerate() {
        draw_text(&mut rgb, n, 6 * scale, 6 * scale + li * 7 * scale, scale, line, [200, 220, 235]);
    }
    rgb
}

fn draw_field(field: &Field, hdr: &mut [f32], n: usize, basis: &[[f32; 32]; 3], raw: bool) {
    let cell_rgb = field_cell_rgb(field, basis, raw);
    // Bilinear upsampling of cell centers to pixels (periodic).
    for py in 0..n {
        let fy = (py as f64 + 0.5) / n as f64 * field.gy as f64 - 0.5;
        let y0 = fy.floor();
        let ty = (fy - y0) as f32;
        let y0 = (y0 as i64).rem_euclid(field.gy as i64) as usize;
        let y1 = (y0 + 1) % field.gy;
        for px in 0..n {
            let fx = (px as f64 + 0.5) / n as f64 * field.gx as f64 - 0.5;
            let x0 = fx.floor();
            let tx = (fx - x0) as f32;
            let x0 = (x0 as i64).rem_euclid(field.gx as i64) as usize;
            let x1 = (x0 + 1) % field.gx;
            let p = (py * n + px) * 3;
            for k in 0..3 {
                let g = |x: usize, y: usize| cell_rgb[(y * field.gx + x) * 3 + k];
                hdr[p + k] += (1.0 - tx) * (1.0 - ty) * g(x0, y0) + tx * (1.0 - ty) * g(x1, y0) + (1.0 - tx) * ty * g(x0, y1) + tx * ty * g(x1, y1);
            }
        }
    }
}

/// Controlled bloom: quarter-resolution threshold + separable box blur, added back softly.
fn bloom(hdr: &mut [f32], n: usize, strength: f32) {
    let q = (n / 4).max(1);
    let mut low = vec![0.0f32; q * q * 3];
    for y in 0..q {
        for x in 0..q {
            for k in 0..3 {
                let mut s = 0.0;
                for dy in 0..4 {
                    for dx in 0..4 {
                        let (sx, sy) = ((x * 4 + dx).min(n - 1), (y * 4 + dy).min(n - 1));
                        s += (hdr[(sy * n + sx) * 3 + k] - 0.6).max(0.0);
                    }
                }
                low[(y * q + x) * 3 + k] = s / 16.0;
            }
        }
    }
    let mut tmp = low.clone();
    for _ in 0..3 {
        box_blur(&low, &mut tmp, q, 3, true);
        box_blur(&tmp, &mut low, q, 3, false);
    }
    for y in 0..n {
        for x in 0..n {
            let (lx, ly) = ((x / 4).min(q - 1), (y / 4).min(q - 1));
            for k in 0..3 {
                hdr[(y * n + x) * 3 + k] += strength * 4.0 * low[(ly * q + lx) * 3 + k];
            }
        }
    }
}

fn box_blur(src: &[f32], dst: &mut [f32], q: usize, r: usize, horizontal: bool) {
    let norm = 1.0 / (2 * r + 1) as f32;
    for y in 0..q {
        for x in 0..q {
            for k in 0..3 {
                let mut s = 0.0;
                for d in 0..=2 * r {
                    let o = (d as i64 - r as i64) as i64;
                    let (sx, sy) = if horizontal { ((x as i64 + o).rem_euclid(q as i64) as usize, y) } else { (x, (y as i64 + o).rem_euclid(q as i64) as usize) };
                    s += src[(sy * q + sx) * 3 + k];
                }
                dst[(y * q + x) * 3 + k] = s * norm;
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
        '-' => "000000111000000", '%' => "101001010100101", '(' => "010100100100010", ')' => "010001001001010",
        _ => "000000000000000",
    }
}

fn draw_text(rgb: &mut [u8], n: usize, x0: usize, y0: usize, s: usize, text: &str, col: [u8; 3]) {
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
                        if x < n && y < n {
                            let p = (y * n + x) * 3;
                            rgb[p..p + 3].copy_from_slice(&col);
                        }
                    }
                }
            }
        }
    }
}

pub fn write_png(path: &std::path::Path, rgb: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header().map_err(std::io::Error::other)?;
    wr.write_image_data(rgb).map_err(std::io::Error::other)?;
    Ok(())
}

pub fn parse_hex(s: &str) -> [f32; 3] {
    let s = s.trim_start_matches('#');
    let p = |i: usize| u8::from_str_radix(s.get(i..i + 2).unwrap_or("00"), 16).unwrap_or(0) as f32 / 255.0;
    // Background is given in sRGB; convert to linear for compositing.
    [p(0).powf(2.2), p(2).powf(2.2), p(4).powf(2.2)]
}
