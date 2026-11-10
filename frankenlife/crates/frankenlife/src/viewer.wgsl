// Live viewer. Field colors are the fl-view formulas (basis uploaded from chem_basis).
// Organisms are splats in the same palette. Nothing here is fed back into the world.

struct FieldU {
    gx: u32,
    gy: u32,
    raw: u32,
    stride: u32,
    area: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Cam {
    center: vec2<f32>,
    world: vec2<f32>,
    screen: vec2<f32>,
    zoom: f32,
    reach: f32,
    inv2s2: f32,
    point: f32,
    grid: vec2<f32>,
    bg: vec4<f32>,
    flags: vec4<f32>, // filmic, encode sRGB, bloom, pad
    hud: vec2<f32>,
    pad1: vec2<f32>,
}

@group(0) @binding(0) var<uniform> field_u: FieldU;
@group(0) @binding(1) var<storage, read> energy: array<f32>;
@group(0) @binding(2) var<storage, read> moment: array<f32>;
@group(0) @binding(3) var<storage, read> basis: array<f32>;
@group(0) @binding(4) var<storage, read_write> field_rgb: array<f32>;

@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> sample_rgb: array<f32>;
@group(0) @binding(2) var hdr_tex: texture_2d<f32>;
@group(0) @binding(3) var hud_tex: texture_2d<f32>;
@group(0) @binding(4) var hud_samp: sampler;

fn basis_at(row: u32, k: u32) -> f32 {
    return basis[row * 32u + k];
}

fn chem(cell: u32, e: f32) -> vec3<f32> {
    var c = vec3<f32>(0.0);
    for (var r = 0u; r < 3u; r++) {
        var s = 0.0;
        for (var k = 0u; k < 32u; k++) {
            s += basis_at(r, k) * moment[cell * 32u + k] / e;
        }
        c[r] = s;
    }
    let m = max(max(abs(c.x), abs(c.y)), max(abs(c.z), 1e-6));
    let base = vec3<f32>(0.25, 0.35, 0.55);
    let scale = vec3<f32>(0.75, 0.65, 0.60);
    return clamp(base + scale * max(c / m, vec3<f32>(-0.3)), vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(64)
fn color_field(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x + gid.y * field_u.stride;
    let cells = field_u.gx * field_u.gy;
    if (c >= cells) {
        return;
    }
    let e = energy[c];
    let r = e / field_u.area;
    var rgb = vec3<f32>(0.0);
    if (field_u.raw != 0u) {
        let v = 1.0 - exp(-r * 0.5);
        rgb = vec3<f32>(v);
    } else if (e > 0.0) {
        let glow = (1.0 - exp(-r * 0.35)) * 0.22;
        rgb = chem(c, e) * glow;
    }
    field_rgb[c * 3u] = rgb.x;
    field_rgb[c * 3u + 1u] = rgb.y;
    field_rgb[c * 3u + 2u] = rgb.z;
}

fn wrap_i(v: i32, n: i32) -> i32 {
    let m = v % n;
    if (m < 0) {
        return m + n;
    }
    return m;
}

fn field_texel(x: i32, y: i32) -> vec3<f32> {
    let w = i32(cam.grid.x);
    let h = i32(cam.grid.y);
    let xx = wrap_i(x, w);
    let yy = wrap_i(y, h);
    let i = u32(yy * w + xx) * 3u;
    return vec3<f32>(sample_rgb[i], sample_rgb[i + 1u], sample_rgb[i + 2u]);
}

fn field_bilinear(world_xy: vec2<f32>) -> vec3<f32> {
    let u = world_xy.x / cam.world.x;
    let v = world_xy.y / cam.world.y;
    let fx = fract(u) * cam.grid.x - 0.5;
    let fy = fract(v) * cam.grid.y - 0.5;
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    let c00 = field_texel(x0, y0);
    let c10 = field_texel(x0 + 1, y0);
    let c01 = field_texel(x0, y0 + 1);
    let c11 = field_texel(x0 + 1, y0 + 1);
    return mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec2<f32>,
}

@vertex
fn vs_field(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = vec2<f32>(-1.0, -1.0);
    if (vi == 1u) {
        p = vec2<f32>(3.0, -1.0);
    } else if (vi == 2u) {
        p = vec2<f32>(-1.0, 3.0);
    }
    let uv = p * 0.5 + 0.5;
    // Framebuffer y grows downward; NDC y grows upward. uv.y here is NDC-up,
    // so flip it to match the CPU rasterizer (world +y toward the bottom).
    let pix = vec2<f32>(uv.x * cam.screen.x, (1.0 - uv.y) * cam.screen.y);
    let world = cam.center + (pix - cam.screen * 0.5) / cam.zoom;
    var o: VsOut;
    o.clip = vec4<f32>(p, 0.0, 1.0);
    o.world = world;
    return o;
}

@fragment
fn fs_field(i: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(field_bilinear(i.world), 1.0);
}

struct OrgOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) delta: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) bright: f32,
}

@vertex
fn vs_org(
    @location(0) xy: vec2<f32>,
    @location(1) rgb: vec3<f32>,
    @location(2) bright: f32,
    @builtin(vertex_index) vi: u32,
) -> OrgOut {
    var q = vec2<f32>(-1.0, -1.0);
    if (vi == 1u || vi == 4u || vi == 5u) {
        q.x = 1.0;
    }
    if (vi == 2u || vi == 3u || vi == 5u) {
        q.y = 1.0;
    }
    let pix = (xy - cam.center) * cam.zoom + cam.screen * 0.5 + q * cam.reach;
    let ndc = vec2<f32>(pix.x / cam.screen.x * 2.0 - 1.0, 1.0 - pix.y / cam.screen.y * 2.0);
    var o: OrgOut;
    o.clip = vec4<f32>(ndc, 0.0, 1.0);
    o.delta = q * cam.reach;
    o.color = rgb;
    o.bright = bright;
    return o;
}

@fragment
fn fs_org(i: OrgOut) -> @location(0) vec4<f32> {
    if (cam.point > 0.5) {
        if (max(abs(i.delta.x), abs(i.delta.y)) > 0.5) {
            discard;
        }
        return vec4<f32>(i.color, 1.0);
    }
    let w = exp(-dot(i.delta, i.delta) * cam.inv2s2) * i.bright;
    if (w < 1e-3) {
        discard;
    }
    return vec4<f32>(i.color * w, 1.0);
}

@vertex
fn vs_quad(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = vec2<f32>(-1.0, -1.0);
    if (vi == 1u) {
        p = vec2<f32>(3.0, -1.0);
    } else if (vi == 2u) {
        p = vec2<f32>(-1.0, 3.0);
    }
    var o: VsOut;
    o.clip = vec4<f32>(p, 0.0, 1.0);
    o.world = p * 0.5 + 0.5;
    return o;
}

fn encode(c: vec3<f32>) -> vec3<f32> {
    if (cam.flags.y > 0.5) {
        return pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    }
    return c;
}

@fragment
fn fs_tonemap(@builtin(position) pix: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(pix.xy);
    let c = textureLoad(hdr_tex, p, 0).rgb;
    var glow = vec3<f32>(0.0);
    if (cam.flags.z > 0.0) {
        let r = 7;
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) {
                    continue;
                }
                let s = textureLoad(hdr_tex, p + vec2<i32>(dx, dy) * r, 0).rgb;
                glow += max(s - vec3<f32>(0.6), vec3<f32>(0.0));
            }
        }
        glow /= 8.0;
    }
    let v = c + cam.flags.z * 4.0 * glow;
    var t = v;
    if (cam.flags.x > 0.5) {
        t = vec3<f32>(1.0) - exp(-v * 1.1);
    } else {
        t = min(v, vec3<f32>(1.0));
    }
    let bg = cam.bg.rgb;
    let outc = bg + (vec3<f32>(1.0) - bg) * t;
    return vec4<f32>(encode(outc), 1.0);
}

@vertex
fn vs_hud(@builtin(vertex_index) vi: u32) -> VsOut {
    var q = vec2<f32>(0.0, 0.0);
    if (vi == 1u || vi == 4u || vi == 5u) {
        q.x = 1.0;
    }
    if (vi == 2u || vi == 3u || vi == 5u) {
        q.y = 1.0;
    }
    let pix = q * cam.hud;
    let ndc = vec2<f32>(pix.x / cam.screen.x * 2.0 - 1.0, 1.0 - pix.y / cam.screen.y * 2.0);
    var o: VsOut;
    o.clip = vec4<f32>(ndc, 0.0, 1.0);
    o.world = q;
    return o;
}

@fragment
fn fs_hud(i: VsOut) -> @location(0) vec4<f32> {
    return textureSample(hud_tex, hud_samp, i.world);
}
