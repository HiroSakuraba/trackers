# FrankenLife × Franken stack (integration workspace)

This workspace connects the authoritative `fl-core` model to the stack the
design plan requires. It is kept separate from the core workspace because the
upstream crates pin `nightly-2026-08-31` (see `rust-toolchain.toml`), while
`fl-core` builds on stable. Every upstream dependency is pinned by git revision
to what was compiled and tested here:

| Upstream | Revision | Used for |
|---|---|---|
| Asupersync | `460306de2aa4` | runtime, region-owned tasks, bounded `mpsc` control, `watch` render epochs, cancellation checkpoints |
| FrankenTorch | `287e14d335dd` (the plan's pin) | `bmm_tensor_contiguous_f32_into` population inference with independent weights |
| FrankenNumPy | `20159e89c4ba` (the plan's pin) | `NdLayout` field descriptors, `ArrayStorage::F32`, `fnp-io` `.npz` export |
| FrankenManim | `a12661bc59ce` | native PNG frames: field `ImageMobject` + bucketed `DotCloud`s via the camera route |

## Crates

- **`fl-torch`**: `TorchBackend` implements fl-core's `ControllerBackend`.
  Genomes live in persistent per-topology weight slabs (`[slot, out, in]`, the
  same layout as fl-core genomes, so upload is a copy). A layer for the whole
  population is one batched matmul `Y[b] = W[b]·x[b]`, then bias and `tanh`.
  Genomes upload on first use and are evicted after they stop being evaluated.
- **`fl-fnp`**: describes the field with FrankenNumPy layouts, checks fl-core's
  buffers against them every tick in M0, and exports a committed epoch as `.npz`.
- **`fl-manim`**: `RenderEpoch` (immutable display projection of a committed
  tick) rendered by FrankenManim as one image plus ≤ 12 dot clouds.
- **`frankenlife-m0`**: the plan's M0 program (§22.1). An Asupersync runtime
  owns one region with three tasks: the world owner (sole writer of scientific
  state), a renderer (observer), and a driver (control commands over a bounded
  channel). Render epochs go over a latest-value `watch` channel, so a slow
  renderer skips epochs instead of slowing the world.

## Run

```bash
cd frankenlife/integration
cargo test --release -p fl-torch -p fl-fnp -p fl-manim
cargo run --release -p frankenlife-m0 -- --ticks 256 --frame-every 64       # reference drives, FrankenTorch compared
cargo run --release -p frankenlife-m0 -- --config ../presets/vivid-field-ecology.toml --backend torch --ticks 384
```

`--backend reference|torch|compare`. Output: `runs/m0/m0_report.json`, PNG frames
in `frames/`, and `final_epoch.npz`. The first build compiles all four
upstream workspaces (about 30 minutes on 4 cores).

## Measured here (4-core Linux container, release)

| Run | Result |
|---|---|
| Reference preset, 4,096 organisms × 9,988 params, 256 ticks, `compare` | Max \|Δlogit\| FrankenTorch vs reference over 1,048,576 evaluations: **4.3e-7**. State hash under Asupersync with the renderer attached equals the plain fl-core run (`52ac89749a88dd21`). 5/5 epochs rendered. 4,096 genome uploads total (slabs persist), 160 MB resident weights. |
| Vivid preset, 384 ticks, `reference` | 23.4 ticks/s (world owner single-threaded), hash equals the control run. |
| Vivid preset, 384 ticks, `torch` | 32.0 ticks/s; same population/births at this horizon; different state hash (expected: separate determinism class). |

## Known gaps

- `World::step` runs on the owner task, not Asupersync's blocking pool, and
  FrankenTorch's CPU matmul uses its own Rayon pool. That executor boundary is
  reported in `m0_report.json` (plan §3.4); it is not Asupersync-scheduled
  data parallelism.
- `DotCloud` has one color per cloud, so organism colors are quantized to 12
  buckets. A per-point color/radius bulk primitive belongs upstream (plan §22.4).
- FrankenManim's camera route shows the field image dimmer and bluer than the
  CLI rasterizer; the color handling of image content hasn't been matched yet.
- No Metal path on this Linux host; no FrankenSQLite catalog or durable
  command journal yet.
