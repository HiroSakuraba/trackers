# FrankenLife — first living world (FieldEcologyV1)

This directory implements the **first useful slice** of the *FrankenLife Design
Plan* (§22.2, §26.2 steps 2–3, Appendix E): one complete cycle in which

source energy enters a field → organisms sense it → **independent neural
controllers** decide → chemistry causes gain/loss and motion → paid emission
changes the field → reproduction produces a **mutated child** → death releases
resources → the committed world is rendered.

It is a self-contained Rust workspace. `fl-core`, `fl-view` and the CLI use
`#![forbid(unsafe_code)]`. The WebAssembly boundary `fl-wasm` uses
`#![deny(unsafe_code)]` with one allowed construct, the `#[no_mangle]`
export attribute (the lint counts it); it performs no unsafe operations and
uses no JS-glue crates.

```
frankenlife/
├── presets/
│   ├── reference-field-ecology.toml   # Appendix A starter profile (one documented deviation)
│   └── vivid-field-ecology.toml       # declared viability-calibrated profile
└── crates/
    ├── fl-core/       # authoritative model: config, RNG, field, sensors, MLPs, genomes, tick, ledger, checkpoints
    ├── fl-gpu/        # wgpu GPU inference backend (fused MLP shader, GPU-resident weight slabs)
    ├── fl-view/       # declared display mappings (lineage / chemistry / field colors) shared by all views
    ├── fl-wasm/       # WebAssembly boundary for the browser Studio
    └── frankenlife/   # CLI: validate, run, view, inspect, metrics, headless renderer
studio/
    studio.template.html, build.sh   # single-file browser Studio (wasm inlined)
```

## Studio (browser)

`./studio/build.sh` compiles `fl-core` to `wasm32-unknown-unknown` (SIMD128)
and inlines it into `studio/frankenlife-studio.html`, a self-contained page
that runs the same engine live: play/pause/step, ticks-per-frame, zoom and
pan on the torus, lineage / chemistry / raw-field views, click-to-inspect
(energy, ancestry, network, sensing scale, decoded decisions, affinity), erase
and feed brushes applied at the next tick boundary, an in-memory snapshot to
branch from, and honest clocks (achieved ticks/s and speed versus real time are
shown separately from the requested ticks per frame). Browser runs are
single-threaded; at 2,048 founders with 4,548-parameter networks headless
Chromium here manages ~55 ticks/s (≈0.4× real time).

Needs `rustup target add wasm32-unknown-unknown`.

## Quick start

```bash
cd frankenlife
cargo test --release                      # 47 tests (the GPU tests skip without an adapter)
cargo run --release -- validate           # derived capacity & cost report for the reference preset
cargo run --release -- run --config presets/vivid-field-ecology.toml \
    --seconds 45 --frame-every 4 --size 720 --out runs/vivid
ffmpeg -framerate 32 -i runs/vivid/frames/frame_%05d.png -pix_fmt yuv420p runs/vivid.mp4
```

`run` writes `metrics.jsonl` (every `metrics_interval_ticks`), PNG frames,
`inspection.json` (the most derived organisms: energy, ancestry, topology,
sensing scale, held raw outputs and their decoding, current observation
vector, mutation record) and `manifest.json` (full effective config, model
version, validation report, ledger, totals, interventions, final state hash).

GPU inference (wgpu: DX12 / Vulkan / Metal) — see [WINDOWS-GPU.md](WINDOWS-GPU.md):

```bash
cargo run --release -- gpu-check                     # open the GPU, verify against the CPU, benchmark
cargo run --release -- run --config presets/vivid-field-ecology.toml --backend gpu --seconds 30
cargo run --release -- run --config presets/target-20k-field-ecology.toml --backend gpu-fused --frame-every 0
```

`--backend gpu-resident` keeps the field on the GPU across steps: transport,
sensing, inference, uptake exchange and deposits run there, with deterministic
per-cell reductions (counting sort by cell, bins sorted by organism index), so
two runs on the same GPU are bit-identical and exact checkpoint/resume still
works. `--backend gpu-field` runs only the field step on the GPU;
`--backend gpu-fused` also moves the sensor pyramid and per-organism sensing to
the GPU (inputs never leave it); it agrees with CPU sensing to ~1e-5 on logits
and ~6e-5 relative on the motion drive and density.

`--backend gpu-realtime` is that path with one GPU wait for the front half of
a tick and one for the back half; on a GTX 1080 it runs the 5,000-organism
preset at about 190 steps/s (128 is real time). See
[WINDOWS-GPU.md](WINDOWS-GPU.md).

## Live viewer

`view` opens a desktop window on the same tick as `run`, including
`gpu-realtime`. It draws the committed world with the `fl-view` lineage,
chemistry and field colors. Frames are not saved. Pause, single-step, and the
live-tuning keys (energy, source speed, diffusion, decay, mutation rate, source
chemistry) are the ways a watched run can diverge; left alone for the same
config, seed and tick count, the final state hash matches `run`.

Press `P`, type a number and press Enter to restart the run with that many
organisms (Backspace edits, Esc cancels). The restart builds a fresh world from
the launch settings and seed, so it discards the current run and any live
tuning; the window, camera and speed stay. Counts the config cannot hold (for
example over its memory budget) are refused with a message on screen.

```bash
cargo run --release -- view --config presets/target-5k-field-ecology.toml \
    --backend gpu-realtime --seconds 86400 --out runs/rt5-live
```

`--seconds 0` (the default) runs until the window closes. The Windows command,
controls, and what the title bar is counting are in
[WINDOWS-GPU.md](WINDOWS-GPU.md). `--screenshot frame.png` draws one frame
headlessly and exits, for a machine with no display.

Performance targets from the revised plan: **20,000 organisms first**
(`presets/target-20k-field-ecology.toml`, 256² field), 80,000 as the stretch goal.
Measure with `--frame-every 0`; every run reports per-phase step timings
(console, `metrics.jsonl`, `manifest.json` → `"timing"`), and `gpu-check`
reports host prep, GPU submit/wait, and an optional timestamp span separately.
Submit/wait is not GPU kernel time.

`--profile` is off by default. It adds CPU stage timers (sensing, chemistry/field,
networks, transfers/sync, other, render, checkpoint) and writes `profile.json`
next to the run. It does not change the tick. On a GPU that already supports
`TIMESTAMP_QUERY`, resident uptake and finish also record a timestamp span in
the wait those calls already perform. The Windows command for the 20k
gpu-resident profile is in [WINDOWS-GPU.md](WINDOWS-GPU.md). A cloud or software
GPU timing is not a GTX 1080 measurement.

The stretch workload (80,000 organisms, 1024² field):
`presets/target-80k-field-ecology.toml` (a workload definition, not a qualified
ecology). The plan itself, revised to the 80,000 target, is in
[docs/FrankenLife_Design_Plan.html](docs/FrankenLife_Design_Plan.html).

Checkpoints and branches:

```bash
# exact checkpoints every 10 simulated seconds (+ one at the end)
cargo run --release -- run --config presets/vivid-field-ecology.toml --seconds 30 --checkpoint-every 10 --out runs/a
# resume: bit-identical to never having stopped (any worker count)
cargo run --release -- run --resume runs/a/checkpoints/tick_0000003840.flck --seconds 30 --out runs/a2
# branch: same checkpoint, plus an intervention, for comparison with the control
cargo run --release -- run --resume runs/a/checkpoints/tick_0000003840.flck --seconds 30 --erase 3840:64:64:30 --out runs/a-erased
# inspect organisms inside a checkpoint
cargo run --release -- inspect runs/a/checkpoints/tick_0000003840.flck --top 3
```

Other options: `--view composite|chemistry|field|raw`, `--workers N`,
`--seed S`, `--exhibition` (stable-order birth rejection instead of pausing at
capacity), `--erase TICK:X:Y:RADIUS` (repeatable erase-brush intervention,
residual energy recycled to the field).

## What the model does

Everything below follows the plan's equations; section numbers refer to it.

| Area | Implementation |
|---|---|
| Field (§4.2, §4.5) | 128×128 periodic grid, F32 storage, energy `e ≥ 0` + 32 signed moments with `‖m‖ ≤ e`; positive five-point diffusion with automatic subcycling; exponential decay charged to dissipation; moving seeded Gaussian sources (smooth-step torus paths, normalized truncated minimum-image kernel, fixed total power, midpoint-time sampling). |
| Exchange (§5.1–5.2) | `grid-cic-v1` bilinear stencils with merged wrapped cells; two-pass demand/grant `α_c = min(1, e_c/A_c)` so no cell grants more than it holds and thread order is irrelevant; proportional moment removal. |
| Chemistry (§5.3) | `η = η_max tanh(k a·z − b)` applied **per granted packet** (fixture C.9 rejects mixture chemistry); toxicity clipped to `E + G` (C.2). |
| Motion (§5.4) | Chemotaxis only — no pairwise forces: 33 heritable gradient sensitivities, uptake-gated, radially bounded acceleration, analytic `expm1` drag, speed limit, affordability by scaling the whole velocity (C.12). |
| Emission (§5.5) | Gate/direction/rate decoded from logits, timestep-consistent fraction of post-motion energy, efficiency loss dissipated, deposited post-motion and visible next tick. |
| Sensors (§6.1) | Density pyramid (2×2 averages of densities, C.10), log-linear scale blending of raw values and analytic bilinear derivatives *before* normalization; exact 104-input ordering. |
| Controllers (§6.2–6.6) | Each organism owns its own weights. Catalog `104→{8, 32, 64·32, 96·64, 128·96}→36` (1,164–29,316 params), tanh hidden, **linear** output, sigmoid only in decoding (C.11); stable-ID cadence phases, forced newborn evaluation, held outputs re-decoded against current energy (C.13). | The catalog's last entry is a recurrent `8 × 32` network (12,980 params): it reads 16 extra memory inputs and writes 16 extra memory outputs (squashed by `tanh`, kept per organism, applied at each evaluation, empty at birth, saved in checkpoints). Presets: `lively-recurrent-field-ecology.toml`. It runs on `cpu`, `gpu-fused`, `gpu-field`, `gpu-resident` and `gpu-realtime`; the controller-only `gpu` backend refuses it. **Half-precision weights:** set `controller.canonical_weight_dtype = "f16"` and every genome stores its weights as 16-bit floats (rounded on creation and mutation; arithmetic stays f32). That halves genome memory on the CPU, in checkpoints and in the recurrent network's GPU slab (26 KB instead of 52 KB per organism). On a GTX 1080 the f32 slab ran out of GPU memory at 160,000 organisms and the f16 slab ran them; speed is unchanged (the network kernel is not memory-bound). |
| Genomes (§7) | Fan-in-scaled init; strict mutation with ≤ 8 addressed candidates (replacement > perturbation, affinity renormalized, motility projected, log-scale reflected), effective-no-op rejection (C.16); topology mutation with the function-preserving widening map and the declared depth-change map. |
| Lifecycle (§7.4–7.6) | Hazard → `expm1` probability, one Bernoulli per eligible tick, viability checks, birth cost, cooldowns, disk-uniform child offsets; exhaustion re-checked after every budget stage; residual-energy recycling. |
| Tick (§8) | All 10 phases in order, computed into a **candidate** state; publication only after invariant checks. A capacity pause or numerical fault leaves the committed world bit-identical. |
| Ledger (§4.4) | Per tick and cumulative `ΔE = I − O − D + ε` with a 9-way dissipation breakdown; residual reported, never silently corrected. |
| Backends (§10, §18.3) | `ControllerBackend` port (`World::step_with`): the scalar reference is built in; `integration/fl-torch` provides FrankenTorch and `crates/fl-gpu` a wgpu GPU backend (`--backend gpu`; weights sharded into binding-sized pages; logits within 7.2e-7 of the reference at 80,000 × 9,988 params on a software Vulkan driver). A reference run through the port reproduces the built-in state hash. |
| Randomness (§16.2) | Counter-based Philox4x32-10 addressed by `(lane, stable id, tick, candidate, draw)` — never by slot or schedule. State hashes are identical for 1 and 4 worker threads. |
| Capacity (§7.5) | Live-count and logical-genome-byte quotas; research `pause-before-commit` or exhibition `reject-births-stable-order`. |
| Checkpoints (§16.3) | `FLCK` v1 container with FNV-1a checksum: effective config, field planes, all organism hot state incl. held outputs/phases/cooldowns/energy history, genome bank, source paths, ID counters, pending future commands, ledger. `save(load(x)) == x`; resume is bit-identical (C.13 test, and CLI hash check). Refuses foreign model versions and corrupt files. |
| Interventions (§14.3) | `EraseDisk` (remove / dissipate / recycle) and `EnergyBrush` commands applied at their recorded tick boundary, with ledger entries. |

### Known-answer fixtures (Appendix C)

Implemented as tests: C.1 contended uptake, C.2 toxicity clipping, C.3
opposing signatures, C.4 paid emission, C.5 birth split, C.6 diffusion and
subcycling, C.7 view neutrality (inspection doesn't change state hashes), C.9
packet-vs-mixture chemistry (`+0.639355310069` vs `−0.047643210759`), C.10
constant-density sensing across scales and quotient derivatives, C.11
logit/sigmoid and small-rate `expm1` arithmetic, C.12 exhaustion after
motion and at maintenance, C.13 newborn forced evaluation, stable phases and
off-phase checkpoint resume with pending future-tick commands,
C.15 integer-tick aging past the F32 horizon, C.16 no-op mutation rejection,
widening preservation and every directed catalog transition, C.17 source
normalization at several resolutions and path continuity. Plus Philox
known-answer vectors, ledger closure, determinism, capacity rollback and
config refusal of unknown keys/families.

## Measured behavior (4-core container, CPU reference, release build)

| Preset | Population | Steps/s (wall) | Notes |
|---|---|---|---|
| reference (Appendix A) | 4,096 founders, 9,988 params each | ~55 | Random start declines slowly (3,350 after 30 s), organisms nearly static (mean speed ≈ 0.004 L/s): most absorbed energy is lost to poorly matched chemistry. |
| vivid (calibrated) | 4,096 → ~2,500 steady | ~60–100 | Births ≈ deaths after ~25 s, mean speed ≈ 0.43 L/s, clustering and filaments along plume boundaries, generation 6 by 40 s. |

Cumulative ledger residuals stay around 1e-4…1e-3 energy units against tens
of thousands of units transferred (F32 field rounding). These are measurements
from this container only, not benchmarks.

## What is *not* done (honest gap list)

The Franken stack integration lives in [`integration/`](integration/README.md)
(separate nightly workspace, upstream pinned by revision). Its M0 program runs
the world under an **Asupersync** runtime, checks the field against
**FrankenNumPy** layout descriptors and exports `.npz`, evaluates independent
networks with **FrankenTorch** batched matmul (max |Δlogit| vs reference 4.3e-7
at 4,096 × 9,988 params), and renders committed epochs with **FrankenManim**.
Still missing:

- Asupersync-scheduled data parallelism (the tick runs on the owner task;
  FrankenTorch uses its own Rayon pool), a durable write-ahead command journal,
  and lab/cancellation-injection tests.
- FrankenTorch Metal (no macOS host here), fused population kernels, F16 genomes.
- A FrankenManim live Studio, cameras/timeline and Cinema export; per-point
  colors in `DotCloud` (organisms are bucketed into 12 colors).
- **FrankenSQLite** run catalog and crash recovery; checkpoints are plain files
  storing full F32 genomes (~165 MB at 4,096 organisms) without deduplication.
- Scale: 20k (first performance target) and 80k (stretch) qualification, deterministic GPU reductions,
  timestep-refinement studies, ablations, seed gallery.

`mutation.rng` in the presets is `fl-philox-semantic-v1` (this crate's own
address mapping) rather than Appendix A's `fnp-philox-semantic-v1`.
`frankenlife validate` lists the declared-but-unimplemented infrastructure
keys explicitly.

Next steps per §26.2: move tick computation onto Asupersync-admitted workers,
add the FrankenSQLite catalog and journal, and profile toward the 25k profile.
