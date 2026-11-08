# FrankenLife — first living world (FieldEcologyV1)

This directory implements the **first useful slice** of the *FrankenLife Design
Plan* (§22.2, §26.2 steps 2–3, Appendix E): one complete cycle in which

source energy enters a field → organisms sense it → **independent neural
controllers** decide → chemistry causes gain/loss and motion → paid emission
changes the field → reproduction produces a **mutated child** → death releases
resources → the committed world is rendered.

It is a self-contained Rust workspace. Both crates use `#![forbid(unsafe_code)]`.

```
frankenlife/
├── presets/
│   ├── reference-field-ecology.toml   # Appendix A starter profile (one documented deviation)
│   └── vivid-field-ecology.toml       # declared viability-calibrated profile
└── crates/
    ├── fl-core/       # authoritative model: config, RNG, field, sensors, MLPs, genomes, tick, ledger
    └── frankenlife/   # CLI: validate, run, metrics, inspection, headless renderer
```

## Quick start

```bash
cd frankenlife
cargo test --release                      # 40 unit/known-answer/world fixtures
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
| Controllers (§6.2–6.6) | Each organism owns its own weights. Catalog `104→{8, 32, 64·32, 96·64, 128·96}→36` (1,164–29,316 params), tanh hidden, **linear** output, sigmoid only in decoding (C.11); stable-ID cadence phases, forced newborn evaluation, held outputs re-decoded against current energy (C.13). |
| Genomes (§7) | Fan-in-scaled init; strict mutation with ≤ 8 addressed candidates (replacement > perturbation, affinity renormalized, motility projected, log-scale reflected), effective-no-op rejection (C.16); topology mutation with the function-preserving widening map and the declared depth-change map. |
| Lifecycle (§7.4–7.6) | Hazard → `expm1` probability, one Bernoulli per eligible tick, viability checks, birth cost, cooldowns, disk-uniform child offsets; exhaustion re-checked after every budget stage; residual-energy recycling. |
| Tick (§8) | All 10 phases in order, computed into a **candidate** state; publication only after invariant checks. A capacity pause or numerical fault leaves the committed world bit-identical. |
| Ledger (§4.4) | Per tick and cumulative `ΔE = I − O − D + ε` with a 9-way dissipation breakdown; residual reported, never silently corrected. |
| Randomness (§16.2) | Counter-based Philox4x32-10 addressed by `(lane, stable id, tick, candidate, draw)` — never by slot or schedule. State hashes are identical for 1 and 4 worker threads. |
| Capacity (§7.5) | Live-count and logical-genome-byte quotas; research `pause-before-commit` or exhibition `reject-births-stable-order`. |
| Interventions (§14.3) | `EraseDisk` (remove / dissipate / recycle) and `EnergyBrush` commands applied at their recorded tick boundary, with ledger entries. |

### Known-answer fixtures (Appendix C)

Implemented as tests: C.1 contended uptake, C.2 toxicity clipping, C.3
opposing signatures, C.4 paid emission, C.5 birth split, C.6 diffusion and
subcycling, C.7 view neutrality (inspection doesn't change state hashes), C.9
packet-vs-mixture chemistry (`+0.639355310069` vs `−0.047643210759`), C.10
constant-density sensing across scales and quotient derivatives, C.11
logit/sigmoid and small-rate `expm1` arithmetic, C.12 exhaustion after
motion and at maintenance, C.13 newborn forced evaluation and stable phases,
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

The plan names a specific stack. None of it is linked here, and nothing in
this directory should be described as using it:

- **Asupersync** runtime ownership, command durability, cancellation lab tests —
  the world is driven by a plain loop; parallel sensing/inference uses
  `std::thread::scope` (deterministic, execution-only).
- **FrankenTorch / FrankenNumPy** typed storage, population kernels, Metal —
  the scalar CPU evaluator here is the "reference evaluator" role only.
- **FrankenManim** bulk primitives, live Studio, cameras, Cinema export — a
  headless CPU rasterizer renders committed epochs to PNG instead.
- **FrankenSQLite** catalog, checkpoints/resume, branch-and-compare, replay
  journal (M2) — not implemented; metrics are JSONL.
- Scale: 25k (M2) and 400k (M4) profiles, deterministic GPU reductions,
  F16 canonical genomes, timestep-refinement studies, ablations, seed gallery.

`mutation.rng` in the presets is `fl-philox-semantic-v1` (this crate's own
address mapping) rather than Appendix A's `fnp-philox-semantic-v1`.
`frankenlife validate` lists the declared-but-unimplemented infrastructure
keys explicitly.

The natural next step per §26.2 is step 4: put the world owner behind
Asupersync and publish immutable render epochs to a FrankenManim bulk adapter,
keeping `fl-core` as the authoritative model and fixture oracle.
