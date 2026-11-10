# Running FrankenLife on a Windows NVIDIA GPU

The `fl-gpu` crate evaluates every organism's neural network on the GPU through
[wgpu](https://wgpu.rs) (DX12 or Vulkan). It was developed and checked on a
software Vulkan driver; speed figures only mean something on real hardware.

## 1. Install the tools (once)

1. **NVIDIA driver**: a current Game Ready or Studio driver (GTX 1080 is supported).
2. **Visual Studio Build Tools**: from <https://visualstudio.microsoft.com/visual-cpp-build-tools/>,
   select the **Desktop development with C++** workload (Rust needs the MSVC linker).
3. **Rust**: run `rustup-init.exe` from <https://rustup.rs> and accept the defaults
   (stable toolchain). The integration workspace's nightly toolchain is **not** needed.
4. **Git**: <https://git-scm.com/download/win>.

## 2. Get the code

```powershell
git clone -b ccr-a6abe6fc-z1wtl2 https://github.com/HiroSakuraba/trackers.git
cd trackers\frankenlife
```

## 3. Check the GPU

```powershell
cargo run --release -- gpu-check
```

The first build takes a few minutes. Expected output (numbers will differ):

```
adapter      : NVIDIA GeForce GTX 1080 (Dx12, DiscreteGpu)
max |dlogit| : ~1e-6 vs CPU reference (OK)
GPU batch    : ... ms  ->  ... network evals/s, ... GB/s of weights
CPU batch    : ... ms (scalar reference, one thread)  ->  GPU is ...x faster
```

`OK` means the GPU matches the CPU reference. To compare DX12 with Vulkan:

```powershell
$env:WGPU_BACKEND = "vulkan"; cargo run --release -- gpu-check
$env:WGPU_BACKEND = "dx12";   cargo run --release -- gpu-check
Remove-Item Env:WGPU_BACKEND
```

The plan's full target (80,000 independent networks, ~3 GB of weights):
`cargo run --release -- gpu-check --organisms 80000`. Other sizes:
`cargo run --release -- gpu-check --organisms 32768 --hidden 128,96`.

## 4. Run a world on the GPU

```powershell
cargo run --release -- run --config presets\vivid-field-ecology.toml --backend gpu --seconds 30 --frame-every 32 --out runs\gpu
```

**GPU-resident field** (`--backend gpu-resident`, the fastest path): the field
stays on the GPU between steps. Field transport, sensing, inference, uptake
exchange (deterministic per-cell sums) and deposits all run there; only small
per-organism results and a per-cell summary come back each step. The field is
copied to the CPU only for frames, checkpoints and the final report.

```powershell
cargo run --release -- run --config presets\target-20k-field-ecology.toml --backend gpu-resident --seconds 6 --frame-every 0 --out runs\t20r
```

**GPU field step too** (`--backend gpu-field`): field diffusion, decay,
sources and tiny-cell clearing also run on the GPU, and fused sensing reuses
the field already there. Exchange and lifecycle still run on the CPU, so the
field is copied back once per step for now.

```powershell
cargo run --release -- run --config presets\target-20k-field-ecology.toml --backend gpu-field --seconds 6 --frame-every 0 --out runs\t20g
```

**Fused GPU sensing + inference** (`--backend gpu-fused`): the field is uploaded
once per step, the sensor pyramid is built on the GPU, and each organism's
inputs are computed and fed to its network without leaving the GPU. Compare it
with `--backend gpu` (CPU sensing, GPU inference):

```powershell
cargo run --release -- run --config presets\target-20k-field-ecology.toml --backend gpu-fused --seconds 6 --frame-every 0 --out runs\t20f
```

**Performance measurement at the first target (20,000 organisms)**, no images, so
only simulation time is counted:

```powershell
cargo run --release -- run --config presets\target-20k-field-ecology.toml --backend gpu --seconds 10 --frame-every 0 --out runs\t20
cargo run --release -- run --config presets\target-20k-field-ecology.toml --backend cpu --seconds 10 --frame-every 0 --out runs\t20cpu
cargo run --release -- gpu-check --organisms 20000
```

Each run prints, every simulated second, the mean step time split into field,
sensing/inference (and the GPU backend's share), exchange and lifecycle, against
the 7.8125 ms budget for 128 steps/s; the same numbers are in `metrics.jsonl` and
`manifest.json` under `"timing"`. `gpu-check` splits each batch into host
preparation, GPU submit/wait, and an optional `TIMESTAMP_QUERY` span. The
submit/wait number is CPU time blocked on the GPU, not GPU kernel time.

## 5. Stage profile on the GTX 1080 (one command)

From `trackers\frankenlife`, the same 20k preset, 6 simulated seconds, no frames.
`--profile` does not change the preset, the tick, or the science. It writes
`runs\t20r-profile\profile.json` (milliseconds, with `total_ns`) and prints a
short summary. Omit `--profile` for a wall-clock run whose GPU command stream
matches the unprofiled path.

```powershell
cargo run --release -- run --config presets\target-20k-field-ecology.toml --backend gpu-resident --seconds 6 --frame-every 0 --profile --out runs\t20r-profile
```

Read `cpu_wall_ms` as CPU `Instant` time (sensing, chemistry/field, networks,
transfers/sync, other, render, checkpoint). `transfers_sync` and any
`cpu_wall_blocked_on_gpu` counter include submit and the waits the step already
used; they are not kernel time. `gpu_timestamp_query` spans are optional and
are omitted when the adapter has no `TIMESTAMP_QUERY`. This build does not
require `TIMESTAMP_QUERY_INSIDE_ENCODERS` or `TIMESTAMP_QUERY_INSIDE_PASSES`.
Resident begin, deposit, and commit stay submit-without-wait. A software
Vulkan driver or a cloud GPU is not a measurement of this GTX 1080. A prior
single run on this machine (about 54 s wall for gpu-resident, exit 0, final
population near 19809) is one observation, not a diagnosis. Real time at 128 Hz
is 7.8125 ms per step.

With `--profile`, `profile.json` also splits the resident uptake wall into
`resident_uptake_pack`, `resident_uptake_upload_encode`, and
`resident_uptake_wait_readback` (the three already sum into the uptake wall;
do not add them again). `resident_uptake_pack` is now only the compact header
(cell, weights, demand, affinity slot). Affinity floats stay in a resident
buffer and a device copy expands them into the same item records as before.
`resident_deposit_timestamp_span` is the deposit GPU timestamp span, read in
the finish poll that already waits. `sense_or_mlp_dispatches` is a count of
compute dispatches, and `sense_or_mlp_readback_copy` is the scatter out of the
mapped sense/MLP rows (there is no second intermediate copy). Sense slot and
scale stay resident; the per-tick upload is position, due, and the five
intrinsic inputs, expanded on device into the same 12-float job. The
unprofiled command stream does not write the deposit timestamp queries.
Profiling does not change results.

`presets/target-10k-field-ecology.toml` is the 20k preset with
`initial_count` 10000 and `maximum_count` 12000. Nothing else differs.

```powershell
cargo run --release -- run --config presets\target-10k-field-ecology.toml --backend gpu-resident --seconds 6 --frame-every 0 --out runs\t10r
cargo run --release -- run --config presets\target-10k-field-ecology.toml --backend gpu-resident --seconds 6 --frame-every 0 --profile --out runs\t10r-profile
cargo run --release -- run --config presets\target-20k-field-ecology.toml --backend gpu-resident --seconds 6 --frame-every 0 --out runs\t20r-overlap
cargo run --release -- run --config presets\target-20k-field-ecology.toml --backend gpu-resident --seconds 6 --frame-every 0 --profile --out runs\t20r-overlap-profile
```

The 20k unprofiled run should keep population 19809 and hash
`611e5a19ee18a552`. `resident_uptake_pack` should fall well below the previous
2.46 ms/tick, and `sense_or_mlp_host_prep` should fall below the previous
3.76 ms/tick.

## Realtime at 5,000 organisms (`--backend gpu-realtime`)

`gpu-realtime` is the resident path with the GPU waits and submissions of a
tick cut down. Same rules and tick order. The one arithmetic difference from
`gpu-resident`: the uptake request `dt * umax * r / (K + r)` is evaluated in
f32 on the device (so uptake can follow sensing with no readback between),
instead of f64 on the host. Runs therefore differ from `gpu-resident` at
rounding level and then diverge the way any two rounding-different runs do.
Two `gpu-realtime` runs on one machine are bit-identical, and resume
reproduces the uninterrupted hash.

What changed per tick:

- One wait for the front half (field accounting, sensing, inference, uptake)
  and one for the back half (deposits, finish), instead of four or more.
- About four queue submissions instead of nine. Every topology group's sense
  pass sits in one submission; the commit copy rides with the next tick.
- Weight and affinity slots are remembered per organism in job order, so the
  steady state does no hash lookups and does not touch the genomes.

Host-side savings that apply to every backend (results unchanged bit for bit):
the tick no longer clones every genome reference, and the per-organism upkeep
rate and parameter count are cached instead of recomputed from the genome.

`presets/target-5k-field-ecology.toml` is the 20k preset at one quarter area:
128 by 128 field, 5,000 founders, cap 6,000, sources scaled to match.

```powershell
cargo run --release -- run --config presets\target-5k-field-ecology.toml --backend gpu-realtime --seconds 30 --frame-every 0 --out runs\rt5
cargo run --release -- run --config presets\target-5k-field-ecology.toml --backend gpu-realtime --seconds 6 --frame-every 0 --profile --out runs\rt5-profile
```

Measured on the GTX 1080 (driver 582.78), simulation only, 2026-11-09, steady
ticks at about 5,000 organisms; real time is 7.81 ms per step:

| backend | ms per step | steps/s |
| --- | --- | --- |
| `gpu-resident` before this change | 7.0 to 8.0 | 124 to 142 |
| `gpu-realtime` | 5.0 to 5.2 | 192 to 198 |

A 30 simulated second run finished in 17.6 s of wall time. With `--profile`
the step is about 0.3 ms slower, because the pyramid and deposits go back to
their own submissions so their timestamp spans stay readable.

A one-pass sensor pyramid (every level straight from the field) was tried and
removed: bit-identical, but its span on this card was 2.2 ms against 0.44 ms
for the level-by-level passes.

## Watch it live

`view` is the same simulation as `run`, in a window. Vulkan on the GTX 1080:

```powershell
$env:WGPU_BACKEND = "vulkan"
cargo run --release -- view --config presets\target-5k-field-ecology.toml --backend gpu-realtime --seconds 86400 --out runs\rt5-live
```

`--seconds 0` runs until you close the window. At startup the viewer prints
one line: surface format, color space, present mode, pixel size, and DPI
scale. Present mode is mailbox or fifo. Immediate present is not used: on
Windows, Vulkan immediate present skips the desktop compositor and the client
area stays the window's white background while the title bar keeps updating.
Real time for this preset is
128 ticks per simulated second. The window title and the overlay show tick,
sim time, population, achieved steps/s, and that rate divided by 128 (speed
versus real time).

| key / mouse | action |
| --- | --- |
| Space | pause / resume |
| `.` or `N` | pause and advance one tick |
| `1` | real time (1×) |
| `2` / `4` | 2× / 4× real time |
| `0` | as fast as possible |
| Tab | composite → chemistry → field → raw |
| drag | pan the torus |
| wheel | zoom, keeping the cursor's point fixed |
| click | name the nearest organism (does not change the world) |
| click a source's ring | pick that source for the chemistry keys (click it again, or empty space, to clear) |
| `Q` / `W` | energy: source power down / up (×1.25 per press) |
| `A` / `S` | source movement speed down / up (×1.25) |
| `D` / `F` | diffusion (how fast energy spreads) down / up (×1.25) |
| `C` / `V` | decay (how fast uneaten energy fades) down / up (×1.25) |
| `Z` / `X` | mutation rate down / up (×1.5; scales all five mutation settings together) |
| `J` / `K` | drift the picked source's chemistry by 5° / 25° toward a random new direction (every source when none is picked) |
| `R` | put energy, speed, diffusion, decay and mutation back to their launch values (chemistry drift is not undone) |
| Esc | quit and write `manifest.json` |

Drawing samples the GPU-resident field between ticks and does not read it
back. It does not change the tick. A run you do not pause, single-step,
erase, or tune finishes with the same population and state hash as `run` for
the same config, seed, backend and `--seconds`. The manifest records
`simulation_mean_step_ms` (the tick only) and `wall_steps_per_second` (tick
plus drawing). Compare the title's steps/s with a `--frame-every 0` run.

### Live tuning

The Q–X keys change the running world. Each press is a command at the next tick
boundary, so it takes effect cleanly and is kept in the world's configuration
(checkpoints save it). The second HUD line shows the requested values, and the
small rings mark where the sources are. Sources that change speed stay where
they are; they do not jump. Every change is also appended to
`controls.jsonl` in the run folder and listed under `live_changes` in
`manifest.json`, because a tuned run no longer matches a `run` of the same config.

For scripted runs and screenshots, `--press TICK:KEYS` presses keys at a tick
(`@N` picks source N), for example
`view --press 0:wwww --press 600:@2kk --screenshot shot.png --seconds 10`.

## Other runs

The full 80,000-organism target world:

```powershell
cargo run --release -- run --config presets\target-80k-field-ecology.toml --backend gpu --seconds 5 --frame-every 64 --out runs\t80
```

It works, but expect it to be slow until the field moves to the GPU: its
1024² field (33 planes) and the sensing for 80,000 organisms still run on the CPU.

Frames go to `runs\gpu\frames`, plus `metrics.jsonl` and `manifest.json`. For a
pure speed measurement use `--frame-every 0` (frames are drawn on the CPU).
Compare with `--backend cpu`.

## What to expect and current limits

- `--backend gpu` still does sensing, the field and uptake on the CPU.
  `--backend gpu-realtime` (and `gpu-resident`) keep the field on the GPU;
  `view` draws that copy without reading it back. The plain `gpu` backend
  gets faster by less than inference alone.
- A GPU run agrees with the CPU reference to about 1e-6 per logit, not bitwise,
  so its trajectory (and state hash) differs from a CPU run with the same seed.
- Weights are split into pages that each fit one GPU storage binding, so
  population is limited by VRAM, not by a single buffer. The plan's
  80,000-organism target needs about 3 GB of F32 weights at 9,988 parameters
  each, which fits the 1080's 8 GB; roughly 150,000 such organisms is the ceiling.
- If anything fails, send the full `gpu-check` output.

## Recurrent (memory) network

`presets\lively-recurrent-field-ecology.toml` is the lively evolving preset with every
organism running an 8-layer, 32-wide recurrent network (12,980 weights). Each
evaluation reads 16 memory values from the last one and writes 16 new ones.
Memory starts empty at birth, is not inherited, and is saved in checkpoints.

```powershell
cargo run --release -- view --config presets\lively-recurrent-field-ecology.toml --backend gpu-realtime --seconds 0 --out runs\rec-live
```

Weights and memory for this network live in the same resident slabs as the others
(about 52 KB per organism). `--backend gpu` (the controller-only kernel) refuses recurrent
networks; the fused backends (`gpu-fused` and later) run them.

