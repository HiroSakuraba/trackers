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
do not add them again). `resident_deposit_timestamp_span` is the deposit GPU
timestamp span, read in the finish poll that already waits. `sense_or_mlp_dispatches`
is a count of compute dispatches, and `sense_or_mlp_readback_copy` is only the
CPU copy after that wait returns. The unprofiled command stream does not record
these and does not write the deposit timestamp queries.

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

- Only controller inference runs on the GPU so far. Sensing, the field, uptake
  and rendering are still on the CPU, so the whole tick gets faster by less than
  inference alone does. Moving the field kernels over is the next step.
- A GPU run agrees with the CPU reference to about 1e-6 per logit, not bitwise,
  so its trajectory (and state hash) differs from a CPU run with the same seed.
- Weights are split into pages that each fit one GPU storage binding, so
  population is limited by VRAM, not by a single buffer. The plan's
  80,000-organism target needs about 3 GB of F32 weights at 9,988 parameters
  each, which fits the 1080's 8 GB; roughly 150,000 such organisms is the ceiling.
- If anything fails, send the full `gpu-check` output.
