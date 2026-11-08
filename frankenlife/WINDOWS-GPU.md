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
the 7.8 ms budget for 128 steps/s; the same numbers are in `metrics.jsonl` and
`manifest.json` under `"timing"`. `gpu-check` now also splits each batch into
host preparation, GPU round trip and kernel time (GPU timestamps). Please send
the console output of all three.

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
