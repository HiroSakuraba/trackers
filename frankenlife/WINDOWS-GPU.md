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

Bigger populations or networks: `cargo run --release -- gpu-check --organisms 32768 --hidden 128,96`.

## 4. Run a world on the GPU

```powershell
cargo run --release -- run --config presets\vivid-field-ecology.toml --backend gpu --seconds 30 --frame-every 32 --out runs\gpu
```

Frames go to `runs\gpu\frames`, plus `metrics.jsonl` and `manifest.json`. For a
pure speed measurement use `--frame-every 0` (frames are drawn on the CPU).
Compare with `--backend cpu`.

## What to expect and current limits

- Only controller inference runs on the GPU so far. Sensing, the field, uptake
  and rendering are still on the CPU, so the whole tick gets faster by less than
  inference alone does. Moving the field kernels over is the next step.
- A GPU run agrees with the CPU reference to about 1e-6 per logit, not bitwise,
  so its trajectory (and state hash) differs from a CPU run with the same seed.
- At 9,988 parameters per organism, one weight buffer holds about 80,000
  organisms per topology on DX12/Vulkan; beyond that sharding is needed (not
  implemented yet). 8 GB of VRAM fits roughly 150,000 such organisms, so the
  plan's 400,000-organism target does not fit on this card at F32.
- If anything fails, send the full `gpu-check` output.
