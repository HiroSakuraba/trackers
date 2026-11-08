#!/usr/bin/env bash
# Build the FrankenLife Studio: fl-core compiled to WebAssembly (SIMD128),
# inlined as base64 into a single self-contained HTML page.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
cd "$root"
RUSTFLAGS="-C target-feature=+simd128" cargo build --release -p fl-wasm --target wasm32-unknown-unknown
wasm="target/wasm32-unknown-unknown/release/fl_wasm.wasm"
out="${1:-$here/frankenlife-studio.html}"
python3 - "$here/studio.template.html" "$wasm" "$out" <<'PY'
import base64, sys
tpl, wasm, out = sys.argv[1:]
html = open(tpl).read().replace("__WASM_BASE64__", base64.b64encode(open(wasm, "rb").read()).decode())
open(out, "w").write(html)
print(f"{out}: {len(html)/1e6:.2f} MB")
PY
