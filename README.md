# mlc-cli: Build MLC-LLM & TVM from source, Run, and Deploy.

[![mlc-cli-tutorial](https://vumbnail.com/1157423379.jpg)](https://vimeo.com/1157423379)

## 🚀 Why use this?

> Prebuilt wheels for Mac & Linux on https://mlc.ai/wheels are broken or not compatible

**mlc-cli** solves this by:

- **Automating Source Builds:** Compiles TVM and MLC from source by following prompts.
- **Fixing Dependency Hell:** Manages `conda` environments and hidden paths automatically.
- **Artifact Caching:** Caches the heavy TVM compilation so you just build once and reuse wheels.
- **Reproducible Builds:** Pins MLC-LLM and TVM to known-good commits so a bad upstream push cannot silently break your build.
- **Custom Fork Support:** Easily point to your own Git forks/branches to test research code.
- **Non-Interactive Mode:** Run any command with flags for CI/CD pipelines or one-liner scripts.

---

## 🛠️ Prerequisites

- **Go** (1.21+) — If you don't have Go installed:
  ```bash
  # Linux
  ./install_go.sh

  # macOS
  brew install go
  ```
- **Git**
- **Conda** (Optional, the tool can install it for you)

---

## ⚡ Quick Start / Current Recommended Flow

```bash
git clone https://github.com/ballinyouup/mlc-cli.git
cd mlc-cli
```

**Build the binary (recommended over `go run .`):**

```bash
go build -o mlc-cli .
```

**Full Linux CUDA flow (validated with Python 3.13 / CUDA sm86):**

```bash
# 1. Build TVM + MLC from source and install wheels
./mlc-cli build --action full --cuda y --cuda-arch 86 --cublas y --cutlass y

# 2. Quantize a raw model (weights must be in models/<name>/)
./mlc-cli quantize --model models/Llama-3-8B-Instruct --quant q4f16_1 --template llama-3 --device cuda

# 3. Pre-compile model library for your GPU
./mlc-cli compile --model dist/Llama-3-8B-Instruct-q4f16_1-MLC --quant q4f16_1 --device cuda --output dist/libs/llama3.so

# 4. Run (uses pre-compiled library; no JIT overhead)
./mlc-cli run --model-name Llama-3-8B-Instruct-q4f16_1-MLC --device cuda --model-lib dist/libs/llama3.so
```

> **macOS note:** Script code parity with Linux has been improved. Runtime validation on a real Mac host is still pending. See [macOS Notes](#-macos-notes).

---

## 🔄 Typical Workflow

```
1. Build    →  Build TVM & MLC from pinned source, install wheels into mlc-cli-venv
2. Quantize →  Convert raw model weights to MLC format (dist/<name>-<quant>-MLC/)
3. Compile  →  Pre-compile model library for your device (dist/libs/<name>.so)
4. Run      →  Chat with zero JIT overhead
```

---

## 🐧 Linux CUDA Flow

The following flow was validated with Python 3.13, CUDA sm86, and the pinned MLC/TVM SHAs.

### 1. Build

```bash
# Full build + install (TVM + MLC source build)
./mlc-cli build --action full --cuda y --cuda-arch 86 --cublas y --cutlass y

# Build-only (skip install, keep wheels in wheels/)
./mlc-cli build --action build-only --cuda y --cuda-arch 86

# Install pre-built wheels only (after a previous build)
./mlc-cli build --action install-wheels
```

**Build environments:**
- `mlc-llm-venv` (or `tvm-build-venv`) — build-time conda env, Python 3.13, cmake, rust
- `mlc-cli-venv` — runtime conda env where wheels are installed

### 2. Quantize

```bash
./mlc-cli quantize \
  --model models/Llama-3-8B-Instruct \
  --quant q4f16_1 \
  --template llama-3 \
  --device cuda
```

Output lands in `dist/<model-name>-<quant>-MLC/`.

### 3. Compile

```bash
./mlc-cli compile \
  --model dist/Llama-3-8B-Instruct-q4f16_1-MLC \
  --quant q4f16_1 \
  --device cuda \
  --output dist/libs/llama3-cuda.so
```

The compile script runs via `conda run --no-capture-output -n mlc-cli-venv python -m mlc_llm compile`, so it works correctly when invoked from the Go subprocess (no silent `conda activate` failure).

### 4. Run

```bash
# With pre-compiled library (recommended — no JIT delay)
./mlc-cli run \
  --model-name Llama-3-8B-Instruct-q4f16_1-MLC \
  --device cuda \
  --model-lib dist/libs/llama3-cuda.so

# With JIT compilation (slower first launch)
./mlc-cli run \
  --model-name Llama-3-8B-Instruct-q4f16_1-MLC \
  --device cuda
```

The run script searches `models/` first, then `dist/` as a fallback. You do **not** need to create a `models/` symlink when the quantized output already lives under `dist/`.

### Validated Linux CUDA flow

The following steps passed end-to-end on a Linux CUDA machine with Python 3.13:

1. ✅ `cp313` wheel build and install (no `cp313t` confusion)
2. ✅ TVM import verified with `from tvm import register_global_func`
3. ✅ `mlc_llm` import
4. ✅ Quantize (`convert_weight` + `gen_config`)
5. ✅ Compile (`mlc_llm compile` via `conda run`)
6. ✅ Run (`mlc_llm chat` via `conda run`)

---

## 🍎 macOS Notes

> **Runtime validation is pending on a real Mac host.** Script code parity with Linux has been improved but has not been end-to-end runtime tested.

**What changed (code parity improvements):**

| Area | Linux | macOS | Status |
|---|---|---|---|
| Compile invocation | `conda run python -m mlc_llm compile` | `conda run python -m mlc_llm compile` | ✅ Aligned |
| TVM env vars at compile time | Always exported | Exported when `tvm/` exists | ✅ Aligned |
| TVM ref pinning (relax mode) | SHA checkout + sync | SHA checkout + sync | ✅ Aligned |
| CLI env preflight check at run | `conda env list` check | `conda env list` check | ✅ Aligned |
| LLVM version (TVM build) | N/A | `LLVM_VERSION=19` from `versions.sh` | ✅ Centralized |
| TVM env vars at **run** time | Exported | Not exported | ⏳ Deferred (bundled mode OK) |

**macOS build (Metal backend):**

```bash
# Full build + install on Mac with Metal
./mlc-cli build --action full --metal y --build-wheels y --force-clone y
```

**macOS run:**

```bash
./mlc-cli run --model-name Qwen3-1.7B-q4f16_1-MLC --device metal
```

If you are using the **relax TVM source mode** on macOS and `run` fails with a TVM import error, set these manually before running:

```bash
export TVM_HOME="$(pwd)/tvm"
export PYTHONPATH="${TVM_HOME}/python:${PYTHONPATH:-}"
export DYLD_LIBRARY_PATH="${TVM_HOME}/build/lib:${DYLD_LIBRARY_PATH:-}"
```

---

## 📁 Quantize / Compile / Run Artifact Locations

| Artifact | Default path | Notes |
|---|---|---|
| Raw model weights | `models/<name>/` | HuggingFace checkout or manual download |
| Quantized MLC model | `dist/<name>-<quant>-MLC/` | Output of `quantize` command |
| Compiled model library | `dist/libs/<name>-<device>.so` | Output of `compile` command |
| Pre-built wheels | `wheels/` | Output of `build --action build-only` |

**Run model resolution order:**
1. `models/<model-name>/` — direct model directory
2. `dist/<model-name>/` — quantized MLC output directory

No symlink needed between `models/` and `dist/`.

---

## 📖 Commands Reference

### Build

| Action | Description |
|---|---|
| `full` | Clone, build TVM & MLC, then install wheels |
| `build-only` | Clone and build without installing wheels |
| `install-wheels` | Install pre-built wheels (skips build config prompts) |

```bash
# Full build + install on Linux with CUDA
mlc-cli build --action full --cuda y --cuda-arch 86 --cublas y --cutlass y

# Full build + install on Mac with Metal
mlc-cli build --action full --metal y --build-wheels y --force-clone y

# Install pre-built wheels only
mlc-cli build --action install-wheels

# Build only, no install, with a custom repo
mlc-cli build --action build-only --git-repo https://github.com/your-fork/mlc-llm
```

Run `mlc-cli build --help` for all available flags.

### Run

```bash
# Run with JIT compilation
mlc-cli run --model-name Qwen3-1.7B-q4f16_1-MLC --device metal

# Run with a pre-compiled model library (no JIT)
mlc-cli run --model-name Qwen3-1.7B-q4f16_1-MLC --device metal --model-lib dist/libs/qwen.so

# Run with a URL (auto-clones model)
mlc-cli run --model-url https://huggingface.co/mlc-ai/Qwen3-1.7B-q4f16_1-MLC --model-name Qwen3-1.7B-q4f16_1-MLC --device cuda
```

### Compile Model

Pre-compile a model library (`.so`) so the runtime skips JIT on every launch.

```bash
# Compile for Metal (macOS)
mlc-cli compile --model dist/Qwen3-1.7B-q4f16_1-MLC --quant q4f16_1 --device metal --output dist/libs/qwen-metal.so

# Compile for CUDA (Linux)
mlc-cli compile --model dist/Llama-3-8B-Instruct-q4f16_1-MLC --quant q4f16_1 --device cuda --output dist/libs/llama3-cuda.so
```

### Quantize Model

Convert raw HuggingFace model weights to MLC format with quantization.

**Quantization format: `q{A}f{B}_{id}`**

| Code | Description |
|---|---|
| `q4f16_1` | 4-bit group quantization, float16, NK layout |
| `q4f16_ft` | 4-bit FasterTransformer quantization, float16 |
| `q4f32_1` | 4-bit group quantization, float32, NK layout |
| `q3f16_1` | 3-bit group quantization, float16, NK layout |
| `q8f16_1` | 8-bit group quantization, float16, NK layout |
| `q0f16` | No quantization, float16 |
| `q0f32` | No quantization, float32 |

```bash
# Quantize a model
mlc-cli quantize --model models/Llama-3-8B-Instruct --quant q4f16_1 --template llama-3 --device cuda

# Quantize with custom output directory
mlc-cli quantize --model models/phi-2 --quant q0f16 --template phi-2 --output dist/phi-2-q0f16-MLC
```

---

## 🐛 Troubleshooting

### Wrong Python version / ABI in wheel

**Symptom:** `pip install` complains about wheel ABI mismatch (e.g. `cp311` wheel in a `cp313` env).

**Cause (old behavior):** Install scripts used loose globs like `mlc*.whl`, which would pick the first wheel alphabetically regardless of ABI.

**Current behavior:** Install scripts filter by `PYTHON_CP_TAG` (e.g. `cp313-cp313`). If you have stale wheels from a previous ABI in `wheels/`, the install will fail clearly instead of silently installing the wrong one.

**Fix:** Remove old wheels from `wheels/` before re-installing:
```bash
rm wheels/mlc_llm-*cp311*.whl wheels/mlc_llm-*cp312*.whl  # remove stale ABI wheels
```

### Ambiguous MLC wheel selection

**Symptom:** Install fails with _"Multiple ABI-matching MLC wheels exist and none matches MLC_LLM_REF short SHA"_.

**Cause:** Multiple `mlc_llm-*-cp313-cp313-*.whl` files exist in `wheels/` from repeated builds with different `MLC_LLM_REF` values.

**Fix:** Keep only the wheel that matches the current `MLC_LLM_REF`. The expected filename contains `g2008fe83`:
```bash
ls wheels/mlc_llm-*cp313*.whl     # list candidates
rm wheels/mlc_llm-<stale-sha>*.whl
```

### TVM import error (`cannot import name 'register_global_func'`)

**Symptom:** Running quantize, compile, or run fails with:
```
ImportError: cannot import name 'register_global_func' from 'tvm'
```

**Cause:** The conda env resolved to the wrong TVM (e.g. system pip TVM or a different wheel) instead of the source-build TVM.

**Fix (Linux):** The TVM env vars are now exported automatically by the run/compile/quantize scripts. If you are running these scripts directly (not via `mlc-cli`), export manually:
```bash
export TVM_HOME="$(pwd)/tvm"
export PYTHONPATH="${TVM_HOME}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${TVM_HOME}/build/lib:$(pwd)/mlc-llm/build/lib:${LD_LIBRARY_PATH:-}"
```

**Fix (macOS):** Same, but use `DYLD_LIBRARY_PATH`:
```bash
export TVM_HOME="$(pwd)/tvm"
export PYTHONPATH="${TVM_HOME}/python:${PYTHONPATH:-}"
export DYLD_LIBRARY_PATH="${TVM_HOME}/build/lib:$(pwd)/mlc-llm/build/lib:${DYLD_LIBRARY_PATH:-}"
```

### `conda run` picks up system Python instead of env Python

**Symptom:** `mlc_llm` is not found even though the wheel is installed.

**Cause (old behavior):** Some scripts used `conda activate` in non-interactive subshells (how Go launches scripts), which silently fails — `CONDA_PREFIX` stays empty and `python` resolves to `/usr/bin/python`.

**Current behavior:** All compile and run scripts use `conda run --no-capture-output -n <env>` which guarantees the correct env Python regardless of shell interactivity.

### MLC_LLM_REF / TVM_REF not applied

**Symptom:** Build pulls upstream HEAD instead of the pinned commit.

**Cause (old behavior):** `MLC_LLM_REF` was defined in `versions.sh` but never applied by build scripts. `TVM_REF` was a live branch name (`"mlc"`).

**Current behavior:** Both refs are pinned to validated SHAs. Build scripts checkout the pinned SHA after `git clone`, and also sync any existing `mlc-llm/` or `tvm/` directory to the pinned SHA if it has drifted.

**To re-pin to a new revision:** Update `MLC_LLM_REF` or `TVM_REF` in `scripts/config/versions.sh` and re-run the build. The scripts will detect the SHA mismatch and update the checkout.

### CLI env not found at run time

**Symptom:** `run` or `compile` fails with _"CLI environment 'mlc-cli-venv' not found. Please run build first."_

**Fix:** Run the install step to create the CLI env:
```bash
./mlc-cli build --action install-wheels
```

---

## 🧰 Developer / Maintainer Notes

### Reproducible Build Pins

All pins live in `scripts/config/versions.sh` — the single source of truth.

| Variable | Value | Purpose |
|---|---|---|
| `PYTHON_VERSION` | `3.13` | Python version for all conda envs |
| `PYTHON_CP_TAG` | `cp313` | CPython ABI tag derived from `PYTHON_VERSION` |
| `MLC_LLM_REF` | `2008fe8343e1f40ef89ee57b9287aebcf1b86c98` | Known-good mlc-llm commit |
| `TVM_REF` | `b628d91fac716679db539884a55f8c6651f54dea` | Known-good mlc-ai/relax commit |
| `LLVM_VERSION` | `19` | LLVM major version for macOS TVM build |
| `CUDA_ARCH_DEFAULT` | `86` | Default CUDA SM arch (RTX 30xx / A10) |
| `CMAKE_MIN_VERSION` | `3.24` | Minimum CMake required by mlc-llm |

**How pins are applied:**
- Fresh clones: scripts checkout the pinned SHA immediately after `git clone`.
- Existing checkouts: scripts compare `git rev-parse HEAD` to the pinned SHA and sync if they differ.
- To use a different revision, edit `versions.sh`. Do not pass raw SHAs via CLI flags.

### Python and ABI Policy

The build enforces **Python 3.13 with standard CPython ABI** (`cp313`), not the free-threading variant (`cp313t`).

Two conda specs enforce this in every environment creation call:

```bash
# From scripts/config/versions.sh
PYTHON_PACKAGE_SPEC="python=3.13"
PYTHON_ABI_SPEC="python_abi=3.13=*_cp313"
```

`PYTHON_ABI_SPEC` is the critical guard: without it, conda-forge may resolve `python=3.13` to `cp313t` (free-threading) on some platforms, which produces incompatible wheels.

**Go constant sync:** `DefaultPythonVersion = "3.13"` in `versions_defaults.go` mirrors `PYTHON_VERSION` in `versions.sh`. The consistency checker (Check 6) will fail if these drift.

### Wheel Selection Policy

Wheel installation auto-discovery follows this priority order:

| Priority | Condition | Action |
|---|---|---|
| 1 | Explicit `MLC_WHEEL` path argument provided | Use it directly; fail if file is missing |
| 2 | Exactly one `mlc_llm-*-cp313-cp313-*.whl` in `wheels/` matching `g<short-sha>` | Use it; log the match |
| 3 | Multiple wheels match the short SHA | **Fail** — remove stale wheels and retry |
| 4 | No SHA match but exactly one ABI-matching wheel exists | Warn and use it |
| 5 | Multiple ABI-matching wheels, no SHA match | **Fail** — ambiguous selection |
| 6 | No ABI-matching wheel found | **Fail** — report missing wheel |

The short SHA is derived from `MLC_LLM_REF`: `g${MLC_LLM_REF:0:8}` → `g2008fe83`.

**TVM wheel selection:** Up to one `tvm-*-cp313-cp313-*.whl` is expected. If multiple exist, the first (alphabetically) is used with a warning. TVM wheels are optional in bundled mode because the `mlc_llm` wheel includes TVM internally.

This logic lives in `scripts/lib/wheel_selection.sh` and is sourced by all four install scripts to avoid duplication.

### Version Consistency Checker

Run at any time to verify all centralized values are in sync:

```bash
bash scripts/check_version_consistency.sh
```

Checks enforced:
1. No hardcoded `python=X.Y` in scripts (use `PYTHON_PACKAGE_SPEC` / `PYTHON_ABI_SPEC`)
2. No hardcoded mlc-llm GitHub URL in scripts (use `MLC_LLM_REPO`)
3. No hardcoded `mlc-ai/relax.git` URL in scripts (use `TVM_REPO`)
4. No hardcoded CUDA arch `86` as a default (use `CUDA_ARCH_DEFAULT`)
5. No hardcoded mlc-llm URL literal in Go (use `DefaultMlcLLMRepo`)
6. `DefaultPythonVersion` in Go matches `PYTHON_VERSION` in shell
7. No hardcoded `llvmdev` version in scripts (use `LLVM_VERSION`)

---

## 🏗️ Supported Platforms

| Platform | Status | Notes |
|---|---|---|
| **Linux (CUDA)** | ✅ Runtime validated | Python 3.13, sm86 validated end-to-end |
| **macOS (M-series)** | ⏳ Code parity improved | Runtime validation on Mac host pending |
| **Android** | ✅ Manual | Manual deployment required (see below) |
| **Windows** | ⚠️ WSL only | Use WSL |

---

## ⏳ Known Deferred Improvements

These items are known gaps that are intentionally deferred. They do not affect normal usage.

| Item | Risk | Status |
|---|---|---|
| Post-compile output file validation | Low | Deferred — compile exit code is checked but output `.so` size is not |
| Quantize artifact manifest (JSON) | Low | Deferred — no machine-readable record of quant parameters |
| `linux_install_cuda.sh` hardcoded CUDA 13/Ubuntu 24 | Low | Deferred — one-time setup script; see header comment in file for how to adjust |
| `pip install build` unversioned in build scripts | Low | Deferred — PyPA `build` is stable; pin if wheel naming changes |
| macOS `run` TVM source env vars | Low | Deferred — only affects relax-TVM source mode; bundled mode is fine |

---

## 📱 Android Development

> **Note:** The Deploy menu option is currently a placeholder. Follow the manual instructions below for Android deployment.

1. Use `mlc-cli` to build the `tvm` and `mlc` libraries from source first.
2. Open `./android/MLCChat` in **Android Studio**.
3. Connect your device.
4. **Build → Make Project**.
5. **Run → Run 'app'**.

---

## 🧪 Verified Models

- `mlc-ai/Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC`

---

## 📄 License

MIT
