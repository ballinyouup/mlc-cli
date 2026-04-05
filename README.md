# mlc-cli: Build MLC-LLM & TVM from source, Run, and Deploy.

[![mlc-cli-tutorial](https://vumbnail.com/1157423379.jpg)](https://vimeo.com/1157423379)

## 🚀 Why use this?

> Prebuilt wheels for Mac & Linux on https://mlc.ai/wheels are broken or not compatible

**mlc-cli** solves this by:

- **Automating Source Builds:** Compiles TVM and MLC from source by following prompts.
- **Fixing Dependency Hell:** Manages `conda` environments and hidden paths automatically.
- **Artifact Caching:** Caches the heavy TVM compilation so you just build once and reuse wheels.
- **Custom Fork Support:** Easily point to your own Git forks/branches to test research code.
- **Non-Interactive Mode:** Run any command with flags for CI/CD pipelines or one-liner scripts.

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

## ⚡ Getting Started

### 1. Installation

```bash
git clone https://github.com/ballinyouup/mlc-cli.git
cd mlc-cli
```

### 2. Usage

**Interactive mode** — launch the menu-driven CLI:

```bash
go run .
```

**Non-interactive mode** — pass a subcommand with flags:

```bash
go run . <command> [flags]
```

Or build the binary first:

```bash
go build -o mlc-cli .
./mlc-cli <command> [flags]
```

## 📖 Commands

### Build

Build TVM/MLC from source and install wheels. The build menu offers three sub-options:

| Action | Description |
| :--- | :--- |
| **Full Build + Install** | Clone, build TVM & MLC, then install wheels |
| **Build Only** | Clone and build without installing wheels |
| **Install Wheels Only** | Install pre-built wheels (skips all build config prompts) |

**Features:**

- **Force re-clone:** If `mlc-llm/` or `tvm/` directories already exist, the tool asks whether to keep them or delete and re-clone.
- **TVM dependency ordering:** The MLC wheel install automatically installs the TVM wheel first to avoid missing dependency errors.
- **macOS flashinfer fix:** The build script automatically comments out `flashinfer-python` from `requirements.txt` (requires NVIDIA libraries not available on macOS).

**Non-interactive examples:**

```bash
# Full build + install on Mac with Metal
mlc-cli build --action full --metal y --build-wheels y --force-clone y

# Full build + install on Linux with CUDA
mlc-cli build --action full --cuda y --cuda-arch 86 --cublas y --cutlass y

# Install pre-built wheels only
mlc-cli build --action install-wheels

# Build only, no install, with a custom repo
mlc-cli build --action build-only --git-repo https://github.com/your-fork/mlc-llm
```

Run `mlc-cli build --help` for all available flags.

### Run

Chat with a model. The tool clones the model from HuggingFace if needed, then launches the MLC chat interface.

Supports **pre-compiled model libraries** to skip JIT compilation at runtime — when prompted, select "Yes (use compiled library)" and provide the `.so` path.

**Non-interactive examples:**

```bash
# Run with JIT compilation
mlc-cli run --model-name Qwen3-1.7B-q4f16_1-MLC --device metal

# Run with a pre-compiled model library (no JIT)
mlc-cli run --model-name Qwen3-1.7B-q4f16_1-MLC --device metal --model-lib dist/libs/qwen.so

# Run with a URL (auto-clones)
mlc-cli run --model-url https://huggingface.co/mlc-ai/Qwen3-1.7B-q4f16_1-MLC --model-name Qwen3-1.7B-q4f16_1-MLC --device cuda

# Low memory profile
mlc-cli run --model-name Qwen3-1.7B-q4f16_1-MLC --device metal --profile low
```

### Compile Model

Pre-compile a model library (`.so` file) so the runtime doesn't need to JIT compile on every launch. This is the **code generation** step — it uses TVM to produce optimized GPU code for your specific device.

In interactive mode, the tool lists all models in your `models/` directory and all supported quantization options.

**Non-interactive examples:**

```bash
# Compile for Metal (macOS)
mlc-cli compile --model models/Qwen3-1.7B-q4f16_1-MLC --quant q4f16_1 --device metal --output dist/libs/qwen-metal.so

# Compile for CUDA (Linux)
mlc-cli compile --model models/Llama-3-8B-Instruct --quant q4f16_1 --device cuda
```

### Quantize Model

Convert raw model weights (e.g. from HuggingFace) to MLC format with quantization. This is the **data transformation** step — it compresses model weights.

In interactive mode, the tool lists all models in your `models/` directory and all supported quantization options.

**Quantization naming format: `q{A}f{B}_{id}`**

The format follows the convention from the MLC-LLM quantization docs:

| Part | Meaning | Examples |
| :--- | :--- | :--- |
| `q{A}` | **Weight bits** — number of bits for storing weights. Lower = smaller model, less VRAM, some quality loss. | `q4` = 4-bit, `q3` = 3-bit, `q8` = 8-bit |
| `q0` | **No quantization** — weights kept at full precision | `q0f16` = full precision float16 |
| `f{B}` | **Activation bits** — number of bits for storing activations during inference | `f16` = float16, `f32` = float32 |
| `_{id}` | **Algorithm identifier** — distinguishes different quantization algorithms (e.g. symmetric, non-symmetric, AWQ) | `_1` = default group quant, `_ft` = FasterTransformer, `_awq` = AWQ |

MLC-LLM also supports **weight-activation FP8 quantization** on CUDA with formats like `e4m3_e4m3_f16` and `e5m2_e5m2_f16`, where both weights and activations are quantized to FP8.

**Supported quantizations:**

| Code | Description |
| :--- | :--- |
| `q4f16_1` | 4-bit group quantization, float16, NK layout |
| `q4f16_ft` | 4-bit FasterTransformer quantization, float16 |
| `q4f32_1` | 4-bit group quantization, float32, NK layout |
| `q3f16_1` | 3-bit group quantization, float16, NK layout |
| `q8f16_1` | 8-bit group quantization, float16, NK layout |
| `q0f16` | No quantization, float16 |
| `q0f32` | No quantization, float32 |

**Non-interactive examples:**

```bash
# Quantize a model
mlc-cli quantize --model models/Llama-3-8B-Instruct --quant q4f16_1 --template llama-3 --device metal

# Quantize with custom output directory
mlc-cli quantize --model models/phi-2 --quant q0f16 --template phi-2 --output dist/phi-2-q0f16-MLC
```

## 🔄 Typical Workflow

```
1. Build    →  Build TVM & MLC from source, install wheels
2. Quantize →  Convert raw model weights to MLC format
3. Compile  →  Pre-compile model library for your device
4. Run      →  Chat with zero JIT overhead
```

## 📱 Android Development

> **Note:** The Deploy menu option is currently a placeholder. Follow the manual instructions below for Android deployment.

This CLI prepares the environment required to build the Android APK.

1. Use `mlc-cli` to build the `tvm` and `mlc` libraries from source first.
2. Open `./android/MLCChat` in **Android Studio**.
3. Connect your device.
4. **Build → Make Project**.
5. **Run → Run 'app'**.

## 🏗️ Supported Architectures

| Platform | Status | Notes |
| :--- | :--- | :--- |
| **Linux** | ✅ Verified | CUDA, ROCm, Vulkan supported |
| **Mac (M1/M2/M3)** | ✅ Verified | Metal support |
| **Android** | ✅ Verified | Manual deployment required |
| **Windows** | ⚠️ Requires WSL | Use WSL or Git Bash |

## 🧪 Verified Models

The tool is tested with the following HuggingFace models:

- `mlc-ai/Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC`

## 🐛 Known Issues

1. **Deploy Menu:** The "Deploy" option in interactive mode shows a placeholder message. See the Android Development section for deployment instructions.

2. **CUDA on macOS:** CUDA-specific options (cutlass, flashinfer, etc.) are not available on macOS. The script automatically disables these.

## 📄 License

MIT
