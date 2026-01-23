# mlc-cli: The Missing Build Tool for MLC-LLM

> **Simple automated builds for MLC-LLM.**


[![mlc-cli-tutorial](https://vumbnail.com/1157423379.jpg)](https://vimeo.com/1157423379)

## 🚀 Why use this?
> Prebuilt wheels for Mac & Linux on https://mlc.ai/wheels are broken or not compatible

**mlc-cli** solves this by:
* **Automating Source Builds:** compiles TVM and MLC from source by following prompts.
* **Fixing Dependency Hell:** Manages `conda` environments and hidden paths automatically.
* **Artifact Caching:** Caches the heavy TVM compilation so you just build once and reuse wheels.
* **Custom Fork Support:** Easily point to your own Git forks/branches to test research code.

## 🛠️ Prerequisites
* [Go](https://go.dev/dl/) (1.20+)
* Git
* Conda (Optional, the tool can install it for you)

## ⚡ Getting Started

### 1. Installation
Clone the repository:
```bash
git clone [https://github.com/yourusername/mlc-cli.git](https://github.com/yourusername/mlc-cli.git)
cd mlc-cli
```

### 2. Usage
Run the interactive CLI:
```bash
go run .
```

### 3. The Workflow
1.  **Select "Build":** The tool will detect your OS (Linux/Mac) and GPU (CUDA/Metal).
2.  **Configure Source:** Select current mlc/llm repo or input your own.
3.  **Wait for Magic:** The tool compiles the binaries and installs the Python wheels.
4.  **Select "Run":** Launch the chat interface with your compiled model.

## 📱 Android Development
This CLI prepares the environment required to build the Android APK.

1.  Use `mlc-cli` to build the `tvm` and `mlc` libraries from source first.
2.  Open `./android/MLCChat` in **Android Studio**.
3.  Connect your device.
4.  **Build → Make Project**.
5.  **Run → Run 'app'**.

## 🏗️ Supported Architectures

| Platform  | Status |
| :--- | :--- |
| **Linux** | ✅ Verified |
| **Mac (M1/M2/M3)**  ✅ Verified |
| **Android**  ✅ Verified |

## 🧪 Verified Models
The tool is tested with the following HuggingFace models:
* `mlc-ai/Qwen3-1.7B-q4f16_1-MLC` (Fastest for testing)
* `mlc-ai/Qwen3-4B-q4f16_1-MLC`
* `mlc-ai/Llama-3-8B-Instruct-q3f16_1-MLC`

## 📄 License
MIT
