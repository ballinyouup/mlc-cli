### Overview
The goal of this project is to make it easier to work with MLC LLM repo 
for tasks like building from source with source code modifications, 
compiling models to MLC format, and running the models with custom model configs.

## Getting Started

```bash
  go run .
```
Options:
- ```-v``` - Shows all the output (ex: ```go run . -v```)
### Platform
- Linux

### Steps:
1. Install Conda
2. Install CUDA (Use ./linux_cuda.sh)
3. Create the environments (Use ./linux_env.sh)
4. Download prebuilt tvm from https://mlc.ai/wheels (mlc_ai_nightly_cu130-0.20.dev537-py3-none-manylinux_2_28_x86_64.whl) and place in wheels folder
5. Create a GitHub personal access token and when asked for GitHub repo from cli, paste https://[YOUR_TOKEN]@github.com/[YOUR_REPO]
6. Build MLC LLM (Use ./linux_build.sh)
7. Generate Config (Use ./linux_gen_config.sh)
8. Run Model (Use ./linux_run_model.sh) Only use MLC format models ex: https://huggingface.co/mlc-ai/Qwen3-1.7B-q4f16_1-MLC

### GPU Runtime
- Cuda
- None (CPU)

## Deployments
- CLI - Execute Run Command
- Android

## Android
```bash
# After Installation
Open folder ./android/MLCChat as an Android Studio Project
Connect your Android device to your machine. In the menu bar of Android Studio, click “Build → Make Project”.
Once the build is finished, click “Run → Run ‘app’” and you will see the app launched on your phone.
```
