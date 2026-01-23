### Overview
The goal of this project is to make it easier to work with MLC LLM repo 
for tasks like building from source with source code modifications, 
compiling models to MLC format, and running the models with custom model configs.

## Getting Started

```bash
  go run .
```

### Platform
- Linux
- Mac

### Steps:

1. Install Conda
2. Install CUDA (Use ./linux_cuda.sh)
3. Select Build and follow all the prompts

#### Model Examples
   1. https://huggingface.co/mlc-ai/Qwen3-1.7B-q4f16_1-MLC
   2. https://huggingface.co/mlc-ai/Qwen3-4B-q4f16_1-MLC
   3. https://huggingface.co/mlc-ai/Llama-3-8B-Instruct-q3f16_1-MLC

### GPU Runtime
- Cuda (Linux)
- Metal (Mac)
- None (CPU)

## Deployments
- CLI
- Android

## Android
```bash
# After Installation
Open folder ./android/MLCChat as an Android Studio Project
Connect your Android device to your machine. In the menu bar of Android Studio, click “Build → Make Project”.
Once the build is finished, click “Run → Run ‘app’” and you will see the app launched on your phone.
```
