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
- MacOS
- Linux (TODO)
- Windows (TODO)

### GPU Runtime
- Metal Apple M1/M2
- Cuda (TODO)
- RocM (TODO)
- Vulkan (TODO)
- OpenCL (TODO)
- None

## Deployments
- WebLLM (TODO)
- REST Server (TODO)
- CLI - Execute Run Command
- Android
- iOS (TODO)

## Android
```bash
# After Installation
Open folder ./android/MLCChat as an Android Studio Project
Connect your Android device to your machine. In the menu bar of Android Studio, click “Build → Make Project”.
Once the build is finished, click “Run → Run ‘app’” and you will see the app launched on your phone.
```

### Possible Conflicts (Mac OS)
During installation, any global packages installed through homebrew, like cmake, pip, conda, may have conflicts during installation.
```bash
  which cmake # try uninstalling from homebrew
  which python # should point to miniconda or conda, not homebrew
  open ~/.zshrc # comment out alias python=/opt/homebrew/bin/python3
```
