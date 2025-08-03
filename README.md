### Overview
The goal of this project is to make it easier to work with MLC LLM repo 
for tasks like building from source with source code modifications, 
compiling models to MLC format, and running the models with custom model configs.

## Getting Started

```bash
  go run .
	
  Options:
  -v verbose Shows all the output
```

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

### Possible Conflicts
During installation, any global packages installed through homebrew, like cmake, pip, conda, may have conflicts during installation.
```bash
  which cmake # try uninstalling from homebrew
  which python # should point to miniconda or conda, not homebrew
  open ~/.zshrc # comment out alias python=/opt/homebrew/bin/python3
```