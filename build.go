package main

import (
	"fmt"
	"github.com/manifoldco/promptui"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

func installTVM(userOS string, cliEnv string) {
	prompt := promptui.Select{
		Label: "Install tvm:",
		Items: []string{"Pre-built", "From source"},
	}
	_, resp, err := prompt.Run()
	if err != nil {
		cliError("Error getting install tvm response: ", err)
	}
	if userOS == "MacOS" {
		if resp == "Pre-built" {
			cmd := exec.Command("conda", "run", "-n", cliEnv, "pip", "install", "--pre", "-U", "-f", "https://mlc.ai/wheels", "mlc-ai-nightly-cpu")
			osToCmdOutput(cmd)
			err := cmd.Run()
			if err != nil {
				cliError("Error installing TVM: ", err)
			}
		} else {
			// TODO: Build & Install TVM from source
		}
	}
}

func promptMLCRepo() {
	prompt := promptui.Prompt{
		Label:   "Enter MLC repo (press enter for 'https://github.com/mlc-ai/mlc-llm.git')",
		Default: "https://github.com/mlc-ai/mlc-llm.git",
	}
	repo, err := prompt.Run()
	if err != nil {
		cliError("Error getting MLC repo: ", err)
	}

	// Check if the repo folder already exists and if we should clone
	cloneRepo := false
	if _, err := os.Stat("mlc-llm"); err == nil {
		prompt := promptui.Select{
			Label: "MLC repo already exists. Delete?",
			Items: []string{"Yes", "No"},
		}
		_, resp, err := prompt.Run()
		if err != nil {
			cliError("Error getting clone MLC repo response: ", err)
		}
		if resp == "Yes" {
			err := os.RemoveAll("mlc-llm")
			if err != nil {
				cliError("Error deleting MLC repo: ", err)
			}
			cloneRepo = true
		}
	} else if os.IsNotExist(err) { // Repo doesn't exist
		cloneRepo = true
	} else {
		cliError("Error checking for mlc-llm directory: ", err)
	}

	if cloneRepo {
		cmd := exec.Command("git", "clone", "--recursive", repo)
		osToCmdOutput(cmd)
		err := cmd.Run()
		if err != nil {
			cliError("Error cloning MLC repo: ", err)
		}
	}
}

func generateConfig(buildEnv string) {
	CUDA := "n"
	ROCM := "n"
	VULKAN := "n"
	METAL := "n"
	OPENCL := "n"
	tvmPrompt := promptui.Prompt{
		Label:   "Enter TVM_SOURCE_DIR in absolute path. If not specified, 3rdparty/tvm will be used by default",
		Default: "",
	}
	TvmSourceDir, err := tvmPrompt.Run()
	if err != nil {
		cliError("Error getting TVM source directory: ", err)
	}
	runtimePrompt := promptui.Select{
		Label: "Select GPU runtime (or None for CPU-only)",
		Items: []string{"CUDA", "ROCM", "VULKAN", "METAL", "OPENCL", "None"},
	}
	_, gpuRuntime, err := runtimePrompt.Run()
	if err != nil {
		cliError("Error getting GPU runtime: ", err)
	}
	switch gpuRuntime {
	case "CUDA":
		CUDA = "y"
	case "ROCM":
		ROCM = "y"
	case "VULKAN":
		VULKAN = "y"
	case "METAL":
		METAL = "y"
	case "OPENCL":
		OPENCL = "y"
	}

	answers := []string{
		TvmSourceDir, // 1. TVM_SOURCE_DIR
		CUDA,         // 2. Use CUDA
		ROCM,         // 3. Use ROCM
		VULKAN,       // 4. Use Vulkan
		METAL,        // 5. Use Metal
		OPENCL,       // 6. Use OpenCL
	}

	input := strings.Join(answers, "\n") + "\n"

	commandString := fmt.Sprintf("source $(conda info --base)/etc/profile.d/conda.sh && (conda activate %s && echo -e '%s' | python3 ../cmake/gen_cmake_config.py)", buildEnv, input)
	cmd := exec.Command("bash", "-c", commandString)
	cmd.Dir = "mlc-llm/build"
	osToCmdOutput(cmd)

	err = cmd.Run()
	if err != nil {
		cliError("Error generating config: ", err)
	}
}

func buildMLC(buildEnv string) {
	cmakeCmd := exec.Command("conda", "run", "-n", buildEnv, "cmake", "..")
	cmakeCmd.Dir = "mlc-llm/build"
	osToCmdOutput(cmakeCmd)
	err := cmakeCmd.Run()
	if err != nil {
		cliError("Error running cmake: ", err)
	}

	nCores := runtime.NumCPU() // Get the number of CPU cores
	//fmt.Printf("Building MLC with %d parallel jobs...\n", nCores)
	buildCmd := exec.Command("conda", "run", "-n", buildEnv, "cmake", "--build", ".", "--parallel", fmt.Sprintf("%d", nCores))
	buildCmd.Dir = "mlc-llm/build"
	osToCmdOutput(buildCmd)
	err = buildCmd.Run()
	if err != nil {
		cliError("Error building MLC: ", err)
	}
}

func installMLC(cliEnv string) {
	installCmd := exec.Command("conda", "run", "-n", cliEnv, "pip", "install", ".")
	installCmd.Dir = "mlc-llm/python"
	osToCmdOutput(installCmd)
	err := installCmd.Run()
	if err != nil {
		cliError("Error installing MLC: ", err)
	}
}
