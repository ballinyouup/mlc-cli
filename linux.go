package main

import (
	"os/exec"

	"github.com/manifoldco/promptui"
)

type LinuxPlatform struct {
	BasePlatform
}

func (linux *LinuxPlatform) GenerateConfig() {
	CUDA := "n"
	ROCM := "n"
	VULKAN := "n"
	OPENCL := "n"

	// Ask about prebuilt TVM
	prebuiltTvmPrompt := promptui.Select{
		Label: "Use prebuilt TVM?",
		Items: []string{"No (build from source)", "Yes (use prebuilt)"},
	}
	_, prebuiltTvmChoice, err := prebuiltTvmPrompt.Run()
	if err != nil {
		cliError("Error getting prebuilt TVM choice: ", err)
	}
	usePrebuiltTvm := "n"
	TvmSourceDir := ""
	if prebuiltTvmChoice == "Yes (use prebuilt)" {
		usePrebuiltTvm = "y"
	} else {
		tvmPrompt := promptui.Prompt{
			Label:   "Enter TVM_SOURCE_DIR in absolute path (press enter for 3rdparty/tvm)",
			Default: "",
		}
		TvmSourceDir, err = tvmPrompt.Run()
		if err != nil {
			cliError("Error getting TVM source directory: ", err)
		}
	}

	// Ask about prebuilt MLC LLM
	prebuiltMlcPrompt := promptui.Select{
		Label: "Use prebuilt MLC LLM?",
		Items: []string{"No (build from source)", "Yes (use prebuilt)"},
	}
	_, prebuiltMlcChoice, err := prebuiltMlcPrompt.Run()
	if err != nil {
		cliError("Error getting prebuilt MLC LLM choice: ", err)
	}
	usePrebuiltMlc := "n"
	if prebuiltMlcChoice == "Yes (use prebuilt)" {
		usePrebuiltMlc = "y"
	}

	runtimePrompt := promptui.Select{
		Label: "Select GPU runtime (or None for CPU-only)",
		Items: []string{"CUDA", "ROCM", "VULKAN", "OPENCL", "None"},
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
	case "OPENCL":
		OPENCL = "y"
	}

	loading := createLoader("Generating Config...")
	cmd := exec.Command("bash", "scripts/linux_gen_config.sh", linux.BuildEnv, TvmSourceDir, CUDA, ROCM, VULKAN, OPENCL, usePrebuiltTvm, usePrebuiltMlc)
	cmd.Dir = "."
	osToCmdOutput(cmd)
	err = cmd.Run()
	stopLoader(loading)
	if err != nil {
		cliError("Error generating config: ", err)
	}
	println(Success + "Generated Config")
}

func (linux *LinuxPlatform) BuildMLC() {
	runtimePrompt := promptui.Select{
		Label: "Select build target",
		Items: []string{"CPU", "CUDA"},
	}
	_, buildTarget, err := runtimePrompt.Run()
	if err != nil {
		cliError("Error getting build target: ", err)
	}
	useCuda := "n"
	cudaArch := "86"
	if buildTarget == "CUDA" {
		useCuda = "y"
		// Prompt for CUDA compute capability
		cudaArchPrompt := promptui.Prompt{
			Label:   "Enter CUDA compute capability (e.g., 86 for RTX 3060, 89 for RTX 4090)",
			Default: "86",
		}
		cudaArch, err = cudaArchPrompt.Run()
		if err != nil {
			cliError("Error getting CUDA compute capability: ", err)
		}
	}

	loading := createLoader("Building MLC...")
	cmd := exec.Command("bash", "scripts/linux_build.sh", linux.BuildEnv, useCuda, cudaArch)
	cmd.Dir = "."
	osToCmdOutput(cmd)
	err = cmd.Run()
	stopLoader(loading)
	if err != nil {
		cliError("Error building MLC: ", err)
	}
	println(Success + "Built MLC")
}

func (linux *LinuxPlatform) BuildAndroid() {
	//TODO: Implement Android build
}
