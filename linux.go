package main

import (
	"fmt"
	"github.com/manifoldco/promptui"
	"os/exec"
	"runtime"
	"strings"
)

type LinuxPlatform struct {
	BasePlatform
}

func (linux *LinuxPlatform) InstallTVM() {
	prompt := promptui.Select{
		Label: "Install tvm:",
		Items: []string{"Pre-built", "From source"},
	}
	_, installMethod, err := prompt.Run()
	if err != nil {
		cliError("Error getting install tvm response: ", err)
	}

	if installMethod == "Pre-built" {
		loading := createLoader("Installing Pre-built TVM...")
		cmd := exec.Command("conda", "run", "-n", linux.CliEnv, "pip", "install", "--pre", "-U", "-f", "https://mlc.ai/wheels", "mlc-ai-nightly-cpu")
		osToCmdOutput(cmd)
		err := cmd.Run()
		if err != nil {
			cliError("Error installing TVM: ", err)
		}
		stopLoader(loading)
		println(Success + "Installed TVM")
	} else {
		// TODO: Build & Install TVM from source
	}

}

func (linux *LinuxPlatform) GenerateConfig() {
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
	loading := createLoader("Generating Config...")
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

	commandString := fmt.Sprintf("source $(conda info --base)/etc/profile.d/conda.sh && (conda activate %s && echo -e '%s' | python3 ../cmake/gen_cmake_config.py)", linux.BuildEnv, input)
	cmd := exec.Command("bash", "-c", commandString)
	cmd.Dir = "mlc-llm/build"
	osToCmdOutput(cmd)

	err = cmd.Run()
	if err != nil {
		cliError("Error generating config: ", err)
	}
	stopLoader(loading)
	println(Success + "Generated Config")
}

func (linux *LinuxPlatform) BuildMLC() {
	loading := createLoader("Building MLC...")
	cmakeCmd := exec.Command("conda", "run", "-n", linux.BuildEnv, "cmake", "..")
	cmakeCmd.Dir = "mlc-llm/build"
	osToCmdOutput(cmakeCmd)
	err := cmakeCmd.Run()
	if err != nil {
		cliError("Error running cmake: ", err)
	}

	nCores := runtime.NumCPU() // Get the number of CPU cores
	buildCmd := exec.Command("conda", "run", "-n", linux.BuildEnv, "cmake", "--build", ".", "--parallel", fmt.Sprintf("%d", nCores))
	buildCmd.Dir = "mlc-llm/build"
	osToCmdOutput(buildCmd)
	err = buildCmd.Run()
	if err != nil {
		cliError("Error building MLC: ", err)
	}
	stopLoader(loading)
	println(Success + "Built MLC")
}

func (linux *LinuxPlatform) BuildTVM() {
	//TODO
}

func (linux *LinuxPlatform) BuildAndroid() {
	//TODO
}

func (linux *LinuxPlatform) GetName() string {
	return linux.Name
}

func (linux *LinuxPlatform) GetBuildEnv() string {
	return linux.BuildEnv
}

func (linux *LinuxPlatform) GetCliEnv() string {
	return linux.CliEnv
}
