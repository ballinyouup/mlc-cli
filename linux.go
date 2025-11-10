package main

import (
	"github.com/manifoldco/promptui"
	"os/exec"
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

	loading := createLoader("Installing TVM...")
	installMethodArg := "prebuilt"
	if installMethod == "From source" {
		installMethodArg = "source"
	}

	cmd := exec.Command("bash", "scripts/linux_cli.sh", linux.CliEnv, installMethodArg)
	osToCmdOutput(cmd)
	err = cmd.Run()
	if err != nil {
		cliError("Error installing TVM: ", err)
	}
	stopLoader(loading)
	println(Success + "Installed TVM")
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

	loading := createLoader("Generating Config...")
	cmd := exec.Command("bash", "scripts/linux_gen_config.sh", linux.BuildEnv, TvmSourceDir, CUDA, ROCM, VULKAN, METAL, OPENCL)
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
	cmd := exec.Command("bash", "scripts/linux_build.sh", linux.BuildEnv)
	osToCmdOutput(cmd)
	err := cmd.Run()
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
