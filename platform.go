package main

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/manifoldco/promptui"
)

type Platform interface {
	BuildTVM(pkg string)
	BuildMLC()
	InstallTVM()
	InstallMLC()
	CreatePlatform()
}
type BasePlatform struct {
	Platform
	TVMBuildEnv     string
	MLCBuildEnv     string
	CliEnv          string
	OperatingSystem string
	ModelURL        string
	ModelName       string
	Device          string
	CUDA            string
	ROCM            string
	Vulkan          string
	Metal           string
	OpenCL          string
	Cutlass         string
	CuBLAS          string
	FlashInfer      string
	CUDAArch        string
}

func (p *BasePlatform) build(pkg string) {
	var cmd *exec.Cmd

	if pkg == "mlc" {
		if p.OperatingSystem == "mac" {
			cmd = exec.Command("bash", "scripts/"+p.OperatingSystem+"_build_"+pkg+".sh",
				p.MLCBuildEnv, p.CUDA, p.ROCM, p.Vulkan, p.Metal, p.OpenCL)
		} else {
			cmd = exec.Command("bash", "scripts/"+p.OperatingSystem+"_build_"+pkg+".sh",
				p.MLCBuildEnv, p.CUDA, p.Cutlass, p.CuBLAS, p.ROCM, p.Vulkan, p.OpenCL, p.FlashInfer, p.CUDAArch)
		}
	} else if pkg == "tvm" {
		if p.OperatingSystem == "mac" {
			cmd = exec.Command("bash", "scripts/"+p.OperatingSystem+"_build_"+pkg+".sh", p.TVMBuildEnv)
		} else {
			cmd = exec.Command("bash", "scripts/"+p.OperatingSystem+"_build_"+pkg+".sh", p.CUDAArch)
		}
	} else {
		cmd = exec.Command("bash", "scripts/"+p.OperatingSystem+"_build_"+pkg+".sh", p.TVMBuildEnv)
	}

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	scriptErr := cmd.Run()
	if scriptErr != nil {
		panic(scriptErr)
	}
}

func (p *BasePlatform) install(pkg string) {
	var cmd *exec.Cmd

	scriptPath := "scripts/" + p.OperatingSystem + "_install_" + pkg + ".sh"

	switch pkg {
	case "cuda":
		if CheckCudaInstalled() {
			fmt.Println(Success + "CUDA is already installed, skipping installation.")
			return
		} else if p.OperatingSystem != "linux" {
			fmt.Println("CUDA installation is only supported on Linux.")
			return
		}
		cmd = exec.Command("bash", scriptPath)
	case "mlc", "tvm", "wheels":
		cmd = exec.Command("bash", scriptPath, p.CliEnv)
	default:
		cmd = exec.Command("bash", scriptPath)
	}

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	scriptErr := cmd.Run()
	if scriptErr != nil {
		panic(scriptErr)
	}
}

func (p *BasePlatform) run() {
	computePrompt := promptui.Select{
		Label: "Select compute profile",
		Items: []string{"Really Low", "Low", "Default", "High"},
	}
	_, computeProfile, err := computePrompt.Run()
	if err != nil {
		panic(err)
	}

	var overrides string
	switch computeProfile {
	case "Really Low":
		overrides = "context_window_size=10240;prefill_chunk_size=512"
	case "Low":
		overrides = "context_window_size=20480;prefill_chunk_size=1024"
	case "Default":
		overrides = ""
	case "High":
		overrides = "context_window_size=81920;prefill_chunk_size=4096"
	}

	cmd := exec.Command("bash", "scripts/"+p.OperatingSystem+"_run_model.sh", p.CliEnv, p.ModelURL, p.ModelName, p.Device, overrides)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	scriptErr := cmd.Run()
	if scriptErr != nil {
		panic(scriptErr)
	}
}

func (p *BasePlatform) ConfigureBuildOptions() {

	if p.OperatingSystem == "mac" {
		p.CUDA = "n"
		p.ROCM = "n"
		p.Vulkan = "n"
		p.Metal = "y"
		p.OpenCL = "n"
		p.Cutlass = "n"
		p.CuBLAS = "n"
		p.FlashInfer = "n"
		p.CUDAArch = ""

		metalPrompt := promptui.Select{
			Label: "Enable Metal support?",
			Items: []string{"Yes", "No"},
		}
		_, metalResult, err := metalPrompt.Run()
		if err != nil {
			panic(err)
		}
		if metalResult == "Yes" {
			p.Metal = "y"
		} else {
			p.Metal = "n"
		}

		vulkanPrompt := promptui.Select{
			Label: "Enable Vulkan support?",
			Items: []string{"Yes", "No"},
		}
		_, vulkanResult, err := vulkanPrompt.Run()
		if err != nil {
			panic(err)
		}
		if vulkanResult == "Yes" {
			p.Vulkan = "y"
		} else {
			p.Vulkan = "n"
		}
	} else {
		cudaPrompt := promptui.Select{
			Label: "Enable CUDA support?",
			Items: []string{"Yes", "No"},
		}
		_, cudaResult, err := cudaPrompt.Run()
		if err != nil {
			panic(err)
		}
		if cudaResult == "Yes" {
			p.CUDA = "y"

			cudaArchPrompt := promptui.Prompt{
				Label:   "Enter CUDA compute capability (e.g., 86 for RTX 3060)",
				Default: "86",
			}
			p.CUDAArch, err = cudaArchPrompt.Run()
			if err != nil {
				panic(err)
			}

			cutlassPrompt := promptui.Select{
				Label: "Enable CUTLASS support?",
				Items: []string{"Yes", "No"},
			}
			_, cutlassResult, err := cutlassPrompt.Run()
			if err != nil {
				panic(err)
			}
			if cutlassResult == "Yes" {
				p.Cutlass = "y"
			} else {
				p.Cutlass = "n"
			}

			cublasPrompt := promptui.Select{
				Label: "Enable cuBLAS support?",
				Items: []string{"Yes", "No"},
			}
			_, cublasResult, err := cublasPrompt.Run()
			if err != nil {
				panic(err)
			}
			if cublasResult == "Yes" {
				p.CuBLAS = "y"
			} else {
				p.CuBLAS = "n"
			}

			flashinferPrompt := promptui.Select{
				Label: "Enable FlashInfer support?",
				Items: []string{"Yes", "No"},
			}
			_, flashinferResult, err := flashinferPrompt.Run()
			if err != nil {
				panic(err)
			}
			if flashinferResult == "Yes" {
				p.FlashInfer = "y"
			} else {
				p.FlashInfer = "n"
			}
		} else {
			p.CUDA = "n"
			p.Cutlass = "n"
			p.CuBLAS = "n"
			p.FlashInfer = "n"
			p.CUDAArch = "86"
		}

		rocmPrompt := promptui.Select{
			Label: "Enable ROCm support?",
			Items: []string{"Yes", "No"},
		}
		_, rocmResult, err := rocmPrompt.Run()
		if err != nil {
			panic(err)
		}
		if rocmResult == "Yes" {
			p.ROCM = "y"
		} else {
			p.ROCM = "n"
		}

		vulkanPrompt := promptui.Select{
			Label: "Enable Vulkan support?",
			Items: []string{"Yes", "No"},
		}
		_, vulkanResult, err := vulkanPrompt.Run()
		if err != nil {
			panic(err)
		}
		if vulkanResult == "Yes" {
			p.Vulkan = "y"
		} else {
			p.Vulkan = "n"
		}

		openclPrompt := promptui.Select{
			Label: "Enable OpenCL support?",
			Items: []string{"Yes", "No"},
		}
		_, openclResult, err := openclPrompt.Run()
		if err != nil {
			panic(err)
		}
		if openclResult == "Yes" {
			p.OpenCL = "y"
		} else {
			p.OpenCL = "n"
		}

		p.Metal = "n"
	}
}

func (p *BasePlatform) ConfigureModel() {
	var err error

	modelSourcePrompt := promptui.Select{
		Label: "Select model source",
		Items: []string{"Use local model", "Download from Git (HuggingFace)"},
	}
	_, modelSource, err := modelSourcePrompt.Run()
	if err != nil {
		panic(err)
	}

	if modelSource == "Download from Git (HuggingFace)" {
		modelURLPrompt := promptui.Prompt{
			Label: "Enter model Git URL",
		}
		p.ModelURL, err = modelURLPrompt.Run()
		if err != nil {
			panic(err)
		}

		urlParts := []rune(p.ModelURL)
		lastSlash := -1
		for i := len(urlParts) - 1; i >= 0; i-- {
			if urlParts[i] == '/' {
				lastSlash = i
				break
			}
		}
		if lastSlash != -1 && lastSlash < len(urlParts)-1 {
			p.ModelName = string(urlParts[lastSlash+1:])
		} else {
			p.ModelName = p.ModelURL
		}
	} else {
		entries, err := os.ReadDir("models")
		if err != nil || len(entries) == 0 {
			modelNamePrompt := promptui.Prompt{
				Label:   "Enter local model name (in models/ directory)",
				Default: "",
			}
			p.ModelName, err = modelNamePrompt.Run()
			if err != nil {
				panic(err)
			}
		} else {
			var modelDirs []string
			for _, entry := range entries {
				if entry.IsDir() {
					modelDirs = append(modelDirs, entry.Name())
				}
			}

			if len(modelDirs) == 0 {
				modelNamePrompt := promptui.Prompt{
					Label:   "Enter local model name (in models/ directory)",
					Default: "",
				}
				p.ModelName, err = modelNamePrompt.Run()
				if err != nil {
					panic(err)
				}
			} else {
				modelSelectPrompt := promptui.Select{
					Label: "Select a model from models/ directory",
					Items: modelDirs,
				}
				_, p.ModelName, err = modelSelectPrompt.Run()
				if err != nil {
					panic(err)
				}
			}
		}
		p.ModelURL = ""
	}

	deviceDefault := "metal"
	if p.OperatingSystem == "linux" {
		deviceDefault = "cuda"
	}

	devicePrompt := promptui.Prompt{
		Label:   "Enter device type",
		Default: deviceDefault,
	}
	p.Device, err = devicePrompt.Run()
	if err != nil {
		panic(err)
	}
}

func CheckAndInstallConda(operatingSystem string) {
	cmd := exec.Command("conda", "--version")
	err := cmd.Run()

	if err != nil {
		installPrompt := promptui.Select{
			Label: "Conda is not installed. Would you like to install it?",
			Items: []string{"Yes", "No"},
		}
		_, result, err := installPrompt.Run()
		if err != nil {
			panic(err)
		}

		if result == "Yes" {
			installCmd := exec.Command("bash", "scripts/"+operatingSystem+"_install_conda.sh")
			installCmd.Stdout = os.Stdout
			installCmd.Stderr = os.Stderr
			installErr := installCmd.Run()
			if installErr != nil {
				panic(installErr)
			}
		} else {
			panic("Conda is required to proceed. Please install conda and try again.")
		}
	}
}

func CheckCudaInstalled() bool {
	cmd := exec.Command("nvcc", "--version")
	err := cmd.Run()
	return err == nil
}

// CreatePlatform static function
func CreatePlatform() BasePlatform {
	OperatingSystem := ""
	TvmBuildEnv := ""
	MLCBuildEnv := ""
	CliEnv := ""
	var err error = nil

	// Set Operating System
	osPrompt := promptui.Select{
		Label: "Select a MLC build environment",
		Items: []string{"mac", "linux"},
	}
	_, OperatingSystem, err = osPrompt.Run()
	if err != nil {
		panic(err)
	}

	// Check and install conda if needed
	CheckAndInstallConda(OperatingSystem)

	// Set TVM Build Environment Name
	TvmBuildEnvPrompt := promptui.Prompt{
		Label:   "Enter a TVM build environment name",
		Default: "tvm-build-venv",
	}

	TvmBuildEnv, err = TvmBuildEnvPrompt.Run()
	if err != nil {
		panic(err)
	}

	// Set MLC Build Environment Name
	MLCBuildEnvPrompt := promptui.Prompt{
		Label:   "Enter a MLC build environment name",
		Default: "mlc-build-venv",
	}

	MLCBuildEnv, err = MLCBuildEnvPrompt.Run()
	if err != nil {
		panic(err)
	}

	// Set CLI Environment Name
	CliEnvPrompt := promptui.Prompt{
		Label:   "Enter a CLI environment name",
		Default: "mlc-cli-venv",
	}

	CliEnv, err = CliEnvPrompt.Run()
	if err != nil {
		panic(err)
	}

	return BasePlatform{
		OperatingSystem: OperatingSystem,
		TVMBuildEnv:     TvmBuildEnv,
		MLCBuildEnv:     MLCBuildEnv,
		CliEnv:          CliEnv,
		ModelURL:        "",
		ModelName:       "",
		Device:          "",
		CUDA:            "",
		ROCM:            "",
		Vulkan:          "",
		Metal:           "",
		OpenCL:          "",
		Cutlass:         "",
		CuBLAS:          "",
		FlashInfer:      "",
		CUDAArch:        "",
	}
}
