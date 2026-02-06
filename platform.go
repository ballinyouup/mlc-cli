package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"

	"github.com/manifoldco/promptui"
)

type Platform struct {
	TVMBuildEnv     string
	MLCBuildEnv     string
	CliEnv          string
	OperatingSystem string
	GitHubRepo      string
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

func (p *Platform) build(pkg string) {
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

func (p *Platform) install(pkg string) {
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

func handlePromptError(err error) {
	if errors.Is(err, promptui.ErrInterrupt) {
		fmt.Println("\nExiting...")
		os.Exit(0)
	}
	panic(err)
}

func promptYesNo(label string) string {
	prompt := promptui.Select{
		Label: label,
		Items: []string{"Yes", "No"},
	}
	_, result, err := prompt.Run()
	if err != nil {
		handlePromptError(err)
	}
	if result == "Yes" {
		return "y"
	}
	return "n"
}

func (p *Platform) run() {
	computePrompt := promptui.Select{
		Label: "Select compute profile",
		Items: []string{"Really Low", "Low", "Default", "High"},
	}
	_, computeProfile, err := computePrompt.Run()
	if err != nil {
		handlePromptError(err)
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

func (p *Platform) ConfigureGitHubRepo() {
	gitHubRepoPrompt := promptui.Prompt{
		Label:   "Enter GitHub repository URL",
		Default: "https://github.com/mlc-ai/mlc-llm",
	}
	var err error
	p.GitHubRepo, err = gitHubRepoPrompt.Run()
	if err != nil {
		handlePromptError(err)
	}
}

func (p *Platform) ConfigureBuildOptions() {
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

		p.Metal = promptYesNo("Enable Metal support?")
		p.Vulkan = promptYesNo("Enable Vulkan support?")
	} else {
		p.CUDA = promptYesNo("Enable CUDA support?")

		if p.CUDA == "y" {
			cudaArchPrompt := promptui.Prompt{
				Label:   "Enter CUDA compute capability (e.g., 86 for RTX 3060)",
				Default: "86",
			}
			var err error
			p.CUDAArch, err = cudaArchPrompt.Run()
			if err != nil {
				handlePromptError(err)
			}

			p.Cutlass = promptYesNo("Enable CUTLASS support?")
			p.CuBLAS = promptYesNo("Enable cuBLAS support?")
			p.FlashInfer = promptYesNo("Enable FlashInfer support?")
		} else {
			p.CUDA = "n"
			p.Cutlass = "n"
			p.CuBLAS = "n"
			p.FlashInfer = "n"
			p.CUDAArch = "86"
		}

		p.ROCM = promptYesNo("Enable ROCm support?")
		p.Vulkan = promptYesNo("Enable Vulkan support?")
		p.OpenCL = promptYesNo("Enable OpenCL support?")
		p.Metal = "n"
	}
}

func extractModelNameFromURL(url string) string {
	urlParts := []rune(url)
	lastSlash := -1
	for i := len(urlParts) - 1; i >= 0; i-- {
		if urlParts[i] == '/' {
			lastSlash = i
			break
		}
	}
	if lastSlash != -1 && lastSlash < len(urlParts)-1 {
		return string(urlParts[lastSlash+1:])
	}
	return url
}

func promptModelURL() string {
	modelURLPrompt := promptui.Prompt{
		Label: "Enter model Git URL",
	}
	url, err := modelURLPrompt.Run()
	if err != nil {
		handlePromptError(err)
	}
	return url
}

func getLocalModelDirectories() []string {
	entries, err := os.ReadDir("models")
	if err != nil {
		return nil
	}

	var modelDirs []string
	for _, entry := range entries {
		if entry.IsDir() {
			modelDirs = append(modelDirs, entry.Name())
		}
	}
	return modelDirs
}

func promptLocalModelName() string {
	modelNamePrompt := promptui.Prompt{
		Label:   "Enter local model name (in models/ directory)",
		Default: "",
	}
	name, err := modelNamePrompt.Run()
	if err != nil {
		handlePromptError(err)
	}
	return name
}

func selectLocalModel() string {
	modelDirs := getLocalModelDirectories()

	if len(modelDirs) == 0 {
		return promptLocalModelName()
	}

	modelSelectPrompt := promptui.Select{
		Label: "Select a model from models/ directory",
		Items: modelDirs,
	}
	_, modelName, err := modelSelectPrompt.Run()
	if err != nil {
		handlePromptError(err)
	}
	return modelName
}

func (p *Platform) configureRemoteModel() {
	p.ModelURL = promptModelURL()
	p.ModelName = extractModelNameFromURL(p.ModelURL)
}

func (p *Platform) configureLocalModel() {
	p.ModelName = selectLocalModel()
	p.ModelURL = ""
}

func (p *Platform) configureDevice() {
	deviceDefault := "metal"
	if p.OperatingSystem == "linux" {
		deviceDefault = "cuda"
	}

	devicePrompt := promptui.Prompt{
		Label:   "Enter device type",
		Default: deviceDefault,
	}
	var err error
	p.Device, err = devicePrompt.Run()
	if err != nil {
		handlePromptError(err)
	}
}

func (p *Platform) ConfigureModel() {
	modelSourcePrompt := promptui.Select{
		Label: "Select model source",
		Items: []string{"Use local model", "Download from Git (HuggingFace)"},
	}
	_, modelSource, err := modelSourcePrompt.Run()
	if err != nil {
		handlePromptError(err)
	}

	if modelSource == "Download from Git (HuggingFace)" {
		p.configureRemoteModel()
	} else {
		p.configureLocalModel()
	}

	p.configureDevice()
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
			handlePromptError(err)
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
func CreatePlatform() Platform {
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
		handlePromptError(err)
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
		handlePromptError(err)
	}

	// Set MLC Build Environment Name
	MLCBuildEnvPrompt := promptui.Prompt{
		Label:   "Enter a MLC build environment name",
		Default: "mlc-build-venv",
	}

	MLCBuildEnv, err = MLCBuildEnvPrompt.Run()
	if err != nil {
		handlePromptError(err)
	}

	// Set CLI Environment Name
	CliEnvPrompt := promptui.Prompt{
		Label:   "Enter a CLI environment name",
		Default: "mlc-cli-venv",
	}

	CliEnv, err = CliEnvPrompt.Run()
	if err != nil {
		handlePromptError(err)
	}

	return Platform{
		OperatingSystem: OperatingSystem,
		TVMBuildEnv:     TvmBuildEnv,
		MLCBuildEnv:     MLCBuildEnv,
		CliEnv:          CliEnv,
		GitHubRepo:      "",
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
