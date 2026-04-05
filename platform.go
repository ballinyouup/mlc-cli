package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/manifoldco/promptui"
)

// Platform holds all configuration for the build/run environment
type Platform struct {
	OperatingSystem string
	TVMBuildEnv     string
	MLCBuildEnv     string
	CliEnv          string
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
	TVMSource       string
	BuildWheels     string
	ForceClone      string
	InstallMode     string
}

// build executes the build script for the specified package
func (p *Platform) build(pkg string) {
	var cmd *exec.Cmd
	scriptPath := "scripts/" + p.OperatingSystem + "_build_" + pkg + ".sh"

	// Check if script exists
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		reportError("Build script not found: %s", err)
		return
	}

	if pkg == "mlc" {
		if p.OperatingSystem == "mac" {
			cmd = exec.Command("bash", scriptPath,
				p.MLCBuildEnv, p.CUDA, p.ROCM, p.Vulkan, p.Metal, p.OpenCL, p.TVMSource, p.BuildWheels, p.ForceClone)
		} else {
			cmd = exec.Command("bash", scriptPath,
				p.MLCBuildEnv, p.CUDA, p.Cutlass, p.CuBLAS, p.ROCM, p.Vulkan, p.OpenCL, p.FlashInfer, p.CUDAArch, p.GitHubRepo, p.TVMSource, p.BuildWheels, p.ForceClone)
		}
	} else if pkg == "tvm" {
		if p.OperatingSystem == "mac" {
			cmd = exec.Command("bash", scriptPath, p.TVMBuildEnv, p.TVMSource, p.BuildWheels, p.ForceClone)
		} else {
			cmd = exec.Command("bash", scriptPath, p.CUDAArch, p.TVMSource, p.BuildWheels, p.ForceClone)
		}
	} else {
		cmd = exec.Command("bash", scriptPath, p.TVMBuildEnv)
	}

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	fmt.Printf("\n📦 Building %s...\n", pkg)
	if err := cmd.Run(); err != nil {
		reportError("%s build failed", err)
		return
	}
	fmt.Printf("%sBuild of %s completed successfully.\n", Success, pkg)
}

// install executes the install script for the specified package
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
		// Check if script exists
		if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
			fmt.Println(Warning + "CUDA install script not found, skipping.")
			return
		}
		cmd = exec.Command("bash", scriptPath)
	case "tvm", "wheels":
		cmd = exec.Command("bash", scriptPath, p.CliEnv)
	case "mlc":
		cmd = exec.Command("bash", scriptPath, p.CliEnv, p.TVMSource, p.InstallMode)
	default:
		cmd = exec.Command("bash", scriptPath)
	}

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	fmt.Printf("\n📥 Installing %s...\n", pkg)
	if err := cmd.Run(); err != nil {
		reportError("%s installation failed", err)
		return
	}
	fmt.Printf("%sInstallation of %s completed successfully.\n", Success, pkg)
}

// run executes the model run script
func (p *Platform) run() {
	scriptPath := "scripts/" + p.OperatingSystem + "_run_model.sh"

	// Ask if user has a pre-compiled model library
	modelLibPath := ""
	compiledPrompt := promptui.Select{
		Label: "Use a pre-compiled model library? (skips JIT compilation at runtime)",
		Items: []string{"No (JIT compile at runtime)", "Yes (use compiled library)"},
	}
	_, compiledChoice, err := compiledPrompt.Run()
	if err != nil {
		handlePromptError(err)
	}
	if compiledChoice == "Yes (use compiled library)" {
		libPrompt := promptui.Prompt{
			Label:   "Enter path to compiled model library (.so/.dylib)",
			Default: "dist/",
		}
		modelLibPath, err = libPrompt.Run()
		if err != nil {
			handlePromptError(err)
		}
	}

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
	case "High":
		overrides = "context_window_size=81920;prefill_chunk_size=4096"
	default:
		overrides = ""
	}

	fmt.Printf("\n🚀 Running model on %s...\n", p.Device)
	cmd := exec.Command("bash", scriptPath, p.CliEnv, p.ModelURL, p.ModelName, p.Device, overrides, modelLibPath)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		reportError("Model run failed", err)
	}
}

// ConfigureGitHubRepo prompts for GitHub repository URL
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

// ConfigureRepoAction checks for existing repos and prompts for action
func (p *Platform) ConfigureRepoAction() {
	// Check if mlc-llm or tvm directories already exist
	mlcExists := false
	tvmExists := false
	if _, err := os.Stat("mlc-llm"); err == nil {
		mlcExists = true
	}
	if _, err := os.Stat("tvm"); err == nil {
		tvmExists = true
	}

	if mlcExists || tvmExists {
		var existingDirs string
		if mlcExists && tvmExists {
			existingDirs = "mlc-llm and tvm"
		} else if mlcExists {
			existingDirs = "mlc-llm"
		} else {
			existingDirs = "tvm"
		}

		repoActionPrompt := promptui.Select{
			Label: fmt.Sprintf("Existing repo(s) found: %s. What would you like to do?", existingDirs),
			Items: []string{
				"Keep existing (skip clone)",
				"Delete and re-clone",
			},
		}
		_, repoAction, err := repoActionPrompt.Run()
		if err != nil {
			handlePromptError(err)
		}
		if repoAction == "Delete and re-clone" {
			p.ForceClone = "y"
		}
	}
}

// ConfigureBuildOptions prompts for build configuration
func (p *Platform) ConfigureBuildOptions() {
	// Prompt for TVM source selection
	tvmSourcePrompt := promptui.Select{
		Label: "Select TVM source",
		Items: []string{
			"Use bundled TVM (from mlc-llm/3rdparty)",
			"Use mlc-ai/relax stable (clones mlc branch)",
			"Use custom TVM (from repo_root/tvm)",
		},
	}
	_, tvmSourceSelection, err := tvmSourcePrompt.Run()
	if err != nil {
		handlePromptError(err)
	}

	switch tvmSourceSelection {
	case "Use mlc-ai/relax stable (clones mlc branch)":
		p.TVMSource = "relax"
	case "Use custom TVM (from repo_root/tvm)":
		p.TVMSource = "custom"
	default:
		p.TVMSource = "bundled"
	}

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

// ConfigureWheelBuildOption prompts for wheel building
func (p *Platform) ConfigureWheelBuildOption() {
	p.BuildWheels = promptYesNo("Build Python wheels after compilation?")
}

// configureRemoteModel sets up a remote model from URL
func (p *Platform) configureRemoteModel() {
	p.ModelURL = promptModelURL()
	p.ModelName = extractModelNameFromURL(p.ModelURL)
}

// configureLocalModel sets up a local model
func (p *Platform) configureLocalModel() {
	p.ModelName = selectLocalModel()
	p.ModelURL = ""
}

// configureDevice prompts for device selection
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

	// Validate device
	if !isValidDevice(p.Device) {
		fmt.Printf(Warning+"Invalid device '%s'. Using default: %s\n", p.Device, deviceDefault)
		p.Device = deviceDefault
	}
}

// ConfigureModel prompts for model source selection
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

// CheckAndInstallConda verifies conda is installed and offers to install if not
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
			scriptPath := "scripts/" + operatingSystem + "_install_conda.sh"
			// Check if script exists
			if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
				reportError("Conda install script not found: %s", err)
				return
			}
			installCmd := exec.Command("bash", scriptPath)
			installCmd.Stdout = os.Stdout
			installCmd.Stderr = os.Stderr
			if err := installCmd.Run(); err != nil {
				reportError("Conda installation failed", err)
			}
		} else {
			reportError("Conda is required to proceed. Please install conda and try again.", nil)
		}
	}
}

// CheckCudaInstalled checks if CUDA is installed
func CheckCudaInstalled() bool {
	cmd := exec.Command("nvcc", "--version")
	err := cmd.Run()
	return err == nil
}

// CreatePlatform creates a new Platform instance with user prompts
func CreatePlatform() *Platform {
	operatingSystem := ""
	tvmBuildEnv := ""
	mlcBuildEnv := ""
	cliEnv := ""

	osPrompt := promptui.Select{
		Label: "Select Operating System",
		Items: []string{"mac", "linux"},
	}
	_, operatingSystem, err := osPrompt.Run()
	if err != nil {
		handlePromptError(err)
	}

	CheckAndInstallConda(operatingSystem)

	tvmBuildEnvPrompt := promptui.Prompt{
		Label:   "Enter a TVM build environment name",
		Default: "tvm-build-venv",
	}
	tvmBuildEnv, err = tvmBuildEnvPrompt.Run()
	if err != nil {
		handlePromptError(err)
	}

	mlcBuildEnvPrompt := promptui.Prompt{
		Label:   "Enter a MLC build environment name",
		Default: "mlc-build-venv",
	}
	mlcBuildEnv, err = mlcBuildEnvPrompt.Run()
	if err != nil {
		handlePromptError(err)
	}

	cliEnvPrompt := promptui.Prompt{
		Label:   "Enter a CLI environment name",
		Default: "mlc-cli-venv",
	}
	cliEnv, err = cliEnvPrompt.Run()
	if err != nil {
		handlePromptError(err)
	}

	return &Platform{
		OperatingSystem: operatingSystem,
		TVMBuildEnv:     tvmBuildEnv,
		MLCBuildEnv:     mlcBuildEnv,
		CliEnv:          cliEnv,
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
		TVMSource:       "",
		BuildWheels:     "y",
		ForceClone:      "n",
		InstallMode:     "source",
	}
}

// Helper functions

// handlePromptError handles errors from promptui
func handlePromptError(err error) {
	if errors.Is(err, promptui.ErrInterrupt) {
		fmt.Println("\nExiting...")
		os.Exit(0)
	}
	reportError("Prompt error", err)
}

// promptYesNo prompts for a yes/no selection
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

// reportError prints an error message and exits
func reportError(msg string, err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s%s: %v\n", Error, msg, err)
	} else {
		fmt.Fprintf(os.Stderr, "%s%s\n", Error, msg)
	}
	os.Exit(1)
}

// extractModelNameFromURL extracts the model name from a URL
func extractModelNameFromURL(url string) string {
	return filepath.Base(url)
}

// promptModelURL prompts for a model URL
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

// getLocalModelDirectories returns a list of model directories in models/
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

// promptLocalModelName prompts for a local model name
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

// selectLocalModel prompts the user to select a local model
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
