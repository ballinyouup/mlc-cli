package main

import (
	"fmt"
	"github.com/manifoldco/promptui"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

type Platform interface {
	InstallTVM()
	GenerateConfig()
	BuildMLC()
	InstallMLC()
	PromptMLCRepo()
	RunMLCModel()
	CreateEnvironments()
	ClearEnvironments()
	CreateDirectories()
	// CheckDependencies() error // Check if all dependencies are installed
}
type BasePlatform struct {
	BuildEnv string
	CliEnv   string
	Name     string
}

func (platform *BasePlatform) CheckCUDA() {
	cmd := exec.Command("nvidia-smi")
	err := cmd.Run()
	if err != nil {
		cliError("CUDA not found. Install CUDA and try again.", err)
	}
}

func (platform *BasePlatform) CheckROCM() {
	cmd := exec.Command("rocm-smi")
	err := cmd.Run()
	if err != nil {
		cliError("ROCM not found. Install ROCM and try again.", err)
	}
}

func (platform *BasePlatform) CheckVulkan() {
	cmd := exec.Command("vulkaninfo")
	err := cmd.Run()
	if err != nil {
		cliError("Vulkan not found. Install Vulkan and try again.", err)
	}
}

func (platform *BasePlatform) PromptMLCRepo() {
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

func (platform *BasePlatform) InstallMLC() {
	installCmd := exec.Command("conda", "run", "-n", platform.CliEnv, "pip", "install", ".")
	installCmd.Dir = "mlc-llm/python"
	osToCmdOutput(installCmd)
	err := installCmd.Run()
	if err != nil {
		cliError("Error installing MLC: ", err)
	}
}

func (platform *BasePlatform) RunMLCModel() {
	urlPrompt := promptui.Prompt{
		Label: "Enter Huggingface url",
	}
	url, err := urlPrompt.Run()
	if err != nil {
		cliError("Error getting url: ", err)
	}

	modelName := strings.Split(url, "/")[4]
	modelPath := "mlc-llm/models/" + modelName

	if !strings.Contains(url, "huggingface.co") {
		cliError("Invalid url. Must be a huggingface url", nil)
	} else if !strings.Contains(url, "MLC") {
		continuePrompt := promptui.Select{
			Label: "Huggingface url does not contain MLC. Continue?",
			Items: []string{"Yes", "No"},
		}
		_, choice, err := continuePrompt.Run()
		if err != nil {
			cliError("Error getting choice: ", err)
		}
		if choice == "No" {
			os.Exit(0)
		}
	}

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Println("Cloning model...")
		gitCmd := exec.Command("conda", "run", "-n", platform.CliEnv, "git", "clone", url)
		gitCmd.Dir = "mlc-llm/models/"
		osToCmdOutput(gitCmd)
		err = gitCmd.Run()
		if err != nil {
			cliError("Error cloning model: ", err)
		}

		gitPullCmd := exec.Command("conda", "run", "-n", platform.CliEnv, "git", "lfs", "pull")
		gitPullCmd.Dir = modelPath
		osToCmdOutput(gitPullCmd)
		err = gitPullCmd.Run()
		if err != nil {
			cliError("Error pulling model: ", err)
		}
	} else {
		fmt.Println("Model directory already exists, skipping clone.")
	}

	runCmdString := fmt.Sprintf("source $(conda info --base)/etc/profile.d/conda.sh && (conda activate %s && mlc_llm chat .)", platform.CliEnv)
	runCmd := exec.Command("bash", "-c", runCmdString)
	runCmd.Dir = modelPath
	runCmd.Stdin = os.Stdin
	runCmd.Stdout = os.Stdout
	osToCmdOutput(runCmd)
	err = runCmd.Run()
	if err != nil {
		cliError("Error running MLC: ", err)
	}
}

func (platform *BasePlatform) CreateEnvironments() {
	createBuildEnvCmd := exec.Command("conda", "create", "-n", platform.BuildEnv, "-c", "conda-forge", "--yes", "cmake=3.29", "rust", "git", "python=3.11", "pip", "sentencepiece", "git-lfs")
	osToCmdOutput(createBuildEnvCmd)
	err := createBuildEnvCmd.Run()
	if err != nil {
		cliError("Error creating build environment: ", err)
	}
	createCliEnvCmd := exec.Command("conda", "create", "-n", platform.CliEnv, "-c", "conda-forge", "--yes", "cmake=3.29", "rust", "git", "python=3.11", "pip", "sentencepiece", "git-lfs")
	osToCmdOutput(createCliEnvCmd)
	err = createCliEnvCmd.Run()
	if err != nil {
		cliError("Error creating cli environment: ", err)
	}
}

func (platform *BasePlatform) ClearEnvironments() {
	prompt := promptui.Select{
		Label: "Clear existing environments?",
		Items: []string{"Yes", "No"},
	}
	_, resp, err := prompt.Run()
	if err != nil {
		cliError("Error getting clear environments response: ", err)
	}
	if resp == "Yes" {
		fmt.Println("Removing build environment:", platform.BuildEnv)
		removeBuildCmd := exec.Command("conda", "env", "remove", "--name", platform.BuildEnv, "--yes")
		osToCmdOutput(removeBuildCmd)
		_ = removeBuildCmd.Run() // Ignore the error if the environment doesn't exist

		fmt.Println("Removing CLI environment:", platform.CliEnv)
		removeCliCmd := exec.Command("conda", "env", "remove", "--name", platform.CliEnv, "--yes")
		osToCmdOutput(removeCliCmd)
		_ = removeCliCmd.Run() // Ignore the error if the environment doesn't exist
	}
}

func (platform *BasePlatform) CreateDirectories() {
	if err := os.MkdirAll("mlc-llm/build", 0755); err != nil {
		cliError("Error creating build directory: ", err)
	}

	if err := os.MkdirAll("mlc-llm/models", 0755); err != nil {
		cliError("Error creating models directory: ", err)
	}
}

func PromptEnvNames() []string {
	buildPrompt := promptui.Prompt{
		Label:   "Enter build environment name (press enter for 'mlc-llm-venv')",
		Default: "mlc-llm-venv",
	}
	buildResult, buildErr := buildPrompt.Run()

	if buildErr != nil {
		cliError("Error getting build environment name: ", buildErr)
	}

	cliPrompt := promptui.Prompt{
		Label:   "Enter cli environment name (press enter for 'mlc-cli-venv')",
		Default: "mlc-cli-venv",
	}

	cliResult, cliErr := cliPrompt.Run()
	if cliErr != nil {
		cliError("Error getting cli environment name: ", cliErr)
	}

	return []string{buildResult, cliResult}
}

func CreatePlatform() Platform {
	envNames := PromptEnvNames()
	switch runtime.GOOS {
	case "darwin":
		return &MacOSPlatform{
			BasePlatform: BasePlatform{
				Name:     "MacOS",
				BuildEnv: envNames[0],
				CliEnv:   envNames[1],
			},
		}
	case "linux":
		return &LinuxPlatform{
			BasePlatform: BasePlatform{
				Name:     "Linux",
				BuildEnv: envNames[0],
				CliEnv:   envNames[1],
			},
		}
	case "windows":
		return &WindowsPlatform{
			BasePlatform: BasePlatform{
				Name:     "Windows",
				BuildEnv: envNames[0],
				CliEnv:   envNames[1],
			},
		}
	default:
		fmt.Printf("Unsupported OS: %s\n", runtime.GOOS)
		os.Exit(1)
	}
	return nil
}
