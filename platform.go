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
	BuildTVM()
	GenerateConfig()
	BuildMLC()
	InstallMLC()
	PromptMLCRepo()
	RunMLCModel()
	CreateEnvironments()
	ClearEnvironments()
	CreateDirectories()
	BuildAndroid()
	GetName() string
	GetBuildEnv() string
	GetCliEnv() string
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
		loading := createLoader("Cloning MLC repo...")
		cmd := exec.Command("git", "clone", "--recursive", repo)
		osToCmdOutput(cmd)
		err := cmd.Run()
		if err != nil {
			cliError("Error cloning MLC repo: ", err)
		}
		stopLoader(loading)
		println(Success + "Cloned MLC repo")
	}
}

func (platform *BasePlatform) InstallMLC() {
	loading := createLoader("Installing MLC to environment...")

	commandString := fmt.Sprintf("source $(conda info --base)/etc/profile.d/conda.sh && conda activate %s && cd python && pip install --no-deps .", platform.CliEnv)
	installCmd := exec.Command("bash", "-c", commandString)
	installCmd.Dir = "mlc-llm"
	osToCmdOutput(installCmd)
	err := installCmd.Run()
	if err != nil {
		cliError("Error installing MLC: ", err)
	}
	stopLoader(loading)
	println(Success + "Installed MLC")
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
		commandString := fmt.Sprintf("source $(conda info --base)/etc/profile.d/conda.sh && conda activate %s && git clone %s && cd %s && git lfs pull", platform.CliEnv, url, modelName)
		gitCmd := exec.Command("bash", "-c", commandString)
		gitCmd.Dir = "mlc-llm/models/"
		osToCmdOutput(gitCmd)
		err = gitCmd.Run()
		if err != nil {
			cliError("Error cloning and pulling model: ", err)
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
	loading := createLoader("Creating Environments...")

	// Create both environments in a single shell command like the shell script
	commandString := fmt.Sprintf(`source $(conda info --base)/etc/profile.d/conda.sh && \
		conda create -n %s -c conda-forge --yes "cmake=3.29" rust git python=3.11 pip sentencepiece git-lfs && \
		conda create -n %s -c conda-forge --yes "cmake=3.29" rust git python=3.11 pip sentencepiece git-lfs`,
		platform.BuildEnv, platform.CliEnv)

	cmd := exec.Command("bash", "-c", commandString)
	osToCmdOutput(cmd)
	err := cmd.Run()
	if err != nil {
		cliError("Error creating environments: ", err)
	}

	stopLoader(loading)
	println(Success + "Created Environments: " + platform.BuildEnv + ", " + platform.CliEnv)
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
		loading := createLoader(fmt.Sprintf("Removing Environments %s, %s ...", platform.BuildEnv, platform.CliEnv))

		commandString := fmt.Sprintf(`source $(conda info --base)/etc/profile.d/conda.sh && \
			echo -e "\ny\ny" | conda env remove --name %s && \
			echo -e "\ny\ny" | conda env remove --name %s`,
			platform.BuildEnv, platform.CliEnv)

		cmd := exec.Command("bash", "-c", commandString)
		osToCmdOutput(cmd)
		_ = cmd.Run() // Ignore the error if the environments don't exist

		stopLoader(loading)
	}
	println(Success + "Cleared Environments: " + platform.BuildEnv + ", " + platform.CliEnv)
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
		println(Success + "Created Platform: MacOS")
		return &MacOSPlatform{
			BasePlatform: BasePlatform{
				Name:     "MacOS",
				BuildEnv: envNames[0],
				CliEnv:   envNames[1],
			},
		}
	case "linux":
		println(Success + "Created Platform: Linux")
		return &LinuxPlatform{
			BasePlatform: BasePlatform{
				Name:     "Linux",
				BuildEnv: envNames[0],
				CliEnv:   envNames[1],
			},
		}
	case "windows":
		println(Success + "Created Platform: Windows")
		return &WindowsPlatform{
			BasePlatform: BasePlatform{
				Name:     "Windows",
				BuildEnv: envNames[0],
				CliEnv:   envNames[1],
			},
		}
	default:
		cliError(Error+"Error creating platform: "+runtime.GOOS, nil)
		os.Exit(1)
	}
	return nil
}
