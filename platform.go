package main

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"

	"github.com/manifoldco/promptui"
)

type Platform interface {
	GenerateConfig()
	BuildMLC()
	InstallMLC()
	PromptMLCRepo()
	RunMLCModel()
	CreateEnvironments()
	ClearEnvironments()
	CreateDirectories()
	BuildAndroid()
}
type BasePlatform struct {
	BuildEnv string
	CliEnv   string
	Name     string
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
		cmd := exec.Command("git", "clone", "--recurse-submodules", repo, "mlc-llm")
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
	cmd := exec.Command("bash", "scripts/linux_install_mlc.sh", platform.CliEnv)
	cmd.Dir = "."
	osToCmdOutput(cmd)
	err := cmd.Run()
	stopLoader(loading)
	if err != nil {
		cliError("Error installing MLC: ", err)
	}
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

	cmd := exec.Command("bash", "scripts/linux_run_model.sh", platform.CliEnv, url, modelName)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err = cmd.Run()
	if err != nil {
		cliError("Error running MLC: ", err)
	}
}

func (platform *BasePlatform) CreateEnvironments() {
	loading := createLoader("Creating Environments...")
	cmd := exec.Command("bash", "scripts/linux_env.sh", platform.CliEnv, platform.BuildEnv, "no")
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
		cmd := exec.Command("bash", "scripts/linux_env.sh", platform.CliEnv, platform.BuildEnv, "yes")
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

	if err := os.MkdirAll("models", 0755); err != nil {
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
