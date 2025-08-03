package main

import (
	"fmt"
	"github.com/manifoldco/promptui"
	"os"
	"os/exec"
	"runtime"
)

func getOS() string {
	detectedOS := ""
	switch runtime.GOOS {
	case "darwin":
		detectedOS = "MacOS"
	case "linux":
		detectedOS = "Linux"
	case "windows":
		detectedOS = "Windows"
	default:
		fmt.Printf("Unsupported OS: %s\n", runtime.GOOS)
		os.Exit(1)
	}

	prompt := promptui.Select{
		Label: "Detected OS: " + detectedOS + ". Is this correct?",
		Items: []string{"Yes", "No"},
	}
	_, resp, err := prompt.Run()
	if err != nil {
		cliError("Error getting OS confirmation: ", err)
	}

	if resp == "Yes" {
		return detectedOS
	}

	// Fallback to manual selection if "No"
	prompt = promptui.Select{
		Label: "Select OS",
		Items: []string{"MacOS", "Windows", "Linux"},
	}
	_, userOS, err := prompt.Run()
	if err != nil {
		cliError("Error getting OS: ", err)
	}
	return userOS
}

func promptEnvNames() []string {
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

func createEnvironments(buildEnv string, cliEnv string) {
	createBuildEnvCmd := exec.Command("conda", "create", "-n", buildEnv, "-c", "conda-forge", "--yes", "cmake=3.29", "rust", "git", "python=3.11", "pip", "sentencepiece", "git-lfs")
	osToCmdOutput(createBuildEnvCmd)
	err := createBuildEnvCmd.Run()
	if err != nil {
		cliError("Error creating build environment: ", err)
	}
	createCliEnvCmd := exec.Command("conda", "create", "-n", cliEnv, "-c", "conda-forge", "--yes", "cmake=3.29", "rust", "git", "python=3.11", "pip", "sentencepiece", "git-lfs")
	osToCmdOutput(createCliEnvCmd)
	err = createCliEnvCmd.Run()
	if err != nil {
		cliError("Error creating cli environment: ", err)
	}
}

func clearEnvironments(buildEnv string, cliEnv string) {
	prompt := promptui.Select{
		Label: "Clear existing environments?",
		Items: []string{"Yes", "No"},
	}
	_, resp, err := prompt.Run()
	if err != nil {
		cliError("Error getting clear environments response: ", err)
	}
	if resp == "Yes" {
		fmt.Println("Removing build environment:", buildEnv)
		removeBuildCmd := exec.Command("conda", "env", "remove", "--name", buildEnv, "--yes")
		osToCmdOutput(removeBuildCmd)
		_ = removeBuildCmd.Run() // Ignore the error if the environment doesn't exist

		fmt.Println("Removing CLI environment:", cliEnv)
		removeCliCmd := exec.Command("conda", "env", "remove", "--name", cliEnv, "--yes")
		osToCmdOutput(removeCliCmd)
		_ = removeCliCmd.Run() // Ignore the error if the environment doesn't exist
	}
}

func createDirectories() {
	if err := os.MkdirAll("mlc-llm/build", 0755); err != nil {
		cliError("Error creating build directory: ", err)
	}

	if err := os.MkdirAll("mlc-llm/models", 0755); err != nil {
		cliError("Error creating models directory: ", err)
	}
}
