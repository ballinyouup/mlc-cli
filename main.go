package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"

	"github.com/manifoldco/promptui"
)

const (
	Green   = "\033[32m"
	Reset   = "\033[0m"
	Red     = "\033[31m"
	Success = "[" + Green + "✓" + Reset + "] "
	Error   = "[" + Red + "✗" + Reset + "] "
)

func cliError(msg string, err error) {
	fmt.Println(Error + msg + err.Error())
	os.Exit(1)
}

func main() {
	fmt.Println("Welcome to MLC-LLM CLI!")

	prompt := promptui.Select{
		Label: "Options",
		Items: []string{"Build", "Run", "Quantize Model", "Deploy"},
	}
	_, selection, err := prompt.Run()
	if err != nil {
		if errors.Is(err, promptui.ErrInterrupt) {
			fmt.Println("\nExiting...")
			os.Exit(0)
		}
		cliError("Error getting selection: ", err)
	}

	if selection == "Build" {
		platform := CreatePlatform()
		platform.ConfigureGitHubRepo()
		platform.ConfigureBuildOptions()
		promptInstall(platform, "cuda")
		promptBuild(platform, "tvm")
		promptBuild(platform, "mlc")
		promptInstall(platform, "tvm")
		promptInstall(platform, "mlc")

	} else if selection == "Run" {
		platform := CreatePlatform()
		platform.ConfigureModel()
		platform.run()

	} else if selection == "Quantize Model" {
		platform := CreatePlatform()
		promptQuantizeModel(platform)

	} else if selection == "Deploy" {
	}
}

func promptQuantizeModel(platform Platform) {
	promptPath := promptui.Prompt{
		Label: "Enter Hugging Face Model Path (or local path)",
	}
	modelPath, err := promptPath.Run()
	if err != nil {
		cliError("Input failed", err)
	}

	promptOut := promptui.Prompt{
		Label:   "Enter Output Directory",
		Default: "dist/model-output",
	}
	outputDir, err := promptOut.Run()
	if err != nil {
		cliError("Input failed", err)
	}

	promptQuant := promptui.Select{
		Label: "Select Quantization Level",
		Items: []string{
			"q4f16_1 (Standard - 4-bit, Balanced)",
			"q3f16_1 (Small - 3-bit, Low VRAM)",
			"q0f16   (None - 16-bit, High Quality)",
		},
	}
	_, result, err := promptQuant.Run()
	if err != nil {
		cliError("Selection failed", err)
	}

	var quantCode string
	switch result {
	case "q3f16_1 (Small - 3-bit, Low VRAM)":
		quantCode = "q3f16_1"
	case "q0f16   (None - 16-bit, High Quality)":
		quantCode = "q0f16"
	default:
		quantCode = "q4f16_1"
	}

	promptTemplate := promptui.Select{
		Label: "Select Conversation Template",
		Items: []string{"llama-3", "chatml", "mistral_default", "phi-2", "gemma", "qwen2"},
	}
	_, convTemplate, err := promptTemplate.Run()
	if err != nil {
		cliError("Selection failed", err)
	}

	fmt.Printf("\n🚀 Starting Quantization [%s] using env [%s]...\n", quantCode, platform.CliEnv)

	cmd := exec.Command("conda", "run", "--no-capture-output", "-n", platform.CliEnv,
		"python", "-m", "mlc_llm", "convert_weight",
		modelPath,
		"--quantization", quantCode,
		"-o", outputDir)

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		cliError("Quantization failed: ", err)
	}

	fmt.Println("\n📄 Generating config...")
	cmdConfig := exec.Command("conda", "run", "--no-capture-output", "-n", platform.CliEnv,
		"python", "-m", "mlc_llm", "gen_config", modelPath,
		"--quantization", quantCode,
		"--conv-template", convTemplate,
		"-o", outputDir)

	cmdConfig.Stdout = os.Stdout
	cmdConfig.Stderr = os.Stderr

	if err := cmdConfig.Run(); err != nil {
		cliError("Config generation failed: ", err)
	}

	fmt.Println("\n" + Success + "Quantization Complete! Model saved to " + outputDir)
}

func promptInstall(platform Platform, pkg string) {
	prompt := promptui.Select{
		Label: "Install " + pkg + "?",
		Items: []string{"Yes", "No"},
	}

	_, result, err := prompt.Run()
	if err != nil {
		if errors.Is(err, promptui.ErrInterrupt) {
			fmt.Println("\nExiting...")
			os.Exit(0)
		}
		cliError("Error getting selection: ", err)
	}
	if result == "Yes" {
		platform.install(pkg)
	}
}

func promptBuild(platform Platform, pkg string) {
	prompt := promptui.Select{
		Label: "Build " + pkg + " from source?",
		Items: []string{"Yes", "No"},
	}

	_, result, err := prompt.Run()
	if err != nil {
		if errors.Is(err, promptui.ErrInterrupt) {
			fmt.Println("\nExiting...")
			os.Exit(0)
		}
		cliError("Error getting selection: ", err)
	}
	if result == "Yes" {
		platform.build(pkg)
	}
}