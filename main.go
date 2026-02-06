package main

import (
	"errors"
	"fmt"
	"os"

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
		Items: []string{"Build", "Run", "Deploy"},
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
	} else if selection == "Deploy" {
	}
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
