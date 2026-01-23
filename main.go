package main

import (
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
		cliError("Error getting selection: ", err)
	}
	if selection == "Build" {
		platform := CreatePlatform()
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

func promptInstall(platform BasePlatform, pkg string) {
	prompt := promptui.Select{
		Label: "Install " + pkg + "?",
		Items: []string{"Yes", "No"},
	}

	_, result, err := prompt.Run()
	if err != nil {
		cliError("Error getting selection: ", err)
	}
	if result == "Yes" {
		platform.install(pkg)
	}
}

func promptBuild(platform BasePlatform, pkg string) {
	prompt := promptui.Select{
		Label: "Build " + pkg + " from source?",
		Items: []string{"Yes", "No"},
	}

	_, result, err := prompt.Run()
	if err != nil {
		cliError("Error getting selection: ", err)
	}
	if result == "Yes" {
		platform.build(pkg)
	}
}
