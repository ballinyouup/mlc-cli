package main

import (
	"flag"
	"fmt"
	"github.com/manifoldco/promptui"
	"os"
	"os/exec"
)

var verbose bool

func osToCmdOutput(cmd *exec.Cmd) {
	if verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
}

func cliError(msg string, err error) {
	fmt.Println(msg + err.Error())
	os.Exit(1)
}

func main() {
	verboseFlag := flag.Bool("v", false, "Enable verbose output for all steps")
	flag.Parse()
	verbose = *verboseFlag

	fmt.Println("Welcome to MLC-LLM CLI!")

	prompt := promptui.Select{
		Label: "Options",
		Items: []string{"Build", "Run"},
	}
	_, selection, err := prompt.Run()
	if err != nil {
		cliError("Error getting selection: ", err)
	}
	if selection == "Build" {
		platform := CreatePlatform()
		platform.ClearEnvironments()
		platform.CreateEnvironments()
		platform.InstallTVM()
		platform.PromptMLCRepo()
		platform.CreateDirectories()
		platform.GenerateConfig()
		platform.BuildMLC()
		platform.InstallMLC()
	} else if selection == "Run" {
		//envNames := PromptEnvNames()
		//cliEnv := envNames[1]
		platform := CreatePlatform()
		platform.CreateDirectories()
		platform.RunMLCModel()
	}
	println("MLC-LLM setup complete!")
}
