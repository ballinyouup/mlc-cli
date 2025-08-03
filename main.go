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
	}
	cmd.Stderr = os.Stderr
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
		userOS := getOS()
		envNames := promptEnvNames()
		buildEnv := envNames[0]
		cliEnv := envNames[1]

		clearEnvironments(buildEnv, cliEnv)
		createEnvironments(buildEnv, cliEnv)

		if userOS == "MacOS" {
			installTVM(userOS, cliEnv)
			promptMLCRepo()
			createDirectories()
			generateConfig(buildEnv)
			buildMLC(buildEnv)
			installMLC(cliEnv)
		}
	} else if selection == "Run" {
		envNames := promptEnvNames()
		cliEnv := envNames[1]
		createDirectories()
		runMLCModel(cliEnv)
	}
	println("MLC-LLM setup complete!")
}
