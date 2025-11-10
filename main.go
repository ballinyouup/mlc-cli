package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/manifoldco/promptui"
)

var verbose bool = false

const (
	Green   = "\033[32m"
	Reset   = "\033[0m"
	Red     = "\033[31m"
	Success = "[" + Green + "✓" + Reset + "] "
	Error   = "[" + Red + "✗" + Reset + "] "
)

func osToCmdOutput(cmd *exec.Cmd) {
	if verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
}

func cliError(msg string, err error) {
	fmt.Println(Error + msg + err.Error())
	os.Exit(1)
}

func createLoader(message string) chan bool {
	loading := make(chan bool)
	go spinner(message, loading)
	return loading
}

func stopLoader(loader chan bool) {
	loader <- true
	time.Sleep(50 * time.Millisecond) // Work around for race condition
}

func spinner(message string, stop chan bool) {
	for {
		select {
		case <-stop:
			fmt.Printf("\r%s\r", strings.Repeat(" ", len(message)+5))
			return
		default:
			for _, char := range `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` {
				fmt.Printf("\r%s%c%s %s", Green, char, Reset, message)
				time.Sleep(100 * time.Millisecond)
			}
		}
	}
}

func main() {
	verboseFlag := flag.Bool("v", false, "Enable verbose output for all steps")
	flag.Parse()
	verbose = *verboseFlag

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
		// TODO: Ask the user if we should keep current environments

		platform.ClearEnvironments()
		platform.CreateEnvironments()
		platform.PromptMLCRepo()
		platform.CreateDirectories()
		platform.GenerateConfig()
		platform.BuildMLC()
		platform.InstallMLC() // This now installs both MLC and TVM
		println("MLC-LLM setup complete!")

	} else if selection == "Run" {
		platform := CreatePlatform()
		platform.CreateDirectories()
		platform.RunMLCModel()
	} else if selection == "Deploy" {
		prompt := promptui.Select{
			Label: "Options",
			Items: []string{"WebLLM", "REST API", "iOS", "Android"},
		}
		_, selection, err := prompt.Run()
		if err != nil {
			cliError("Error getting selection: ", err)
		}

		if selection == "Android" {
			platform := CreatePlatform()
			platform.BuildAndroid()
			println("MLC-LLM Android build complete!")
		}
	}
}
