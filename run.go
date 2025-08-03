package main

import (
	"fmt"
	"github.com/manifoldco/promptui"
	"os"
	"os/exec"
	"strings"
)

func runMLCModel(cliEnv string) {
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
		gitCmd := exec.Command("conda", "run", "-n", cliEnv, "git", "clone", url)
		gitCmd.Dir = "mlc-llm/models/"
		osToCmdOutput(gitCmd)
		err = gitCmd.Run()
		if err != nil {
			cliError("Error cloning model: ", err)
		}

		gitPullCmd := exec.Command("conda", "run", "-n", cliEnv, "git", "lfs", "pull")
		gitPullCmd.Dir = modelPath
		osToCmdOutput(gitPullCmd)
		err = gitPullCmd.Run()
		if err != nil {
			cliError("Error pulling model: ", err)
		}
	} else {
		fmt.Println("Model directory already exists, skipping clone.")
	}

	runCmdString := fmt.Sprintf("source $(conda info --base)/etc/profile.d/conda.sh && (conda activate %s && mlc_llm chat .)", cliEnv)
	runCmd := exec.Command("bash", "-c", runCmdString)
	runCmd.Dir = modelPath
	runCmd.Stdin = os.Stdin
	osToCmdOutput(runCmd)
	err = runCmd.Run()
	if err != nil {
		cliError("Error running MLC: ", err)
	}
}
