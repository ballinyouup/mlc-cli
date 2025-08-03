package main

import (
	"fmt"
	"github.com/manifoldco/promptui"
	"os"
	"os/exec"
	"runtime"
	"strings"
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
		Label:   "Enter build environment name (press enter for 'mlc-chat-venv')",
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

func installTVM(userOS string, cliEnv string) {
	prompt := promptui.Select{
		Label: "Install tvm:",
		Items: []string{"Pre-built", "From source"},
	}
	_, resp, err := prompt.Run()
	if err != nil {
		cliError("Error getting install tvm response: ", err)
	}
	if userOS == "MacOS" {
		if resp == "Pre-built" {
			cmd := exec.Command("conda", "run", "-n", cliEnv, "pip", "install", "--pre", "-U", "-f", "https://mlc.ai/wheels", "mlc-ai-nightly-cpu")
			osToCmdOutput(cmd)
			err := cmd.Run()
			if err != nil {
				cliError("Error installing TVM: ", err)
			}
		} else {
			// TODO: Build & Install TVM from source
		}
	}
}

func promptMLCRepo() {
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
		cmd := exec.Command("git", "clone", "--recursive", repo)
		osToCmdOutput(cmd)
		err := cmd.Run()
		if err != nil {
			cliError("Error cloning MLC repo: ", err)
		}
	}
}

func generateConfig(buildEnv string) {
	CUDA := "n"
	ROCM := "n"
	VULKAN := "n"
	METAL := "n"
	OPENCL := "n"
	tvmPrompt := promptui.Prompt{
		Label:   "Enter TVM_SOURCE_DIR in absolute path. If not specified, 3rdparty/tvm will be used by default",
		Default: "",
	}
	TvmSourceDir, err := tvmPrompt.Run()
	if err != nil {
		cliError("Error getting TVM source directory: ", err)
	}
	runtimePrompt := promptui.Select{
		Label: "Select GPU runtime (or None for CPU-only)",
		Items: []string{"CUDA", "ROCM", "VULKAN", "METAL", "OPENCL", "None"},
	}
	_, gpuRuntime, err := runtimePrompt.Run()
	if err != nil {
		cliError("Error getting GPU runtime: ", err)
	}
	switch gpuRuntime {
	case "CUDA":
		CUDA = "y"
	case "ROCM":
		ROCM = "y"
	case "VULKAN":
		VULKAN = "y"
	case "METAL":
		METAL = "y"
	case "OPENCL":
		OPENCL = "y"
	}

	answers := []string{
		TvmSourceDir, // 1. TVM_SOURCE_DIR
		CUDA,         // 2. Use CUDA
		ROCM,         // 3. Use ROCM
		VULKAN,       // 4. Use Vulkan
		METAL,        // 5. Use Metal
		OPENCL,       // 6. Use OpenCL
	}

	input := strings.Join(answers, "\n") + "\n"

	commandString := fmt.Sprintf("source $(conda info --base)/etc/profile.d/conda.sh && (conda activate %s && echo -e '%s' | python3 ../cmake/gen_cmake_config.py)", buildEnv, input)
	cmd := exec.Command("bash", "-c", commandString)
	cmd.Dir = "mlc-llm/build"
	osToCmdOutput(cmd)

	err = cmd.Run()
	if err != nil {
		cliError("Error generating config: ", err)
	}
}

func buildMLC(buildEnv string) {
	cmakeCmd := exec.Command("conda", "run", "-n", buildEnv, "cmake", "..")
	cmakeCmd.Dir = "mlc-llm/build"
	osToCmdOutput(cmakeCmd)
	err := cmakeCmd.Run()
	if err != nil {
		cliError("Error running cmake: ", err)
	}

	nCores := runtime.NumCPU() // Get the number of CPU cores
	//fmt.Printf("Building MLC with %d parallel jobs...\n", nCores)
	buildCmd := exec.Command("conda", "run", "-n", buildEnv, "cmake", "--build", ".", "--parallel", fmt.Sprintf("%d", nCores))
	buildCmd.Dir = "mlc-llm/build"
	osToCmdOutput(buildCmd)
	err = buildCmd.Run()
	if err != nil {
		cliError("Error building MLC: ", err)
	}
}

func installMLC(cliEnv string) {
	installCmd := exec.Command("conda", "run", "-n", cliEnv, "pip", "install", ".")
	installCmd.Dir = "mlc-llm/python"
	osToCmdOutput(installCmd)
	err := installCmd.Run()
	if err != nil {
		cliError("Error installing MLC: ", err)
	}
}

func osToCmdOutput(cmd *exec.Cmd) {
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
}

func cliError(msg string, err error) {
	fmt.Println(msg + err.Error())
	os.Exit(1)
}

func main() {
	userOS := getOS()
	envNames := promptEnvNames()
	buildEnv := envNames[0]
	cliEnv := envNames[1]

	clearEnvironments(buildEnv, cliEnv)
	createEnvironments(buildEnv, cliEnv)

	if userOS == "MacOS" {
		installTVM(userOS, cliEnv)
		promptMLCRepo()
		if err := os.MkdirAll("mlc-llm/build", 0755); err != nil {
			cliError("Error creating build directory: ", err)
		}
		generateConfig(buildEnv)
		buildMLC(buildEnv)
		installMLC(cliEnv)
	}

	println("MLC-LLM setup complete!")
}
