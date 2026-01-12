package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"

	"github.com/manifoldco/promptui"
)

type MacOSPlatform struct {
	BasePlatform
}

func (mac *MacOSPlatform) GenerateConfig() {
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
		Label: "Select runtime",
		Items: []string{"Metal (GPU)", "None (CPU)"},
	}
	_, gpuRuntime, err := runtimePrompt.Run()
	if err != nil {
		cliError("Error getting GPU runtime: ", err)
	}

	if gpuRuntime == "Metal (GPU)" {
		METAL = "y"
	}

	loading := createLoader("Generating Config...")
	cmd := exec.Command("bash", "scripts/mac_gen_config.sh", mac.BuildEnv, TvmSourceDir, CUDA, ROCM, VULKAN, METAL, OPENCL)
	cmd.Dir = "."
	osToCmdOutput(cmd)
	err = cmd.Run()
	stopLoader(loading)
	if err != nil {
		cliError("Error generating config: ", err)
	}
	println(Success + "Generated Config")
}

func (mac *MacOSPlatform) BuildMLC() {
	cleanBuildPrompt := promptui.Select{
		Label: "Clean build? (Required after TVM updates to fix compilation bugs)",
		Items: []string{"No (incremental)", "Yes (clean rebuild)"},
	}
	_, cleanBuildChoice, err := cleanBuildPrompt.Run()
	if err != nil {
		cliError("Error getting clean build choice: ", err)
	}

	cleanBuild := "no"
	if cleanBuildChoice == "Yes (clean rebuild)" {
		cleanBuild = "yes"
	}

	loading := createLoader("Building MLC...")
	cmd := exec.Command("bash", "scripts/mac_build.sh", mac.BuildEnv, cleanBuild)
	cmd.Dir = "."
	osToCmdOutput(cmd)
	err = cmd.Run()
	stopLoader(loading)
	if err != nil {
		cliError("Error building MLC: ", err)
	}
	println(Success + "Built MLC")
}

func (mac *MacOSPlatform) BuildAndroid() {
	loading := createLoader("Building Android...")
	// Check Dependencies using proper conda sourcing instead of conda run -n
	installRustString := "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"

	// Check for rustc, cargo, and rustup using single shell command with conda activation
	checkDepsCmd := fmt.Sprintf("source $(conda info --base)/etc/profile.d/conda.sh && conda activate %s && which rustc && which cargo && which rustup", mac.CliEnv)
	cmd := exec.Command("bash", "-c", checkDepsCmd)
	if err := cmd.Run(); err != nil {
		cliError("Rust toolchain is not installed.\n"+installRustString, err)
	}

	// Get Paths
	wd, _ := os.Getwd()
	ndkRoot := filepath.Join(os.Getenv("HOME"), "Library/Android/sdk/ndk")
	entries, _ := os.ReadDir(ndkRoot)
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
	androidNDK := filepath.Join(ndkRoot, entries[len(entries)-1].Name())

	// Use proper conda sourcing for the build command
	buildCmdString := fmt.Sprintf("source $(conda info --base)/etc/profile.d/conda.sh && conda activate %s && mlc_llm package", mac.CliEnv)
	buildCmd := exec.Command("bash", "-c", buildCmdString)
	buildCmd.Dir = "mlc-llm/android/MLCChat"
	buildCmd.Env = append(os.Environ(),
		"MLC_LLM_SOURCE_DIR="+filepath.Join(wd, "mlc-llm"),
		"TVM_SOURCE_DIR="+filepath.Join(wd, "mlc-llm/3rdparty/tvm"),
		"ANDROID_NDK="+androidNDK,
		"TVM_NDK_CC="+filepath.Join(
			androidNDK,
			"toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android24-clang",
		),
		"JAVA_HOME=/Applications/Android Studio.app/Contents/jbr/Contents/Home",
	)

	osToCmdOutput(buildCmd)
	if err := buildCmd.Run(); err != nil {
		cliError("Error building MLC: ", err)
	}
	stopLoader(loading)
	println(Success + "Built Android")
	println("Next steps:")
	println("Open folder ./android/MLCChat as an Android Studio Project")
	println("Connect your Android device to your machine. In the menu bar of Android Studio, click “Build → Make Project”.")
	println("Once the build is finished, click “Run → Run ‘app’” and you will see the app launched on your phone.")
}

func (mac *MacOSPlatform) GetName() string {
	return mac.Name
}

func (mac *MacOSPlatform) GetBuildEnv() string {
	return mac.BuildEnv
}

func (mac *MacOSPlatform) GetCliEnv() string {
	return mac.CliEnv
}
