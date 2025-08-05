package main

import (
	"fmt"
	"github.com/manifoldco/promptui"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

type MacOSPlatform struct {
	BasePlatform
}

func (mac *MacOSPlatform) InstallTVM() {
	prompt := promptui.Select{
		Label: "Install tvm:",
		Items: []string{"Pre-built", "From source"},
	}
	_, installMethod, err := prompt.Run()
	if err != nil {
		cliError("Error getting install tvm response: ", err)
	}

	if installMethod == "Pre-built" {
		loading := createLoader("Installing Pre-built TVM...")
		cmd := exec.Command("conda", "run", "-n", mac.CliEnv, "pip", "install", "--pre", "-U", "-f", "https://mlc.ai/wheels", "mlc-ai-nightly-cpu")
		osToCmdOutput(cmd)
		err := cmd.Run()
		if err != nil {
			cliError("Error installing TVM: ", err)
		}
		stopLoader(loading)
		println(Success + "Installed TVM")
	} else {
		// TODO: Build & Install TVM from source
	}

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
		Label: "Select GPU runtime (or None for CPU-only)",
		Items: []string{"CUDA", "ROCM", "VULKAN", "METAL", "OPENCL", "None"},
	}
	_, gpuRuntime, err := runtimePrompt.Run()
	loading := createLoader("Generating Config...")
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

	commandString := fmt.Sprintf("source $(conda info --base)/etc/profile.d/conda.sh && (conda activate %s && echo -e '%s' | python3 ../cmake/gen_cmake_config.py)", mac.BuildEnv, input)
	cmd := exec.Command("bash", "-c", commandString)
	cmd.Dir = "mlc-llm/build"
	osToCmdOutput(cmd)

	err = cmd.Run()
	if err != nil {
		cliError("Error generating config: ", err)
	}
	stopLoader(loading)
	println(Success + "Generated Config")
}

func (mac *MacOSPlatform) BuildMLC() {
	loading := createLoader("Building MLC...")
	cmakeCmd := exec.Command("conda", "run", "-n", mac.BuildEnv, "cmake", "..")
	cmakeCmd.Dir = "mlc-llm/build"
	osToCmdOutput(cmakeCmd)
	err := cmakeCmd.Run()
	if err != nil {
		cliError("Error running cmake: ", err)
	}

	nCores := runtime.NumCPU() // Get the number of CPU cores
	buildCmd := exec.Command("conda", "run", "-n", mac.BuildEnv, "cmake", "--build", ".", "--parallel", fmt.Sprintf("%d", nCores))
	buildCmd.Dir = "mlc-llm/build"
	osToCmdOutput(buildCmd)
	err = buildCmd.Run()
	if err != nil {
		cliError("Error building MLC: ", err)
	}
	stopLoader(loading)
	println(Success + "Built MLC")
}

func (mac *MacOSPlatform) BuildTVM() {
	// TODO
}

func (mac *MacOSPlatform) BuildAndroid() {
	loading := createLoader("Building Android...")
	// Check Dependencies
	installRustString := "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
	rustcCmd := exec.Command("conda", "run", "-n", mac.CliEnv, "which", "rustc")
	if err := rustcCmd.Run(); err != nil {
		cliError("rustc is not installed.\n"+installRustString, err)
	}
	cargoCmd := exec.Command("conda", "run", "-n", mac.CliEnv, "which", "cargo")
	if err := cargoCmd.Run(); err != nil {
		cliError("cargo is not installed.\n"+installRustString, err)
	}
	rustupCmd := exec.Command("conda", "run", "-n", mac.CliEnv, "which", "rustup")
	if err := rustupCmd.Run(); err != nil {
		cliError("rustup is not installed.\n"+installRustString, err)
	}

	// Get Paths
	wd, _ := os.Getwd()
	ndkRoot := filepath.Join(os.Getenv("HOME"), "Library/Android/sdk/ndk")
	entries, _ := os.ReadDir(ndkRoot)
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
	androidNDK := filepath.Join(ndkRoot, entries[len(entries)-1].Name())

	buildCmd := exec.Command("conda", "run", "-n", mac.CliEnv, "mlc_llm", "package")
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
