package main

// =============================================================================
// versions_defaults.go — Go-side mirror of scripts/config/versions.sh
// =============================================================================
// These constants mirror the values in scripts/config/versions.sh.
// When you change a value in versions.sh you MUST update the matching
// constant here, and vice versa.  The consistency check script
// scripts/check_version_consistency.sh will catch drift.
//
// Go cannot source shell scripts at compile time, so we keep this thin
// mirror rather than parsing the shell file at runtime.
// =============================================================================

const (
	// DefaultMlcLLMRepo is the default GitHub repo for mlc-llm.
	// Mirror of MLC_LLM_REPO in scripts/config/versions.sh.
	DefaultMlcLLMRepo = "https://github.com/mlc-ai/mlc-llm"

	// DefaultCUDAArch is the default CUDA compute capability.
	// Mirror of CUDA_ARCH_DEFAULT in scripts/config/versions.sh.
	DefaultCUDAArch = "86"

	// DefaultPythonVersion is the Python version used for build and CLI envs.
	// Mirror of PYTHON_VERSION in scripts/config/versions.sh.
	// NOTE: This constant is informational only — Go does not create conda
	// environments directly. The shell scripts use PYTHON_VERSION.
	DefaultPythonVersion = "3.13"
)
