#!/usr/bin/env bash
# =============================================================================
# scripts/check_version_consistency.sh
# =============================================================================
# Fail if hardcoded version/dependency values appear in active scripts outside
# the allowed files.  Run this in CI or pre-commit to catch drift from
# scripts/config/versions.sh early.
#
# Usage: bash scripts/check_version_consistency.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load canonical values from versions.sh so we know what to look for.
source "${SCRIPT_DIR}/config/versions.sh"

ERRORS=0
RED='\033[1;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

fail() { echo -e "${RED}[FAIL]${NC} $1"; ERRORS=$((ERRORS + 1)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
pass() { echo -e "${GREEN}[OK]${NC}  $1"; }

# =============================================================================
# Files exempt from all checks
# These are the single source of truth or intentional documentation.
# =============================================================================
ALLOWED_FILES=(
    "scripts/config/versions.sh"
    "scripts/check_version_consistency.sh"
    "versions_defaults.go"
    # README / DOCS intentionally document these values; exclude from checks.
    "README.md"
    "DOCS.md"
    "mlc-llm-repo.md"
)

# Create a temp file with allowed files for grep -Fv -f
EXCLUDE_FILE=$(mktemp)
trap 'rm -f "$EXCLUDE_FILE"' EXIT
for f in "${ALLOWED_FILES[@]}"; do
    echo "$f" >> "$EXCLUDE_FILE"
done

# =============================================================================
# Helper: search active scripts only, excluding exempt files
# =============================================================================
search_scripts() {
    # $1 = grep pattern; $2 = description
    local pattern="$1"
    local desc="$2"

    # Search scripts/ and *.go; then filter out exempt file paths
    local hits
    hits=$(grep -rn --include="*.sh" --include="*.go" \
        -e "${pattern}" \
        "${REPO_ROOT}/scripts" "${REPO_ROOT}"/*.go 2>/dev/null \
        | grep -Fv -f "${EXCLUDE_FILE}" || true)

    if [[ -n "$hits" ]]; then
        fail "${desc}"
        echo "$hits" | sed 's/^/    /'
        return 1
    fi
    pass "${desc}"
    return 0
}

echo "=== Version Consistency Check ==="
echo "Canonical values (from scripts/config/versions.sh):"
echo "  PYTHON_VERSION    = ${PYTHON_VERSION}"
echo "  LLVM_VERSION      = ${LLVM_VERSION}"
echo "  MLC_LLM_REPO      = ${MLC_LLM_REPO}"
echo "  TVM_REPO          = ${TVM_REPO}"
echo "  TVM_REF           = ${TVM_REF}"
echo "  CUDA_ARCH_DEFAULT = ${CUDA_ARCH_DEFAULT}"
echo ""

# =============================================================================
# Check 1: No hardcoded python=X.YY in scripts (other than via variable)
# Match literal  python=3.11  or  python=3.13  but NOT  python="${PYTHON_VERSION}"
# =============================================================================
search_scripts \
    'python=3\.[0-9]\+[^$"{]' \
    "No hardcoded python=X.Y in scripts (use PYTHON_PACKAGE_SPEC / PYTHON_ABI_SPEC)"

# =============================================================================
# Check 2: No hardcoded MLC-LLM GitHub URL (outside allowed files)
# Match literal https://github.com/mlc-ai/mlc-llm but NOT as part of a variable
# assignment in versions.sh.
# =============================================================================
search_scripts \
    'https://github\.com/mlc-ai/mlc-llm[^$"{]' \
    "No hardcoded mlc-llm GitHub URL in scripts (use MLC_LLM_REPO)"

# =============================================================================
# Check 3: No hardcoded relax.git URL in scripts
# =============================================================================
search_scripts \
    'https://github\.com/mlc-ai/relax\.git[^$"{]' \
    "No hardcoded mlc-ai/relax.git URL in scripts (use TVM_REPO)"

# =============================================================================
# Check 4: No hardcoded CUDA arch 86 as bare string default in scripts
# Match  :-86}  or  :-86"  style defaults, which bypass CUDA_ARCH_DEFAULT.
# Matches  ${5:-86}  or  ${9:-86}  etc.
# =============================================================================
search_scripts \
    ':-86[}"]' \
    "No hardcoded CUDA arch '86' as fallback default (use CUDA_ARCH_DEFAULT)"

# =============================================================================
# Check 5: Go files must not contain duplicate raw URL string literals
# (versions_defaults.go is exempt via ALLOWED_FILES above)
# =============================================================================
search_scripts \
    '"https://github\.com/mlc-ai/mlc-llm"' \
    "No hardcoded mlc-llm URL string literal in Go (use DefaultMlcLLMRepo)"

# =============================================================================
# Check 6: DefaultPythonVersion in versions_defaults.go matches PYTHON_VERSION
# =============================================================================
GO_PYTHON_VERSION=$(grep 'DefaultPythonVersion.*=' "${REPO_ROOT}/versions_defaults.go" | sed 's/.*= *"\(.*\)".*/\1/' || true)
if [[ -z "${GO_PYTHON_VERSION}" ]]; then
    fail "Could not parse DefaultPythonVersion from versions_defaults.go"
elif [[ "${GO_PYTHON_VERSION}" != "${PYTHON_VERSION}" ]]; then
    fail "DefaultPythonVersion in versions_defaults.go (${GO_PYTHON_VERSION}) does not match PYTHON_VERSION in versions.sh (${PYTHON_VERSION})"
else
    pass "DefaultPythonVersion in Go matches PYTHON_VERSION in shell (${PYTHON_VERSION})"
fi

# =============================================================================
# Check 7: No hardcoded llvmdev=NN in scripts
# Match literal llvmdev= followed by digits
# =============================================================================
search_scripts \
    'llvmdev=[0-9]\+' \
    "No hardcoded llvmdev version in scripts (use LLVM_VERSION)"

# =============================================================================
# Check 8: No bare 'pip install' (should be 'python -m pip install')
# Exempt: pip install build  (used in build scripts for the build tool itself,
# which is fine — it installs into the already-activated conda env).
# =============================================================================
echo ""
echo "=== Bare pip install check (informational) ==="
BARE_PIP=$(grep -rn --include="*.sh" \
    -e '^\s*pip install' \
    "${REPO_ROOT}/scripts" 2>/dev/null \
    | grep -Fv -f "${EXCLUDE_FILE}" || true)

if [[ -n "$BARE_PIP" ]]; then
    warn "Bare 'pip install' found (should use 'python -m pip install' where pip may not be on PATH):"
    echo "$BARE_PIP" | sed 's/^/    /'
    echo "  These are warnings, not errors. Verify that the conda env is active when these run."
else
    pass "No bare 'pip install' found in scripts/"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
if [[ $ERRORS -gt 0 ]]; then
    echo -e "${RED}=== FAILED: ${ERRORS} check(s) failed ===${NC}"
    echo "Fix the issues above, then re-run this script."
    exit 1
else
    echo -e "${GREEN}=== PASSED: All version consistency checks passed ===${NC}"
fi
