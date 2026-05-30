#!/usr/bin/env bash

# Shared helper to select the correct MLC wheel based on ABI and MLC_LLM_REF.
# It expects the following variables to be set by the caller:
# - WHEELS_DIR
# - PYTHON_CP_TAG
# - MLC_LLM_REF (can be empty)
# - MLC_WHEEL (optional, for explicit override)

find_mlc_wheel() {
    MLC_WHEEL_PATH=""
    
    # 1. Explicit override
    if [[ -n "${MLC_WHEEL:-}" ]]; then
        if type log_info &>/dev/null; then
            log_info "Using explicit MLC_WHEEL argument: ${MLC_WHEEL}"
        else
            echo "Using explicit MLC_WHEEL argument: ${MLC_WHEEL}"
        fi
        
        if [[ ! -f "${MLC_WHEEL}" ]]; then
            if type log_error &>/dev/null; then
                log_error "Explicit MLC_WHEEL path does not exist: ${MLC_WHEEL}"
            else
                echo "Error: Explicit MLC_WHEEL path does not exist: ${MLC_WHEEL}" >&2
            fi
            exit 1
        fi
        MLC_WHEEL_PATH="${MLC_WHEEL}"
        return 0
    fi

    # 2. Auto-discovery
    local SHORT_SHA="${MLC_LLM_REF:0:8}"
    local MLC_WHEELS=($(ls "${WHEELS_DIR}"/mlc_llm-*-${PYTHON_CP_TAG}-${PYTHON_CP_TAG}-*.whl 2>/dev/null || true))
    local SHA_MATCHES=()
    if [[ -n "${SHORT_SHA}" ]]; then
        SHA_MATCHES=($(printf '%s\n' "${MLC_WHEELS[@]}" | grep "g${SHORT_SHA}" || true))
    fi

    if [[ -n "${SHORT_SHA}" ]] && [[ ${#SHA_MATCHES[@]} -eq 1 ]]; then
        MLC_WHEEL_PATH="${SHA_MATCHES[0]}"
        if type log_info &>/dev/null; then
            log_info "Selected MLC wheel matching MLC_LLM_REF short SHA (g${SHORT_SHA}): $(basename "${MLC_WHEEL_PATH}")"
        else
            echo "Selected MLC wheel matching MLC_LLM_REF short SHA (g${SHORT_SHA}): $(basename "${MLC_WHEEL_PATH}")"
        fi
    elif [[ -n "${SHORT_SHA}" ]] && [[ ${#SHA_MATCHES[@]} -gt 1 ]]; then
        if type log_error &>/dev/null; then
            log_error "Multiple MLC wheels match short SHA g${SHORT_SHA} in ${WHEELS_DIR}. Remove stale wheels and retry."
        else
            echo "Error: Multiple MLC wheels match short SHA g${SHORT_SHA} in ${WHEELS_DIR}. Remove stale wheels and retry." >&2
        fi
        printf '  %s\n' "${SHA_MATCHES[@]}" >&2
        exit 1
    elif [[ ${#MLC_WHEELS[@]} -eq 1 ]]; then
        MLC_WHEEL_PATH="${MLC_WHEELS[0]}"
        if [[ -n "${SHORT_SHA}" ]]; then
            if type log_warning &>/dev/null; then
                log_warning "No MLC wheel matches MLC_LLM_REF short SHA (g${SHORT_SHA}). Using only available ABI-matching wheel: $(basename "${MLC_WHEEL_PATH}")"
            else
                echo "Warning: No MLC wheel matches MLC_LLM_REF short SHA (g${SHORT_SHA}). Using only available ABI-matching wheel: $(basename "${MLC_WHEEL_PATH}")" >&2
            fi
        fi
    elif [[ ${#MLC_WHEELS[@]} -gt 1 ]]; then
        if type log_error &>/dev/null; then
            if [[ -n "${SHORT_SHA}" ]]; then
                log_error "Multiple ABI-matching MLC wheels exist and none matches MLC_LLM_REF short SHA (g${SHORT_SHA}). Delete stale wheels or pass an explicit wheel path."
            else
                log_error "Multiple ABI-matching MLC wheels exist. Delete stale wheels or pass an explicit wheel path."
            fi
        else
            if [[ -n "${SHORT_SHA}" ]]; then
                echo "Error: Multiple ABI-matching MLC wheels exist and none matches MLC_LLM_REF short SHA (g${SHORT_SHA}). Delete stale wheels or pass an explicit wheel path." >&2
            else
                echo "Error: Multiple ABI-matching MLC wheels exist. Delete stale wheels or pass an explicit wheel path." >&2
            fi
        fi
        printf '  %s\n' "${MLC_WHEELS[@]}" >&2
        exit 1
    fi
}
