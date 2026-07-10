#!/usr/bin/env bash

prepend_path_if_dir() {
    local var_name="$1"
    local new_path="$2"

    if [[ -d "${new_path}" ]]; then
        local current_value="${!var_name:-}"
        if [[ -n "${current_value}" ]]; then
            export "${var_name}=${new_path}:${current_value}"
        else
            export "${var_name}=${new_path}"
        fi
    fi
}

setup_tvm_runtime_env() {
    local repo_root="$1"

    export MLC_CLI_PATH="${MLC_CLI_PATH:-${repo_root}}"
    export TVM_SOURCE="${TVM_SOURCE:-bundled}"

    case "${TVM_SOURCE}" in
        bundled)
            export TVM_HOME="${MLC_CLI_PATH}/mlc-llm/3rdparty/tvm"
            ;;
        relax|standalone)
            export TVM_HOME="${MLC_CLI_PATH}/tvm"
            ;;
        custom)
            if [[ -z "${TVM_HOME:-}" ]]; then
                echo "[ERROR] TVM_SOURCE=custom requires TVM_HOME" >&2
                return 1
            fi
            ;;
        *)
            echo "[ERROR] Unsupported TVM_SOURCE=${TVM_SOURCE}" >&2
            return 1
            ;;
    esac

    prepend_path_if_dir PYTHONPATH "${TVM_HOME}/python"

    prepend_path_if_dir LD_LIBRARY_PATH "${TVM_HOME}/build/lib"
    prepend_path_if_dir LD_LIBRARY_PATH "${TVM_HOME}/build"
    prepend_path_if_dir LD_LIBRARY_PATH "${MLC_CLI_PATH}/mlc-llm/build/lib"
    prepend_path_if_dir LD_LIBRARY_PATH "${MLC_CLI_PATH}/mlc-llm/build"

    if [[ ! -d "${TVM_HOME}/include" ]]; then
        echo "[WARN] TVM include directory not found: ${TVM_HOME}/include" >&2
    fi
}
