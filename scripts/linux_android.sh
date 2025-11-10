#!/bin/bash

source "$HOME/.cargo/env"
NDK_VERSION=$(ls -1 "$HOME/Library/Android/sdk/ndk/" | sort -V | tail -1)
export ANDROID_NDK="$HOME/Library/Android/sdk/ndk/$NDK_VERSION"
export TVM_NDK_CC="$ANDROID_NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android24-clang"
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
export MLC_LLM_SOURCE_DIR="$(pwd)/mlc-llm"
export TVM_SOURCE_DIR="$MLC_LLM_SOURCE_DIR/3rdparty/tvm"

cd mlc-llm/android/MLCChat
MLC_JIT_POLICY=REDO mlc_llm package