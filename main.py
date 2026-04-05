#!/usr/bin/env python3
"""
MLC-LLM Example Script

This script demonstrates how to use MLC-LLM for model inference.
It can be used as a quick test after building and installing the wheels.

Usage:
    python main.py [--model MODEL_PATH] [--device DEVICE] [--prompt PROMPT]

Examples:
    python main.py
    python main.py --model ./models/Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC
    python main.py --device cuda --prompt "What is machine learning?"
"""

import argparse
import os
import sys
from pathlib import Path

# Set TVM library path to find compiled TVM libraries
repo_root = Path(__file__).parent
tvm_build_path = repo_root / "tvm" / "build"
if tvm_build_path.exists():
    os.environ["TVM_LIBRARY_PATH"] = str(tvm_build_path)

try:
    from mlc_llm import MLCEngine
except ImportError as e:
    print(f"Error: Failed to import mlc_llm: {e}")
    print("\nPlease ensure MLC-LLM is installed:")
    print("  1. Run './mlc-cli build' to build from source")
    print("  2. Or run './mlc-cli install-wheels' to install pre-built wheels")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run MLC-LLM model inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC",
        help="Path to the MLC model directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (metal, cuda, vulkan, rocm, cpu). Auto-detected if not specified."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the meaning of life?",
        help="Prompt to send to the model"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    return parser.parse_args()


def detect_device():
    """Detect the best available device."""
    import platform
    
    system = platform.system()
    
    if system == "Darwin":
        # macOS - prefer Metal
        return "metal"
    elif system == "Linux":
        # Linux - check for CUDA
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            if result.returncode == 0:
                return "cuda"
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
        return "cpu"
    else:
        return "cpu"


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at: {model_path}")
        print("\nAvailable models in ./models/:")
        models_dir = Path("./models")
        if models_dir.exists():
            for model in models_dir.iterdir():
                if model.is_dir():
                    print(f"  - {model.name}")
        print("\nDownload a model with:")
        print("  ./mlc-cli run --model-url <huggingface-url>")
        sys.exit(1)
    
    # Detect device if not specified
    device = args.device or detect_device()
    
    print(f"Loading model: {model_path}")
    print(f"Device: {device}")
    print(f"Prompt: {args.prompt}")
    print("-" * 50)
    
    # Create engine
    try:
        engine = MLCEngine(str(model_path), device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Run chat completion in OpenAI API format
    print("\nResponse:")
    print("-" * 50)
    
    try:
        for response in engine.chat.completions.create(
            messages=[{"role": "user", "content": args.prompt}],
            model=str(model_path),
            stream=True,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ):
            for choice in response.choices:
                if choice.delta.content:
                    print(choice.delta.content, end="", flush=True)
    except KeyboardInterrupt:
        print("\n\n[Generation interrupted by user]")
    finally:
        print("\n")
        engine.terminate()
        print("-" * 50)
        print("Session ended.")


if __name__ == "__main__":
    main()
