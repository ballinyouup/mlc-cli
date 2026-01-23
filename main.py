import os
from pathlib import Path

# Set TVM library path to find compiled TVM libraries
repo_root = Path(__file__).parent
tvm_build_path = repo_root / "tvm" / "build"
if tvm_build_path.exists():
    os.environ["TVM_LIBRARY_PATH"] = str(tvm_build_path)

from mlc_llm import MLCEngine

# Create engine
model = "./models/Qwen3-1.7B-q4f16_1-MLC"
engine = MLCEngine(model, device="metal")

# Run chat completion in OpenAI API.
for response in engine.chat.completions.create(
        messages=[{"role": "user", "content": "What is the meaning of life?"}],
        model=model,
        stream=True,
):
    for choice in response.choices:
        print(choice.delta.content, end="", flush=True)
print("\n")

engine.terminate()