.. _introduction-to-mlc-llm:

Introduction to MLC LLM
=======================

.. contents:: Table of Contents
:local:
:depth: 2

MLC LLM is a machine learning compiler and high-performance deployment
engine for large language models.  The mission of this project is to enable everyone to develop,
optimize, and deploy AI models natively on everyone's platforms. 

This page is a quick tutorial to introduce how to try out MLC LLM, and the steps to
deploy your own models with MLC LLM.

Installation
------------

:ref:`MLC LLM <install-mlc-packages>` is available via pip.
It is always recommended to install it in an isolated conda virtual environment.

To verify the installation, activate your virtual environment, run

.. code:: bash

python -c "import mlc_llm; print(mlc_llm.__path__)"

You are expected to see the installation path of MLC LLM Python package.


Chat CLI
--------

As the first example, we try out the chat CLI in MLC LLM with 4-bit quantized 8B Llama-3 model.
You can run MLC chat through a one-liner command:

.. code:: bash

    mlc_llm chat HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC

It may take 1-2 minutes for the first time running this command.
After waiting, this command launch a chat interface where you can enter your prompt and chat with the model.

.. code::

You can use the following special commands:
/help               print the special commands
/exit               quit the cli
/stats              print out the latest stats (token/sec)
/reset              restart a fresh chat
/set [overrides]    override settings in the generation config. For example,
`/set temperature=0.5;max_gen_len=100;stop=end,stop`
Note: Separate stop words in the `stop` option with commas (,).
Multi-line input: Use escape+enter to start a new line.

user: What's the meaning of life
assistant:
What a profound and intriguing question! While there's no one definitive answer, I'd be happy to help you explore some perspectives on the meaning of life.

The concept of the meaning of life has been debated and...


The figure below shows what run under the hood of this chat CLI command.
For the first time running the command, there are three major phases.

- **Phase 1. Pre-quantized weight download.** This phase automatically downloads pre-quantized Llama-3 model from `Hugging Face <https://huggingface.co/mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC>`_ and saves it to your local cache directory.
- **Phase 2. Model compilation.** This phase automatically optimizes the Llama-3 model to accelerate model inference on GPU with techniques of machine learning compilation in `Apache TVM <https://llm.mlc.ai/docs/install/tvm.html>`_ compiler, and generate the binary model library that enables the execution language models on your local GPU.
- **Phase 3. Chat runtime.** This phase consumes the model library built in phase 2 and the model weights downloaded in phase 1, launches a platform-native chat runtime to drive the execution of Llama-3 model.

We cache the pre-quantized model weights and compiled model library locally.
Therefore, phase 1 and 2 will only execute **once** over multiple runs.

.. figure:: /_static/img/project-workflow.svg
:width: 700
:align: center
:alt: Project Workflow

Workflow in MLC LLM

.. note::

If you want to enable tensor parallelism to run LLMs on multiple GPUs,
please specify argument ``--overrides "tensor_parallel_shards=$NGPU"``.
For example,

.. code:: shell

    mlc_llm chat HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC --overrides "tensor_parallel_shards=2"

.. _introduction-to-mlc-llm-python-api:

Python API
----------

In the second example, we run the Llama-3 model with the chat completion Python API of MLC LLM.
You can save the code below into a Python file and run it.

.. code:: python

from mlc_llm import MLCEngine

# Create engine
model = "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC"
engine = MLCEngine(model)

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

.. figure:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/python-engine-api.jpg
:width: 500
:align: center

MLC LLM Python API

This code example first creates an :class:`mlc_llm.MLCEngine` instance with the 4-bit quantized Llama-3 model.
**We design the Python API** :class:`mlc_llm.MLCEngine` **to align with OpenAI API**,
which means you can use :class:`mlc_llm.MLCEngine` in the same way of using
`OpenAI's Python package <https://github.com/openai/openai-python?tab=readme-ov-file#usage>`_
for both synchronous and asynchronous generation.

In this code example, we use the synchronous chat completion interface and iterate over
all the stream responses.
If you want to run without streaming, you can run

.. code:: python

response = engine.chat.completions.create(
messages=[{"role": "user", "content": "What is the meaning of life?"}],
model=model,
stream=False,
)
print(response)

You can also try different arguments supported in `OpenAI chat completion API <https://platform.openai.com/docs/api-reference/chat/create>`_.
If you would like to do concurrent asynchronous generation, you can use :class:`mlc_llm.AsyncMLCEngine` instead.

.. note::

If you want to enable tensor parallelism to run LLMs on multiple GPUs,
please specify argument ``model_config_overrides`` in MLCEngine constructor.
For example,

.. code:: python

    from mlc_llm import MLCEngine
    from mlc_llm.serve.config import EngineConfig

    model = "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC"
    engine = MLCEngine(
        model,
        engine_config=EngineConfig(tensor_parallel_shards=2),
    )


REST Server
-----------

For the third example, we launch a REST server to serve the 4-bit quantized Llama-3 model
for OpenAI chat completion requests. The server can be launched in command line with

.. code:: bash

mlc_llm serve HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC

The server is hooked at ``http://127.0.0.1:8000`` by default, and you can use ``--host`` and ``--port``
to set a different host and port.
When the server is ready (showing ``INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)``),
we can open a new shell and send a cURL request via the following command:

.. code:: bash

curl -X POST \
-H "Content-Type: application/json" \
-d '{
"model": "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC",
"messages": [
{"role": "user", "content": "Hello! Our project is MLC LLM. What is the name of our project?"}
]
}' \
http://127.0.0.1:8000/v1/chat/completions

The server will process this request and send back the response.
Similar to :ref:`introduction-to-mlc-llm-python-api`, you can pass argument ``"stream": true``
to request for stream responses.

.. note::

If you want to enable tensor parallelism to run LLMs on multiple GPUs,
please specify argument ``--overrides "tensor_parallel_shards=$NGPU"``.
For example,

.. code:: shell

    mlc_llm serve HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC --overrides "tensor_parallel_shards=2"

.. _introduction-deploy-your-own-model:

Deploy Your Own Model
---------------------

So far we have been using pre-converted models weights from Hugging Face.
This section introduces the core workflow regarding how you can *run your own models with MLC LLM*.

We use the `Phi-2 <https://huggingface.co/microsoft/phi-2>`_ as the example model.
Assuming the Phi-2 model is downloaded and placed under ``models/phi-2``,
there are two major steps to prepare your own models.

- **Step 1. Generate MLC config.** The first step is to generate the configuration file of MLC LLM.

  .. code:: bash

  export LOCAL_MODEL_PATH=models/phi-2   # The path where the model resides locally.
  export MLC_MODEL_PATH=dist/phi-2-MLC/  # The path where to place the model processed by MLC.
  export QUANTIZATION=q0f16              # The choice of quantization.
  export CONV_TEMPLATE=phi-2             # The choice of conversation template.
  mlc_llm gen_config $LOCAL_MODEL_PATH \
  --quantization $QUANTIZATION \
  --conv-template $CONV_TEMPLATE \
  -o $MLC_MODEL_PATH

  The config generation command takes in the local model path, the target path of MLC output,
  the conversation template name in MLC and the quantization name in MLC.
  Here the quantization ``q0f16`` means float16 without quantization,
  and the conversation template ``phi-2`` is the Phi-2 model's template in MLC.

  If you want to enable tensor parallelism on multiple GPUs, add argument
  ``--tensor-parallel-shards $NGPU`` to the config generation command.

    - `The full list of supported quantization in MLC <https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/quantization/quantization.py#L29>`_. You can try different quantization methods with MLC LLM. Typical quantization methods are ``q4f16_1`` for 4-bit group quantization, ``q4f16_ft`` for 4-bit FasterTransformer format quantization.
    - `The full list of conversation template in MLC <https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/interface/gen_config.py#L276>`_.

- **Step 2. Convert model weights.** In this step, we convert the model weights to MLC format.

  .. code:: bash

  mlc_llm convert_weight $LOCAL_MODEL_PATH \
  --quantization $QUANTIZATION \
  -o $MLC_MODEL_PATH

  This step consumes the raw model weights and converts them to for MLC format.
  The converted weights will be stored under ``$MLC_MODEL_PATH``,
  which is the same directory where the config file generated in Step 1 resides.

Now, we can try to run your own model with chat CLI:

.. code:: bash

mlc_llm chat $MLC_MODEL_PATH

For the first run, model compilation will be triggered automatically to optimize the
model for GPU accelerate and generate the binary model library.
The chat interface will be displayed after model JIT compilation finishes.
You can also use this model in Python API, MLC serve and other use scenarios.

(Optional) Compile Model Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In previous sections, model libraries are compiled when the :class:`mlc_llm.MLCEngine` launches,
which is what we call "JIT (Just-in-Time) model compilation".
In some cases, it is beneficial to explicitly compile the model libraries.
We can deploy LLMs with reduced dependencies by shipping the library for deployment without going through compilation.
It will also enable advanced options such as cross-compiling the libraries for web and mobile deployments.


Below is an example command of compiling model libraries in MLC LLM:

.. code:: bash

  export MODEL_LIB=$MLC_MODEL_PATH/lib.so  # ".dylib" for Intel Macs.
                                            # ".dll" for Windows.
                                            # ".wasm" for web.
                                            # ".tar" for iPhone/Android.
  mlc_llm compile $MLC_MODEL_PATH -o $MODEL_LIB

At runtime, we need to specify this model library path to use it. For example,

.. code:: bash

  # For chat CLI
  mlc_llm chat $MLC_MODEL_PATH --model-lib $MODEL_LIB
  # For REST server
  mlc_llm serve $MLC_MODEL_PATH --model-lib $MODEL_LIB

.. code:: python

  from mlc_llm import MLCEngine

  # For Python API
  model = "models/phi-2"
  model_lib = "models/phi-2/lib.so"
  engine = MLCEngine(model, model_lib=model_lib)

:ref:`compile-model-libraries` introduces the model compilation command in detail,
where you can find instructions and example commands to compile model to different
hardware backends, such as WebGPU, iOS and Android.

Universal Deployment
--------------------

MLC LLM is a high-performance universal deployment solution for large language models,
to enable native deployment of any large language models with native APIs with compiler acceleration
So far, we have gone through several examples running on a local GPU environment.
The project supports multiple kinds of GPU backends.

You can use `--device` option in compilation and runtime to pick a specific GPU backend.
For example, if you have an NVIDIA or AMD GPU, you can try to use the option below
to run chat through the vulkan backend. Vulkan-based LLM applications run in less typical
environments (e.g. SteamDeck).

.. code:: bash

    mlc_llm chat HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC --device vulkan

The same core LLM runtime engine powers all the backends, enabling the same model to be deployed across backends as
long as they fit within the memory and computing budget of the corresponding hardware backend.
We also leverage machine learning compilation to build backend-specialized optimizations to
get out the best performance on the targetted backend when possible, and reuse key insights and optimizations
across backends we support.

Please checkout the what to do next sections below to find out more about different deployment scenarios,
such as WebGPU-based browser deployment, mobile and other settings.

Summary and What to Do Next
---------------------------

To briefly summarize this page,

- We went through three examples (chat CLI, Python API, and REST server) of MLC LLM,
- we introduced how to convert model weights for your own models to run with MLC LLM, and (optionally) how to compile your models.
- We also discussed the universal deployment capability of MLC LLM.

Next, please feel free to check out the pages below for quick start examples and more detailed information
on specific platforms

- :ref:`Quick start examples <quick-start>` for Python API, chat CLI, REST server, web browser, iOS and Android.
- Depending on your use case, check out our API documentation and tutorial pages:

  - :ref:`webllm-runtime`
  - :ref:`deploy-rest-api`
  - :ref:`deploy-cli`
  - :ref:`deploy-python-engine`
  - :ref:`deploy-ios`
  - :ref:`deploy-android`
  - :ref:`deploy-ide-integration`

- :ref:`Convert model weight to MLC format <convert-weights-via-MLC>`, if you want to run your own models.
- :ref:`Compile model libraries <compile-model-libraries>`, if you want to deploy to web/iOS/Android or control the model optimizations.
- Report any problem or ask any question: open new issues in our `GitHub repo <https://github.com/mlc-ai/mlc-llm/issues>`_.

.. _quick-start:

Quick Start
===========

Examples
--------

To begin with, try out MLC LLM support for int4-quantized Llama3 8B.
It is recommended to have at least 6GB free VRAM to run it.

.. tabs::

  .. tab:: Python

    **Install MLC LLM**. :ref:`MLC LLM <install-mlc-packages>` is available via pip.
    It is always recommended to install it in an isolated conda virtual environment.

    **Run chat completion in Python.** The following Python script showcases the Python API of MLC LLM:

    .. code:: python

      from mlc_llm import MLCEngine

      # Create engine
      model = "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC"
      engine = MLCEngine(model)

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

    .. Todo: link the colab notebook when ready:

    **Documentation and tutorial.** Python API reference and its tutorials are :ref:`available online <deploy-python-engine>`.

    .. figure:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/python-engine-api.jpg
      :width: 600
      :align: center

      MLC LLM Python API

  .. tab:: REST Server

    **Install MLC LLM**. :ref:`MLC LLM <install-mlc-packages>` is available via pip.
    It is always recommended to install it in an isolated conda virtual environment.

    **Launch a REST server.** Run the following command from command line to launch a REST server at ``http://127.0.0.1:8000``.

    .. code:: shell

      mlc_llm serve HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC

    **Send requests to server.** When the server is ready (showing ``INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)``),
    open a new shell and send a request via the following command:

    .. code:: shell

      curl -X POST \
        -H "Content-Type: application/json" \
        -d '{
              "model": "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC",
              "messages": [
                  {"role": "user", "content": "Hello! Our project is MLC LLM. What is the name of our project?"}
              ]
        }' \
        http://127.0.0.1:8000/v1/chat/completions

    **Documentation and tutorial.** Check out :ref:`deploy-rest-api` for the REST API reference and tutorial.
    Our REST API has complete OpenAI API support.

    .. figure:: https://raw.githubusercontent.com/mlc-ai/web-data/main/images/mlc-llm/tutorials/python-serve-request.jpg
      :width: 600
      :align: center

      Send HTTP request to REST server in MLC LLM

  .. tab:: Command Line

    **Install MLC LLM**. :ref:`MLC LLM <install-mlc-packages>` is available via pip.
    It is always recommended to install it in an isolated conda virtual environment.

    For Windows/Linux users, make sure to have latest :ref:`Vulkan driver <vulkan_driver>` installed.

    **Run in command line**.

    .. code:: bash

      mlc_llm chat HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC


    If you are using windows/linux/steamdeck and would like to use vulkan,
    we recommend installing necessary vulkan loader dependency via conda
    to avoid vulkan not found issues.

    .. code:: bash

      conda install -c conda-forge gcc libvulkan-loader


  .. tab:: Web Browser

    `WebLLM <https://webllm.mlc.ai/#chat-demo>`__. MLC LLM generates performant code for WebGPU and WebAssembly,
    so that LLMs can be run locally in a web browser without server resources.

    **Download pre-quantized weights**. This step is self-contained in WebLLM.

    **Download pre-compiled model library**. WebLLM automatically downloads WebGPU code to execute.

    **Check browser compatibility**. The latest Google Chrome provides WebGPU runtime and `WebGPU Report <https://webgpureport.org/>`__ as a useful tool to verify WebGPU capabilities of your browser.

    .. figure:: https://blog.mlc.ai/img/redpajama/web.gif
      :width: 300
      :align: center

      MLC LLM on Web

  .. tab:: iOS

    **Install MLC Chat iOS**. It is available on AppStore:

    .. image:: https://developer.apple.com/assets/elements/badges/download-on-the-app-store.svg
      :width: 135
      :target: https://apps.apple.com/us/app/mlc-chat/id6448482937

    |

    **Note**. The larger model might take more VRAM, try start with smaller models first.

    **Tutorial and source code**. The source code of the iOS app is fully `open source <https://github.com/mlc-ai/mlc-llm/tree/main/ios>`__,
    and a :ref:`tutorial <deploy-ios>` is included in documentation.

    .. figure:: https://blog.mlc.ai/img/redpajama/ios.gif
      :width: 300
      :align: center

      MLC Chat on iOS

  .. tab:: Android

    **Install MLC Chat Android**. A prebuilt is available as an APK:

    .. image:: https://seeklogo.com/images/D/download-android-apk-badge-logo-D074C6882B-seeklogo.com.png
      :width: 135
      :target: https://github.com/mlc-ai/binary-mlc-llm-libs/releases/download/Android-09262024/mlc-chat.apk

    |

    **Note**. The larger model might take more VRAM, try start with smaller models first.
    The demo is tested on

    - Samsung S23 with Snapdragon 8 Gen 2 chip
    - Redmi Note 12 Pro with Snapdragon 685
    - Google Pixel phones

    **Tutorial and source code**. The source code of the android app is fully `open source <https://github.com/mlc-ai/mlc-llm/tree/main/android>`__,
    and a :ref:`tutorial <deploy-android>` is included in documentation.

    .. figure:: https://blog.mlc.ai/img/android/android-recording.gif
      :width: 300
      :align: center

      MLC LLM on Android


What to Do Next
---------------

- Check out :ref:`introduction-to-mlc-llm` for the introduction of a complete workflow in MLC LLM.
- Depending on your use case, check out our API documentation and tutorial pages:

  - :ref:`webllm-runtime`
  - :ref:`deploy-rest-api`
  - :ref:`deploy-cli`
  - :ref:`deploy-python-engine`
  - :ref:`deploy-ios`
  - :ref:`deploy-android`
  - :ref:`deploy-ide-integration`

- :ref:`convert-weights-via-MLC`, if you want to run your own models.
- :ref:`compile-model-libraries`, if you want to deploy to web/iOS/Android or control the model optimizations.
- Report any problem or ask any question: open new issues in our `GitHub repo <https://github.com/mlc-ai/mlc-llm/issues>`_.

.. _compile-model-libraries:

Compile Model Libraries
=======================

To run a model with MLC LLM in any platform, we need:

1. **Model weights** converted to MLC format (e.g. `RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC <https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/tree/main>`__.)
2. **Model library** that comprises the inference logic

This page describes how to compile a model library with MLC LLM. Model compilation optimizes
the model inference for a given platform, allowing users bring their own new model
architecture, use different quantization modes, and customize the overall model
optimization flow.



Notably, in many cases you do not need to explicit call compile.

- If you are using the Python API, you can skip specifying ``model_lib`` and
  the system will JIT compile the library.

- If you are building iOS/android package, checkout :ref:`package-libraries-and-weights`,
  which provides a simpler high-level command that leverages the compile behind the scheme.


This page is still helpful to understand the compilation flow behind the scheme,
or be used to explicit create model libraries.
We compile ``RedPajama-INCITE-Chat-3B-v1`` with ``q4f16_1`` as an example for all platforms.

.. note::
    Before you proceed, make sure you followed :ref:`install-tvm`, a required
    backend to compile models with MLC LLM.

    Please also follow the instructions in :ref:`deploy-cli` / :ref:`deploy-python-engine` to obtain
    the CLI app / Python API that can be used to chat with the compiled model.


.. contents:: Table of Contents
    :depth: 1
    :local:

1. Verify Installation
----------------------

**Step 1. Verify mlc_llm**

We use the python package ``mlc_llm`` to compile models. This can be installed by
following :ref:`install-mlc-packages`, either by building from source, or by
installing the prebuilt package. Verify ``mlc_llm`` installation in command line via:

.. code:: bash

    $ mlc_llm --help
    # You should see help information with this line
    usage: MLC LLM Command Line Interface. [-h] {compile,convert_weight,gen_config}

.. note::
    If it runs into error ``command not found: mlc_llm``, try ``python -m mlc_llm --help``.

**Step 2. Verify TVM**

To compile models, you also need to follow :ref:`install-tvm`.
Here we verify ``tvm`` quickly with command line (for full verification, see :ref:`tvm-validate`):

.. code:: bash

    $ python -c "import tvm; print(tvm.__file__)"
    /some-path/lib/python3.13/site-packages/tvm/__init__.py

1. Clone from HF and convert_weight
-----------------------------------

This replicates :ref:`convert-weights-via-MLC`, see that page for more details.

You can be under the mlc-llm repo, or your own working directory. Note that all platforms
can share the same compiled/quantized weights.

.. code:: shell

    # Create directory
    mkdir -p dist/models && cd dist/models
    # Clone HF weights
    git lfs install
    git clone https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1
    cd ../..
    # Convert weight
    mlc_llm convert_weight ./dist/models/RedPajama-INCITE-Chat-3B-v1/ \
        --quantization q4f16_1 \
        -o dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC

2. Generate mlc-chat-config and compile
---------------------------------------

A model library is specified by:

 - The model architecture (e.g. ``llama-2``, ``gpt-neox``)
 - Quantization (e.g. ``q4f16_1``, ``q0f32``)
 - Metadata (e.g. ``context_window_size``, ``sliding_window_size``, ``prefill-chunk-size``), which affects memory planning
 - Platform (e.g. ``cuda``, ``webgpu``, ``iOS``)

All these knobs are specified in ``mlc-chat-config.json`` generated by ``gen_config``.

.. code:: shell

    # Create output directory for the model library compiled
    mkdir dist/libs

.. tabs::

    .. group-tab:: Linux - CUDA

        .. code:: shell

            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_llm gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ \
                --quantization q4f16_1 --conv-template redpajama_chat \
                -o dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/
            # 2. compile: compile model library with specification in mlc-chat-config.json
            mlc_llm compile ./dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/mlc-chat-config.json \
                --device cuda -o dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-cuda.so


    .. group-tab:: Metal

        For M-chip Mac:

        .. code:: shell

            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_llm gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ \
                --quantization q4f16_1 --conv-template redpajama_chat \
                -o dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/
            # 2. compile: compile model library with specification in mlc-chat-config.json
            mlc_llm compile ./dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/mlc-chat-config.json \
                --device metal -o dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-metal.so

        Cross-Compiling for Intel Mac on M-chip Mac:

        .. code:: shell

            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_llm gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ \
                --quantization q4f16_1 --conv-template redpajama_chat \
                -o dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/
            # 2. compile: compile model library with specification in mlc-chat-config.json
            mlc_llm compile ./dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/mlc-chat-config.json \
                --device metal:x86-64 -o dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-metal_x86_64.dylib

        For Intel Mac:

        .. code:: shell

            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_llm gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ \
                --quantization q4f16_1 --conv-template redpajama_chat \
                -o dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/
            # 2. compile: compile model library with specification in mlc-chat-config.json
            mlc_llm compile ./dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/mlc-chat-config.json \
                --device metal -o dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-metal_x86_64.dylib


    .. group-tab:: Vulkan

        For Linux:

        .. code:: shell

            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_llm gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ \
                --quantization q4f16_1 --conv-template redpajama_chat \
                -o dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/
            # 2. compile: compile model library with specification in mlc-chat-config.json
            mlc_llm compile ./dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/mlc-chat-config.json \
                --device vulkan -o dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-vulkan.so

        For Windows:

        .. code:: shell

            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_llm gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ \
                --quantization q4f16_1 --conv-template redpajama_chat \
                -o dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/
            # 2. compile: compile model library with specification in mlc-chat-config.json
            mlc_llm compile ./dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/mlc-chat-config.json \
                --device vulkan -o dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-vulkan.dll

    .. group-tab:: iOS/iPadOS

        You need a Mac to compile models for it.

        .. code:: shell

            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_llm gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ --quantization q4f16_1 \
                --conv-template redpajama_chat --context-window-size 768 \
                -o dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/
            # 2. compile: compile model library with specification in mlc-chat-config.json
            mlc_llm compile ./dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/mlc-chat-config.json \
                --device iphone -o dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-iphone.tar

        .. note::
            If it runs into error

            .. code:: text

                Compilation error:
                xcrun: error: unable to find utility "metal", not a developer tool or in PATH
                xcrun: error: unable to find utility "metallib", not a developer tool or in PATH

            , please check and make sure you have Command Line Tools for Xcode installed correctly.
            You can use ``xcrun metal`` to validate: when it prints ``metal: error: no input files``, it means the Command Line Tools for Xcode is installed and can be found, and you can proceed with the model compiling.

    .. group-tab:: Android

        .. code:: shell

            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_llm gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ --quantization q4f16_1 \
                --conv-template redpajama_chat --context-window-size 768 \
                -o dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/
            # 2. compile: compile model library with specification in mlc-chat-config.json
            mlc_llm compile ./dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/mlc-chat-config.json \
                --device android -o dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-android.tar

    .. group-tab:: WebGPU

        .. code:: shell

            # 1. gen_config: generate mlc-chat-config.json and process tokenizers
            mlc_llm gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ \
                --quantization q4f16_1 --conv-template redpajama_chat \
                -o dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/
            # 2. compile: compile model library with specification in mlc-chat-config.json
            mlc_llm compile ./dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/mlc-chat-config.json \
                --device webgpu -o dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-webgpu.wasm

        .. note::
            To compile for webgpu, you need to build from source when installing ``mlc_llm``. Besides, you also need to follow :ref:`install-web-build`.
            Otherwise, it would run into error

            .. code:: text

                RuntimeError: Cannot find libraries: wasm_runtime.bc

        .. note::
            For webgpu, when compiling larger models like ``Llama-2-7B``, you may want to add ``--prefill-chunk-size 1024`` or lower ``--context-window-size`` to decrease memory usage.
            Otherwise, you may run into issues like:

            .. code:: text

                TypeError: Failed to execute 'createBuffer' on 'GPUDevice': Failed to read the 'size' property from
                'GPUBufferDescriptor': Value is outside the 'unsigned long long' value range.

.. note::

    For the ``conv-template``, `conversation_template.py <https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/conversation_template.py>`__
    contains a full list of conversation templates that MLC provides. If the model you are adding
    requires a new conversation template, you would need to add your own.
    Follow `this PR <https://github.com/mlc-ai/mlc-llm/pull/2163>`__ as an example.
    However, adding your own template would require you :ref:`build mlc_llm from source <mlcchat_build_from_source>`
    in order for it to be recognized by the runtime.

    For more details, please see :ref:`configure-mlc-chat-json`.

3. Verify output and chat
-------------------------

By executing the compile command above, we generate the model weights, model lib, and a chat config.
We can check the output with the commands below:

.. tabs::

    .. group-tab:: Linux - CUDA

        .. code:: shell

            ~/mlc-llm > ls dist/libs
              RedPajama-INCITE-Chat-3B-v1-q4f16_1-cuda.so      # ===> the model library

            ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC
              mlc-chat-config.json                             # ===> the chat config
              tensor-cache.json                               # ===> the model weight info
              params_shard_0.bin                               # ===> the model weights
              params_shard_1.bin
              ...
              tokenizer.json                                   # ===> the tokenizer files
              tokenizer_config.json

        We can now chat with the model using the command line interface (CLI) app or the Python API.

        .. code:: shell

            python
            >>> from mlc_llm import MLCEngine
            >>> engine = MLCEngine(model="./dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
            ...   model_lib="./dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-cuda.so")
            >>> engine.chat.completions.create(
            ...   messages=[{"role": "user", "content": "hello"}]
            ... )
            ChatCompletionResponse(
              choices=[ChatCompletionResponseChoice(
                message=ChatCompletionMessage(
                  content="Hi! How can I assist you today?", role='assistant'
                )
              )],
              ...
            )

    .. group-tab:: Metal

        .. code:: shell

            ~/mlc-llm > ls dist/libs
              RedPajama-INCITE-Chat-3B-v1-q4f16_1-metal.so     # ===> the model library (will be -metal_x86_64.dylib for Intel Mac)

            ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC
              mlc-chat-config.json                             # ===> the chat config
              tensor-cache.json                               # ===> the model weight info
              params_shard_0.bin                               # ===> the model weights
              params_shard_1.bin
              ...
              tokenizer.json                                   # ===> the tokenizer files
              tokenizer_config.json

        We can now chat with the model using the command line interface (CLI) app or the Python API.

        .. code:: shell

            python
            >>> from mlc_llm import MLCEngine
            >>> engine = MLCEngine(model="./dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
            ...   model_lib="./dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-metal.so")
            >>> engine.chat.completions.create(
            ...   messages=[{"role": "user", "content": "hello"}]
            ... )
            ChatCompletionResponse(
              choices=[ChatCompletionResponseChoice(
                message=ChatCompletionMessage(
                  content="Hi! How can I assist you today?", role='assistant'
                )
              )],
              ...
            )


    .. group-tab:: Vulkan

        .. code:: shell

            ~/mlc-llm > ls dist/libs
              RedPajama-INCITE-Chat-3B-v1-q4f16_1-vulkan.so    # ===> the model library (will be .dll for Windows)

            ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC
              mlc-chat-config.json                             # ===> the chat config
              tensor-cache.json                               # ===> the model weight info
              params_shard_0.bin                               # ===> the model weights
              params_shard_1.bin
              ...
              tokenizer.json                                   # ===> the tokenizer files
              tokenizer_config.json

        We can now chat with the model using the command line interface (CLI) app or the Python API.

        .. code:: shell

            python
            >>> from mlc_llm import MLCEngine
            >>> engine = MLCEngine(model="./dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
            ...   model_lib="./dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-vulkan.so")
            >>> engine.chat.completions.create(
            ...   messages=[{"role": "user", "content": "hello"}]
            ... )
            ChatCompletionResponse(
              choices=[ChatCompletionResponseChoice(
                message=ChatCompletionMessage(
                  content="Hi! How can I assist you today?", role='assistant'
                )
              )],
              ...
            )

    .. group-tab:: iOS/iPadOS

        .. code:: shell

            ~/mlc-llm > ls dist/libs
              RedPajama-INCITE-Chat-3B-v1-q4f16_1-iphone.tar   # ===> the model library

            ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC
              mlc-chat-config.json                             # ===> the chat config
              tensor-cache.json                               # ===> the model weight info
              params_shard_0.bin                               # ===> the model weights
              params_shard_1.bin
              ...
              tokenizer.json                                   # ===> the tokenizer files
              tokenizer_config.json

        The model lib ``dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-iphone.tar``
        will be packaged as a static library into the iOS app. Checkout :ref:`deploy-ios` for more details.

    .. group-tab:: Android

        .. code:: shell

            ~/mlc-llm > ls dist/libs
              RedPajama-INCITE-Chat-3B-v1-q4f16_1-android.tar  # ===> the model library

            ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC
              mlc-chat-config.json                             # ===> the chat config
              tensor-cache.json                               # ===> the model weight info
              params_shard_0.bin                               # ===> the model weights
              params_shard_1.bin
              ...
              tokenizer.json                                   # ===> the tokenizer files
              tokenizer_config.json

        The model lib ``dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-android.tar``
        will be packaged as a static library into the android app. Checkout :ref:`deploy-android` for more details.

    .. group-tab:: WebGPU

        .. code:: shell

            ~/mlc-llm > ls dist/libs
              RedPajama-INCITE-Chat-3B-v1-q4f16_1-webgpu.wasm  # ===> the model library

            ~/mlc-llm > ls dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC
              mlc-chat-config.json                             # ===> the chat config
              tensor-cache.json                               # ===> the model weight info
              params_shard_0.bin                               # ===> the model weights
              params_shard_1.bin
              ...
              tokenizer.json                                   # ===> the tokenizer files
              tokenizer_config.json

        To use this in WebGPU runtime, checkout :ref:`webllm-runtime`.

Compile Commands for More Models
--------------------------------

This section lists compile commands for more models that you can try out. Note that this can be easily
generalized to any model variant, as long as mlc-llm supports the architecture.

.. tabs::

    .. tab:: Model: Llama-2-7B

        Please `request for access <https://huggingface.co/meta-llama>`_ to the Llama-2 weights from Meta first.
        After granted access, first create directory ``dist/models`` and download the model to the directory.
        For example, you can run the following code:

        .. code:: shell

            mkdir -p dist/models && cd dist/models
            git lfs install
            git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
            cd ../..

        Then convert the HF weights into MLC-compatible weights. Note that all platforms
        can share the same compiled/quantized weights.

        .. code:: shell

            mlc_llm convert_weight ./dist/models/Llama-2-7b-chat-hf/ --quantization q4f16_1 -o dist/Llama-2-7b-chat-hf-q4f16_1-MLC

        Afterwards, run the following command to generate mlc config and compile the model.

        .. code:: shell

            # Create output directory for the model library compiled
            mkdir dist/libs

        .. tabs::

            .. tab:: Target: CUDA

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/Llama-2-7b-chat-hf/ --quantization q4f16_1 \
                        --conv-template llama-2 -o dist/Llama-2-7b-chat-hf-q4f16_1-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/Llama-2-7b-chat-hf-q4f16_1-MLC/mlc-chat-config.json \
                        --device cuda -o dist/libs/Llama-2-7b-chat-hf-q4f16_1-cuda.so

            .. tab:: Metal

                For M-chip Mac:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/Llama-2-7b-chat-hf/ --quantization q4f16_1 \
                        --conv-template llama-2 -o dist/Llama-2-7b-chat-hf-q4f16_1-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/Llama-2-7b-chat-hf-q4f16_1-MLC/mlc-chat-config.json \
                        --device metal -o dist/libs/Llama-2-7b-chat-hf-q4f16_1-metal.so

                Cross-Compiling for Intel Mac on M-chip Mac:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/RedPajama-INCITE-Chat-3B-v1/ \
                        --quantization q4f16_1 --conv-template redpajama_chat \
                        -o dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/mlc-chat-config.json \
                        --device metal:x86-64 -o dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-metal_x86_64.dylib

                For Intel Mac:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/Llama-2-7b-chat-hf/ --quantization q4f16_1 \
                        --conv-template llama-2 -o dist/Llama-2-7b-chat-hf-q4f16_1-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/Llama-2-7b-chat-hf-q4f16_1-MLC/mlc-chat-config.json \
                        --device metal -o dist/libs/Llama-2-7b-chat-hf-q4f16_1-metal_x86_64.dylib

            .. tab:: Vulkan

                For Linux:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/Llama-2-7b-chat-hf/ --quantization q4f16_1 \
                        --conv-template llama-2 -o dist/Llama-2-7b-chat-hf-q4f16_1-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/Llama-2-7b-chat-hf-q4f16_1-MLC/mlc-chat-config.json \
                        --device vulkan -o dist/libs/Llama-2-7b-chat-hf-q4f16_1-vulkan.so

                For Windows:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/Llama-2-7b-chat-hf/ --quantization q4f16_1 \
                        --conv-template llama-2 -o dist/Llama-2-7b-chat-hf-q4f16_1-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/Llama-2-7b-chat-hf-q4f16_1-MLC/mlc-chat-config.json \
                        --device vulkan -o dist/libs/Llama-2-7b-chat-hf-q4f16_1-vulkan.dll

            .. tab:: WebGPU

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/Llama-2-7b-chat-hf/ --quantization q4f16_1 \
                        --context-window-size 2048 --conv-template llama-2 -o dist/Llama-2-7b-chat-hf-q4f16_1-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/Llama-2-7b-chat-hf-q4f16_1-MLC/mlc-chat-config.json \
                        --device webgpu -o dist/libs/Llama-2-7b-chat-hf-q4f16_1-webgpu.wasm

                .. note::
                    To compile for webgpu, you need to build from source when installing ``mlc_llm``. Besides, you also need to follow :ref:`install-web-build`.
                    Otherwise, it would run into error

                    .. code:: text

                        RuntimeError: Cannot find libraries: wasm_runtime.bc

            .. tab:: iPhone/iPad

                You need a Mac to compile models for it.

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/Llama-2-7b-chat-hf/ --quantization q4f16_1 \
                        --conv-template llama-2 --context-window-size 768 -o dist/Llama-2-7b-chat-hf-q4f16_1-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/Llama-2-7b-chat-hf-q4f16_1-MLC/mlc-chat-config.json \
                        --device iphone -o dist/libs/Llama-2-7b-chat-hf-q4f16_1-iphone.tar

            .. tab:: Android

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/Llama-2-7b-chat-hf/ --quantization q4f16_1 \
                        --conv-template llama-2 --context-window-size 768 -o dist/Llama-2-7b-chat-hf-q4f16_1-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/Llama-2-7b-chat-hf-q4f16_1-MLC/mlc-chat-config.json \
                        --device android -o dist/libs/Llama-2-7b-chat-hf-q4f16_1-android.tar

    .. tab:: Mistral-7B-Instruct-v0.2

        Note that Mistral uses sliding window attention (SWA). Thus, instead of specifying
        ``context-window-size``, we specify ``sliding-window-size``.

        First create directory ``dist/models`` and download the model to the directory.
        For example, you can run the following code:

        .. code:: shell

            mkdir -p dist/models && cd dist/models
            git lfs install
            git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
            cd ../..

        Then convert the HF weights into MLC-compatible weights. Note that all platforms
        can share the same compiled/quantized weights.

        .. code:: shell

            mlc_llm convert_weight ./dist/models/Mistral-7B-Instruct-v0.2/ --quantization q4f16_1 \
                -o dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC

        Afterwards, run the following command to generate mlc config and compile the model.

        .. code:: shell

            # Create output directory for the model library compiled
            mkdir dist/libs

        .. tabs::

            .. tab:: Target: CUDA

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/Mistral-7B-Instruct-v0.2/ --quantization q4f16_1 \
                        --conv-template mistral_default -o dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/mlc-chat-config.json \
                        --device cuda -o dist/libs/Mistral-7B-Instruct-v0.2-q4f16_1-cuda.so

            .. tab:: Metal

                For M-chip Mac:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/Mistral-7B-Instruct-v0.2/ --quantization q4f16_1 \
                        --conv-template mistral_default -o dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/mlc-chat-config.json \
                        --device metal -o dist/libs/Mistral-7B-Instruct-v0.2-q4f16_1-metal.so


                For Intel Mac:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/Mistral-7B-Instruct-v0.2/ --quantization q4f16_1 \
                        --conv-template mistral_default -o dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/mlc-chat-config.json \
                        --device metal -o dist/libs/Mistral-7B-Instruct-v0.2-q4f16_1-metal_x86_64.dylib

            .. tab:: Vulkan

                For Linux:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/Mistral-7B-Instruct-v0.2/ --quantization q4f16_1 \
                        --conv-template mistral_default -o dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/mlc-chat-config.json \
                        --device vulkan -o dist/libs/Mistral-7B-Instruct-v0.2-q4f16_1-vulkan.so

                For Windows:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/Mistral-7B-Instruct-v0.2/ --quantization q4f16_1 \
                        --conv-template mistral_default -o dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/mlc-chat-config.json \
                        --device vulkan -o dist/libs/Mistral-7B-Instruct-v0.2-q4f16_1-vulkan.dll

            .. tab:: WebGPU

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/Mistral-7B-Instruct-v0.2/ --quantization q4f16_1 \
                        --prefill-chunk-size 1024 --conv-template mistral_default \
                        -o dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/mlc-chat-config.json \
                        --device webgpu -o dist/libs/Mistral-7B-Instruct-v0.2-q4f16_1-webgpu.wasm

                .. note::
                    To compile for webgpu, you need to build from source when installing ``mlc_llm``. Besides, you also need to follow :ref:`install-web-build`.
                    Otherwise, it would run into error

                    .. code:: text

                        RuntimeError: Cannot find libraries: wasm_runtime.bc

                .. note::
                    For webgpu, when compiling larger models like ``Llama-2-7B``, you may want to add ``--prefill-chunk-size 1024`` or lower ``--context-window-size`` to decrease memory usage.
                    Otherwise, you may run into issues like:

                    .. code:: text

                        TypeError: Failed to execute 'createBuffer' on 'GPUDevice': Failed to read the 'size' property from
                        'GPUBufferDescriptor': Value is outside the 'unsigned long long' value range.

            .. tab:: iPhone/iPad

                You need a Mac to compile models for it.

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/Mistral-7B-Instruct-v0.2/ --quantization q4f16_1 \
                        --conv-template mistral_default --sliding-window-size 1024 --prefill-chunk-size 128  \
                        -o dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/mlc-chat-config.json \
                        --device iphone -o dist/libs/Mistral-7B-Instruct-v0.2-q4f16_1-iphone.tar

            .. tab:: Android

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/Mistral-7B-Instruct-v0.2/ --quantization q4f16_1 \
                        --conv-template mistral_default --sliding-window-size 1024 --prefill-chunk-size 128 -o dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/mlc-chat-config.json \
                        --device android -o dist/libs/Mistral-7B-Instruct-v0.2-q4f16_1-android.tar

    .. tab:: Other models

        First create directory ``dist/models`` and download the model to the directory.
        For example, you can run the following code:

        .. code:: shell

            mkdir -p dist/models && cd dist/models
            git lfs install
            git clone https://huggingface.co/DISTRIBUTOR/HF_MODEL
            cd ../..

        Then convert the HF weights into MLC-compatible weights. Note that all platforms
        can share the same compiled/quantized weights.

        .. code:: shell

            mlc_llm convert_weight ./dist/models/HF_MODEL/ --quantization q4f16_1 -o dist/OUTPUT-MLC

        Afterwards, run the following command to generate mlc config and compile the model.

        .. code:: shell

            # Create output directory for the model library compiled
            mkdir dist/libs

        .. tabs::

            .. tab:: Target: CUDA

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/HF_MODEL/ --quantization q4f16_1 --conv-template CONV_TEMPLATE -o dist/OUTPUT-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/OUTPUT-MLC/mlc-chat-config.json --device cuda -o dist/libs/OUTPUT-cuda.so

            .. tab:: Metal

                For M-chip Mac:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/HF_MODEL/ --quantization q4f16_1 --conv-template CONV_TEMPLATE -o dist/OUTPUT-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/OUTPUT-MLC/mlc-chat-config.json --device metal -o dist/libs/OUTPUT-metal.so


                For Intel Mac:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/HF_MODEL/ --quantization q4f16_1 --conv-template CONV_TEMPLATE -o dist/OUTPUT-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/OUTPUT-MLC/mlc-chat-config.json --device metal -o dist/libs/OUTPUT-metal_x86_64.dylib

            .. tab:: Vulkan

                For Linux:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/HF_MODEL/ --quantization q4f16_1 --conv-template CONV_TEMPLATE -o dist/OUTPUT-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/OUTPUT-MLC/mlc-chat-config.json --device vulkan -o dist/libs/OUTPUT-vulkan.so

                For Windows:

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/HF_MODEL/ --quantization q4f16_1 --conv-template CONV_TEMPLATE -o dist/OUTPUT-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/OUTPUT-MLC/mlc-chat-config.json --device vulkan -o dist/libs/OUTPUT-vulkan.dll

            .. tab:: WebGPU

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/HF_MODEL/ --quantization q4f16_1 --conv-template CONV_TEMPLATE -o dist/OUTPUT-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/OUTPUT-MLC/mlc-chat-config.json --device webgpu -o dist/libs/OUTPUT-webgpu.wasm

                .. note::
                    To compile for webgpu, you need to build from source when installing ``mlc_llm``. Besides, you also need to follow :ref:`install-web-build`.
                    Otherwise, it would run into error

                    .. code:: text

                        RuntimeError: Cannot find libraries: wasm_runtime.bc

                .. note::
                    For webgpu, when compiling larger models like ``Llama-2-7B``, you may want to add ``--prefill-chunk-size 1024`` or lower ``--context-window-size`` to decrease memory usage.
                    Otherwise, you may run into issues like:

                    .. code:: text

                        TypeError: Failed to execute 'createBuffer' on 'GPUDevice': Failed to read the 'size' property from
                        'GPUBufferDescriptor': Value is outside the 'unsigned long long' value range.

            .. tab:: iPhone/iPad

                You need a Mac to compile models for it.

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/HF_MODEL/ --quantization q4f16_1 --conv-template CONV_TEMPLATE \
                        --context-window-size 768 -o dist/OUTPUT-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/OUTPUT-MLC/mlc-chat-config.json --device iphone -o dist/libs/OUTPUT-iphone.tar

            .. tab:: Android

                .. code:: shell

                    # 1. gen_config: generate mlc-chat-config.json and process tokenizers
                    mlc_llm gen_config ./dist/models/HF_MODEL/ --quantization q4f16_1 --conv-template CONV_TEMPLATE \
                        --context-window-size 768 -o dist/OUTPUT-MLC/
                    # 2. compile: compile model library with specification in mlc-chat-config.json
                    mlc_llm compile ./dist/OUTPUT-MLC/mlc-chat-config.json --device android -o dist/libs/OUTPUT-android.tar

For each model and each backend, the above only provides the most recommended build command (which is the most optimized).
You can also try with different argument values (e.g., different quantization modes, context window size, etc.),
whose build results affect runtime memory requirement, and it is possible that they may not run as
fast and robustly as the provided one when running the model.

.. note::
    Uing 3-bit quantization usually can be overly aggressive and only works for limited settings.
    If you encounter issues where the compiled model does not perform as expected,
    consider utilizing a higher number of bits for quantization (e.g., 4-bit quantization).

If you are interested in distributing the model besides local execution, please checkout :ref:`distribute-compiled-models`.


.. _compile-command-specification:

Compile Command Specification
-----------------------------

As you have seen in the section above, the model compilation is split into three steps: convert weights, generate
``mlc-chat-config.json``, and compile the model. This section describes the list of options that can be used
during compilation.

1. Convert Weight
^^^^^^^^^^^^^^^^^

Weight conversion command follows the pattern below:

.. code:: text

    mlc_llm convert_weight \
        CONFIG \
        --quantization QUANTIZATION_MODE \
        [--model-type MODEL_TYPE] \
        [--device DEVICE] \
        [--source SOURCE] \
        [--source-format SOURCE_FORMAT] \
        --output OUTPUT

Note that ``CONFIG`` is a positional argument. Arguments wrapped with ``[ ]`` are optional.

--CONFIG                            It can be one of the following:

                                    1. Path to a HuggingFace model directory that contains a ``config.json`` or
                                    2. Path to ``config.json`` in HuggingFace format, or
                                    3. The name of a pre-defined model architecture.

                                    A ``config.json`` file in HuggingFace format defines the model architecture, including the vocabulary
                                    size, the number of layers, the hidden size, number of attention heads, etc.
                                    Example: https://huggingface.co/codellama/CodeLlama-7b-hf/blob/main/config.json.

                                    A HuggingFace directory often contains a ``config.json`` which defines the model architecture,
                                    the non-quantized model weights in PyTorch or SafeTensor format, tokenizer configurations,
                                    as well as an optional ``generation_config.json`` provides additional default configuration for
                                    text generation.
                                    Example: https://huggingface.co/codellama/CodeLlama-7b-hf/tree/main.

                                    For existing pre-defined model architecture, see ``MODEL_PRESETS``
                                    `here <https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/compiler/model/model.py>`_.

--quantization QUANTIZATION_MODE    The quantization mode we use to compile.

                                    See :ref:`quantization_mode` for more information.
                                    Available options are: ``q0f16``, ``q0f32``, ``q3f16_1``, ``q4f16_1``, ``q4f32_1``, and
                                    ``q4f16_awq``.

                                    We encourage you to use 4-bit quantization, as the text generated by 3-bit
                                    quantized models may have bad quality depending on the model.

--model-type MODEL_TYPE             Model architecture such as "llama". If not set, it is inferred from ``config.json``.

--device DEVICE                     The device used to do quantization such as "cuda" or "cuda:0". Will detect from
                                    local available GPUs if not specified.

--source SOURCE                     The path to original model weight, infer from ``config`` if missing.

--source-format SOURCE_FORMAT       The format of source model weight, infer from ``config`` if missing.

--output OUTPUT                     The output directory to save the quantized model weight.
                                    Will create ``params_shard_*.bin`` and ```tensor-cache.json``` in this directory.

2. Generate MLC Chat Config
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to compile a model, we first need to generate the ``mlc-chat-config.json``. This file contains specifications
like ``context-window-size`` and ``sliding-window-size``, among others that can alter the model compiled. We also process
tokenizers in this step.

Config generation command follows the pattern below:

.. code:: text

    mlc_llm gen_config \
        CONFIG \
        --quantization QUANTIZATION_MODE \
        [--model-type MODEL_TYPE] \
        --conv-template CONV_TEMPLATE \
        [--context-window-size CONTEXT_WINDOW_SIZE] \
        [--sliding-window-size SLIDING_WINDOW_SIZE] \
        [--prefill-chunk-size PREFILL_CHUNK_SIZE] \
        [--tensor-parallel-shard TENSOR_PARALLEL_SHARDS] \
        --output OUTPUT

Note that ``CONFIG`` is a positional argument. Arguments wrapped with ``[ ]`` are optional.

--CONFIG                                        It can be one of the following:

                                                1. Path to a HuggingFace model directory that contains a ``config.json`` or
                                                2. Path to ``config.json`` in HuggingFace format, or
                                                3. The name of a pre-defined model architecture.

                                                A ``config.json`` file in HuggingFace format defines the model architecture, including the vocabulary
                                                size, the number of layers, the hidden size, number of attention heads, etc.
                                                Example: https://huggingface.co/codellama/CodeLlama-7b-hf/blob/main/config.json.

                                                A HuggingFace directory often contains a ``config.json`` which defines the model architecture,
                                                the non-quantized model weights in PyTorch or SafeTensor format, tokenizer configurations,
                                                as well as an optional ``generation_config.json`` provides additional default configuration for
                                                text generation.
                                                Example: https://huggingface.co/codellama/CodeLlama-7b-hf/tree/main.

                                                For existing pre-defined model architecture, see ``MODEL_PRESETS``
                                                `here <https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/compiler/model/model.py>`_.

--quantization QUANTIZATION_MODE                The quantization mode we use to compile.

                                                See :ref:`quantization_mode` for more information.
                                                Available options are: ``q0f16``, ``q0f32``, ``q3f16_1``, ``q4f16_1``, ``q4f32_1``, and
                                                ``q4f16_awq``.

                                                We encourage you to use 4-bit quantization, as the text generated by 3-bit
                                                quantized models may have bad quality depending on the model.

--model-type MODEL_TYPE                         Model architecture such as "llama". If not set, it is inferred from ``config.json``.

--conv-template CONV_TEMPLATE                   Conversation template. It depends on how the model is tuned. Use "LM" for vanilla base model
                                                For existing pre-defined templates, see ``CONV_TEMPLATES``
                                                `here <https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/model/model.py>`_.

--context-window-size CONTEXT_WINDOW_SIZE       Option to provide the maximum sequence length supported by the model.
                                                This is usually explicitly shown as context length or context window in the model card.
                                                If this option is not set explicitly, by default,
                                                it will be determined by ``context_window_size`` or ``max_position_embeddings`` in ``config.json``,
                                                and the latter is usually inaccurate for some models.

--sliding-window-size SLIDING_WINDOW            (Experimental) The sliding window size in sliding window attention (SWA).
                                                This optional field overrides the ``sliding_window`` in ``config.json`` for
                                                those models that use SWA. Currently only useful when compiling mistral-based models.
                                                This flag subjects to future refactoring.

--prefill-chunk-size PREFILL_CHUNK_SIZE         (Experimental) The chunk size during prefilling. By default,
                                                the chunk size is the same as ``context_window_size`` or ``sliding_window_size``.
                                                This flag subjects to future refactoring.

--tensor-parallel-shard TENSOR_PARALLEL_SHARDS  Number of shards to split the model into in tensor parallelism multi-gpu inference.

--output OUTPUT                                 The output directory for generated configurations, including `mlc-chat-config.json` and tokenizer configuration.

3. Compile Model Library
^^^^^^^^^^^^^^^^^^^^^^^^

After generating ``mlc-chat-config.json``, we can compile the model into a model library (files ending in ``.so``, ``.tar``, etc. that contains
the inference logic of a model).

Model compilation command follows the pattern below:

.. code:: text

    mlc_llm compile \
        MODEL \
        [--quantization QUANTIZATION_MODE] \
        [--model-type MODEL_TYPE] \
        [--device DEVICE] \
        [--host HOST] \
        [--opt OPT] \
        [--system-lib-prefix SYSTEM_LIB_PREFIX] \
        --output OUTPUT \
        [--overrides OVERRIDES]

Note that ``MODEL`` is a positional argument. Arguments wrapped with ``[ ]`` are optional.

--MODEL                                     A path to ``mlc-chat-config.json``, or an MLC model directory that contains ``mlc-chat-config.json``.

--quantization QUANTIZATION_MODE            The quantization mode we use to compile. If unprovided, will infer from ``MODEL``.

                                            See :ref:`quantization_mode` for more information.
                                            Available options are: ``q0f16``, ``q0f32``, ``q3f16_1``, ``q4f16_1``, ``q4f32_1``, and
                                            ``q4f16_awq``.

                                            We encourage you to use 4-bit quantization, as the text generated by 3-bit
                                            quantized models may have bad quality depending on the model.

--model-type MODEL_TYPE                     Model architecture such as "llama". If not set, it is inferred from ``mlc-chat-config.json``.

--device DEVICE                             The GPU device to compile the model to. If not set, it is inferred from GPUs available locally.

--host HOST                                 The host LLVM triple to compile the model to. If not set, it is inferred from the local CPU and OS.
                                            Examples of the LLVM triple:

                                            1) iPhones: arm64-apple-ios;
                                            2) ARM64 Android phones: aarch64-linux-android;
                                            3) WebAssembly: wasm32-unknown-unknown-wasm;
                                            4) Windows: x86_64-pc-windows-msvc;
                                            5) ARM macOS: arm64-apple-darwin.

--opt OPT                                   Optimization flags. MLC LLM maintains a predefined set of optimization flags,
                                            denoted as ``O0``, ``O1``, ``O2``, ``O3``, where ``O0`` means no optimization, ``O2``
                                            means majority of them, and ``O3`` represents extreme optimization that could
                                            potentially break the system.

                                            Meanwhile, optimization flags could be explicitly specified via details knobs, e.g.
                                            ``--opt="cutlass_attn=1;cutlass_norm=0;cublas_gemm=0;cudagraph=0"``.

--system-lib-prefix SYSTEM_LIB_PREFIX       Adding a prefix to all symbols exported. Similar to ``objcopy --prefix-symbols``.
                                            This is useful when compiling multiple models into a single library to avoid symbol
                                            conflicts. Different from objcopy, this takes no effect for shared library.


--output OUTPUT                             The path to the output file. The suffix determines if the output file is a shared library or
                                            objects. Available suffixes:

                                            1) Linux: .so (shared), .tar (objects);
                                            2) macOS: .dylib (shared), .tar (objects);
                                            3) Windows: .dll (shared), .tar (objects);
                                            4) Android, iOS: .tar (objects);
                                            5) Web: .wasm (web assembly).

--overrides OVERRIDES                       Model configuration override. Configurations to override ``mlc-chat-config.json``. Supports
                                            ``context_window_size``, ``prefill_chunk_size``, ``sliding_window``, ``max_batch_size`` and
                                            ``tensor_parallel_shards``. Meanwhile, model config could be explicitly specified via details
                                            knobs, e.g. ``--overrides "context_window_size=1024;prefill_chunk_size=128"``.

Configure Quantization
======================

Quantization Algorithm
----------------------

The default quantization algorithm used in MLC-LLM is grouping quantization method discussed in the papers `The case for 4-bit precision: k-bit Inference Scaling Laws <https://arxiv.org/abs/2212.09720>`__ and `LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models <https://arxiv.org/abs/2206.09557>`__.

.. _quantization_mode:

Quantization Mode
-----------------

In MLC-LLM we use a short code that indicates the quantization mode to use. MLC-LLM supports both
weight-only quantization and weight-activation quantization.

For the weight-only quantization, he format of the code is ``qAfB(_id)``, where ``A`` represents the number
of bits for storing weights and ``B`` represents the number of bits for storing activations.
The ``_id`` is an integer identifier to distinguish different quantization algorithms (e.g. symmetric, non-symmetric, AWQ, etc).

Currently, available options are: ``q0f16``, ``q0f32``, ``q3f16_1``, ``q4f16_1``, ``q4f32_1``, and ``q4f16_awq`` (not stable).

For the weight-activation quantization, currently MLC-LLM supports FP8 quantization on CUDA.
The available options are: ``e4m3_e4m3_f16`` and ``e5m2_e5m2_f16``. In these modes, both weights and activations are quantized to FP8 format.
The output of each layer is in higher precision (FP16) and then requantized to FP8.

.. _calibration:

Calibration
-----------

For ``e4m3_e4m3_f16`` quantization, we need to calibrate the quantization parameters for the activations.
The calibration process is done by running the following command:

1. Compile the calibration model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use the same compilation workflow to compile the model in calibration mode.
The only difference is that we need to specify the quantization mode as ``e4m3_e4m3_f16_calibrate``.

.. code-block:: bash

    mlc_llm gen_config \
        <model-path> \
        --quantization e4m3_e4m3_f16_max_calibrate \
        --output <output-path>

    mlc_llm convert_weights \
        <model-path> \
        --quantization e4m3_e4m3_f16_max_calibrate \
        --output <output-path>

    mlc_llm compile \
        <config-path> \
        --output <output-path>

2. Run the calibration model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will run the calibration model on the dataset such as ShareGPT to collect the statistics of the
activations. The calibration model will updates the quantization parameters in the weights file
in-place. We turn off the cuda graph as it is not yet supported in the calibration process.

.. code-block:: bash

   mlc_llm calibrate \
       <model-path> \
       --model-lib <model-lib-path> \
       --dataset <dataset-path> \
       --num-calibration-samples <num-samples> \
       --opt "cudagraph=0"
       --output <output-path>

3. Compile the quantized model for inference.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After the calibration process, we can compile the model for inference. In this step, we only need
to generate the configuration file using the desired quantization format and compile the model.
Weights are already quantized and calibrated in the previous steps and do not need to be converted again.

.. code-block:: bash

    mlc_llm gen_config \
        <model-path> \
        --quantization e4m3_e4m3_f16 \
        --output <output-path>
    mlc_llm compile \
        <config-path> \
        --output <output-path>

.. _convert-weights-via-MLC:

Convert Model Weights
=====================

To run a model with MLC LLM,
we need to convert model weights into MLC format (e.g. `RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC <https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/tree/main>`_.)
This page walks us through the process of adding a model variant with ``mlc_llm convert_weight``, which
takes a huggingface model as input and converts/quantizes into MLC-compatible weights.

Specifically, we add RedPjama-INCITE-**Instruct**-3B-v1, while MLC already
provides a model library for RedPjama-INCITE-**Chat**-3B-v1, which we can reuse.

This can be extended to, e.g.:

- Add ``OpenHermes-Mistral`` when MLC already supports Mistral
- Add ``Llama-2-uncensored`` when MLC already supports Llama-2

.. note::
    Before you proceed, make sure you followed :ref:`install-tvm`, a required
    backend to compile models with MLC LLM.

    Please also follow the instructions in :ref:`deploy-cli` / :ref:`deploy-python-engine` to obtain
    the CLI app / Python API that can be used to chat with the compiled model.


.. contents:: Table of Contents
    :depth: 1
    :local:

.. _verify_installation_for_compile:

1. Verify installation
----------------------

**Step 1. Verify mlc_llm**

We use the python package ``mlc_llm`` to compile models. This can be installed by
following :ref:`install-mlc-packages`, either by building from source, or by
installing the prebuilt package. Verify ``mlc_llm`` installation in command line via:

.. code:: bash

    $ mlc_llm --help
    # You should see help information with this line
    usage: MLC LLM Command Line Interface. [-h] {compile,convert_weight,gen_config}

.. note::
    If it runs into error ``command not found: mlc_llm``, try ``python -m mlc_llm --help``.

**Step 2. Verify TVM**

To compile models, you also need to follow :ref:`install-tvm`.
Here we verify ``tvm`` quickly with command line (for full verification, see :ref:`tvm-validate`):

.. code:: bash

    $ python -c "import tvm; print(tvm.__file__)"
    /some-path/lib/python3.13/site-packages/tvm/__init__.py


1. Clone from HF and convert_weight
-----------------------------------

You can be under the mlc-llm repo, or your own working directory. Note that all platforms
can share the same compiled/quantized weights. See :ref:`compile-command-specification`
for specification of ``convert_weight``.

.. code:: shell

    # Create directory
    mkdir -p dist/models && cd dist/models
    # Clone HF weights
    git lfs install
    git clone https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1
    cd ../..
    # Convert weight
    mlc_llm convert_weight ./dist/models/RedPajama-INCITE-Instruct-3B-v1/ \
        --quantization q4f16_1 \
        -o dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC

.. _generate_mlc_chat_config:

2. Generate MLC Chat Config
---------------------------

Use ``mlc_llm gen_config`` to generate ``mlc-chat-config.json`` and process tokenizers.
See :ref:`compile-command-specification` for specification of ``gen_config``.

.. code:: shell

    mlc_llm gen_config ./dist/models/RedPajama-INCITE-Instruct-3B-v1/ \
        --quantization q4f16_1 --conv-template redpajama_chat \
        -o dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC/


.. note::
    The file ``mlc-chat-config.json`` is crucial in both model compilation
    and runtime chatting. Here we only care about the latter case.

    You can **optionally** customize
    ``dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC/mlc-chat-config.json`` (checkout :ref:`configure-mlc-chat-json` for more detailed instructions).
    You can also simply use the default configuration.

    `conversation_template <https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/conversation_template>`__
    directory contains a full list of conversation templates that MLC provides. If the model you are adding
    requires a new conversation template, you would need to add your own.
    Follow `this PR <https://github.com/mlc-ai/mlc-llm/pull/2163>`__ as an example. However,
    adding your own template would require you :ref:`build mlc_llm from source <mlcchat_build_from_source>` in order for it
    to be recognized by the runtime.

By now, you should have the following files.

.. code:: shell

    ~/mlc-llm > ls dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC
        mlc-chat-config.json                             # ===> the chat config
        tensor-cache.json                               # ===> the model weight info
        params_shard_0.bin                               # ===> the model weights
        params_shard_1.bin
        ...
        tokenizer.json                                   # ===> the tokenizer files
        tokenizer_config.json

.. _distribute-compiled-models:

(Optional) 3. Upload weights to HF
----------------------------------

Optionally, you can upload what we have to huggingface.

.. code:: shell

    # First, please create a repository on Hugging Face.
    # With the repository created, run
    git lfs install
    git clone https://huggingface.co/my-huggingface-account/my-redpajama3b-weight-huggingface-repo
    cd my-redpajama3b-weight-huggingface-repo
    cp path/to/mlc-llm/dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC/* .
    git add . && git commit -m "Add redpajama-3b instruct model weights"
    git push origin main

This would result in something like `RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC
<https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/tree/main>`_, but
for **Instruct** instead of **Chat**.

Good job, you have successfully distributed the model you compiled.
Next, we will talk about how we can consume the model weights in applications.

Download the Distributed Models
-------------------------------

You can now use the existing mlc tools such as chat/serve/package with the converted weights.

.. code:: shell

    mlc_llm chat HF://my-huggingface-account/my-redpajama3b-weight-huggingface-repo

.. _package-libraries-and-weights:

Package Libraries and Weights
=============================

When we want to build LLM applications with MLC LLM (e.g., iOS/Android apps),
usually we need to build static model libraries and app binding libraries,
and sometimes bundle model weights into the app.
MLC LLM provides a tool for fast model library and weight packaging: ``mlc_llm package``.

This page briefly introduces how to use ``mlc_llm package`` for packaging.
Tutorials :ref:`deploy-ios` and :ref:`deploy-android` contain detailed examples and instructions
on using this packaging tool for iOS and Android deployment.

-----

Introduction
------------

To use ``mlc_llm package``, we must clone the source code of `MLC LLM <https://github.com/mlc-ai/mlc-llm>`_
and `install the MLC LLM and TVM package <https://llm.mlc.ai/docs/install/mlc_llm.html#option-1-prebuilt-package>`_.
Depending on the app we build, there might be some other dependencies, which are described in
corresponding :ref:`iOS <deploy-ios>` and :ref:`Android <deploy-android>` tutorials.

After cloning, the basic usage of ``mlc_llm package`` is as the following.

.. code:: bash

    export MLC_LLM_SOURCE_DIR=/path/to/mlc-llm
    cd /path/to/app  # The app root directory which contains "mlc-package-config.json".
                     # E.g., "ios/MLCChat" or "android/MLCChat"
    mlc_llm package

**The package command reads from the JSON file** ``mlc-package-config.json`` **under the current directory.**
The output of this command is a directory ``dist/``,
which contains the packaged model libraries (under ``dist/lib/``) and weights (under ``dist/bundle/``).
This directory contains all necessary data for the app build.
Depending on the app we build, the internal structure of ``dist/lib/`` may be different.

.. code::

   dist
   ├── lib
   │   └── ...
   └── bundle
       └── ...

The input ``mlc-package-config.json`` file specifies

* the device (e.g., iPhone or Android) to package model libraries and weights for,
* the list of models to package.

Below is an example ``mlc-package-config.json`` file:

.. code:: json

    {
        "device": "iphone",
        "model_list": [
            {
                "model": "HF://mlc-ai/Mistral-7B-Instruct-v0.2-q3f16_1-MLC",
                "model_id": "Mistral-7B-Instruct-v0.2-q3f16_1",
                "estimated_vram_bytes": 3316000000,
                "bundle_weight": true,
                "overrides": {
                    "context_window_size": 512
                }
            },
            {
                "model": "HF://mlc-ai/gemma-2b-it-q4f16_1-MLC",
                "model_id": "gemma-2b-q4f16_1",
                "estimated_vram_bytes": 3000000000,
                "overrides": {
                    "prefill_chunk_size": 128
                }
            }
        ]
    }

This example ``mlc-package-config.json`` specifies "iphone" as the target device.
In the ``model_list``,

* ``model`` points to the Hugging Face repository which contains the pre-converted model weights. Apps will download model weights from the Hugging Face URL.
* ``model_id`` is a unique model identifier.
* ``estimated_vram_bytes`` is an estimation of the vRAM the model takes at runtime.
* ``"bundle_weight": true`` means the model weights of the model will be bundled into the app when building.
* ``overrides`` specifies some model config parameter overrides.


Below is a more detailed specification of the ``mlc-package-config.json`` file.
Each entry in ``"model_list"`` of the JSON file has the following fields:

``model``
   (Required) The path to the MLC-converted model to be built into the app.

   Usually it is a Hugging Face URL (e.g., ``"model": "HF://mlc-ai/phi-2-q4f16_1-MLC"```) that contains the pre-converted model weights.
   For iOS, it can also be a path to a local model directory which contains converted model weights (e.g., ``"model": "../dist/gemma-2b-q4f16_1"``).
   Please check out :ref:`convert-weights-via-MLC` if you want to build local model into the app.

``model_id``
  (Required) A unique local identifier to identify the model.
  It can be an arbitrary one.

``estimated_vram_bytes``
   (Required) Estimated requirements of vRAM to run the model.

``bundle_weight``
   (Optional) A boolean flag indicating whether to bundle model weights into the app.
   If this field is set to true, the ``mlc_llm package`` command will copy the model weights
   to ``dist/bundle/$model_id``.

``overrides``
   (Optional) A dictionary to override the default model context window size (to limit the KV cache size) and prefill chunk size (to limit the model temporary execution memory).
   Example:

   .. code:: json

      {
         "device": "iphone",
         "model_list": [
            {
                  "model": "HF://mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
                  "model_id": "RedPajama-INCITE-Chat-3B-v1-q4f16_1",
                  "estimated_vram_bytes": 2960000000,
                  "overrides": {
                     "context_window_size": 512,
                     "prefill_chunk_size": 128
                  }
            }
         ]
      }

``model_lib``
   (Optional) A string specifying the system library prefix to use for the model.
   Usually this is used when you want to build multiple model variants with the same architecture into the app.
   **This field does not affect any app functionality.**
   The ``"model_lib_path_for_prepare_libs"`` introduced below is also related.
   Example:

   .. code:: json

      {
         "device": "iphone",
         "model_list": [
            {
                  "model": "HF://mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
                  "model_id": "RedPajama-INCITE-Chat-3B-v1-q4f16_1",
                  "estimated_vram_bytes": 2960000000,
                  "model_lib": "gpt_neox_q4f16_1"
            }
         ]
      }


Besides ``model_list`` in ``MLCChat/mlc-package-config.json``,
you can also **optionally** specify a dictionary of ``"model_lib_path_for_prepare_libs"``,
**if you want to use model libraries that are manually compiled**.
The keys of this dictionary should be the ``model_lib`` that specified in model list,
and the values of this dictionary are the paths (absolute, or relative) to the manually compiled model libraries.
The model libraries specified in ``"model_lib_path_for_prepare_libs"`` will be built into the app when running ``mlc_llm package``.
Example:

.. code:: json

   {
      "device": "iphone",
      "model_list": [
         {
               "model": "HF://mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
               "model_id": "RedPajama-INCITE-Chat-3B-v1-q4f16_1",
               "estimated_vram_bytes": 2960000000,
               "model_lib": "gpt_neox_q4f16_1"
         }
      ],
      "model_lib_path_for_prepare_libs": {
         "gpt_neox_q4f16_1": "../../dist/lib/RedPajama-INCITE-Chat-3B-v1-q4f16_1-iphone.tar"
      }
   }

Compilation Cache
-----------------
``mlc_llm package`` leverage a local JIT cache to avoid repetitive compilation of the same input.
It also leverages a local cache to download weights from remote. These caches
are shared across the entire project. Sometimes it is helpful to force rebuild when
we have a new compiler update or when something goes wrong with the cached library.
You can do so by setting the environment variable ``MLC_JIT_POLICY=REDO``

.. code:: bash

   MLC_JIT_POLICY=REDO mlc_llm package

Arguments of ``mlc_llm package``
--------------------------------

Command ``mlc_llm package`` can optionally take the arguments below:

``--package-config``
    A path to ``mlc-package-config.json`` which contains the device and model specification.
    By default, it is the ``mlc-package-config.json`` under the current directory.

``--mlc-llm-source-dir``
    The path to MLC LLM source code (cloned from https://github.com/mlc-ai/mlc-llm).
    By default, it is the ``$MLC_LLM_SOURCE_DIR`` environment variable.
    If neither ``$MLC_LLM_SOURCE_DIR`` or ``--mlc-llm-source-dir`` is specified, error will be reported.

``--output`` / ``-o``
    The output directory of ``mlc_llm package`` command.
    By default, it is ``dist/`` under the current directory.


Summary and What to Do Next
---------------------------

In this page, we introduced the ``mlc_llm package`` command for fast model library and weight packaging.

* It takes input file ``mlc-package-config.json`` which contains the device and model specification for packaging.
* It outputs directory ``dist/``, which contains packaged libraries under ``dist/lib/`` and model weights under ``dist/bundle/``.

Next, please feel free to check out the :ref:`iOS <deploy-ios>` and :ref:`Android <deploy-android>` tutorials for detailed examples of using ``mlc_llm package``.

Install Conda
=============

MLC LLM does not depend on, but generally recommends conda as a generic dependency manager, primarily because it creates unified cross-platform experience to make windows/Linux/macOS development equally easy. Moreover, conda is python-friendly and provides all the python packages needed for MLC LLM, such as numpy.

.. contents:: Table of Contents
    :depth: 2


Install Miniconda
-----------------

**Use installer.** Miniconda, a minimal distribution of conda, comes with out-of-box installer across Windows/macOS/Linux. Please refer to its `official website <https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links>`_ link for detailed instructions.

**Set libmamba as the dependency solver.** The default dependency solver in conda could be slow in certain scenarios, and it is always recommended to upgrade it to libmamba, a faster solver.

.. code-block:: bash
   :caption: Set libmamba as the default solver

   # update conda
   conda update --yes -n base -c defaults conda
   # install `conda-libmamba-solver`
   conda install --yes -n base conda-libmamba-solver
   # set it as the default solver
   conda config --set solver libmamba

.. note::
    Conda is a generic dependency manager, which is not necessarily related to any Python distributions.
    In fact, some of our tutorials recommends to use conda to install cmake, git and rust for its unified experience across OS platforms.


Validate installation
---------------------

**Step 1. Check conda-arch mismatch.** Nowadays macOS runs on two different architectures: arm64 and x86_64, which could particularly lead to many misuses in MLC LLM, where the error message hints about "architecture mismatch". Use the following command to make sure particular conda architecture is installed accordingly:

.. code-block:: bash
   :caption: Check conda architecture

   >>> conda info | grep platform
   # for arm mac
   platform : osx-arm64
   # for x86 mac
   platform : osx-64

**Step 2. Check conda virtual environment.** If you have installed python in your conda virtual environment, make sure conda, Python and pip are all from this environment:

.. code-block:: bash
   :caption: Check conda virtual environment (macOS, Linux)

   >>> echo $CONDA_PREFIX
   /.../miniconda3/envs/mlc-doc-venv
   >>> which python
   /.../miniconda3/envs/mlc-doc-venv/bin/python
   >>> which pip
   /.../miniconda3/envs/mlc-doc-venv/bin/pip

.. code-block:: bat
   :caption: Check conda virtual environment (Windows)

   >>> echo $Env:CONDA_PREFIX
   \...\miniconda3\envs\mlc-doc-venv
   >>> Get-Command python.exe
   \...\miniconda3\envs\mlc-doc-venv\bin\python.exe
   >>> Get-Command pip.exe
   \...\miniconda3\envs\mlc-doc-venv\bin\pip.exe

.. _install-web-build:

Install Wasm Build Environment
==============================

This page describes the steps to setup build environment for WebAssembly and WebGPU builds.

Step 1: Install EMSDK
---------------------

Emscripten is an LLVM-based compiler that compiles C/C++ source code to WebAssembly.
We need to install emscripten for webgpu build.

- Please follow the installation instruction `here <https://emscripten.org/docs/getting_started/downloads.html#installation-instructions-using-the-emsdk-recommended>`__
  to install the latest emsdk.
- Source path/to/emsdk_env.sh so emcc is reachable from PATH and the command emcc works.

Validate that emcc is accessible in shell

.. code:: bash

    emcc --version

.. note::
    We recently found that using the latest ``emcc`` version may run into issues during runtime. Use
    ``./emsdk install 3.1.56`` instead of ``./emsdk install latest`` for now as a workaround.

    The error may look like

    .. code:: text

        Init error, LinkError: WebAssembly.instantiate(): Import #6 module="wasi_snapshot_preview1"
        function="proc_exit": function import requires a callable


Step 2: Set TVM_SOURCE_DIR and MLC_LLM_SOURCE_DIR
-------------------------------------------------

We need to set a path to a tvm source in order to build tvm runtime.
Note that you do not need to build TVM from the source. The source here is only used to build the web runtime component.
Set environment variable in your shell startup profile in to point to ``3rdparty/tvm`` (if preferred, you could also
point to your own TVM address if you installed TVM from source).

Besides, we also need to set ``MLC_LLM_SOURCE_DIR`` so that we can locate ``mlc_wasm_runtime.bc`` when compiling a model library wasm.

.. code:: bash

    export TVM_SOURCE_DIR=/path/to/3rdparty/tvm
    export MLC_LLM_SOURCE_DIR=/path/to/mlc-llm


Step 3: Prepare Wasm Runtime
----------------------------

First, we need to obtain a copy of the mlc-llm source code for the setup script

.. code:: bash

    git clone https://github.com/mlc-ai/mlc-llm.git --recursive
    cd mlc-llm

Now we can prepare wasm runtime using the script in mlc-llm repo

.. code:: bash

    ./web/prep_emcc_deps.sh

We can then validate the outcome

.. code:: bash

    >>> echo ${TVM_SOURCE_DIR}

    /path/set/in/step2

    >>> ls -l ${TVM_SOURCE_DIR}/web/dist/wasm/*.bc

    tvmjs_support.bc
    wasm_runtime.bc
    webgpu_runtime.bc

GPU Drivers and SDKs
====================

.. contents:: Table of Contents
    :depth: 2

MLC LLM is a universal deployment solution that allows efficient CPU/GPU code generation without AutoTVM-based performance tuning. This section focuses on generic GPU environment setup and troubleshooting.

CUDA
----

CUDA is required to compile and run models with CUDA backend.

Installation
^^^^^^^^^^^^

If you have a NVIDIA GPU and you want to use models compiled with CUDA
backend, you should install CUDA, which can be downloaded from
`here <https://developer.nvidia.com/cuda-downloads>`__.

Validate Installation
^^^^^^^^^^^^^^^^^^^^^

To verify you have correctly installed CUDA runtime and NVIDIA driver, run ``nvidia-smi`` in command line and see if you can get the GPU information.

ROCm
----

ROCm is required to compile and run models with ROCm backend.

Installation
^^^^^^^^^^^^

Right now MLC LLM only supports ROCm 6.1/6.2.
If you have AMD GPU and you want to use models compiled with ROCm
backend, you should install ROCm from `here <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.0/install/quick-start.html>`__.

Validate Installation
^^^^^^^^^^^^^^^^^^^^^

To verify you have correctly installed ROCm, run ``rocm-smi`` in command line.
If you see the list of AMD devices printed out in a table, it means the ROCm is correctly installed.

.. _vulkan_driver:

Vulkan Driver
-------------

Installation
^^^^^^^^^^^^

To run pre-trained models (e.g. pulled from MLC-AI's Hugging Face repository) compiled with Vulkan backend, you are expected to install Vulkan driver on your machine.

Please check `this
page <https://www.vulkan.org/tools#vulkan-gpu-resources>`__ and find the
Vulkan driver according to your GPU vendor.

AMD Radeon and Radeon PRO
#########################

For AMD Radeon and Radeon PRO users, please download AMD's drivers from official website (`Linux <https://www.amd.com/en/support/linux-drivers>`__ / `Windows <https://www.amd.com/en/support>`__).
For Linux users, after you installed the ``amdgpu-install`` package, you can follow the instructions in its `documentation <https://amdgpu-install.readthedocs.io/en/latest/install-script.html>`__ to install
the driver. We recommend you installing ROCr OpenCL and PRO Vulkan (proprietary) for best performance, which can be done by running the following command:

.. code:: bash

   amdgpu-install --usecase=graphics,opencl --opencl=rocr --vulkan=pro --no-32

Validate Installation
^^^^^^^^^^^^^^^^^^^^^

To verify whether Vulkan installation is successful or not, you are encouraged to install ``vulkaninfo``, below are the instructions to install ``vulkaninfo`` on different platforms:

.. tabs ::

   .. code-tab :: bash Ubuntu/Debian

      sudo apt-get update
      sudo apt-get install vulkan-tools

   .. code-tab :: bash Windows

      # It comes with your GPU driver

   .. code-tab :: bash Fedora

      sudo dnf install vulkan-tools

   .. code-tab :: bash Arch Linux

      sudo pacman -S vulkan-tools
      # Arch Linux has maintained an awesome wiki page for Vulkan which you can refer to for troubleshooting: https://wiki.archlinux.org/title/Vulkan

   .. code-tab :: bash Other Distributions

      # Please install Vulkan SDK for your platform
      # https://vulkan.lunarg.com/sdk/home


After installation, you can run ``vulkaninfo`` in command line and see if you can get the GPU information.

.. note::
   WSL support for Windows is work-in-progress at the moment. Please do not use WSL on Windows to run Vulkan.

Vulkan SDK
----------

Vulkan SDK is required for compiling models to Vulkan backend. To build TVM compiler from source, you will need to install Vulkan SDK as a dependency, but our :doc:`pre-built wheels <../install/mlc_llm>` already ships with Vulkan SDK.

Check Vulkan SDK installation guide according to your platform:

.. tabs ::

   .. tab :: Windows

      `Getting Started with the Windows Tarball Vulkan SDK <https://vulkan.lunarg.com/doc/sdk/latest/windows/getting_started.html>`__

   .. tab :: Linux

      For Ubuntu user, please check
      `Getting Started with the Ubuntu Vulkan SDK <https://vulkan.lunarg.com/doc/sdk/latest/linux/getting_started_ubuntu.html>`__

      For other Linux distributions, please check
      `Getting Started with the Linux Tarball Vulkan SDK <https://vulkan.lunarg.com/doc/sdk/latest/linux/getting_started.html>`__

   .. tab :: Mac

      `Getting Started with the macOS Vulkan SDK <https://vulkan.lunarg.com/doc/sdk/latest/mac/getting_started.html>`__

Please refer to installation and setup page for next steps to build TVM from source.

OpenCL SDK
----------

OpenCL SDK is only required when you want to build your own models for OpenCL backend. Please refer to `OpenCL's Github Repository <https://github.com/KhronosGroup/OpenCL-SDK>`__ for installation guide of OpenCL-SDK.

Orange Pi 5 (RK3588 based SBC)
------------------------------

OpenCL SDK and Mali GPU driver is required to compile and run models for OpenCL backend.

Installation
^^^^^^^^^^^^

* Download and install the Ubuntu 22.04 for your board from `here <https://github.com/Joshua-Riek/ubuntu-rockchip/releases/tag/v1.22>`__

* Download and install ``libmali-g610.so``

.. code-block:: bash

   cd /usr/lib && sudo wget https://github.com/JeffyCN/mirrors/raw/libmali/lib/aarch64-linux-gnu/libmali-valhall-g610-g6p0-x11-wayland-gbm.so

* Check if file ``mali_csffw.bin`` exist under path ``/lib/firmware``, if not download it with command:

.. code-block:: bash

   cd /lib/firmware && sudo wget https://github.com/JeffyCN/mirrors/raw/libmali/firmware/g610/mali_csffw.bin

* Download OpenCL ICD loader and manually add libmali to ICD

.. code-block:: bash

   sudo apt update
   sudo apt install mesa-opencl-icd
   sudo mkdir -p /etc/OpenCL/vendors
   echo "/usr/lib/libmali-valhall-g610-g6p0-x11-wayland-gbm.so" | sudo tee /etc/OpenCL/vendors/mali.icd

* Download and install ``libOpenCL``

.. code-block:: bash

   sudo apt install ocl-icd-opencl-dev

* Download and install dependencies for Mali OpenCL

.. code-block:: bash

   sudo apt install libxcb-dri2-0 libxcb-dri3-0 libwayland-client0 libwayland-server0 libx11-xcb1

* Download and install clinfo to check if OpenCL successfully installed

.. code-block:: bash

   sudo apt install clinfo

Validate Installation
^^^^^^^^^^^^^^^^^^^^^

To verify you have correctly installed OpenCL runtime and Mali GPU driver, run ``clinfo`` in command line and see if you can get the GPU information.
You are expect to see the following information:

.. code-block:: bash

   $ clinfo
   arm_release_ver: g13p0-01eac0, rk_so_ver: 3
   Number of platforms                               2
      Platform Name                                   ARM Platform
      Platform Vendor                                 ARM
      Platform Version                                OpenCL 2.1 v1.g6p0-01eac0.2819f9d4dbe0b5a2f89c835d8484f9cd
      Platform Profile                                FULL_PROFILE
      ...

.. _install-mlc-packages:

Install MLC LLM Python Package
==============================

.. contents:: Table of Contents
    :local:
    :depth: 2

MLC LLM Python Package can be installed directly from a prebuilt developer package, or built from source.

Option 1. Prebuilt Package
--------------------------

We provide nightly built pip wheels for MLC-LLM via pip.
Select your operating system/compute platform and run the command in your terminal:

.. note::
    ❗ Whenever using Python, it is highly recommended to use **conda** to manage an isolated Python environment to avoid missing dependencies, incompatible versions, and package conflicts.
    Please make sure your conda environment has Python and pip installed.

.. tabs::

    .. tab:: Linux

        .. tabs::

            .. tab:: CPU

                .. code-block:: bash

                    conda activate your-environment
                    python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu

            .. tab:: CUDA 12.8

                .. code-block:: bash

                    conda activate your-environment
                    python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu128 mlc-ai-nightly-cu128

            .. tab:: CUDA 13.0

                .. code-block:: bash

                    conda activate your-environment
                    python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu130 mlc-ai-nightly-cu130

            .. tab:: ROCm 6.1

                .. code-block:: bash

                    conda activate your-environment
                    python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-rocm61 mlc-ai-nightly-rocm61

            .. tab:: ROCm 6.2

                .. code-block:: bash

                    conda activate your-environment
                    python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-rocm62 mlc-ai-nightly-rocm62

            .. tab:: Vulkan

                Supported in all Linux packages. Checkout the following instructions
                to install the latest vulkan loader to avoid vulkan not found issue.

                .. code-block:: bash

                    conda install -c conda-forge gcc libvulkan-loader

        .. note::
            We need git-lfs in the system, you can install it via

            .. code-block:: bash

                conda install -c conda-forge git-lfs

            If encountering issues with GLIBC not found, please install the latest glibc in conda:

            .. code-block:: bash

                conda install -c conda-forge libstdcxx-ng

            Besides, we would recommend using Python 3.13; so if you are creating a new environment,
            you could use the following command:

            .. code-block:: bash

                conda create --name mlc-prebuilt  python=3.13

    .. tab:: macOS

        .. tabs::

            .. tab:: CPU + Metal

                .. code-block:: bash

                    conda activate your-environment
                    python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu

        .. note::

            Always check if conda is installed properly in macOS using the command below:

            .. code-block:: bash

                conda info | grep platform

            It should return "osx-64" for Mac with Intel chip, and "osx-arm64" for Mac with Apple chip.
            We need git-lfs in the system, you can install it via

            .. code-block:: bash

                conda install -c conda-forge git-lfs

    .. tab:: Windows

        .. tabs::

            .. tab:: CPU + Vulkan

                .. code-block:: bash

                    conda activate your-environment
                    python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu

        .. note::
            Please make sure your conda environment comes with python and pip.
            Make sure you also install the following packages,
            vulkan loader, clang, git and git-lfs to enable proper automatic download
            and jit compilation.

            .. code-block:: bash

                conda install -c conda-forge clang libvulkan-loader git-lfs git

            If encountering the error below:

            .. code-block:: bash

                FileNotFoundError: Could not find module 'path\to\site-packages\tvm\tvm.dll' (or one of its dependencies). Try using the full path with constructor syntax.

            It is likely `zstd`, a dependency to LLVM, was missing. Please use the command below to get it installed:

            .. code-block:: bash

                conda install zstd


Then you can verify installation in command line:

.. code-block:: bash

    python -c "import mlc_llm; print(mlc_llm)"
    # Prints out: <module 'mlc_llm' from '/path-to-env/lib/python3.13/site-packages/mlc_llm/__init__.py'>

|

.. _mlcchat_build_from_source:

Option 2. Build from Source
---------------------------

We also provide options to build mlc runtime libraries ``mlc_llm`` from source.
This step is useful when you want to make modification or obtain a specific version of mlc runtime.


**Step 1. Set up build dependency.** To build from source, you need to ensure that the following build dependencies are satisfied:

* CMake >= 3.24
* Git
* `Rust and Cargo <https://www.rust-lang.org/tools/install>`_, required by Hugging Face's tokenizer
* One of the GPU runtimes:

    * CUDA >= 11.8 (NVIDIA GPUs)
    * Metal (Apple GPUs)
    * Vulkan (NVIDIA, AMD, Intel GPUs)

.. code-block:: bash
    :caption: Set up build dependencies in Conda

    # make sure to start with a fresh environment
    conda env remove -n mlc-chat-venv
    # create the conda environment with build dependency
    conda create -n mlc-chat-venv -c conda-forge \
        "cmake>=3.24" \
        rust \
        git \
        python=3.13
    # enter the build environment
    conda activate mlc-chat-venv

.. note::
    For runtime, :doc:`TVM </install/tvm>` compiler is not a dependency for MLCChat CLI or Python API. Only TVM's runtime is required, which is automatically included in `3rdparty/tvm <https://github.com/mlc-ai/mlc-llm/tree/main/3rdparty>`_.
    However, if you would like to compile your own models, you need to follow :doc:`TVM </install/tvm>`.

**Step 2. Configure and build.** A standard git-based workflow is recommended to download MLC LLM, after which you can specify build requirements with our lightweight config generation tool:

.. code-block:: bash
    :caption: Configure and build

    # clone from GitHub
    git clone --recursive https://github.com/mlc-ai/mlc-llm.git && cd mlc-llm/
    # create build directory
    mkdir -p build && cd build
    # generate build configuration
    python ../cmake/gen_cmake_config.py
    # build mlc_llm libraries
    cmake .. && make -j $(nproc) && cd ..

**Step 3. Install via Python.** We recommend that you install ``mlc_llm`` as a Python package, giving you
access to ``mlc_llm.compile``, ``mlc_llm.MLCEngine``, and the CLI.
There are two ways to do so:

    .. tabs ::

       .. code-tab :: bash Install via environment variable

          export MLC_LLM_SOURCE_DIR=/path-to-mlc-llm
          export PYTHONPATH=$MLC_LLM_SOURCE_DIR/python:$PYTHONPATH
          alias mlc_llm="python -m mlc_llm"

       .. code-tab :: bash Install via pip local project

          conda activate your-own-env
          which python # make sure python is installed, expected output: path_to_conda/envs/your-own-env/bin/python
          cd /path-to-mlc-llm/python
          pip install -e .

**Step 4. Validate installation.** You may validate if MLC libarires and mlc_llm CLI is compiled successfully using the following command:

.. code-block:: bash
    :caption: Validate installation

    # expected to see `libmlc_llm.so` and `libtvm_runtime.so`
    ls -l ./build/
    # expected to see help message
    mlc_llm chat -h

Finally, you can verify installation in command line. You should see the path you used to build from source with:

.. code:: bash

   python -c "import mlc_llm; print(mlc_llm)"

.. _install-tvm:

Install TVM Compiler
==========================

.. contents:: Table of Contents
    :local:
    :depth: 2

`TVM Unity <https://discuss.tvm.apache.org/t/establish-tvm-unity-connection-a-technical-strategy/13344>`__, the latest development in Apache TVM, is required to build MLC LLM. Its features include:

- High-performance CPU/GPU code generation instantly without tuning;
- Dynamic shape and symbolic shape tracking by design;
- Supporting both inference and training;
- Productive python-first compiler implementation. As a concrete example, MLC LLM compilation is implemented in pure python using its API.

TVM can be installed directly from a prebuilt developer package, or built from source.

.. _tvm-prebuilt-package:

Option 1. Prebuilt Package
--------------------------

A nightly prebuilt Python package of Apache TVM is provided.

.. note::
    ❗ Whenever using Python, it is highly recommended to use **conda** to manage an isolated Python environment to avoid missing dependencies, incompatible versions, and package conflicts.

.. tabs::

   .. tab:: Linux

      .. tabs::

         .. tab:: CPU

            .. code-block:: bash

              conda activate your-environment
              python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cpu

         .. tab:: CUDA 12.8

            .. code-block:: bash

              conda activate your-environment
              python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cu128

         .. tab:: CUDA 13.0

            .. code-block:: bash

              conda activate your-environment
              python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cu130

         .. tab:: ROCm 6.1

            .. code-block:: bash

              conda activate your-environment
              python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-rocm61

         .. tab:: ROCm 6.2

            .. code-block:: bash

              conda activate your-environment
              python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-rocm62

         .. tab:: Vulkan

            Supported in all Linux packages.

      .. note::

        If encountering issues with GLIBC not found, please install the latest glibc in conda:

        .. code-block:: bash

          conda install -c conda-forge libstdcxx-ng

   .. tab:: macOS

      .. tabs::

         .. tab:: CPU + Metal

            .. code-block:: bash

              conda activate your-environment
              python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cpu

        .. note::

          Always check if conda is installed properly in macOS using the command below:

          .. code-block:: bash

            conda info | grep platform

          It should return "osx-64" for Mac with Intel chip, and "osx-arm64" for Mac with Apple chip.

   .. tab:: Windows

      .. tabs::

         .. tab:: CPU + Vulkan

            .. code-block:: bash

              conda activate your-environment
              python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cpu

      .. note::
        Make sure you also install vulkan loader and clang to avoid vulkan
        not found error or clang not found(needed for jit compile)

        .. code-block:: bash

            conda install -c conda-forge clang libvulkan-loader

        If encountering the error below:

        .. code-block:: bash

            FileNotFoundError: Could not find module 'path\to\site-packages\tvm\tvm.dll' (or one of its dependencies). Try using the full path with constructor syntax.

        It is likely `zstd`, a dependency to LLVM, was missing. Please use the command below to get it installed:

        .. code-block:: bash

            conda install zstd

.. _tvm-build-from-source:

Option 2. Build from Source
---------------------------

While it is generally recommended to always use the prebuilt TVM, if you require more customization, you may need to build it from source. **NOTE.** this should only be attempted if you are familiar with the intricacies of C++, CMake, LLVM, Python, and other related systems.

.. collapse:: Details

    **Step 1. Set up build dependency.** To build from source, you need to ensure that the following build dependencies are met:

    - CMake >= 3.24
    - LLVM >= 15
      - For please install LLVM>=17 for ROCm 6.1 and LLVM>=18 for ROCm 6.2.
    - Git
    - (Optional) CUDA >= 11.8 (targeting NVIDIA GPUs)
    - (Optional) Metal (targeting Apple GPUs such as M1 and M2)
    - (Optional) Vulkan (targeting NVIDIA, AMD, Intel and mobile GPUs)
    - (Optional) OpenCL (targeting NVIDIA, AMD, Intel and mobile GPUs)

    .. note::
        - To target NVIDIA GPUs, either CUDA or Vulkan is required (CUDA is recommended);
        - For AMD and Intel GPUs, Vulkan is necessary;
        - When targeting Apple (macOS, iOS, iPadOS), Metal is a mandatory dependency;
        - Some Android devices only support OpenCL, but most of them support Vulkan.

    To easiest way to manage dependency is via conda, which maintains a set of toolchains including LLVM across platforms. To create the environment of those build dependencies, one may simply use:

    .. code-block:: bash
        :caption: Set up build dependencies in conda

        # make sure to start with a fresh environment
        conda env remove -n tvm-build-venv
        # create the conda environment with build dependency
        conda create -n tvm-build-venv -c conda-forge \
            "llvmdev>=15" \
            "cmake>=3.24" \
            git \
            python=3.13
        # enter the build environment
        conda activate tvm-build-venv

    **Step 2. Configure and build.** Standard git-based workflow are recommended to download Apache TVM, and then specify build requirements in ``config.cmake``:

    .. code-block:: bash
        :caption: Download TVM from GitHub

        # clone from GitHub
        git clone --recursive https://github.com/apache/tvm.git && cd tvm
        # create the build directory
        rm -rf build && mkdir build && cd build
        # specify build requirements in `config.cmake`
        cp ../cmake/config.cmake .

    We want to specifically tweak the following flags by appending them to the end of the configuration file:

    .. code-block:: bash
        :caption: Configure build in ``config.cmake``

        # controls default compilation flags
        echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake
        # LLVM is a must dependency
        echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
        echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake
        # GPU SDKs, turn on if needed
        echo "set(USE_CUDA   OFF)" >> config.cmake
        echo "set(USE_ROCM   OFF)" >> config.cmake
        echo "set(USE_METAL  OFF)" >> config.cmake
        echo "set(USE_VULKAN OFF)" >> config.cmake
        echo "set(USE_OPENCL OFF)" >> config.cmake
        # Below are options for CUDA, turn on if needed
        # CUDA_ARCH is the cuda compute capability of your GPU.
        # Examples: 89 for 4090, 90a for H100/H200, 100a for B200.
        # Reference: https://developer.nvidia.com/cuda-gpus
        echo "set(CMAKE_CUDA_ARCHITECTURES YOUR_CUDA_COMPUTE_CAPABILITY_HERE)" >> config.cmake
        echo "set(USE_CUBLAS ON)" >> config.cmake
        echo "set(USE_CUTLASS ON)" >> config.cmake
        echo "set(USE_THRUST ON)" >> config.cmake
        echo "set(USE_NVTX ON)" >> config.cmake
        # Below is the option for ROCM, turn on if needed
        echo "set(USE_HIPBLAS ON)" >> config.cmake

    .. note::
        ``HIDE_PRIVATE_SYMBOLS`` is a configuration option that enables the ``-fvisibility=hidden`` flag. This flag helps prevent potential symbol conflicts between TVM and PyTorch. These conflicts arise due to the frameworks shipping LLVMs of different versions.

        `CMAKE_BUILD_TYPE <https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html>`_ controls default compilation flag:

        - ``Debug`` sets ``-O0 -g``
        - ``RelWithDebInfo`` sets ``-O2 -g -DNDEBUG`` (recommended)
        - ``Release`` sets ``-O3 -DNDEBUG``

    Once ``config.cmake`` is edited accordingly, kick off build with the commands below:

    .. code-block:: bash
        :caption: Build ``libtvm`` using cmake and cmake

        cmake .. && make -j $(nproc) && cd ..

    A success build should produce ``libtvm`` and ``libtvm_runtime`` under ``/path-tvm/build/`` directory.

    Leaving the build environment ``tvm-build-venv``, there are two ways to install the successful build into your environment:

    .. tabs ::

       .. code-tab :: bash Install via environment variable

          export PYTHONPATH=/path-to-tvm/python:$PYTHONPATH

       .. code-tab :: bash Install via pip local project

          conda activate your-own-env
          conda install python # make sure python is installed
          cd /path-to-tvm/python
          pip install -e .

.. `|` adds a blank line

|

.. _tvm-validate:

Validate TVM Installation
-------------------------

Using a compiler infrastructure with multiple language bindings could be error-prone.
Therefore, it is highly recommended to validate TVM installation before use.

**Step 1. Locate TVM Python package.** The following command can help confirm that TVM is properly installed as a python package and provide the location of the TVM python package:

.. code-block:: bash

    >>> python -c "import tvm; print(tvm.__file__)"
    /some-path/lib/python3.13/site-packages/tvm/__init__.py

**Step 2. Confirm which TVM library is used.** When maintaining multiple build or installation of TVM, it becomes important to double check if the python package is using the proper ``libtvm`` with the following command:

.. code-block:: bash

    >>> python -c "import tvm; print(tvm.base._LIB)"
    <CDLL '/some-path/lib/python3.13/site-packages/tvm/libtvm.dylib', handle 95ada510 at 0x1030e4e50>

**Step 3. Reflect TVM build option.** Sometimes when downstream application fails, it could likely be some mistakes with a wrong TVM commit, or wrong build flags. To find it out, the following commands will be helpful:

.. code-block:: bash

    >>> python -c "import tvm; print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))"
    ... # Omitted less relevant options
    GIT_COMMIT_HASH: 4f6289590252a1cf45a4dc37bce55a25043b8338
    HIDE_PRIVATE_SYMBOLS: ON
    USE_LLVM: llvm-config --link-static
    LLVM_VERSION: 15.0.7
    USE_VULKAN: OFF
    USE_CUDA: OFF
    CUDA_VERSION: NOT-FOUND
    USE_OPENCL: OFF
    USE_METAL: ON
    USE_ROCM: OFF

.. note::
    ``GIT_COMMIT_HASH`` indicates the exact commit of the TVM build, and it can be found on GitHub via ``https://github.com/mlc-ai/relax/commit/$GIT_COMMIT_HASH``.

**Step 4. Check device detection.** Sometimes it could be helpful to understand if TVM could detect your device at all with the following commands:

.. code-block:: bash

    >>> python -c "import tvm; print(tvm.metal().exist)"
    True # or False
    >>> python -c "import tvm; print(tvm.cuda().exist)"
    False # or True
    >>> python -c "import tvm; print(tvm.vulkan().exist)"
    False # or True

Please note that the commands above verify the presence of an actual device on the local machine for the TVM runtime (not the compiler) to execute properly. However, TVM compiler can perform compilation tasks without requiring a physical device. As long as the necessary toolchain, such as NVCC, is available, TVM supports cross-compilation even in the absence of an actual device.