# CPM.cu

<strong>[中文版本](./README_ZH.md) | English</strong>

CPM.cu is a lightweight, high-performance CUDA implementation for LLMs, optimized for end-device inference and featuring cutting-edge techniques in **sparse architecture**, **speculative sampling** and **quantization**.

<div id="news"></div>

## 🔥 Project Updates

- [2025.06.06] Optimized for [MiniCPM4](https://github.com/openbmb/minicpm).
    - Support InfLLM-v2 attention kernel
    - Support sliding-window for the MTP layer, optimized for long context
    - Support quantization for the MTP layer
- [2025.05.29] Support Quantization at [SpecMQuant](https://github.com/AI9Stars/SpecMQuant).
    - Support Marlin GPTQ kernel for the LLM
    - Support Speculative Sampling for quantized LLM
- [2025.03.01] Release the first version at [FR-Spec](https://github.com/thunlp/FR-Spec).
    - SOTA Speculative Sampling Implementation
    - Support FR-Spec: Frequency-Ranked Speculative Sampling
    - Support Tree-based verification of Speculative Sampling in Flash-Attention
    - Support Static memory management and memory reuse
    - Support Fused kernels
    - Support Chunked prefill
    - Support CUDA Graph

<div id="demo"></div>

## Demo

https://github.com/user-attachments/assets/ab36fd7a-485b-4707-b72f-b80b5c43d024

<div id="getstart"></div>

## Getting Started

- [Installation](#install)
- [Model Weights](#modelweights)
- [Quick Start](#example)

<div id="install"></div>

## Installation

### Install from source

```bash
git clone https://github.com/OpenBMB/CPM.cu.git --recursive
cd CPM.cu
python3 setup.py install
```

<div id="modelweights"></div>

## Prepare Model

Please follow [MiniCPM4's README](https://github.com/openbmb/minicpm) to download the model weights.

<div id="example"></div>

## Quick Start

We provide a simple example to show how to use CPM.cu to generate text.

```bash
python3 tests/test_generate.py --prompt-file <your prompt file>
```

If you don't ​​specify​​ the model path, the scripts will load the model from ​​OpenBMB's Hugging Face repository​​.
If you want to use local paths, we recommend keeping all model filenames unchanged and placing them in the same directory. This way, you can run the model by specifying the directory with the -p parameter. Otherwise, we suggest modifying the paths in the code accordingly.

If you don't ​​specify​​ the prompt file, a default ​​Haystack task​​ with ​​15K context length​​ will be provided.
You can use --help to learn more ​​about the script's features​​.

We also provide a script, `tests/long_prompt_gen.py`, to generate ​​long code summarization.
This script ​​automatically collects code from this repository​​ and prompts ​​the model to "Summarize the code."​

```bash
python3 tests/long_prompt_gen.py # generate prompt.txt (for more details, use --help)
python3 tests/test_generate.py --prompt-file prompt.txt
```

The output should be of the following format:

```bash
Generated text (streaming output):
--------------------------------------------------
Prefilling: 100.0% (106850/106850 tokens) @ 6565.3 tokens/s - Complete!

<Generated Output HERE>
==================================================
Stream Generation Summary:
==================================================
Prefill length: 106850
Prefill time: 16.36 s
Prefill tokens/s: 6530.77
Mean accept length: 2.50
Decode length: 118
Decode time: 0.76 s
Decode tokens/s: 154.59
```

Where:

- the `Prefill` and `Decode` speed are output by (length, time and token/s).
- the `Mean accept length` is the average length of the accepted tokens when using Speculative Sampling.

## Code Structure

```bash
CPM.cu/
├── src/
│   ├── flash_attn/ # attention kernels: sparse, tree-verification, etc.
│   ├── model/
│   │   ├── minicpm4/ # minicpm4 model
│   │   ├── w4a16_gptq_marlin/ # marlin kernel
│   │   └── ... # common layers
│   ├── entry.cu # pybind: bind cuda and python
│   └── ...
├── cpmcu/ # python interface
└── ...
```

## More
We provide a word frequency generation script for FR-Spec, located at "scripts/fr_spec/gen_fr_index.py". You can run it as follows:
```
python scripts/fr_spec/gen_fr_index.py --model_path <your_model_path>
```
You can modify the code to use your own dataset. If your task is in a specific vertical domain, constructing word frequencies tailored to that domain can significantly improve processing speed.


## Acknowledgments

Our `src/flash_attn` folder modified based on [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/v2.6.3/csrc/flash_attn).

We have drawn inspiration from the following repositories:

- [EAGLE](https://github.com/SafeAILab/EAGLE)
- [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)

## Citation

Please cite our paper if you find our work valuable.

```
@article{zhao2025fr,
  title={FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling},
  author={Zhao, Weilin and Pan, Tengyu and Han, Xu and Zhang, Yudi and Sun, Ao and Huang, Yuxiang and Zhang, Kaihuo and Zhao, Weilun and Li, Yuxuan and Wang, Jianyong and others},
  journal={arXiv preprint arXiv:2502.14856},
  year={2025}
}

@article{zhang2025specmqaunt,
  title={Speculative Decoding Meets Quantization: Compatibility Evaluation and Hierarchical Framework Design},
  author={Zhang, Yudi and Zhao, Weilin and Han, Xu and Zhao, Tiejun and Xu, Wang and Cao, Hailong and Zhu, Conghui},
  journal={arXiv preprint arXiv:2505.22179},
  year={2025}
}

@article{minicpm4,
  title={MiniCPM4: Ultra-Efficient LLMs on End Devices},
  author={MiniCPM},
  year={2025}
}
```