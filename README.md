<div align="center">
    <h1>Confucius4-TTS-CPU</h1>
    <p><b>CPU-optimized fork of Confucius4-TTS — no GPU needed.</b></p>
</div>

<div align="center">
    <a href="./README.zh.md"><img src="https://img.shields.io/badge/README-中文版本-red"></a>
    &nbsp;&nbsp;
    <a href="https://github.com/netease-youdao/Confucius4-TTS"><img src="https://img.shields.io/badge/Upstream-Confucius4--TTS-blue"></a>
    &nbsp;&nbsp;
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
</div>

---

## What is this?

A community fork of [Confucius4-TTS](https://github.com/netease-youdao/Confucius4-TTS) (the multilingual zero-shot TTS engine by NetEase Youdao), optimized for **CPU-only inference and easy Linux deployment**.

## Why this fork?

The original project requires an NVIDIA GPU with CUDA 12.6 for inference. The BigVGAN vocoder relies on custom CUDA kernels that need `nvcc` and `ninja` to compile. This makes it difficult or impossible to deploy on:

- Linux servers without GPUs (VPS, cloud VMs)
- CI/CD pipelines
- Low-cost edge devices
- Any environment where installing CUDA toolchain is not practical

This fork removes all CUDA dependencies so the model runs on **pure CPU with standard PyTorch**.

## Installation

```bash
# 1. Clone
git clone https://github.com/nowszhao/Confucius4-TTS-CPU.git
cd Confucius4-TTS-CPU

# 2. Create venv
python3.10 -m venv venv && source venv/bin/activate

# 3. Install CPU PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install dependencies
pip install transformers safetensors huggingface_hub PyYAML librosa soundfile \
    scipy tqdm matplotlib inflect jaconv pykakasi sentencepiece wetext
```

> Model weights are auto-downloaded from HuggingFace on first run. Tokenizer files and `wav2vec2bert_stats.pt` are included in `checkpoints/`.

## Usage

### CLI

```bash
python -m confuciustts.cli.run_inference \
    --prompt-wav reference.wav \
    --text "Hello, this is a test." \
    --lang en \
    --output output.wav \
    --device cpu \
    --verbose
```

### Python API

```python
from confuciustts.cli.inference import ConfuciusTTS

model = ConfuciusTTS(
    config_path="config/inference_config.yaml",
    device="cpu",
)
audio = model.generate(
    text="你好，世界。",
    lang="zh",
    prompt_wav="reference.wav",
)
```

## Performance

CPU inference runs at approximately **10x real-time**. For example, a 3-second sentence takes ~30 seconds on a modern CPU core. Audio quality is **identical** to GPU inference — the only difference is speed.

If you have a GPU, install GPU PyTorch (`pip install -r requirements.txt`) and the model will use it automatically.

## Upstream

This project is a fork of [Confucius4-TTS](https://github.com/netease-youdao/Confucius4-TTS) by NetEase Youdao. For original features, benchmarks, and training guides, see the [upstream repo](https://github.com/netease-youdao/Confucius4-TTS).
