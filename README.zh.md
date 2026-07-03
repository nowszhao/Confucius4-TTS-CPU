<div align="center">
    <h1>Confucius4-TTS-CPU</h1>
    <p><b>Confucius4-TTS 的 CPU 优化分支 — 无需 GPU。</b></p>
</div>

<div align="center">
    <a href="./README.md"><img src="https://img.shields.io/badge/README-EN-red"></a>
    &nbsp;&nbsp;
    <a href="https://github.com/netease-youdao/Confucius4-TTS"><img src="https://img.shields.io/badge/上游-Confucius4--TTS-blue"></a>
    &nbsp;&nbsp;
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
</div>

---

## 这是什么？

基于 [Confucius4-TTS](https://github.com/netease-youdao/Confucius4-TTS)（网易有道开源的多语种零样本 TTS 引擎）的社区分支，专门优化了 **CPU 推理和 Linux 部署**。

## 为什么要 Fork？

原项目推理需要 NVIDIA GPU + CUDA 12.6。其中 BigVGAN 声码器依赖自定义 CUDA kernel，必须安装 `nvcc` 和 `ninja` 才能编译运行。这导致无法在以下场景部署：

- 无 GPU 的 Linux 服务器（VPS、云虚拟机）
- CI/CD 流水线
- 低成本边缘设备
- 任何不方便安装 CUDA 工具链的环境

本分支移除了所有 CUDA 依赖，模型可**只用标准 PyTorch 在纯 CPU 上运行**。

## 安装

```bash
# 1. 克隆
git clone https://github.com/nowszhao/Confucius4-TTS-CPU.git
cd Confucius4-TTS-CPU

# 2. 创建虚拟环境
python3.10 -m venv venv && source venv/bin/activate

# 3. 安装 CPU 版 PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. 安装其他依赖
pip install transformers safetensors huggingface_hub PyYAML librosa soundfile \
    scipy tqdm matplotlib inflect jaconv pykakasi sentencepiece wetext
```

> 模型权重首次运行时自动从 HuggingFace 下载。分词器文件和 `wav2vec2bert_stats.pt` 已包含在 `checkpoints/` 目录中。

## 使用

### 命令行

```bash
python -m confuciustts.cli.run_inference \
    --prompt-wav reference.wav \
    --text "你好，这是一个测试。" \
    --lang zh \
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

## 性能

CPU 推理速度约为 **10 倍实时**。例如 3 秒的句子在普通 CPU 上大约需要 30 秒生成。音质与 GPU 推理**完全一致**，区别仅在于速度。

如果你有 GPU，安装 GPU 版 PyTorch（`pip install -r requirements.txt`）即可自动使用 GPU 加速。

## 上游项目

本项目为 [Confucius4-TTS](https://github.com/netease-youdao/Confucius4-TTS)（网易有道）的分支。原始功能、性能基准测试和训练指南请查看[上游仓库](https://github.com/netease-youdao/Confucius4-TTS)。
