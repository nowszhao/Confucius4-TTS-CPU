<div align="center">
    <img src="./resources/Confucius4-TTS.png" alt="Confucius4-TTS" width="35%">
    <h1>Confucius4-TTS: a Multilingual and Cross-Lingual Zero-Shot TTS Engine</h1>
    <p><b>One voice. Any language.</b></p>
</div>

<div align="center">
    <a href="./README.zh.md"><img src="https://img.shields.io/badge/README-中文版本-red"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://confucius4-tts.youdao.com/gradio"><img src="https://img.shields.io/badge/Demo-在线体验-purple"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://2901733926.github.io/Confucius4-TTS/"><img src="https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=GitHub&style=flat-square"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
</div>
<br>

📢 **Note:** Code and model weights are currently under preparation and will be released soon. In the meantime, you can try our online demo at [https://confucius4-tts.youdao.com/gradio](https://confucius4-tts.youdao.com/gradio).

Confucius4-TTS is an advanced LLM-based text-to-speech (TTS) system designed for multilingual and cross-lingual speech synthesis. Built on a speech encoder + large language model (LLM) architecture, Confucius4-TTS enables high-quality speech generation while preserving speaker identity across languages.

**✨ Key Features**

- **14 Languages Supported**: Chinese, English, Japanese, Korean, German, French, Spanish, Indonesian, Italian, Thai, Portuguese, Russian, Malay and Vietnamese *(more coming soon)*
- **Unconstrained Voice Cloning**: No reference transcript required
- **Cross-Lingual Voice Transfer**: Unaccented speech synthesis across 14 languages
- **Zero-Shot Voice Transfer**: Clone voices without additional training
- **Seamless Emotion Transfer**: Clone the feeling, not just the voice
- **Robust Generalization**: Stable performance in real-world multilingual scenarios

With strong cross-lingual generalization, Confucius4-TTS allows users to seamlessly switch languages while keeping the same voice, delivering fluent, natural, and expressive speech.

## Contents

- [Performance](#-performance)
- [Citation](#citation)

## 📊 Performance

Confucius4-TTS achieves competitive results on multilingual and cross-lingual zero-shot TTS benchmarks, with strong intelligibility and speaker similarity across multiple languages.

> Lower is better for WER/CER (↓), and higher is better for SIM (↑).

### CV3-eval Cross-lingual

<details>
<summary><b>CV3-eval Cross-lingual Results (click to expand)</b></summary>

| Direction | Metric | Confucius4-TTS | F5-TTS† | Spark-TTS | CosyVoice2† | CosyVoice3-0.5B† | CosyVoice3-0.5B + DiffRO† | CosyVoice3-1.5B† | CosyVoice3-1.5B + DiffRO† |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| en→zh | WER↓ | **6.71** | 11.60 | 12.40 | 13.50 | 8.48 | 5.16 | 8.01 | 5.09 |
| ja→zh | WER↓ | 4.93 | – | – | 48.10 | 6.86 | 3.22 | 6.78 | **3.05** |
| ko→zh | WER↓ | 1.46 | – | – | 7.70 | 5.24 | **1.03** | 3.30 | 1.06 |
| zh→en | WER↓ | **3.19** | 5.57 | 7.36 | 17.10 | 6.83 | 4.41 | 5.39 | 4.20 |
| ja→en | WER↓ | **3.44** | – | – | 11.20 | 5.86 | 4.78 | 5.94 | 4.19 |
| ko→en | WER↓ | **3.42** | – | – | 13.10 | 18.30 | 7.91 | 13.70 | 7.08 |

† Requires reference text.

</details>

### X-Voice Benchmark

<details>
<summary><b>X-Voice Cross-lingual Results (click to expand)</b></summary>

| Direction | Metric | Confucius4-TTS | X-Voice | OmniVoice† | IndexTTS2 |
|---|---|---:|---:|---:|---:|
| de→zh | WER↓ | **2.86** | 3.07 | 13.10 | 3.46 |
|  | SIM↑ | 0.569 | 0.516 | **0.691** | 0.544 |
| en→zh | WER↓ | 3.27 | **3.06** | 4.03 | 3.78 |
|  | SIM↑ | 0.504 | 0.443 | **0.544** | 0.485 |
| fr→zh | WER↓ | **2.74** | 3.01 | 18.10 | 3.53 |
|  | SIM↑ | 0.550 | 0.518 | **0.686** | 0.543 |
| ja→zh | WER↓ | 3.50 | **3.39** | 79.10 | 4.11 |
|  | SIM↑ | 0.637 | 0.629 | **0.709** | 0.650 |
| ko→zh | WER↓ | **2.86** | 3.13 | 11.88 | 2.90 |
|  | SIM↑ | 0.649 | 0.655 | **0.718** | 0.650 |
| th→zh | WER↓ | 2.87 | **2.79** | 3.30 | 3.08 |
|  | SIM↑ | 0.623 | 0.614 | **0.661** | 0.622 |
| vi→zh | WER↓ | **2.75** | 2.78 | 10.51 | 2.98 |
|  | SIM↑ | 0.640 | 0.641 | **0.701** | 0.641 |

† Requires reference text.

</details>

### Seed-TTS-eval

<details>
<summary><b>Seed-TTS-eval English & Chinese Zero-shot Results (click to expand)</b></summary>

| Language | Metric | Confucius4-TTS | Qwen3-TTS | FishAudio S2† | OmniVoice† | VoxCPM2† | X-Voice |
|---|---|---:|---:|---:|---:|---:|---:|
| English | WER↓ | 1.47 | 1.24 | **0.99** | 1.60 | 1.84 | 1.91 |
|  | SIM↑ | 0.702 | 0.714 | – | 0.741 | **0.753** | 0.627 |
| Chinese | CER↓ | 1.09 | 0.77 | **0.54** | 0.84 | 0.97 | 1.47 |
|  | SIM↑ | 0.749 | 0.770 | – | 0.777 | **0.795** | 0.746 |

† Requires reference text.

</details>

### MiniMax-Multilingual-Test

<details>
<summary><b>MiniMax-Multilingual-Test Results (click to expand)</b></summary>

| Language | Metric | Confucius4-TTS | ElevenLab | Qwen3-TTS | FishAudio S2† | OmniVoice† | VoxCPM2† | X-Voice |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| German | WER↓ | 0.89 | 0.57 | 1.24 | **0.55** | 0.96 | 0.68 | 2.00 |
|  | SIM↑ | 0.767 | 0.614 | 0.768 | 0.767 | **0.812** | 0.803 | 0.763 |
| French | WER↓ | 3.81 | 5.22 | **2.86** | 3.05 | 3.35 | 4.53 | 4.73 |
|  | SIM↑ | 0.697 | 0.535 | 0.716 | 0.698 | **0.801** | 0.735 | 0.746 |
| Indonesian | WER↓ | 1.79 | 1.06 | – | 1.46 | 1.97 | **1.08** | 1.47 |
|  | SIM↑ | 0.754 | 0.660 | – | 0.763 | **0.805** | 0.800 | 0.725 |
| Korean | WER↓ | 2.20 | 1.87 | 1.76 | **1.18** | 2.65 | 1.96 | 2.27 |
|  | SIM↑ | 0.790 | 0.700 | 0.790 | 0.817 | 0.828 | **0.833** | 0.788 |
| Thai | WER↓ | **2.42** | 73.94 | – | 4.23 | 3.98 | 2.96 | 4.71 |
|  | SIM↑ | 0.736 | 0.588 | – | 0.786 | **0.841** | 0.840 | 0.791 |
| Japanese | WER↓ | 4.26 | 10.65 | 3.82 | **2.76** | 4.03 | 4.63 | 7.13 |
|  | SIM↑ | 0.775 | 0.738 | 0.771 | 0.796 | **0.828** | **0.828** | 0.765 |
| Vietnamese | WER↓ | 1.99 | 73.42 | – | 7.41 | **1.37** | 3.31 | 1.40 |
|  | SIM↑ | 0.764 | 0.369 | – | 0.740 | **0.805** | 0.806 | 0.672 |
| Italian | WER↓ | 1.58 | 1.74 | 0.95 | 1.27 | 2.07 | 1.56 | 2.27 |
|  | SIM↑ | 0.764 | 0.579 | 0.752 | 0.747 | **0.812** | 0.780 | 0.780 |
| Portuguese | WER↓ | 2.04 | 1.33 | 1.53 | **1.14** | 2.51 | 1.94 | 2.61 |
|  | SIM↑ | 0.794 | 0.711 | 0.805 | 0.781 | **0.859** | 0.837 | 0.794 |
| Spanish | WER↓ | **0.95** | 1.08 | 1.13 | 0.91 | 1.03 | 1.44 | 2.91 |
|  | SIM↑ | 0.770 | 0.615 | 0.814 | 0.776 | 0.804 | **0.831** | 0.747 |
| Russian | WER↓ | 4.38 | 3.88 | 3.21 | 2.40 | **2.23** | 3.63 | 6.49 |
|  | SIM↑ | 0.790 | 0.675 | 0.784 | 0.790 | 0.783 | **0.811** | 0.799 |

† Requires reference text.

</details>

---

## Citation

If you find Confucius4-TTS useful in your research or project, please consider citing:

```bibtex
@misc{confucius4tts_2026,
  title        = {Confucius4-TTS: A Multilingual and Cross-Lingual Zero-Shot TTS Engine},
  author       = {{NetEase Youdao}},
  year         = {2026},
  howpublished = {\url{https://github.com/netease-youdao/Confucius4-TTS}},
  note         = {GitHub repository}
}
```
