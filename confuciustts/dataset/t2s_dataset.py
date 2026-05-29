import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer, SeamlessM4TFeatureExtractor


# Disable tokenizers parallelism to avoid deadlocks in DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Language token map for multilingual TTS
LANGUAGE_TOKEN_MAP = {
    # === East Asian Languages ===
    "zh": "请用中文朗读接下来的文字",  # Chinese
    "ja": "请用日语朗读接下来的文字",  # Japanese
    "ko": "请用韩语朗读接下来的文字",  # Korean
    
    # === Southeast Asian Languages ===
    "vi": "请用越南语朗读接下来的文字",  # Vietnamese
    "th": "请用泰语朗读接下来的文字",  # Thai
    "id": "请用印尼语朗读接下来的文字",  # Indonesian
    "ms": "请用马来语朗读接下来的文字",  # Malay
    "tl": "请用菲律宾语朗读接下来的文字",  # Tagalog/Filipino
    "my": "请用缅甸语朗读接下来的文字",  # Burmese
    "km": "请用高棉语朗读接下来的文字",  # Khmer
    "lo": "请用老挝语朗读接下来的文字",  # Lao
    
    # === South Asian Languages ===
    "hi": "请用印地语朗读接下来的文字",  # Hindi
    "bn": "请用孟加拉语朗读接下来的文字",  # Bengali
    "ta": "请用泰米尔语朗读接下来的文字",  # Tamil
    "te": "请用泰卢固语朗读接下来的文字",  # Telugu
    "mr": "请用马拉地语朗读接下来的文字",  # Marathi
    "gu": "请用古吉拉特语朗读接下来的文字",  # Gujarati
    "kn": "请用卡纳达语朗读接下来的文字",  # Kannada
    "ml": "请用马拉雅拉姆语朗读接下来的文字",  # Malayalam
    "pa": "请用旁遮普语朗读接下来的文字",  # Punjabi
    "ur": "请用乌尔都语朗读接下来的文字",  # Urdu
    "ne": "请用尼泊尔语朗读接下来的文字",  # Nepali
    "si": "请用僧伽罗语朗读接下来的文字",  # Sinhala
    
    # === Germanic Languages ===
    "en": "请用英文朗读接下来的文字",  # English
    "de": "请用德语朗读接下来的文字",  # German
    "nl": "请用荷兰语朗读接下来的文字",  # Dutch
    "sv": "请用瑞典语朗读接下来的文字",  # Swedish
    "da": "请用丹麦语朗读接下来的文字",  # Danish
    "no": "请用挪威语朗读接下来的文字",  # Norwegian
    "nb": "请用挪威语朗读接下来的文字",  # Norwegian Bokmål
    "nn": "请用挪威语朗读接下来的文字",  # Norwegian Nynorsk
    "is": "请用冰岛语朗读接下来的文字",  # Icelandic
    "af": "请用南非荷兰语朗读接下来的文字",  # Afrikaans
    "lb": "请用卢森堡语朗读接下来的文字",  # Luxembourgish
    "fy": "请用弗里斯兰语朗读接下来的文字",  # Frisian
    
    # === Romance Languages ===
    "fr": "请用法语朗读接下来的文字",  # French
    "es": "请用西班牙语朗读接下来的文字",  # Spanish
    "pt": "请用葡萄牙语朗读接下来的文字",  # Portuguese
    "it": "请用意大利语朗读接下来的文字",  # Italian
    "ro": "请用罗马尼亚语朗读接下来的文字",  # Romanian
    "ca": "请用加泰罗尼亚语朗读接下来的文字",  # Catalan
    "gl": "请用加利西亚语朗读接下来的文字",  # Galician
    "oc": "请用奥克语朗读接下来的文字",  # Occitan
    "la": "请用拉丁语朗读接下来的文字",  # Latin
    
    # === Slavic Languages ===
    "ru": "请用俄语朗读接下来的文字",  # Russian
    "uk": "请用乌克兰语朗读接下来的文字",  # Ukrainian
    "pl": "请用波兰语朗读接下来的文字",  # Polish
    "cs": "请用捷克语朗读接下来的文字",  # Czech
    "sk": "请用斯洛伐克语朗读接下来的文字",  # Slovak
    "bg": "请用保加利亚语朗读接下来的文字",  # Bulgarian
    "sr": "请用塞尔维亚语朗读接下来的文字",  # Serbian
    "hr": "请用克罗地亚语朗读接下来的文字",  # Croatian
    "sl": "请用斯洛文尼亚语朗读接下来的文字",  # Slovenian
    "mk": "请用马其顿语朗读接下来的文字",  # Macedonian
    "bs": "请用波斯尼亚语朗读接下来的文字",  # Bosnian
    "be": "请用白俄罗斯语朗读接下来的文字",  # Belarusian
    
    # === Baltic Languages ===
    "lt": "请用立陶宛语朗读接下来的文字",  # Lithuanian
    "lv": "请用拉脱维亚语朗读接下来的文字",  # Latvian
    
    # === Finno-Ugric Languages ===
    "fi": "请用芬兰语朗读接下来的文字",  # Finnish
    "et": "请用爱沙尼亚语朗读接下来的文字",  # Estonian
    "hu": "请用匈牙利语朗读接下来的文字",  # Hungarian
    
    # === Celtic Languages ===
    "ga": "请用爱尔兰语朗读接下来的文字",  # Irish
    "cy": "请用威尔士语朗读接下来的文字",  # Welsh
    "gd": "请用苏格兰盖尔语朗读接下来的文字",  # Scottish Gaelic
    "br": "请用布列塔尼语朗读接下来的文字",  # Breton
    
    # === Hellenic Languages ===
    "el": "请用希腊语朗读接下来的文字",  # Greek
    
    # === Albanian ===
    "sq": "请用阿尔巴尼亚语朗读接下来的文字",  # Albanian
    
    # === Basque ===
    "eu": "请用巴斯克语朗读接下来的文字",  # Basque
    
    # === Maltese ===
    "mt": "请用马耳他语朗读接下来的文字",  # Maltese
    
    # === Turkic Languages ===
    "tr": "请用土耳其语朗读接下来的文字",  # Turkish
    "az": "请用阿塞拜疆语朗读接下来的文字",  # Azerbaijani
    "kk": "请用哈萨克语朗读接下来的文字",  # Kazakh
    "uz": "请用乌兹别克语朗读接下来的文字",  # Uzbek
    "tk": "请用土库曼语朗读接下来的文字",  # Turkmen
    "ky": "请用吉尔吉斯语朗读接下来的文字",  # Kyrgyz
    "tt": "请用鞑靼语朗读接下来的文字",  # Tatar
    
    # === Semitic Languages ===
    "ar": "请用阿拉伯语朗读接下来的文字",  # Arabic
    "he": "请用希伯来语朗读接下来的文字",  # Hebrew
    "am": "请用阿姆哈拉语朗读接下来的文字",  # Amharic
    
    # === Iranian Languages ===
    "fa": "请用波斯语朗读接下来的文字",  # Persian/Farsi
    "ps": "请用普什图语朗读接下来的文字",  # Pashto
    "ku": "请用库尔德语朗读接下来的文字",  # Kurdish
    "tg": "请用塔吉克语朗读接下来的文字",  # Tajik
    
    # === Caucasian Languages ===
    "ka": "请用格鲁吉亚语朗读接下来的文字",  # Georgian
    "hy": "请用亚美尼亚语朗读接下来的文字",  # Armenian
    
    # === African Languages ===
    "sw": "请用斯瓦希里语朗读接下来的文字",  # Swahili
    "yo": "请用约鲁巴语朗读接下来的文字",  # Yoruba
    "ha": "请用豪萨语朗读接下来的文字",  # Hausa
    "ig": "请用伊博语朗读接下来的文字",  # Igbo
    "zu": "请用祖鲁语朗读接下来的文字",  # Zulu
    "xh": "请用科萨语朗读接下来的文字",  # Xhosa
    
    # === Mongolian ===
    "mn": "请用蒙古语朗读接下来的文字",  # Mongolian
    
    # === Other Languages ===
    "eo": "请用世界语朗读接下来的文字",  # Esperanto
}



# Audio length limits
MAX_AUDIO_DURATION_SEC = 30
MAX_PROMPT_AUDIO_DURATION_SEC = 15


class T2SDataset(Dataset):
    """TSV-driven dataset for Text-to-Semantic (T2S) autoregressive LM training.

    Reads one or more TSV files (columns: lang, wav_path, norm_text,
    semantic_ids_path, ref_audio_paths) and yields per-sample dicts containing
    tokenized text, pre-extracted semantic codes, and a reference audio clip for
    w2v-BERT speaker conditioning.
    """

    def __init__(
        self,
        data_path: List[str],
        tokenizer: AutoTokenizer,
        w2v_bert_path: str,
        max_text_seq_len: int = 600,
        max_semantic_seq_len: int = 750,
        sample_rate: int = 16000,
        semantic_pad_token: int = 8193,
        start_semantic_token: int = 8192,
        stop_semantic_token: int = 8193,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.max_text_seq_len = max_text_seq_len
        self.max_semantic_seq_len = max_semantic_seq_len
        self.semantic_pad_token = semantic_pad_token
        self.start_semantic_token = start_semantic_token
        self.stop_semantic_token = stop_semantic_token

        # Tokenizer
        self.tokenizer = tokenizer
        self.text_pad_token = getattr(tokenizer, "pad_token_id", 0) if tokenizer.pad_token_id is not None else 0
        self.vocab_size = len(tokenizer)

        # w2v-bert
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
            w2v_bert_path,
        )

        self.column_names = ["lang", "wav_path", "norm_text", "semantic_ids_path", "ref_audio_paths"]
        self.data_list = [self._load_data_file(p) for p in data_path]
        self.num_langs = len(self.data_list)

    def _load_data_file(self, path: str):
        try:
            return load_dataset(
                "csv",
                data_files=path,
                delimiter="\t",
                column_names=self.column_names,
                split="train",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {path}: {e}")

    def __len__(self) -> int:
        return max(len(data) for data in self.data_list) * self.num_langs

    def _get_random_sample(self) -> Dict:
        """Get a random valid sample (fallback for invalid samples)."""
        random_idx = random.randint(0, len(self) - 1)
        return self[random_idx]

    def _get_language_token(self, lang: str) -> Optional[str]:
        return LANGUAGE_TOKEN_MAP.get(lang, f"请用{lang}朗读接下来的文字")

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample by index.

        Returns:
            Dictionary containing text_inputs, semantic_codes, and condition_vector
        """
        try:
            lang_idx = idx % self.num_langs
            data = self.data_list[lang_idx]
            data_idx = (idx // self.num_langs) % len(data)

            sample = data[data_idx]
            lang = sample["lang"]
            norm_text = sample["norm_text"]
            semantic_ids_path = sample["semantic_ids_path"]
            ref_audio_paths_str = sample["ref_audio_paths"]

            semantic_codes = np.load(semantic_ids_path)
            semantic_len = len(semantic_codes)
            if semantic_len == 0 or semantic_len > self.max_semantic_seq_len:
                raise ValueError(f"Invalid semantic length: {semantic_len}")

            lang_token = self._get_language_token(lang)

            full_text = f"You are a helpful assistant. {lang_token}:{norm_text}"

            text_inputs = self.tokenizer.encode(full_text)

            if len(text_inputs) >= self.max_text_seq_len:
                raise ValueError(f"Invalid text length: {len(text_inputs)}")
            if max(text_inputs) >= self.vocab_size:
                raise ValueError(f"Invalid text token ids: max={max(text_inputs)}")

            ref_candidates = ref_audio_paths_str.split(",")
            ref_audio_path = random.choice(ref_candidates)

            prompt_audio, sr = torchaudio.load(ref_audio_path)

            if sr != self.sample_rate:
                prompt_audio = torchaudio.functional.resample(prompt_audio, sr, self.sample_rate)

            prompt_audio_len = prompt_audio.size(1)

            max_prompt_samples = MAX_PROMPT_AUDIO_DURATION_SEC * self.sample_rate
            if prompt_audio_len > max_prompt_samples:
                max_start = prompt_audio_len - max_prompt_samples
                start = random.randint(0, max_start)
                prompt_audio = prompt_audio[:, start:start + max_prompt_samples]
                prompt_audio_len = max_prompt_samples

            return {
                "text_inputs": text_inputs,
                "text_inputs_len": len(text_inputs),
                "semantic_codes": semantic_codes,
                "semantic_codes_len": semantic_len,
                "prompt_audio": prompt_audio,
                "prompt_audio_len": prompt_audio_len,
            }

        except Exception:
            # Return random valid sample on error
            return self._get_random_sample()

    def collate(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        IGNORE_INDEX = -100

        text_tokens = [torch.tensor(ex["text_inputs"], dtype=torch.long) for ex in examples]
        text_lengths = torch.tensor([ex["text_inputs_len"] for ex in examples], dtype=torch.long)

        text_inputs = pad_sequence(text_tokens, batch_first=True, padding_value=self.text_pad_token)

        semantic_inputs_list = []
        semantic_targets_list = []
        semantic_lengths_list = []

        for ex in examples:
            s_codes = torch.tensor(ex["semantic_codes"], dtype=torch.long)

            # input: [BOS, tokens..., EOS]
            si = torch.cat([
                torch.tensor([self.start_semantic_token]),
                s_codes,
                torch.tensor([self.stop_semantic_token])
            ])

            # target: [tokens..., EOS, IGNORE]
            st = torch.cat([
                s_codes,
                torch.tensor([self.stop_semantic_token, IGNORE_INDEX])
            ])

            semantic_inputs_list.append(si)
            semantic_targets_list.append(st)
            semantic_lengths_list.append(ex["semantic_codes_len"] + 2)

        semantic_lengths = torch.tensor(semantic_lengths_list, dtype=torch.long)
        semantic_codes = pad_sequence(semantic_inputs_list, batch_first=True, padding_value=self.semantic_pad_token)

        semantic_targets = pad_sequence(semantic_targets_list, batch_first=True, padding_value=IGNORE_INDEX)

        batch_size = len(examples)

        cond_mask = torch.ones(batch_size, 1, dtype=torch.bool)

        text_seq_range = torch.arange(text_inputs.shape[1]).unsqueeze(0)
        text_attn_mask = text_seq_range < text_lengths.unsqueeze(1)

        # Semantic attention mask
        semantic_seq_range = torch.arange(semantic_codes.shape[1]).unsqueeze(0)
        semantic_attn_mask = semantic_seq_range < semantic_lengths.unsqueeze(1)

        # full Attention Mask: [Condition, Text, Semantic]
        attention_mask = torch.cat([cond_mask, text_attn_mask, semantic_attn_mask], dim=1)

        spk_audio_list = [ex["prompt_audio"].squeeze().numpy() for ex in examples]
        spk_inputs = self.extract_features(
            spk_audio_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )

        return {
            "text_inputs": text_inputs,
            "text_lengths": text_lengths,
            "semantic_codes": semantic_codes,
            "semantic_targets": semantic_targets,
            "semantic_lengths": semantic_lengths,
            "spk_input_features": spk_inputs["input_features"],
            "attention_mask": attention_mask,
            "spk_attention_mask": spk_inputs["attention_mask"],
        }


class T2SDataModule(LightningDataModule):
    """
    Lightning DataModule for Text2Semantic training.
    """

    def __init__(
        self,
        train_data_path: List[str],
        val_data_path: Optional[List[str]],
        tokenizer,
        w2v_bert_path: str,
        batch_size: int = 16,
        num_workers: int = 4,
        max_text_seq_len: int = 600,
        max_semantic_seq_len: int = 750,
        sample_rate: int = 16000,
        semantic_pad_token: int = 8193,
        start_semantic_token: int = 8192,
        stop_semantic_token: int = 8193,
    ):
        """
        Initialize Text2Semantic data module.

        Args:
            train_data_path: List of paths to training TSV files
            val_data_path: List of paths to validation TSV files (optional)
            tokenizer: Text tokenizer
            w2v_bert_path: Path to w2v-bert-2.0 model directory
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            max_text_seq_len: Maximum text token length
            max_semantic_seq_len: Maximum semantic token length
            sample_rate: Audio sample rate (16kHz for semantic features)
            semantic_pad_token: Padding token for semantic codes
            start_semantic_token: Start token for semantic codes
            stop_semantic_token: Stop token for semantic codes
        """
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path or train_data_path
        self.tokenizer = tokenizer
        self.w2v_bert_path = w2v_bert_path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_text_seq_len = max_text_seq_len
        self.max_semantic_seq_len = max_semantic_seq_len
        self.sample_rate = sample_rate
        self.semantic_pad_token = semantic_pad_token
        self.start_semantic_token = start_semantic_token
        self.stop_semantic_token = stop_semantic_token

        # Datasets (initialized in setup)
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training/validation/testing.

        Args:
            stage: Current stage ("fit", "validate", "test", or None)
        """
        if stage == "fit" or stage is None:
            # Training dataset
            self.train_dataset = T2SDataset(
                data_path=self.train_data_path,
                tokenizer=self.tokenizer,
                w2v_bert_path=self.w2v_bert_path,
                max_text_seq_len=self.max_text_seq_len,
                max_semantic_seq_len=self.max_semantic_seq_len,
                sample_rate=self.sample_rate,
                semantic_pad_token=self.semantic_pad_token,
                start_semantic_token=self.start_semantic_token,
                stop_semantic_token=self.stop_semantic_token,
            )

            # Validation dataset
            self.val_dataset = T2SDataset(
                data_path=self.val_data_path,
                tokenizer=self.tokenizer,
                w2v_bert_path=self.w2v_bert_path,
                max_text_seq_len=self.max_text_seq_len,
                max_semantic_seq_len=self.max_semantic_seq_len,
                sample_rate=self.sample_rate,
                semantic_pad_token=self.semantic_pad_token,
                start_semantic_token=self.start_semantic_token,
                stop_semantic_token=self.stop_semantic_token,
            )

    def train_dataloader(self):
        """Create training dataloader."""
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate,
            drop_last=True,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        from torch.utils.data import DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate,
            drop_last=False,
        )
