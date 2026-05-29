"""
this code is modified from https://github.com/FunAudioLLM/CosyVoice
"""

import re
import regex
import inflect
from typing import List, Callable


class TextNormalizer:
    """
    Multilingual text normalizer.

    Supports Chinese, English, and other languages with appropriate
    normalization strategies for each.
    """

    def __init__(self):
        self.inflect_parser = inflect.engine()

        # Regex patterns
        self.chinese_pattern = re.compile(r'[一-鿿]+')
        self.punctuation_pattern = r'^[\p{P}\p{S}]*$'

        # Try to load WeText for advanced normalization
        self.zh_normalizer = None
        self.en_normalizer = None
        try:
            from wetext import Normalizer
            self.zh_normalizer = Normalizer(remove_erhua=False)
            self.en_normalizer = Normalizer()
        except ImportError:
            pass

    def contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return bool(self.chinese_pattern.search(text))

    def is_only_punctuation(self, text: str) -> bool:
        """Check if text contains only punctuation."""
        return bool(regex.fullmatch(self.punctuation_pattern, text))

    def replace_corner_marks(self, text: str) -> str:
        """Replace special corner mark symbols."""
        text = text.replace('²', '平方')
        text = text.replace('³', '立方')
        return text

    def remove_brackets(self, text: str) -> str:
        """Remove meaningless brackets and symbols."""
        text = text.replace('（', '').replace('）', '')
        text = text.replace('【', '').replace('】', '')
        text = text.replace('`', '').replace('`', '')
        text = text.replace("——", " ")
        return text

    def remove_blank_between_chinese(self, text: str) -> str:
        """Remove blank spaces between Chinese characters."""
        out_str = []
        for i, c in enumerate(text):
            if c == " ":
                # Keep space only between ASCII characters
                if i > 0 and i < len(text) - 1:
                    if (text[i + 1].isascii() and text[i + 1] != " " and
                        text[i - 1].isascii() and text[i - 1] != " "):
                        out_str.append(c)
            else:
                out_str.append(c)
        return "".join(out_str)

    def spell_out_numbers(self, text: str) -> str:
        """Spell out Arabic numerals to words (for English)."""
        new_text = []
        start = None

        for i, c in enumerate(text):
            if not c.isdigit():
                if start is not None:
                    num_str = self.inflect_parser.number_to_words(text[start:i])
                    new_text.append(num_str)
                    start = None
                new_text.append(c)
            else:
                if start is None:
                    start = i

        if start is not None and start < len(text):
            num_str = self.inflect_parser.number_to_words(text[start:])
            new_text.append(num_str)

        return ''.join(new_text)

    def normalize_chinese(self, text: str) -> str:
        """Normalize Chinese text."""
        # Use WeText if available
        if self.zh_normalizer is not None:
            text = self.zh_normalizer.normalize(text)

        # Remove spaces between Chinese characters
        text = self.remove_blank_between_chinese(text)

        # Replace corner marks
        text = self.replace_corner_marks(text)

        # Normalize punctuation
        text = text.replace(".", "。")
        text = text.replace(" - ", "，")

        # Remove brackets
        text = self.remove_brackets(text)

        # Replace trailing commas with period
        text = re.sub(r'[，,、]+$', '。', text)

        return text

    def normalize_english(self, text: str) -> str:
        """Normalize English text."""
        # Use WeText if available
        if self.en_normalizer is not None:
            text = self.en_normalizer.normalize(text)

        # Spell out numbers
        text = self.spell_out_numbers(text)

        return text

    def normalize(self, text: str, language: str = "auto") -> str:
        """
        Normalize text based on language.

        Args:
            text: Input text
            language: Language code ("zh", "en", "auto")

        Returns:
            Normalized text
        """
        if not text:
            return text

        # Clean whitespace
        text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Auto-detect language if needed
        if language == "auto":
            language = "zh" if self.contains_chinese(text) else "en"

        # Apply language-specific normalization
        if language == "zh":
            text = self.normalize_chinese(text)
        else:
            text = self.normalize_english(text)

        return text

    def segment_text(
        self,
        text: str,
        tokenize_fn: Callable,
        language: str = "zh",
        max_tokens: int = 80,
        min_tokens: int = 60,
        merge_threshold: int = 20,
        split_on_comma: bool = False,
    ) -> List[str]:
        """
        Segment text into sentences based on punctuation and token length.

        Args:
            text: Input text
            tokenize_fn: Function to tokenize text
            language: Language code
            max_tokens: Maximum tokens per segment
            min_tokens: Minimum tokens per segment
            merge_threshold: Merge segments shorter than this
            split_on_comma: Whether to split on commas

        Returns:
            List of text segments
        """
        def calc_length(t: str) -> int:
            if language == "zh":
                return len(t)
            else:
                return len(tokenize_fn(t))

        def should_merge(t: str) -> bool:
            if language == "zh":
                return len(t) < merge_threshold
            else:
                return len(tokenize_fn(t)) < merge_threshold

        # Define punctuation marks
        if language == "zh":
            punctuation = ['。', '？', '！', '；', '：', '.', '?', '!', ';']
        else:
            punctuation = ['.', '?', '!', ';', ':']

        if split_on_comma:
            punctuation.extend(['，', ','])

        # Add period if missing
        if text and text[-1] not in punctuation:
            text += "。" if language == "zh" else "."

        # Split by punctuation
        segments = []
        start = 0

        for i, char in enumerate(text):
            if char in punctuation:
                if len(text[start:i]) > 0:
                    segment = text[start:i] + char

                    # Handle quotation marks
                    if i + 1 < len(text) and text[i + 1] in ['"', '"']:
                        segment += text[i + 1]
                        start = i + 2
                    else:
                        start = i + 1

                    segments.append(segment)

        # Handle single long segment
        if len(segments) == 1 and calc_length(segments[0]) > max_tokens:
            long_text = segments[0][:-1]  # Remove added punctuation
            segments = []
            for i in range(0, len(long_text), max_tokens):
                chunk = long_text[i:i + max_tokens]
                segments.append(chunk)

        # Merge short segments
        final_segments = []
        current = ""

        for seg in segments:
            if calc_length(current + seg) > max_tokens and calc_length(current) > min_tokens:
                final_segments.append(current)
                current = ""
            current = current + seg

        if current:
            if should_merge(current) and final_segments:
                final_segments[-1] = final_segments[-1] + current
            else:
                final_segments.append(current)

        # Clean up Chinese segments
        if language == "zh":
            final_segments = [
                seg[:-1] if seg and seg[-1] in ['。', '；', '：', '.', ';'] else seg
                for seg in final_segments
            ]

        # Filter out punctuation-only segments
        final_segments = [s for s in final_segments if not self.is_only_punctuation(s)]

        return final_segments
