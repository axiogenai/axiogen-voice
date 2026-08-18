"""
splitter.py - Fully Configurable Dynamic Sentence Segmentation & Incremental Stream Buffer.
Zero hardcoding: all patterns, delimiters, abbreviations, and thresholds are dynamically configurable.
"""

import os
import re
from typing import List, Optional, Set, Pattern
from config import CONFIG

# Default abbreviation set (can be extended or overridden via environment or runtime args)
DEFAULT_ABBREVIATIONS: Set[str] = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "inc", "ltd", "corp",
    "co", "e.g", "i.e", "u.s", "u.k", "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "sept", "oct", "nov", "dec", "approx", "dept", "est", "govt"
}

def build_boundary_regex(abbreviations: Optional[Set[str]] = None, custom_pattern: Optional[str] = None) -> Pattern:
    """Dynamically compiles a regex pattern based on provided or default abbreviations."""
    if custom_pattern:
        return re.compile(custom_pattern, re.IGNORECASE)

    abbr_set = abbreviations if abbreviations is not None else DEFAULT_ABBREVIATIONS
    abbr_pattern = '|'.join(re.escape(a) for a in sorted(abbr_set, key=len, reverse=True))
    
    # Pattern detects sentence end while preventing splits on abbreviations, decimals, or initials
    pattern = (
        r'(?<!\b(?:' + abbr_pattern + r')\.)'
        r'(?<!\d\.)'
        r'(?<![A-Z]\.)'
        r'([.!?;\n]+)'
        r'(?:\s+|$)'
    )
    return re.compile(pattern, re.IGNORECASE)

# Module-level default compiled regex
_DEFAULT_SPLIT_REGEX = build_boundary_regex()

def split_sentences(
    text: str,
    max_words_per_chunk: Optional[int] = None,
    abbreviations: Optional[Set[str]] = None,
    regex_pattern: Optional[str] = None
) -> List[str]:
    """
    Dynamically splits text into natural sentences.
    All thresholds (max words, abbreviation lists, regex patterns) are runtime-configurable.
    """
    if not text or not text.strip():
        return []

    max_words = max_words_per_chunk or CONFIG.max_sentence_words
    compiled_regex = build_boundary_regex(abbreviations, regex_pattern) if (abbreviations or regex_pattern) else _DEFAULT_SPLIT_REGEX

    raw = text.strip()
    sentences: List[str] = []
    
    last_pos = 0
    for match in compiled_regex.finditer(raw):
        end_pos = match.end()
        candidate = raw[last_pos:end_pos].strip()
        if candidate:
            sentences.append(candidate)
        last_pos = end_pos
        
    remainder = raw[last_pos:].strip()
    if remainder:
        sentences.append(remainder)

    # Sub-chunk overly long run-on sentences
    final_chunks: List[str] = []
    for s in sentences:
        words = s.split()
        if len(words) <= max_words:
            final_chunks.append(s)
        else:
            # Sub-split long sentences by clause punctuation (, ; :) or space chunks
            parts = re.split(r'([,;:])\s+', s)
            curr = ""
            for p in parts:
                if len((curr + " " + p).split()) > max_words and curr.strip():
                    final_chunks.append(curr.strip())
                    curr = p
                else:
                    curr = (curr + " " + p).strip() if curr else p
            if curr.strip():
                final_chunks.append(curr.strip())

    return [c for c in final_chunks if c]


class IncrementalSentenceBuffer:
    """
    Dynamic token buffer for real-time LLM streaming into TTS.
    Configurable minimum word limit, boundary regex, and flush thresholds.
    """
    def __init__(
        self,
        min_sentence_words: int = 3,
        abbreviations: Optional[Set[str]] = None,
        custom_pattern: Optional[str] = None
    ):
        self.buffer = ""
        self.min_sentence_words = min_sentence_words
        self.split_regex = build_boundary_regex(abbreviations, custom_pattern) if (abbreviations or custom_pattern) else _DEFAULT_SPLIT_REGEX

    def add_token(self, token: str) -> List[str]:
        """Adds incoming token from LLM stream and extracts ready sentences dynamically."""
        self.buffer += token
        return self._extract_ready_sentences()

    def _extract_ready_sentences(self) -> List[str]:
        ready = []
        while True:
            match = self.split_regex.search(self.buffer)
            if not match:
                break
            
            end_idx = match.end()
            sentence = self.buffer[:end_idx].strip()
            if sentence and len(sentence.split()) >= self.min_sentence_words:
                ready.append(sentence)
                self.buffer = self.buffer[end_idx:]
            else:
                break

        return ready

    def flush(self) -> Optional[str]:
        """Flushes any remaining text when LLM completes generation."""
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining if remaining else None
