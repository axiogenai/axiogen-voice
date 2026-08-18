"""
splitter.py - Robust Sentence Segmentation & Incremental Buffer for Axiogen Voice TTS.

Handles:
- Sentence boundary detection (. ! ? ;)
- Decimal numbers (3.14) without false splitting
- Common abbreviations (Mr., Mrs., Dr., Prof., Inc., Ltd., etc.)
- Dialogue & quotes ("Hello!", 'Yes.')
- Newlines & variable whitespace
- Incremental token buffer for LLM streaming integration
"""

import re
from typing import List, Generator, Optional

# Known abbreviations that do not mark sentence boundaries
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "inc", "ltd", "corp",
    "co", "e.g", "i.e", "u.s", "u.k", "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "sept", "oct", "nov", "dec", "approx", "dept", "est", "govt"
}

# Regex to detect sentence boundaries
# Matches punctuation (. ! ? ; \n) followed by whitespace or end of string,
# while ignoring decimal numbers ($1.50, 3.14).
SENTENCE_SPLIT_REGEX = re.compile(
    r'(?<!\b(?:' + '|'.join(re.escape(a) for a in ABBREVIATIONS) + r')\.)'
    r'(?<!\d\.)'
    r'(?<![A-Z]\.)'
    r'([.!?;\n]+)'
    r'(?:\s+|$)',
    re.IGNORECASE
)

def split_sentences(text: str, max_words_per_chunk: int = 25) -> List[str]:
    """
    Splits text into coherent sentence units.
    If a sentence is unusually long (> max_words_per_chunk),
    sub-splits by commas or conjunctions for low-latency streaming.
    """
    if not text or not text.strip():
        return []

    raw = text.strip()
    sentences: List[str] = []
    
    # Split using boundary regex
    last_pos = 0
    for match in SENTENCE_SPLIT_REGEX.finditer(raw):
        end_pos = match.end()
        candidate = raw[last_pos:end_pos].strip()
        if candidate:
            sentences.append(candidate)
        last_pos = end_pos
        
    remainder = raw[last_pos:].strip()
    if remainder:
        sentences.append(remainder)

    # Sub-chunk overly long sentences (e.g. run-on sentences)
    final_chunks: List[str] = []
    for s in sentences:
        words = s.split()
        if len(words) <= max_words_per_chunk:
            final_chunks.append(s)
        else:
            # Split by comma or semicolon
            parts = re.split(r'([,;:])\s+', s)
            curr = ""
            for p in parts:
                if len((curr + " " + p).split()) > max_words_per_chunk and curr.strip():
                    final_chunks.append(curr.strip())
                    curr = p
                else:
                    curr = (curr + " " + p).strip() if curr else p
            if curr.strip():
                final_chunks.append(curr.strip())

    return [c for c in final_chunks if c]


class IncrementalSentenceBuffer:
    """
    Stateful buffer for real-time LLM token streaming into TTS.
    Accumulates incoming words/tokens and yields complete sentences as soon as boundaries appear.
    """
    def __init__(self, min_sentence_words: int = 3):
        self.buffer = ""
        self.min_sentence_words = min_sentence_words

    def add_token(self, token: str) -> List[str]:
        """Adds a token/chunk from LLM and returns newly completed sentences (if any)."""
        self.buffer += token
        return self._extract_ready_sentences()

    def _extract_ready_sentences(self) -> List[str]:
        ready = []
        while True:
            match = SENTENCE_SPLIT_REGEX.search(self.buffer)
            if not match:
                break
            
            end_idx = match.end()
            sentence = self.buffer[:end_idx].strip()
            # If valid sentence
            if sentence and len(sentence.split()) >= self.min_sentence_words:
                ready.append(sentence)
                self.buffer = self.buffer[end_idx:]
            else:
                # If too short (e.g. "Ok."), check if we have more text
                break

        return ready

    def flush(self) -> Optional[str]:
        """Flushes remaining text in buffer when stream ends."""
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining if remaining else None
