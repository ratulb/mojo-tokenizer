"""
Mojo Tokenizer Library

Fast, pure Mojo tokenization for LLM inference pipelines.
Supports BPE encoding with tiktoken and HuggingFace format compatibility.

Basic Usage:
    from mojo_tokenizer import BPETokenizer

    # Load a tiktoken vocabulary
    var tokenizer = BPETokenizer.from_tiktoken("path/to/vocab.tiktoken")

    # Encode text to tokens
    var tokens = tokenizer.encode("Hello, world!")

    # Decode tokens back to text
    var text = tokenizer.decode(tokens)

    # Check cache performance
    print("Cache hit rate:", tokenizer.cache_hit_rate())

Chat Templates:
    from mojo_tokenizer.chat import ChatMessage, llama3_template, apply_chat_template

    var messages = List[ChatMessage]()
    messages.append(ChatMessage.system("You are a helpful assistant."))
    messages.append(ChatMessage.user("Hello!"))

    var formatted = apply_chat_template(messages, llama3_template())

Features:
    - Pure Mojo implementation (no Python dependencies)
    - BPE (Byte Pair Encoding) algorithm
    - Tiktoken format support (OpenAI compatible)
    - HuggingFace tokenizer.json support
    - Special token handling
    - Batch encoding/decoding
    - Word-level LRU caching (80%+ hit rate)
    - SIMD-optimized string operations
    - Word-level LRU caching (80%+ hit rate)
    - SIMD-optimized string operations

Performance:
    - 100k+ tokens/sec on M3 Ultra
    - <100ms vocabulary loading
    - Move semantics for zero-copy operations

Part of mojo-contrib: https://github.com/atsentia/mojo-contrib
"""

# Core types
from .tokenizer import Tokenizer, Token

# BPE implementation
from .bpe import BPETokenizer

# Vocabulary management
from .vocab import Vocabulary, MergeRule

# Special tokens
from .special_tokens import SpecialTokens, SpecialToken

# Format loaders
from .formats import load_tiktoken, load_huggingface

# Caching
from .cache import TokenCache, MergeCache

# Phase 2: Byte trie for direct lookup
from .byte_trie import ByteTrie, TrieNode, TrieLookupResult
