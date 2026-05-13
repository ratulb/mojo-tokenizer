"""
Basic usage example for mojo-tokenizer.

This example demonstrates:
1. Creating a BPE tokenizer
2. Adding special tokens
3. Encoding text to tokens
4. Decoding tokens back to text
"""

from mojo_tokenizer import BPETokenizer, Token


def main() raises:
    print("=== mojo-tokenizer Basic Usage ===\n")

    # Create a tokenizer
    print("1. Creating BPE tokenizer...")
    var tokenizer = BPETokenizer()

    # Add special tokens (typically these come from vocabulary files)
    print("2. Adding special tokens...")
    tokenizer.add_special_token("<|endoftext|>", 50256)
    tokenizer.add_special_token("<|im_start|>", 100264)
    tokenizer.add_special_token("<|im_end|>", 100265)

    print("   Vocabulary size:", tokenizer.vocab_size())

    # In a real application, you would load a vocabulary:
    # var tokenizer = BPETokenizer.from_tiktoken("cl100k_base.tiktoken")
    # var tokenizer = BPETokenizer.from_huggingface("tokenizer.json")

    print("\n3. Example Token struct usage:")
    var token = Token(id=42, text="hello")
    print("   Created token:", token)

    var special_token = Token(id=50256, text="<|endoftext|>", is_special=True)
    print("   Created special token:", special_token)

    print("\n=== Done! ===")
    print("\nNote: Full encoding/decoding requires loading a vocabulary file.")
    print("See README.md for instructions on loading tiktoken or HuggingFace tokenizers.")
