"""Tests for mojo-tokenizer library."""

from mojo_tokenizer import BPETokenizer, Token, Vocabulary, SpecialTokens


def test_token_creation() raises:
    """Test Token struct creation and properties."""
    var token = Token(id=42, text="hello")

    if token.id != 42:
        raise Error("Token ID should be 42")

    if token.text != "hello":
        raise Error("Token text should be 'hello'")

    if token.is_special:
        raise Error("Token should not be special by default")

    # Test special token
    var special = Token(id=100, text="<|endoftext|>", is_special=True)

    if not special.is_special:
        raise Error("Special token should be marked as special")

    print("test_token_creation PASSED")


def test_vocabulary_basic() raises:
    """Test basic vocabulary operations."""
    var vocab = Vocabulary()

    # Test empty vocab
    if vocab.size() != 0:
        raise Error("Empty vocab should have size 0")

    # Add tokens
    vocab.add_token("hello", 0)
    vocab.add_token("world", 1)

    if vocab.size() != 2:
        raise Error("Vocab should have 2 tokens")

    # Test lookups
    if vocab.get_id("hello") != 0:
        raise Error("'hello' should have ID 0")

    if vocab.get_id("world") != 1:
        raise Error("'world' should have ID 1")

    if vocab.get_id("unknown") != -1:
        raise Error("Unknown token should return -1")

    if vocab.get_text(0) != "hello":
        raise Error("ID 0 should be 'hello'")

    if vocab.get_text(999) != "":
        raise Error("Unknown ID should return empty string")

    print("test_vocabulary_basic PASSED")


def test_vocabulary_merges() raises:
    """Test BPE merge rule handling."""
    var vocab = Vocabulary()

    # Add merge rules
    vocab.add_merge("ab", 0)  # Highest priority
    vocab.add_merge("cd", 1)
    vocab.add_merge("abcd", 2)  # Lower priority

    # Test merge lookups
    if vocab.get_merge_rank("ab") != 0:
        raise Error("'ab' merge should have rank 0")

    if vocab.get_merge_rank("cd") != 1:
        raise Error("'cd' merge should have rank 1")

    if vocab.get_merge_rank("xy") != -1:
        raise Error("Unknown merge should return -1")

    print("test_vocabulary_merges PASSED")


def test_special_tokens() raises:
    """Test special token handling."""
    var special = SpecialTokens()

    # Test empty
    if special.size() != 0:
        raise Error("Empty special tokens should have size 0")

    # Add special tokens
    special.add("<|endoftext|>", 50256)
    special.add("<|im_start|>", 100264)
    special.add("<|im_end|>", 100265)

    if special.size() != 3:
        raise Error("Should have 3 special tokens")

    # Test lookups
    if special.get_id("<|endoftext|>") != 50256:
        raise Error("<|endoftext|> should have ID 50256")

    if special.get_text(100264) != "<|im_start|>":
        raise Error("ID 100264 should be <|im_start|>")

    if not special.is_special("<|endoftext|>"):
        raise Error("<|endoftext|> should be recognized as special")

    if special.is_special("hello"):
        raise Error("'hello' should not be special")

    print("test_special_tokens PASSED")


def test_special_token_splitting() raises:
    """Test splitting text on special tokens."""
    var special = SpecialTokens()
    special.add("<|endoftext|>", 50256)

    # Test split
    var segments = special.split_on_special("Hello<|endoftext|>World")

    if len(segments) != 3:
        raise Error("Should have 3 segments, got " + String(len(segments)))

    if segments[0].text != "Hello" or segments[0].is_special:
        raise Error("First segment should be 'Hello' (not special)")

    if segments[1].text != "<|endoftext|>" or not segments[1].is_special:
        raise Error("Second segment should be '<|endoftext|>' (special)")

    if segments[2].text != "World" or segments[2].is_special:
        raise Error("Third segment should be 'World' (not special)")

    print("test_special_token_splitting PASSED")


def test_bpe_tokenizer_creation() raises:
    """Test BPE tokenizer creation."""
    var tokenizer = BPETokenizer()

    # Empty tokenizer should have 0 vocab
    if tokenizer.vocab_size() != 0:
        raise Error("Empty tokenizer should have vocab_size 0")

    # Add a special token
    tokenizer.add_special_token("<|endoftext|>", 50256)

    if tokenizer.vocab_size() != 1:
        raise Error("Tokenizer with 1 special token should have vocab_size 1")

    print("test_bpe_tokenizer_creation PASSED")


def test_empty_encoding() raises:
    """Test encoding empty string."""
    var tokenizer = BPETokenizer()
    var tokens = tokenizer.encode("")

    if len(tokens) != 0:
        raise Error("Empty string should encode to empty list")

    print("test_empty_encoding PASSED")


def main() raises:
    """Run all tests."""
    print("Running mojo-tokenizer tests...\n")

    test_token_creation()
    test_vocabulary_basic()
    test_vocabulary_merges()
    test_special_tokens()
    test_special_token_splitting()
    test_bpe_tokenizer_creation()
    test_empty_encoding()

    print("\n All tests passed!")
