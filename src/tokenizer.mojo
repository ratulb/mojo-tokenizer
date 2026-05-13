"""
Core tokenizer trait and types.

This module defines the Tokenizer trait that all tokenizer implementations
must conform to, plus the Token type for representing individual tokens.
"""


struct Token:
    """Represents a single token with its ID and text representation."""

    var id: Int
    """The numeric token ID."""

    var text: String
    """The text representation of the token."""

    var is_special: Bool
    """Whether this is a special token (e.g., [CLS], [SEP], <|endoftext|>)."""

    def __init__(out self, id: Int, text: String, is_special: Bool = False):
        """Create a new Token.

        Args:
            id: The numeric token ID.
            text: The text representation.
            is_special: Whether this is a special token.
        """
        self.id = id
        self.text = text
        self.is_special = is_special

    def __str__(self) -> String:
        """Return string representation of the token."""
        if self.is_special:
            return "Token(id=" + String(self.id) + ", text='" + self.text + "', special=True)"
        return "Token(id=" + String(self.id) + ", text='" + self.text + "')"


trait Tokenizer:
    """
    Base trait for all tokenizer implementations.

    All tokenizers must implement encode, decode, and vocab_size methods.
    This enables polymorphic tokenizer usage across different algorithms
    (BPE, WordPiece, SentencePiece, etc.).
    """

    def encode(mut self, text: String) raises -> List[Int]:
        """
        Encode text into a list of token IDs.

        Args:
            text: The input text to tokenize.

        Returns:
            A list of integer token IDs.

        Raises:
            Error if encoding fails (e.g., unknown characters).

        Note:
            Uses mut self to allow internal caching of token lookups.
        """
        ...

    def decode(self, tokens: List[Int]) raises -> String:
        """
        Decode a list of token IDs back into text.

        Args:
            tokens: The list of token IDs to decode.

        Returns:
            The reconstructed text string.

        Raises:
            Error if decoding fails (e.g., invalid token ID).
        """
        ...

    def vocab_size(self) -> Int:
        """
        Return the vocabulary size.

        Returns:
            The total number of tokens in the vocabulary.
        """
        ...
