"""
BitField for O(1) bit operations in backtracking BPE.

Used to track reachable positions during backtracking encoding.
All bits are initialized to 1 (all positions reachable initially).

Ported from rs-bpe's bitfield.rs implementation.

Usage:
    var bf = BitField(100)  # 100 bits, all set to 1
    print(bf.is_set(50))    # True
    bf.clear(50)
    print(bf.is_set(50))    # False
    print(bf.successor(49)) # Next set bit after 49
"""


struct BitField(Movable, Copyable):
    """
    Bit field with efficient predecessor/successor queries.

    All bits initialized to 1. Supports:
    - O(1) is_set check
    - O(1) clear operation
    - O(1) amortized successor/predecessor (scans at most ~128 bits)
    """

    var words: List[UInt64]
    """Storage: 64 bits per word."""

    var _num_bits: Int
    """Total number of bits."""

    fn __init__(out self, bits: Int):
        """
        Create a bitfield with all bits set to 1.

        Args:
            bits: Number of bits to allocate.
        """
        self._num_bits = bits
        var num_words = (bits + 63) // 64
        self.words = List[UInt64](capacity=num_words)
        # Initialize all bits to 1 (all positions reachable)
        for _ in range(num_words):
            self.words.append(~UInt64(0))  # All 1s

    fn __copyinit__(out self, copy: Self):
        """Copy constructor."""
        self.words = copy.words.copy()
        self._num_bits = copy._num_bits

    fn __moveinit__(out self, deinit take: Self):
        """Move constructor."""
        self.words = take.words^
        self._num_bits = take._num_bits

    fn is_set(self, bit: Int) -> Bool:
        """
        Check if a bit is set.

        Args:
            bit: Bit position to check.

        Returns:
            True if bit is set (1), False otherwise.
        """
        var word_idx = bit // 64
        var bit_idx = bit % 64
        return (self.words[word_idx] & (UInt64(1) << UInt64(bit_idx))) != 0

    fn clear(mut self, bit: Int):
        """
        Clear a bit (set to 0).

        Args:
            bit: Bit position to clear.
        """
        var word_idx = bit // 64
        var bit_idx = bit % 64
        self.words[word_idx] &= ~(UInt64(1) << UInt64(bit_idx))

    fn set(mut self, bit: Int):
        """
        Set a bit (set to 1).

        Args:
            bit: Bit position to set.
        """
        var word_idx = bit // 64
        var bit_idx = bit % 64
        self.words[word_idx] |= UInt64(1) << UInt64(bit_idx)

    fn successor(self, bit: Int) -> Int:
        """
        Find the next set bit >= given position.

        This is used to find the next reachable position.
        Assumes there is always a successor (caller's responsibility).

        Args:
            bit: Starting position.

        Returns:
            Position of next set bit >= bit.
        """
        var word_idx = bit // 64
        var bit_idx = bit % 64

        # Check current word from bit_idx onwards
        var word = self.words[word_idx] >> UInt64(bit_idx)
        if word != 0:
            return _trailing_zeros(word) + bit

        # Scan subsequent words
        word_idx += 1
        while word_idx < len(self.words):
            var w = self.words[word_idx]
            if w != 0:
                return _trailing_zeros(w) + word_idx * 64
            word_idx += 1

        # Should not reach here if caller ensures successor exists
        return self._num_bits

    fn predecessor(self, bit: Int) -> Int:
        """
        Find the previous set bit <= given position.

        This is used to find the previous reachable position during backtracking.
        Assumes there is always a predecessor (caller's responsibility).

        Args:
            bit: Starting position.

        Returns:
            Position of previous set bit <= bit.
        """
        var word_idx = bit // 64
        var bit_idx = bit % 64

        # Check current word from bit_idx downwards
        # Shift left to put our bit at position 63
        var word = self.words[word_idx] << UInt64(63 - bit_idx)
        if word != 0:
            return bit - _leading_zeros(word)

        # Scan previous words
        while word_idx > 0:
            word_idx -= 1
            var w = self.words[word_idx]
            if w != 0:
                return word_idx * 64 + 63 - _leading_zeros(w)

        # Should not reach here if caller ensures predecessor exists
        return 0

    fn num_bits(self) -> Int:
        """Get total number of bits."""
        return self._num_bits


@always_inline
fn _trailing_zeros(x: UInt64) -> Int:
    """Count trailing zeros (position of lowest set bit)."""
    if x == 0:
        return 64
    var n = 0
    var v = x
    # Binary search for trailing zeros
    if (v & 0xFFFFFFFF) == 0:
        n += 32
        v >>= 32
    if (v & 0xFFFF) == 0:
        n += 16
        v >>= 16
    if (v & 0xFF) == 0:
        n += 8
        v >>= 8
    if (v & 0xF) == 0:
        n += 4
        v >>= 4
    if (v & 0x3) == 0:
        n += 2
        v >>= 2
    if (v & 0x1) == 0:
        n += 1
    return n


@always_inline
fn _leading_zeros(x: UInt64) -> Int:
    """Count leading zeros (64 - 1 - position of highest set bit)."""
    if x == 0:
        return 64
    var n = 0
    var v = x
    # Binary search for leading zeros
    if (v & 0xFFFFFFFF00000000) == 0:
        n += 32
        v <<= 32
    if (v & 0xFFFF000000000000) == 0:
        n += 16
        v <<= 16
    if (v & 0xFF00000000000000) == 0:
        n += 8
        v <<= 8
    if (v & 0xF000000000000000) == 0:
        n += 4
        v <<= 4
    if (v & 0xC000000000000000) == 0:
        n += 2
        v <<= 2
    if (v & 0x8000000000000000) == 0:
        n += 1
    return n
