"""
Byte Trie for direct token lookup.

Phase 2 optimization: Skip BPE iteration for tokens that exist directly
in the vocabulary. Most tokens are short (1-4 bytes) and can be looked
up in O(n) time via trie traversal instead of O(n²) BPE iteration.

For a word like "the" (3 bytes → 1 token):
- BPE approach: Multiple merge iterations to find final token
- Trie approach: Single traversal, direct token ID

Expected gain: 2-3x for common words (skip BPE for ~80% of tokens).
"""


struct TrieNode(Movable, Copyable):
    """A node in the byte trie."""

    var children: List[Int]
    """Child node indices (256 entries, -1 = no child)."""

    var token_id: Int
    """Token ID if this node completes a token, -1 otherwise."""

    var is_terminal: Bool
    """Whether this node represents a complete token."""

    fn __init__(out self):
        """Create an empty trie node."""
        self.children = List[Int](capacity=256)
        for _ in range(256):
            self.children.append(-1)
        self.token_id = -1
        self.is_terminal = False

    fn __copyinit__(out self, copy: Self):
        """Copy constructor."""
        self.children = copy.children.copy()
        self.token_id = copy.token_id
        self.is_terminal = copy.is_terminal

    fn __moveinit__(out self, deinit take: Self):
        """Move constructor."""
        self.children = take.children^
        self.token_id = take.token_id
        self.is_terminal = take.is_terminal


struct ByteTrie(Movable, Copyable):
    """
    Byte trie for direct byte sequence → token ID lookup.

    Provides O(n) lookup where n is the byte sequence length,
    compared to O(n²) for BPE iteration on cache misses.

    Usage:
        var trie = ByteTrie()
        trie.insert(token_bytes, token_id)
        var result = trie.lookup(input_bytes)
        if result.found:
            # Use result.token_id directly, skip BPE
    """

    var nodes: List[TrieNode]
    """All trie nodes. Index 0 is always the root."""

    var _size: Int
    """Number of tokens in the trie."""

    fn __init__(out self):
        """Create an empty byte trie."""
        self.nodes = List[TrieNode]()
        # Create root node
        self.nodes.append(TrieNode())
        self._size = 0

    fn __copyinit__(out self, copy: Self):
        """Copy constructor."""
        self.nodes = copy.nodes.copy()
        self._size = copy._size

    fn __moveinit__(out self, deinit take: Self):
        """Move constructor."""
        self.nodes = take.nodes^
        self._size = take._size

    fn insert(mut self, token_bytes: List[UInt8], token_id: Int):
        """
        Insert a token into the trie.

        Args:
            token_bytes: The byte sequence representing the token.
            token_id: The token ID to associate with this sequence.
        """
        var node_idx = 0  # Start at root

        for i in range(len(token_bytes)):
            var byte_val = Int(token_bytes[i])

            # Get or create child node
            var child_idx = self.nodes[node_idx].children[byte_val]
            if child_idx < 0:
                # Create new child node
                child_idx = len(self.nodes)
                self.nodes.append(TrieNode())
                self.nodes[node_idx].children[byte_val] = child_idx

            node_idx = child_idx

        # Mark this node as a terminal with the token ID
        self.nodes[node_idx].token_id = token_id
        self.nodes[node_idx].is_terminal = True
        self._size += 1

    fn insert_string(mut self, token: String, token_id: Int):
        """
        Insert a token string into the trie.

        Args:
            token: The token string.
            token_id: The token ID to associate with this token.
        """
        var bytes = token.as_bytes()
        var byte_list = List[UInt8](capacity=len(bytes))
        for i in range(len(bytes)):
            byte_list.append(bytes[i])
        self.insert(byte_list, token_id)

    fn lookup(self, input_bytes: List[UInt8]) -> TrieLookupResult:
        """
        Look up a byte sequence in the trie.

        Returns the longest matching token starting at position 0.

        Args:
            input_bytes: The byte sequence to look up.

        Returns:
            TrieLookupResult with found status, token_id, and match length.
        """
        var node_idx = 0  # Start at root
        var last_match_id = -1
        var last_match_len = 0

        for i in range(len(input_bytes)):
            var byte_val = Int(input_bytes[i])
            var child_idx = self.nodes[node_idx].children[byte_val]

            if child_idx < 0:
                # No more matches possible
                break

            node_idx = child_idx

            # Check if this is a complete token
            if self.nodes[node_idx].is_terminal:
                last_match_id = self.nodes[node_idx].token_id
                last_match_len = i + 1

        return TrieLookupResult(
            last_match_id >= 0,
            last_match_id,
            last_match_len
        )

    fn lookup_at_offset(self, input_bytes: List[UInt8], offset: Int) -> TrieLookupResult:
        """
        Look up a byte sequence starting at offset (zero-copy).

        This is the fast path for encoding - avoids allocating a new list
        for each lookup position.

        Args:
            input_bytes: The full byte sequence.
            offset: Starting position for lookup.

        Returns:
            TrieLookupResult with found status, token_id, and match length.
        """
        var node_idx = 0  # Start at root
        var last_match_id = -1
        var last_match_len = 0

        for i in range(offset, len(input_bytes)):
            var byte_val = Int(input_bytes[i])
            var child_idx = self.nodes[node_idx].children[byte_val]

            if child_idx < 0:
                # No more matches possible
                break

            node_idx = child_idx

            # Check if this is a complete token
            if self.nodes[node_idx].is_terminal:
                last_match_id = self.nodes[node_idx].token_id
                last_match_len = i - offset + 1

        return TrieLookupResult(
            last_match_id >= 0,
            last_match_id,
            last_match_len
        )

    fn lookup_exact(self, input_bytes: List[UInt8]) -> Int:
        """
        Look up an exact byte sequence match.

        Args:
            input_bytes: The byte sequence to look up.

        Returns:
            Token ID if exact match found, -1 otherwise.
        """
        var node_idx = 0  # Start at root

        for i in range(len(input_bytes)):
            var byte_val = Int(input_bytes[i])
            var child_idx = self.nodes[node_idx].children[byte_val]

            if child_idx < 0:
                return -1  # No match

            node_idx = child_idx

        # Check if we ended on a terminal node
        if self.nodes[node_idx].is_terminal:
            return self.nodes[node_idx].token_id
        return -1

    fn greedy_tokenize(self, input_bytes: List[UInt8]) -> List[Int]:
        """
        Greedily tokenize a byte sequence using the trie.

        Uses longest-match-first strategy. For bytes not in the trie,
        returns -1 to indicate BPE fallback is needed.

        Args:
            input_bytes: The byte sequence to tokenize.

        Returns:
            List of token IDs (-1 for bytes needing BPE fallback).
        """
        var result = List[Int]()
        var pos = 0
        var n = len(input_bytes)

        while pos < n:
            # Try to find longest match starting at pos
            var node_idx = 0
            var last_match_id = -1
            var last_match_len = 0

            var i = pos
            while i < n:
                var byte_val = Int(input_bytes[i])
                var child_idx = self.nodes[node_idx].children[byte_val]

                if child_idx < 0:
                    break

                node_idx = child_idx

                if self.nodes[node_idx].is_terminal:
                    last_match_id = self.nodes[node_idx].token_id
                    last_match_len = i - pos + 1

                i += 1

            if last_match_id >= 0:
                # Found a match
                result.append(last_match_id)
                pos += last_match_len
            else:
                # No match - need BPE fallback for this byte
                result.append(-1)
                pos += 1

        return result^

    fn size(self) -> Int:
        """Get number of tokens in the trie."""
        return self._size

    fn node_count(self) -> Int:
        """Get total number of nodes in the trie."""
        return len(self.nodes)


struct TrieLookupResult(Movable, Copyable):
    """Result of a trie lookup operation."""

    var found: Bool
    """Whether a match was found."""

    var token_id: Int
    """The token ID if found, -1 otherwise."""

    var match_length: Int
    """Number of bytes matched."""

    fn __init__(out self, found: Bool, token_id: Int, match_length: Int):
        """Create a lookup result."""
        self.found = found
        self.token_id = token_id
        self.match_length = match_length

    fn __copyinit__(out self, copy: Self):
        """Copy constructor."""
        self.found = copy.found
        self.token_id = copy.token_id
        self.match_length = copy.match_length

    fn __moveinit__(out self, deinit take: Self):
        """Move constructor."""
        self.found = take.found
        self.token_id = take.token_id
        self.match_length = take.match_length
