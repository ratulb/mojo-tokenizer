"""
Vocabulary management for tokenizers.

This module handles the mapping between tokens (text) and their IDs,
as well as BPE merge rules that define how tokens combine.
"""


struct MergeRule(Copyable, Movable):
    """Represents a BPE merge rule: two tokens merge into one."""

    var first: String
    """First token in the merge pair."""

    var second: String
    """Second token in the merge pair."""

    var result: String
    """Result of merging first + second."""

    var rank: Int
    """Priority rank (lower = higher priority, applied first)."""

    def __init__(out self, first: String, second: String, rank: Int):
        """Create a new merge rule."""
        self.first = first
        self.second = second
        self.result = first + second
        self.rank = rank

    def __copyinit__(out self, copy: Self):
        self.first = copy.first
        self.second = copy.second
        self.result = copy.result
        self.rank = copy.rank

    def __moveinit__(out self, deinit take: Self):
        self.first = take.first^
        self.second = take.second^
        self.result = take.result^
        self.rank = take.rank


struct Vocabulary(Copyable, Movable):
    """
    Manages the vocabulary mapping between tokens and IDs.

    Stores both the forward mapping (token -> ID) and reverse mapping
    (ID -> token) for efficient lookup in both directions. Also stores
    BPE merge rules for the encoding process.

    Extended with backtracking support tables for O(n) BPE encoding:
    - split_table: How each token decomposes into (left, right)
    - pair_lookup: Which token pairs merge into which result
    - next_prefix_match: Shorter prefix token for backtracking
    """

    var _token_to_id: Dict[String, Int]
    """Map from token text to ID."""

    var _id_to_token: Dict[Int, String]
    """Map from ID to token text."""

    var _id_to_bytes: List[List[UInt8]]
    """Raw bytes for each token ID. Used for trie building to avoid UTF-8 encoding issues."""

    var _merges: Dict[String, Int]
    """Map from merged pair (as string) to rank."""

    var _size: Int
    """Number of tokens in vocabulary."""

    # Backtracking support tables (Phase B optimization)
    var _split_table: List[Tuple[Int, Int]]
    """For each token ID: (left_id, right_id) decomposition.
    If token is original (not merged), stores (id, id)."""

    var _pair_lookup: Dict[Int, Int]
    """Maps encoded pair (token1 * 1000000 + token2) -> merged_token_id.
    Using Int key instead of Tuple for Dict compatibility."""

    var _next_prefix_match: List[Int]
    """For each token ID: ID of next shorter prefix token, or -1 if none."""

    var _backtrack_tables_built: Bool
    """Whether the backtracking tables have been built."""

    def __init__(out self):
        """Create an empty vocabulary."""
        self._token_to_id = Dict[String, Int]()
        self._id_to_token = Dict[Int, String]()
        self._id_to_bytes = List[List[UInt8]]()
        self._merges = Dict[String, Int]()
        self._size = 0
        # Initialize backtracking tables (empty until built)
        self._split_table = List[Tuple[Int, Int]]()
        self._pair_lookup = Dict[Int, Int]()
        self._next_prefix_match = List[Int]()
        self._backtrack_tables_built = False

    def __copyinit__(out self, copy: Self):
        """Copy constructor."""
        self._token_to_id = copy._token_to_id.copy()
        self._id_to_token = copy._id_to_token.copy()
        self._id_to_bytes = copy._id_to_bytes.copy()
        self._merges = copy._merges.copy()
        self._size = copy._size
        # Copy backtracking tables
        self._split_table = copy._split_table.copy()
        self._pair_lookup = copy._pair_lookup.copy()
        self._next_prefix_match = copy._next_prefix_match.copy()
        self._backtrack_tables_built = copy._backtrack_tables_built

    def __moveinit__(out self, deinit take: Self):
        """Move constructor."""
        self._token_to_id = take._token_to_id^
        self._id_to_token = take._id_to_token^
        self._id_to_bytes = take._id_to_bytes^
        self._merges = take._merges^
        self._size = take._size
        # Move backtracking tables
        self._split_table = take._split_table^
        self._pair_lookup = take._pair_lookup^
        self._next_prefix_match = take._next_prefix_match^
        self._backtrack_tables_built = take._backtrack_tables_built

    def add_token(mut self, token: String, id: Int):
        """
        Add a token to the vocabulary.

        Args:
            token: The token text.
            id: The token ID.
        """
        self._token_to_id[token] = id
        self._id_to_token[id] = token
        self._size += 1
        # Ensure _id_to_bytes has space for this ID
        while len(self._id_to_bytes) <= id:
            self._id_to_bytes.append(List[UInt8]())

    def add_token_bytes(mut self, token: String, id: Int, raw_bytes: List[UInt8]):
        """
        Add a token to the vocabulary with raw bytes.

        This is used for tiktoken format where high bytes (>127) need to
        be stored exactly as they appear in the vocabulary file, not as
        UTF-8 encoded strings.

        Args:
            token: The token text (for string-based lookups).
            id: The token ID.
            raw_bytes: The raw bytes of the token (for trie building).
        """
        self._token_to_id[token] = id
        self._id_to_token[id] = token
        self._size += 1
        # Ensure _id_to_bytes has space for this ID
        while len(self._id_to_bytes) <= id:
            self._id_to_bytes.append(List[UInt8]())
        self._id_to_bytes[id] = raw_bytes.copy()

    def get_bytes(self, id: Int) -> List[UInt8]:
        """
        Get the raw bytes for a token ID.

        Args:
            id: The token ID.

        Returns:
            The raw bytes, or empty list if not found.
        """
        if id >= 0 and id < len(self._id_to_bytes):
            return self._id_to_bytes[id].copy()
        return List[UInt8]()

    def has_bytes(self, id: Int) -> Bool:
        """Check if a token ID has raw bytes stored."""
        return id >= 0 and id < len(self._id_to_bytes) and len(self._id_to_bytes[id]) > 0

    def add_merge(mut self, pair: String, rank: Int):
        """
        Add a BPE merge rule.

        Args:
            pair: The concatenated pair (first + second tokens).
            rank: The priority rank (lower = higher priority).
        """
        self._merges[pair] = rank

    def get_id(self, token: String) -> Int:
        """
        Get the ID for a token.

        Args:
            token: The token text.

        Returns:
            The token ID, or -1 if not found.
        """
        if token in self._token_to_id:
            try:
                return self._token_to_id[token]
            except:
                return -1
        return -1

    def get_text(self, id: Int) -> String:
        """
        Get the text for a token ID.

        Args:
            id: The token ID.

        Returns:
            The token text, or empty string if not found.
        """
        if id in self._id_to_token:
            try:
                return self._id_to_token[id]
            except:
                return ""
        return ""

    def get_merge_rank(self, pair: String) -> Int:
        """
        Get the merge rank for a token pair.

        Args:
            pair: The concatenated pair to look up.

        Returns:
            The merge rank, or -1 if no merge rule exists.
        """
        if pair in self._merges:
            try:
                return self._merges[pair]
            except:
                return -1
        return -1

    def has_token(self, token: String) -> Bool:
        """Check if a token exists in the vocabulary."""
        return token in self._token_to_id

    def has_id(self, id: Int) -> Bool:
        """Check if an ID exists in the vocabulary."""
        return id in self._id_to_token

    def size(self) -> Int:
        """Return the number of tokens in the vocabulary."""
        return self._size

    def clear(mut self):
        """Clear all tokens and merges from the vocabulary."""
        self._token_to_id = Dict[String, Int]()
        self._id_to_token = Dict[Int, String]()
        self._merges = Dict[String, Int]()
        self._size = 0
        # Clear backtracking tables
        self._split_table = List[Tuple[Int, Int]]()
        self._pair_lookup = Dict[Int, Int]()
        self._next_prefix_match = List[Int]()
        self._backtrack_tables_built = False

    # =========================================================================
    # Backtracking support methods (Phase B optimization)
    # =========================================================================

    def get_token_len(self, token_id: Int) -> Int:
        """Get the byte length of a token by its ID.

        Uses pre-stored raw bytes if available (tiktoken format),
        otherwise falls back to UTF-8 encoded string length.

        Args:
            token_id: The token ID.

        Returns:
            The byte length of the token, or 0 if not found.
        """
        if token_id >= 0 and token_id < len(self._id_to_bytes):
            return len(self._id_to_bytes[token_id])
        var text = self.get_text(token_id)
        if len(text) > 0:
            return len(text.as_bytes())
        return 0

    def has_backtrack_tables(self) -> Bool:
        """Check if backtracking tables have been built."""
        return self._backtrack_tables_built

    def set_split(mut self, token_id: Int, left: Int, right: Int):
        """
        Set the split decomposition for a token.

        For tokens formed by merging, stores (left_id, right_id).
        For original tokens, stores (token_id, token_id).

        Args:
            token_id: The token ID.
            left: Left component token ID.
            right: Right component token ID.
        """
        # Ensure list is large enough
        while len(self._split_table) <= token_id:
            # Default: token points to itself (original/unmerged)
            self._split_table.append((len(self._split_table), len(self._split_table)))
        self._split_table[token_id] = (left, right)

    def get_split(self, token_id: Int) -> Tuple[Int, Int]:
        """
        Get the split decomposition for a token.

        Args:
            token_id: The token ID.

        Returns:
            (left_id, right_id) decomposition. If token is original, returns (id, id).
        """
        if token_id < len(self._split_table):
            return self._split_table[token_id]
        # Default: token is original (not from merge)
        return (token_id, token_id)

    @staticmethod
    def _encode_pair(token1: Int, token2: Int) -> Int:
        """Encode a pair of tokens as a single Int for Dict lookup."""
        return token1 * 1000000 + token2

    def add_pair_lookup(mut self, token1: Int, token2: Int, merged: Int):
        """
        Add a pair lookup entry.

        Args:
            token1: First token ID.
            token2: Second token ID.
            merged: Resulting merged token ID.
        """
        var key = Self._encode_pair(token1, token2)
        self._pair_lookup[key] = merged

    def get_merged_token(self, token1: Int, token2: Int) -> Int:
        """
        Get the merged token for a pair, if it exists.

        Args:
            token1: First token ID.
            token2: Second token ID.

        Returns:
            Merged token ID, or -1 if no merge exists.
        """
        var key = Self._encode_pair(token1, token2)
        if key in self._pair_lookup:
            try:
                return self._pair_lookup[key]
            except:
                return -1
        return -1

    def set_next_prefix(mut self, token_id: Int, prefix_id: Int):
        """
        Set the next shorter prefix token for a token.

        Args:
            token_id: The token ID.
            prefix_id: ID of next shorter prefix token, or -1 if none.
        """
        # Ensure list is large enough
        while len(self._next_prefix_match) <= token_id:
            self._next_prefix_match.append(-1)
        self._next_prefix_match[token_id] = prefix_id

    def get_next_prefix(self, token_id: Int) -> Int:
        """
        Get the next shorter prefix token for a token.

        Used for backtracking when a token match fails validation.

        Args:
            token_id: The token ID.

        Returns:
            ID of next shorter prefix token, or -1 if none.
        """
        if token_id < len(self._next_prefix_match):
            return self._next_prefix_match[token_id]
        return -1

    def is_valid_token_pair(self, token1: Int, token2: Int) -> Bool:
        """
        Check if two adjacent tokens form a valid BPE pair.

        This is the key validation in the backtracking algorithm.
        Two tokens are valid neighbors if their merge would not have
        happened before either of them was formed.

        Based on rs-bpe's is_valid_token_pair() algorithm.

        Args:
            token1: First token ID.
            token2: Second token ID.

        Returns:
            True if the pair is valid (no premature merge would occur).
        """
        var t1 = token1
        var t2 = token2
        var limit = 0x7FFFFFFF  # Max Int as initial limit

        while True:
            # Check if (t1, t2) can merge
            var merged = self.get_merged_token(t1, t2)
            if merged >= 0 and merged < limit:
                # Would have merged before limit - invalid
                return False

            if t1 > t2:
                # Explore right side of t1
                limit = t1
                var split = self.get_split(t1)
                t1 = split[1]  # Right component
                if t1 == limit:
                    # t1 is original, switch to exploring t2
                    limit = t2 + 1
                    split = self.get_split(t2)
                    t2 = split[0]  # Left component
                    if t2 + 1 == limit:
                        # Both original - valid pair
                        return True
            else:
                # Explore left side of t2
                limit = t2 + 1
                var split = self.get_split(t2)
                t2 = split[0]  # Left component
                if t2 + 1 == limit:
                    # t2 is original, switch to exploring t1
                    limit = t1
                    split = self.get_split(t1)
                    t1 = split[1]  # Right component
                    if t1 == limit:
                        # Both original - valid pair
                        return True

    def mark_backtrack_tables_built(mut self):
        """Mark that backtracking tables have been built."""
        self._backtrack_tables_built = True

    def num_pairs(self) -> Int:
        """Return the number of pair lookups."""
        return len(self._pair_lookup)

    def build_backtrack_tables(mut self):
        """
        Build the backtracking tables from the vocabulary.

        This reverse-engineers the merge/split relationships from the
        token ordering. Must be called after all tokens are added.

        Algorithm (from rs-bpe):
        1. For each token (in order of ID):
           - Find if it can be formed by merging two smaller tokens
           - Record the split and add to pair_lookup
           - Build next_prefix_match for backtracking

        Performance: O(n * m) where n = vocab size, m = avg token length
        Memory: ~1MB additional for 50k token vocabulary
        """
        var n = self._size

        # Initialize tables
        self._split_table = List[Tuple[Int, Int]](capacity=n)
        self._next_prefix_match = List[Int](capacity=n)
        self._pair_lookup = Dict[Int, Int]()

        # For each token in order
        for token_id in range(n):
            # Use raw bytes if available (tiktoken), otherwise fall back to string
            var token_bytes: List[UInt8]
            if self.has_bytes(token_id):
                token_bytes = self.get_bytes(token_id)
            else:
                var token_text = self.get_text(token_id)
                var text_bytes = token_text.as_bytes()
                token_bytes = List[UInt8](capacity=len(text_bytes))
                for j in range(len(text_bytes)):
                    token_bytes.append(text_bytes[j])
            var token_len = len(token_bytes)

            # Default: token is original (points to itself)
            self._split_table.append((token_id, token_id))
            self._next_prefix_match.append(-1)

            # Find next prefix match (longest prefix that's also a token)
            for prefix_len in range(token_len - 1, 0, -1):
                # Get prefix bytes as string (using chr for vocabulary lookup)
                var prefix_str = String("")
                for j in range(prefix_len):
                    prefix_str = prefix_str + String(chr(Int(token_bytes[j])))

                var prefix_id = self.get_id(prefix_str)
                if prefix_id >= 0:
                    # Found a valid prefix token
                    # Note: We don't require prefix_id < token_id here because
                    # backtracking needs to find ANY shorter token, regardless of
                    # when it was merged during BPE training. For example, ' Spe'
                    # (12587) should be reachable from ' Spec' (11197) even though
                    # 12587 > 11197.
                    if self._next_prefix_match[token_id] < 0:
                        self._next_prefix_match[token_id] = prefix_id

                    # Check if remaining bytes form a valid token
                    var rest_str = String("")
                    for j in range(prefix_len, token_len):
                        rest_str = rest_str + String(chr(Int(token_bytes[j])))

                    var rest_id = self.get_id(rest_str)
                    if rest_id >= 0 and rest_id < token_id:
                        # Both parts are valid tokens with lower IDs
                        # Check if this pair is valid according to current tables
                        if self._is_valid_pair_for_building(prefix_id, rest_id):
                            # Record this split
                            self._split_table[token_id] = (prefix_id, rest_id)
                            var pair_key = Self._encode_pair(prefix_id, rest_id)
                            self._pair_lookup[pair_key] = token_id
                            break

        self._backtrack_tables_built = True

    def _is_valid_pair_for_building(self, token1: Int, token2: Int) -> Bool:
        """
        Check if a token pair is valid during table building.

        Simplified version for use during construction.
        Uses only the tables built so far.
        """
        var t1 = token1
        var t2 = token2
        var limit = 0x7FFFFFFF

        # Simple iteration limit to prevent infinite loops
        for _ in range(1000):
            var key = Self._encode_pair(t1, t2)
            if key in self._pair_lookup:
                try:
                    var merged = self._pair_lookup[key]
                    if merged < limit:
                        return False
                except:
                    pass

            if t1 > t2:
                limit = t1
                if t1 < len(self._split_table):
                    var split = self._split_table[t1]
                    var new_t1 = split[1]
                    if new_t1 == t1:
                        limit = t2 + 1
                        if t2 < len(self._split_table):
                            split = self._split_table[t2]
                            var new_t2 = split[0]
                            if new_t2 + 1 == limit:
                                return True
                            t2 = new_t2
                        else:
                            return True
                    else:
                        t1 = new_t1
                else:
                    return True
            else:
                limit = t2 + 1
                if t2 < len(self._split_table):
                    var split = self._split_table[t2]
                    var new_t2 = split[0]
                    if new_t2 + 1 == limit:
                        limit = t1
                        if t1 < len(self._split_table):
                            split = self._split_table[t1]
                            var new_t1 = split[1]
                            if new_t1 == limit:
                                return True
                            t1 = new_t1
                        else:
                            return True
                    else:
                        t2 = new_t2
                else:
                    return True

        # Iteration limit reached - assume valid
        return True
