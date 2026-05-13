"""
Simple JSON parser for tokenizer files.

Focused parser for HuggingFace tokenizer.json format. Extracts:
- model.vocab: Dict[String, Int] (token -> id)
- model.merges: List[String] (merge rules)
- added_tokens: special token configs

Uses SIMD-optimized whitespace skipping for performance.
"""

from ..simd.whitespace import skip_whitespace_simd


struct JsonValue(Copyable, Movable):
    """Represents a parsed JSON value."""

    var _type: Int  # 0=null, 1=bool, 2=int, 3=float, 4=string, 5=array, 6=object
    var _bool_val: Bool
    var _int_val: Int
    var _float_val: Float64
    var _string_val: String
    # Note: arrays and objects are parsed on-demand from raw JSON

    fn __init__(out self):
        """Create a null value."""
        self._type = 0
        self._bool_val = False
        self._int_val = 0
        self._float_val = 0.0
        self._string_val = ""

    fn __copyinit__(out self, copy: Self):
        """Copy constructor."""
        self._type = copy._type
        self._bool_val = copy._bool_val
        self._int_val = copy._int_val
        self._float_val = copy._float_val
        self._string_val = copy._string_val

    fn __moveinit__(out self, deinit take: Self):
        """Move constructor."""
        self._type = take._type
        self._bool_val = take._bool_val
        self._int_val = take._int_val
        self._float_val = take._float_val
        self._string_val = take._string_val^

    fn is_null(self) -> Bool:
        return self._type == 0

    fn is_bool(self) -> Bool:
        return self._type == 1

    fn is_int(self) -> Bool:
        return self._type == 2

    fn is_string(self) -> Bool:
        return self._type == 4

    fn as_bool(self) -> Bool:
        return self._bool_val

    fn as_int(self) -> Int:
        return self._int_val

    fn as_string(self) -> String:
        return self._string_val

    @staticmethod
    fn null() -> Self:
        return Self()

    @staticmethod
    fn from_bool(v: Bool) -> Self:
        var result = Self()
        result._type = 1
        result._bool_val = v
        return result^

    @staticmethod
    fn from_int(v: Int) -> Self:
        var result = Self()
        result._type = 2
        result._int_val = v
        return result^

    @staticmethod
    fn from_string(v: String) -> Self:
        var result = Self()
        result._type = 4
        result._string_val = v
        return result^


struct JsonParser:
    """
    Simple JSON parser for tokenizer files.

    Parses vocab dictionaries, merge lists, and special token configs.
    """

    var _data: String
    var _pos: Int
    var _len: Int

    fn __init__(out self, data: String):
        """Create a parser for the given JSON string."""
        self._data = data
        self._pos = 0
        self._len = len(data)

    fn skip_whitespace(mut self):
        """Skip whitespace using SIMD optimization."""
        self._pos = skip_whitespace_simd(self._data, self._pos)

    fn peek(self) -> String:
        """Peek at current character without consuming."""
        if self._pos >= self._len:
            return ""
        return chr(Int(self._data.as_bytes()[self._pos]))

    fn consume(mut self) -> String:
        """Consume and return current character."""
        if self._pos >= self._len:
            return ""
        var c = chr(Int(self._data.as_bytes()[self._pos]))
        self._pos += 1
        return c

    fn expect(mut self, expected: String) raises:
        """Consume expected character or raise error."""
        var c = self.consume()
        if c != expected:
            raise Error("Expected '" + expected + "' but got '" + c + "' at position " + String(self._pos))

    fn parse_string(mut self) raises -> String:
        """Parse a JSON string value."""
        self.expect("\"")
        var result = String()

        while self._pos < self._len:
            var c = chr(Int(self._data.as_bytes()[self._pos]))
            self._pos += 1

            if c == "\"":
                return result

            if c == "\\":
                # Handle escape sequences
                if self._pos >= self._len:
                    raise Error("Unexpected end of string")
                var next = chr(Int(self._data.as_bytes()[self._pos]))
                self._pos += 1

                if next == "\"":
                    result += "\""
                elif next == "\\":
                    result += "\\"
                elif next == "n":
                    result += "\n"
                elif next == "r":
                    result += "\r"
                elif next == "t":
                    result += "\t"
                elif next == "u":
                    # Unicode escape - parse 4 hex digits
                    if self._pos + 4 > self._len:
                        raise Error("Invalid unicode escape")
                    var hex_str = String()
                    for k in range(self._pos, self._pos + 4):
                        hex_str += chr(Int(self._data.as_bytes()[k]))
                    self._pos += 4
                    var code = self._parse_hex(hex_str)
                    result += chr(code)
                else:
                    result += next
            else:
                result += c

        raise Error("Unterminated string")

    fn _parse_hex(self, s: String) raises -> Int:
        """Parse a 4-digit hex string."""
        var result = 0
        for i in range(len(s)):
            var c = s.as_bytes()[i]
            if c >= UInt8(ord("0")) and c <= UInt8(ord("9")):
                result = result * 16 + (Int(c) - ord("0"))
            elif c >= UInt8(ord("a")) and c <= UInt8(ord("f")):
                result = result * 16 + (Int(c) - ord("a") + 10)
            elif c >= UInt8(ord("A")) and c <= UInt8(ord("F")):
                result = result * 16 + (Int(c) - ord("A") + 10)
            else:
                raise Error("Invalid hex character: " + chr(Int(c)))
        return result

    fn parse_number(mut self) raises -> Int:
        """Parse a JSON number (integer only for vocab IDs)."""
        var result = 0
        var negative = False

        if self.peek() == "-":
            negative = True
            self._pos += 1

        while self._pos < self._len:
            var c = self._data.as_bytes()[self._pos]
            if c >= UInt8(ord("0")) and c <= UInt8(ord("9")):
                result = result * 10 + (Int(c) - ord("0"))
                self._pos += 1
            else:
                break

        if negative:
            result = -result

        return result

    fn parse_bool(mut self) raises -> Bool:
        """Parse true or false."""
        if self._pos + 4 <= self._len and self._check_slice(self._pos, 4, "true"):
            self._pos += 4
            return True
        elif self._pos + 5 <= self._len and self._check_slice(self._pos, 5, "false"):
            self._pos += 5
            return False
        raise Error("Expected 'true' or 'false'")

    fn _check_slice(self, start: Int, length: Int, target: String) -> Bool:
        """Check if a slice of _data matches target by comparing bytes."""
        if start + length > self._len:
            return False
        for i in range(length):
            if self._data.as_bytes()[start + i] != target.as_bytes()[i]:
                return False
        return True

    fn parse_null(mut self) raises:
        """Parse null value."""
        if self._pos + 4 <= self._len and self._check_slice(self._pos, 4, "null"):
            self._pos += 4
            return
        raise Error("Expected 'null'")

    fn skip_value(mut self) raises:
        """Skip over a JSON value (for fields we don't need)."""
        self.skip_whitespace()
        var c = self.peek()

        if c == "\"":
            _ = self.parse_string()
        elif c == "{":
            self.skip_object()
        elif c == "[":
            self.skip_array()
        elif c == "t" or c == "f":
            _ = self.parse_bool()
        elif c == "n":
            self.parse_null()
        elif c == "-" or (c >= "0" and c <= "9"):
            _ = self.parse_number()
        else:
            raise Error("Unexpected character: " + c)

    fn skip_object(mut self) raises:
        """Skip over a JSON object."""
        self.expect("{")
        self.skip_whitespace()

        if self.peek() == "}":
            self._pos += 1
            return

        while True:
            self.skip_whitespace()
            _ = self.parse_string()  # key
            self.skip_whitespace()
            self.expect(":")
            self.skip_value()  # value
            self.skip_whitespace()

            var c = self.peek()
            if c == "}":
                self._pos += 1
                return
            elif c == ",":
                self._pos += 1
            else:
                raise Error("Expected ',' or '}' in object")

    fn skip_array(mut self) raises:
        """Skip over a JSON array."""
        self.expect("[")
        self.skip_whitespace()

        if self.peek() == "]":
            self._pos += 1
            return

        while True:
            self.skip_value()
            self.skip_whitespace()

            var c = self.peek()
            if c == "]":
                self._pos += 1
                return
            elif c == ",":
                self._pos += 1
            else:
                raise Error("Expected ',' or ']' in array")

    fn parse_vocab_dict(mut self) raises -> Dict[String, Int]:
        """Parse a vocab dictionary: {"token": id, ...}."""
        var result = Dict[String, Int]()

        self.expect("{")
        self.skip_whitespace()

        if self.peek() == "}":
            self._pos += 1
            return result^

        while True:
            self.skip_whitespace()
            var key = self.parse_string()
            self.skip_whitespace()
            self.expect(":")
            self.skip_whitespace()
            var value = self.parse_number()

            result[key] = value

            self.skip_whitespace()
            var c = self.peek()
            if c == "}":
                self._pos += 1
                return result^
            elif c == ",":
                self._pos += 1
            else:
                raise Error("Expected ',' or '}' in vocab dict")

    fn parse_merges_array(mut self) raises -> List[String]:
        """Parse a merges array.

        Supports two formats:
        - Old format: ["first second", ...] (space-separated strings)
        - New format: [["first", "second"], ...] (nested arrays)
        """
        var result = List[String]()

        self.expect("[")
        self.skip_whitespace()

        if self.peek() == "]":
            self._pos += 1
            return result^

        while True:
            self.skip_whitespace()
            var c = self.peek()

            var merge: String
            if c == "[":
                # New format: ["first", "second"]
                merge = self._parse_merge_pair()
            elif c == "\"":
                # Old format: "first second"
                merge = self.parse_string()
            else:
                raise Error("Expected '[' or '\"' in merges array, got '" + c + "'")

            result.append(merge)

            self.skip_whitespace()
            c = self.peek()
            if c == "]":
                self._pos += 1
                return result^
            elif c == ",":
                self._pos += 1
            else:
                raise Error("Expected ',' or ']' in merges array")

    fn _parse_merge_pair(mut self) raises -> String:
        """Parse a merge pair array: ["first", "second"] -> "first second"."""
        self.expect("[")
        self.skip_whitespace()

        var first = self.parse_string()

        self.skip_whitespace()
        self.expect(",")
        self.skip_whitespace()

        var second = self.parse_string()

        self.skip_whitespace()
        self.expect("]")

        return first + " " + second

    fn find_field(mut self, field: String) raises -> Bool:
        """Find a field in the current object. Returns True if found."""
        self.skip_whitespace()
        self.expect("{")
        self.skip_whitespace()

        if self.peek() == "}":
            return False

        while True:
            self.skip_whitespace()
            var key = self.parse_string()
            self.skip_whitespace()
            self.expect(":")
            self.skip_whitespace()

            if key == field:
                return True

            # Skip this value and continue searching
            self.skip_value()
            self.skip_whitespace()

            var c = self.peek()
            if c == "}":
                return False
            elif c == ",":
                self._pos += 1
            else:
                raise Error("Expected ',' or '}' in object")


struct AddedToken(Copyable, Movable):
    """Represents an added token from HuggingFace format."""

    var content: String
    var id: Int
    var special: Bool

    fn __init__(out self):
        self.content = ""
        self.id = 0
        self.special = False

    fn __copyinit__(out self, copy: Self):
        self.content = copy.content
        self.id = copy.id
        self.special = copy.special

    fn __moveinit__(out self, deinit take: Self):
        self.content = take.content^
        self.id = take.id
        self.special = take.special


fn parse_added_tokens(mut parser: JsonParser) raises -> List[AddedToken]:
    """Parse the added_tokens array from HuggingFace format."""
    var result = List[AddedToken]()

    parser.expect("[")
    parser.skip_whitespace()

    if parser.peek() == "]":
        parser._pos += 1
        return result^

    while True:
        parser.skip_whitespace()
        var token = _parse_added_token(parser)
        result.append(token^)

        parser.skip_whitespace()
        var c = parser.peek()
        if c == "]":
            parser._pos += 1
            return result^
        elif c == ",":
            parser._pos += 1
        else:
            raise Error("Expected ',' or ']' in added_tokens")


fn _parse_added_token(mut parser: JsonParser) raises -> AddedToken:
    """Parse a single added token object."""
    var token = AddedToken()

    parser.expect("{")
    parser.skip_whitespace()

    if parser.peek() == "}":
        parser._pos += 1
        return token^

    while True:
        parser.skip_whitespace()
        var key = parser.parse_string()
        parser.skip_whitespace()
        parser.expect(":")
        parser.skip_whitespace()

        if key == "content":
            token.content = parser.parse_string()
        elif key == "id":
            token.id = parser.parse_number()
        elif key == "special":
            token.special = parser.parse_bool()
        else:
            parser.skip_value()

        parser.skip_whitespace()
        var c = parser.peek()
        if c == "}":
            parser._pos += 1
            return token^
        elif c == ",":
            parser._pos += 1
        else:
            raise Error("Expected ',' or '}' in added token")
