"""
Base64 encoding/decoding for mojo-tokenizer.

Used for parsing tiktoken vocabulary files where tokens
are stored as base64-encoded byte sequences.
"""

# Standard base64 alphabet
comptime BASE64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


@always_inline
fn _get_decode_value(c: Int) -> Int:
    """Get decode value for a character."""
    # A-Z: 0-25
    if c >= 65 and c <= 90:
        return c - 65
    # a-z: 26-51
    if c >= 97 and c <= 122:
        return c - 97 + 26
    # 0-9: 52-61
    if c >= 48 and c <= 57:
        return c - 48 + 52
    # +: 62
    if c == 43:
        return 62
    # /: 63
    if c == 47:
        return 63
    # Padding or invalid
    return -1


fn base64_decode(encoded: String) raises -> List[UInt8]:
    """
    Decode a base64-encoded string to bytes.

    Args:
        encoded: The base64-encoded string.

    Returns:
        The decoded bytes.

    Raises:
        Error if the input is not valid base64.

    Example:
        var bytes = base64_decode("SGVsbG8=")  # "Hello".
    """
    var result = List[UInt8]()

    if len(encoded) == 0:
        return result^

    # Remove whitespace and validate length
    var clean = String()
    for i in range(len(encoded)):
        var c = chr(Int(encoded.as_bytes()[i]))
        if c != " " and c != "\n" and c != "\r" and c != "\t":
            clean += c

    # Base64 length should be multiple of 4
    if len(clean) % 4 != 0:
        raise Error("Invalid base64: length must be multiple of 4")

    var i = 0
    while i < len(clean):
        # Get 4 characters
        var c0 = Int(clean.as_bytes()[i])
        var c1 = Int(clean.as_bytes()[i + 1])
        var c2 = Int(clean.as_bytes()[i + 2])
        var c3 = Int(clean.as_bytes()[i + 3])

        # Decode to 6-bit values
        var v0 = _get_decode_value(c0)
        var v1 = _get_decode_value(c1)
        var v2 = _get_decode_value(c2)
        var v3 = _get_decode_value(c3)

        if v0 < 0 or v1 < 0:
            raise Error("Invalid base64 character at position " + String(i))

        # First byte: v0 << 2 | v1 >> 4
        result.append(UInt8((v0 << 2) | (v1 >> 4)))

        # Second byte (if not padding)
        if chr(Int(clean.as_bytes()[i + 2])) != "=":
            if v2 < 0:
                raise Error("Invalid base64 character at position " + String(i + 2))
            result.append(UInt8(((v1 & 0x0F) << 4) | (v2 >> 2)))

            # Third byte (if not padding)
            if chr(Int(clean.as_bytes()[i + 3])) != "=":
                if v3 < 0:
                    raise Error("Invalid base64 character at position " + String(i + 3))
                result.append(UInt8(((v2 & 0x03) << 6) | v3))

        i += 4

    return result^


fn base64_encode(data: List[UInt8]) -> String:
    """
    Encode bytes to a base64 string.

    Args:
        data: The bytes to encode.

    Returns:
        The base64-encoded string.
    """
    var result = String()

    var i = 0
    while i < len(data):
        # Get up to 3 bytes
        var b0 = Int(data[i])
        var b1 = 0 if i + 1 >= len(data) else Int(data[i + 1])
        var b2 = 0 if i + 2 >= len(data) else Int(data[i + 2])

        # Convert to 4 base64 characters
        result += BASE64_CHARS[(b0 >> 2) & 0x3F]
        result += BASE64_CHARS[((b0 & 0x03) << 4) | ((b1 >> 4) & 0x0F)]

        if i + 1 < len(data):
            result += BASE64_CHARS[((b1 & 0x0F) << 2) | ((b2 >> 6) & 0x03)]
        else:
            result += "="

        if i + 2 < len(data):
            result += BASE64_CHARS[b2 & 0x3F]
        else:
            result += "="

        i += 3

    return result


fn bytes_to_string(data: List[UInt8]) -> String:
    """
    Convert a list of bytes to a UTF-8 string.

    Args:
        data: The bytes to convert.

    Returns:
        The resulting string.
    """
    var result = String()
    for i in range(len(data)):
        result += chr(Int(data[i]))
    return result


fn string_to_bytes(s: String) -> List[UInt8]:
    """
    Convert a string to a list of UTF-8 bytes.

    Args:
        s: The string to convert.

    Returns:
        The resulting bytes.
    """
    var result = List[UInt8]()
    for i in range(len(s)):
        result.append(s.as_bytes()[i])
    return result^
