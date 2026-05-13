"""
File I/O utilities for mojo-tokenizer.

Provides simple file reading operations needed for loading
vocabulary files in various formats.
"""


def read_file(path: String) raises -> String:
    """
    Read entire file contents as a string.

    Args:
        path: Path to the file to read.

    Returns:
        The file contents as a string.

    Raises:
        Error if the file cannot be read.
    """
    try:
        with open(path, "r") as f:
            return f.read()
    except e:
        raise Error("Failed to read file '" + path + "': " + String(e))


def read_bytes(path: String) raises -> List[UInt8]:
    """
    Read entire file contents as bytes.

    Args:
        path: Path to the file to read.

    Returns:
        The file contents as a list of bytes.

    Raises:
        Error if the file cannot be read.
    """
    try:
        with open(path, "rb") as f:
            var content = f.read_bytes()
            var result = List[UInt8]()
            for i in range(len(content)):
                result.append(content[i])
            return result^
    except e:
        raise Error("Failed to read file '" + path + "': " + String(e))


def read_lines(path: String) raises -> List[String]:
    """
    Read file contents as a list of lines.

    Args:
        path: Path to the file to read.

    Returns:
        List of lines (without trailing newlines).

    Raises:
        Error if the file cannot be read.
    """
    var content = read_file(path)
    var lines = List[String]()
    var current_line = String()
    var content_bytes = content.as_bytes()

    for i in range(len(content)):
        var c = chr(Int(content_bytes[i]))
        if c == "\n":
            lines.append(current_line^)
            current_line = String()
        elif c != "\r":  # Skip carriage returns
            current_line += c

    # Add last line if not empty
    if len(current_line) > 0:
        lines.append(current_line^)

    return lines^


def file_exists(path: String) -> Bool:
    """Check if a file exists."""
    try:
        with open(path, "r") as f:
            _ = f.read(1)
        return True
    except:
        return False
