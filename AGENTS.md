# mojo-tokenizer – AGENTS.md

High-performance BPE tokenizer in Mojo. Package name: `mojo_tokenizer`.

## Setup

```bash
pixi shell                    # activate pixi env first
pixi run build                # build lib/mojo_tokenizer.mojopkg (required before test/use)
pixi run test                 # core tests
```

Mojo version not pinned in `pixi.toml` — only `max` (`>=24.6,<26`). All commands from project root.

## Commands

| Task | Command |
|------|---------|
| Build | `pixi run build` → `lib/mojo_tokenizer.mojopkg` (must build first!) |
| Test | `pixi run test` |
| Format | `pixi run format` or `mojo format src/` |
| Example | `pixi run example` (currently broken — `Token` lacks `Writable`) |

Package must be built before running anything that imports `mojo_tokenizer`. The `-I lib` flag is needed for runtime resolution (already in pixi tasks).

## Quick Start

```mojo
from mojo_tokenizer import BPETokenizer

var tok = BPETokenizer.from_tiktoken("data/o200k_base.tiktoken")
var ids = tok.encode("Hello, world!")  # [13225, 11, 220, 24169, 0]
var text = tok.decode(ids)             # "Hello, world!"
```

## Fixes Applied (May 2026)

Removed broken imports from `src/bpe.mojo` for files that never existed:
- `double_array_trie.mojo`, `heap_bpe.mojo`, `pretokenizer.mojo`
- `InlineBitField` (from `bitfield.mojo`)
- Added `get_token_len()` to `Vocabulary`
- Replaced `pretokenize()` calls with `_split_into_words()`
- Enabled `_use_backtrack = True` for HuggingFace loader
- Fix decode fallback (string bytes when no raw bytes stored)

Removed ~9000 lines of dead code: parallel encode paths, `backtrack_encoder.mojo`, `chat/`, `pipeline/`, `benchmark/`, stale docs, root test files referencing missing modules, and validation test artifacts.

Package compiles and 7 tests pass. Encoder uses ByteTrie-based backtracking (no DoubleArrayTrie/HeapBPE/Pretokenizer).

## Module Architecture

`src/__init__.mojo` exports the public API:
- `BPETokenizer`, `Tokenizer`, `Token` — core types
- `Vocabulary`, `MergeRule` — vocab management
- `SpecialTokens`, `SpecialToken` — special tokens
- `ByteTrie`, `TrieNode` — byte trie for lookup
- `TokenCache`, `MergeCache` — caching layer
- `load_tiktoken`, `load_huggingface` — format loaders

`src/` submodules: `cache/`, `encoding/`, `formats/`, `io/`, `json/`, `simd/`

## Testing

- **Mojo tests**: `tests/test_tokenizer.mojo` — 7 unit tests, pass on `main`
- **Python validation**: `tests/test_tokenizer_validation.py`, `tests/test_huggingface_validation.py` — need `tiktoken`, `transformers` packages

## Data Files

- `data/o200k_base.tiktoken` — o200k_base vocabulary (~200K lines, base64)
- `data/huggingface/{Qwen_Qwen2-1.5B,mistralai_Mistral-7B-v0.1}/` — HF tokenizer.json
