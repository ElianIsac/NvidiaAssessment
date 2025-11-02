# WebAssembly Interpreter

A self-contained **C++17** interpreter for the WebAssembly Text Format (`.wat`).  
Implements a lightweight toolchain pipeline: tokenization → parsing → execution.

---

## Build System

The project uses a simple `Makefile` compatible with GNU Make and `clang++`/`g++`.

### Build Targets

| Target | Description |
|---------|-------------|
| `make` | Compiles all sources into `build/wasm_interpreter`. |
| `make clean` | Removes all build artifacts (`build/` directory). |

### Commands

```bash
make clean && make
```

Output binary:
```
build/wasm_interpreter
```

Example run:
```bash
./build/wasm_interpreter tests/wat/03_test_prio2.wat
```

---

## Source Layout

```
include/
  exec.h                  # Runtime interface
  parser.h                # Parser declarations
  tokenizer.h             # Tokenizer interface
  interpreter_types.h     # Core value and module definitions

src/
  exec.cpp                # Execution engine
  parser.cpp              # WAT parser and module builder
  tokenizer.cpp           # Tokenizer implementation
  main.cpp                # CLI entry point
```

---

## Runtime Overview

### Tokenizer
- Converts `.wat` text into a flat list of tokens.  
- Supports line (`;;`) and block (`(; ;)`) comments, and string literals.

### Parser
- Builds a `Module` with functions, globals, memory, tables, and data segments.  
- Resolves control structures, imports, and type signatures.  
- Minimal validation for WebAssembly MVP syntax.

### Executor
- Interprets instructions sequentially using an explicit instruction pointer.  
- Maintains operand and call stacks, memory, and global state.  
- Includes a basic WASI binding for `fd_write` (console output).

---

## Example Output

```
Function #0 executed, memory[0]: 42
Function #1 executed, memory[0]: 15
Executed 12 functions total.
```

---

## Implementation Notes

- **Language:** C++17  
- **Memory model:** single linear buffer (64 KiB pages)  
- **Control flow:** block/loop/if/else resolved at parse-time  
- **Error handling:** fail-fast via `std::runtime_error`  
- **Dependencies:** none (standard library only)

---

## Usage

```bash
./build/wasm_interpreter <path-to-file.wat>
```

All zero-parameter functions are executed sequentially.  
The resulting state (including `memory[0]`) is printed after each function.

---

## Deployment Notes

- No external dependencies — can compile directly on Linux or macOS with `g++`.  
- Build artifacts and binary are written to `build/` (excluded via `.gitignore`).  
- Designed for reproducible builds and static analysis compatibility.  
- Safe for CI/CD environments (no network access or runtime downloads).

---

## License

All code is original and self-contained for evaluation purposes.  
No third-party code or libraries are used.
