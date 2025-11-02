/**
 * @file main.cpp
 * @brief Command-line interface for the WebAssembly interpreter.
 *
 * This executable serves as a lightweight test harness for running .wat modules.
 * It parses the input file, constructs the in-memory module representation,
 * and executes all functions that take no parameters.
 *
 * For each executed function, the interpreter prints:
 *   - The function index
 *   - The value of memory[0] after execution
 *   - Any returned value (if present)
 *
 * Execution errors such as invalid instructions or memory violations are caught
 * and reported gracefully. The tool is primarily intended for quick validation
 * of parser and runtime correctness across multiple test cases.
 */

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "exec.h"
#include "parser.h"

using namespace std;

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: ./wat <file.wat>\n";
        return 1;
    }

    try {
        Module M = parse_module(argv[1]);
        auto globals = M.globals;

        int executed = 0;
        for (size_t i = 0; i < M.funcs.size(); ++i) {
            const Function& fn = M.funcs[i];
            if (!fn.param_types.empty()) {
                continue;
            }

            if (M.memory.size() >= 4) {
                M.memory[0] = M.memory[1] = M.memory[2] = M.memory[3] = 0;
            }

            try {
                MaybeValue ret = exec_fn(M, static_cast<int>(i), globals, M.memory);
                uint32_t v = static_cast<uint32_t>(M.memory[0])
                           | (static_cast<uint32_t>(M.memory[1]) << 8)
                           | (static_cast<uint32_t>(M.memory[2]) << 16)
                           | (static_cast<uint32_t>(M.memory[3]) << 24);
                int32_t mem0 = static_cast<int32_t>(v);
                cout << "Function #" << i << " executed, memory[0]: " << mem0;
                if (ret.has) {
                    cout << " (return: " << format_value(ret.value) << ")";
                }
                cout << "\n";
                executed++;
            } catch (const exception& e) {
                cout << "Function #" << i << " failed: " << e.what() << "\n";
            }
        }
        cout << "\nExecuted " << executed << " functions total.\n";
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 2;
    }
    return 0;
}

