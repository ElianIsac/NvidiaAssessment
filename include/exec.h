#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "interpreter_types.h"

/**
 * Executes a WebAssembly function by index within a module.
 * @param module The parsed WebAssembly module.
 * @param func_index The index of the function to execute.
 * @param globals Reference to mutable global variables.
 * @param memory Linear memory buffer used during execution.
 * @param args Optional argument list for the call (defaults to empty).
 * @return MaybeValue containing the function's return value if any.
 * @throws std::runtime_error on type or memory violations.
 */
MaybeValue exec_fn(Module& module,
                   int func_index,
                   std::unordered_map<std::string, int32_t>& globals,
                   std::vector<uint8_t>& memory,
                   const std::vector<Value>& args = {});

/**
 * Converts a runtime Value into a human-readable string.
 * Example: "i32=42" or "f64=3.14".
 * @param v The Value to format.
 * @return A formatted string representation.
 */
std::string format_value(const Value& v);

/**
 * Host implementation of the WASI function `fd_write`.
 * Supports printing to stdout/stderr for text output from WebAssembly.
 * @param module The module instance invoking the call.
 * @param mem The linear memory buffer of the module.
 * @param args Argument list (fd, iovs_ptr, iovs_len, nwritten_ptr).
 * @return MaybeValue containing an i32 result (0 on success).
 */
MaybeValue wasi_fd_write(Module& module,
                         std::vector<uint8_t>& mem,
                         const std::vector<Value>& args);

