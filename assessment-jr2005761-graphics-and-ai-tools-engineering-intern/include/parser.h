#pragma once

#include <string>

#include "interpreter_types.h"

/**
 * Parses a WebAssembly text (.wat) file into a Module structure.
 * @param path Path to the .wat source file.
 * @return Fully constructed Module ready for execution.
 * @throws std::runtime_error on malformed input.
 */
Module parse_module(const std::string& path);
