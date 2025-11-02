#pragma once

#include <string>
#include <vector>

/**
 * Tokenizes a WebAssembly text (.wat) source string.
 * @param src The full source text.
 * @return A vector of tokens (identifiers, parentheses, literals, etc.).
 */
std::vector<std::string> tokenize(const std::string& src);

