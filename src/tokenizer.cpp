/**
 * @file tokenizer.cpp
 * @brief Splits WebAssembly text (.wat) source into lexical tokens.
 *
 * Handles parentheses, string literals, and both line (`;;`) and block (`(; ;)`)
 * comments while preserving token order for the parser.
 * Returns a flat list of tokens suitable for syntactic analysis in `parser.cpp`.
 */

#include "tokenizer.h"

#include <cctype>

std::vector<std::string> tokenize(const std::string& src) {
    std::vector<std::string> out;
    std::string cur;
    bool inLineComment = false;
    int blockCommentDepth = 0;
    bool inString = false;

    for (size_t i = 0; i < src.size(); ++i) {
        char c = src[i];
        char next = (i + 1 < src.size()) ? src[i + 1] : '\0';

        if (inLineComment) {
            if (c == '\n') inLineComment = false;
            continue;
        }

        if (blockCommentDepth > 0) {
            if (c == '(' && next == ';') {
                blockCommentDepth++;
                i++;
                continue;
            }
            if (c == ';' && next == ')') {
                blockCommentDepth--;
                i++;
            }
            continue;
        }

        if (!inString && c == '(' && next == ';') {
            blockCommentDepth = 1;
            i++;
            continue;
        }

        if (!inString && c == ';' && next == ';') {
            inLineComment = true;
            i++;
            continue;
        }

        if (inString) {
            cur.push_back(c);
            if (c == '"' && (cur.size() == 1 || cur[cur.size() - 2] != '\\')) {
                inString = false;
                out.push_back(cur);
                cur.clear();
            }
            continue;
        }

        if (c == '"') {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
            inString = true;
            cur.push_back(c);
            continue;
        }

        if (std::isspace(static_cast<unsigned char>(c)) || c == '(' || c == ')') {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
            if (c == '(' || c == ')') out.emplace_back(1, c);
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

