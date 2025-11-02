/**
 * @file parser.cpp
 * @brief Parses WebAssembly text (.wat) modules into executable structures.
 *
 * Converts a stream of tokens from `tokenize()` into a structured `Module`
 * containing functions, types, globals, memory, and data segments.
 * Performs minimal validation and prepares the module for execution by `exec.cpp`.
 *
 * Control flow constructs (block, loop, if, else, end) are linked by matching
 * parentheses, and numeric or string literals are parsed into typed values.
 *
 * Errors are reported through `std::runtime_error` for quick debugging.
 */

#include "parser.h"

#include "exec.h"
#include "tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

using namespace std;

static size_t find_matching_paren(const vector<string>& T, size_t l) {
    int depth = 0;
    for (size_t i = l; i < T.size(); ++i) {
        if (T[i] == "(") depth++;
        else if (T[i] == ")") {
            depth--;
            if (depth == 0) return i;
        }
    }
    return T.size();
}

static bool is_int_token(const string& t) {
    if (t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X')) return true;
    if (!t.empty() && (t[0] == '-' || t[0] == '+' || isdigit(static_cast<unsigned char>(t[0])))) return true;
    return false;
}

static int32_t parse_i32(const string& t) {
    if (t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X')) {
        uint32_t v = static_cast<uint32_t>(stoul(t, nullptr, 16));
        return static_cast<int32_t>(v);
    }
    return static_cast<int32_t>(stol(t, nullptr, 0));
}

static int64_t parse_i64(const string& t) {
    if (t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X')) {
        uint64_t v = stoull(t, nullptr, 16);
        return static_cast<int64_t>(v);
    }
    return static_cast<int64_t>(stoll(t, nullptr, 0));
}

static float parse_f32(const string& t) {
    if (t == "inf" || t == "infinity") return numeric_limits<float>::infinity();
    if (t == "-inf" || t == "-infinity") return -numeric_limits<float>::infinity();
    if (t == "nan") return numeric_limits<float>::quiet_NaN();
    if (t == "-nan") return -numeric_limits<float>::quiet_NaN();
    return stof(t);
}

static double parse_f64(const string& t) {
    if (t == "inf" || t == "infinity") return numeric_limits<double>::infinity();
    if (t == "-inf" || t == "-infinity") return -numeric_limits<double>::infinity();
    if (t == "nan") return numeric_limits<double>::quiet_NaN();
    if (t == "-nan") return -numeric_limits<double>::quiet_NaN();
    return stod(t);
}

static ValueType parse_valtype(const string& t) {
    if (t == "i32") return ValueType::I32;
    if (t == "i64") return ValueType::I64;
    if (t == "f32") return ValueType::F32;
    if (t == "f64") return ValueType::F64;
    throw runtime_error("unsupported value type: " + t);
}

int Function::resolve_local(const string& tok) const {
    if (tok.empty()) throw runtime_error("empty local token");
    if (is_int_token(tok)) {
        int idx = parse_i32(tok);
        if (idx < 0 || idx >= total_vars()) throw runtime_error("local index out of range");
        return idx;
    }
    auto it = local_lookup.find(tok);
    if (it == local_lookup.end()) throw runtime_error("unknown local: " + tok);
    return it->second;
}

static vector<uint8_t> parse_string_literal(const string& tok) {
    if (tok.size() < 2 || tok.front() != '"' || tok.back() != '"') {
        throw runtime_error("invalid string literal: " + tok);
    }
    vector<uint8_t> out;
    for (size_t i = 1; i + 1 < tok.size(); ++i) {
        char c = tok[i];
        if (c == '\\') {
            if (i + 1 >= tok.size() - 1) throw runtime_error("unterminated escape in string");
            char d = tok[++i];
            switch (d) {
                case 'n': out.push_back(static_cast<uint8_t>('\n')); break;
                case 'r': out.push_back(static_cast<uint8_t>('\r')); break;
                case 't': out.push_back(static_cast<uint8_t>('\t')); break;
                case '"': out.push_back(static_cast<uint8_t>('"')); break;
                case '\\': out.push_back(static_cast<uint8_t>('\\')); break;
                default:
                    if (isxdigit(static_cast<unsigned char>(d))) {
                        if (i + 1 >= tok.size() - 1 || !isxdigit(static_cast<unsigned char>(tok[i + 1]))) {
                            throw runtime_error("hex escape must have two digits");
                        }
                        char e = tok[++i];
                        string hex;
                        hex.push_back(d);
                        hex.push_back(e);
                        uint8_t v = static_cast<uint8_t>(stoi(hex, nullptr, 16));
                        out.push_back(v);
                    } else {
                        throw runtime_error(string("unsupported escape: \\") + d);
                    }
                    break;
            }
        } else {
            out.push_back(static_cast<uint8_t>(c));
        }
    }
    return out;
}

static string strip_quotes(const string& tok) {
    if (tok.size() >= 2 && tok.front() == '"' && tok.back() == '"') {
        return tok.substr(1, tok.size() - 2);
    }
    return tok;
}

static void skip_optional_result_sig(const vector<string>& T, size_t& k) {
    if (k >= T.size()) return;
    if (T[k] == "(" && k + 1 < T.size() && T[k + 1] == "result") {
        k = find_matching_paren(T, k) + 1;
    }
}

static void parse_decl_list(vector<string>& names,
                            vector<ValueType>& types,
                            const vector<string>& tokens,
                            size_t start,
                            size_t end) {
    string pendingName;
    for (size_t idx = start; idx < end; ++idx) {
        const string& tok = tokens[idx];
        if (tok.empty()) continue;
        if (tok[0] == '$') {
            pendingName = tok;
        } else {
            ValueType vt = parse_valtype(tok);
            if (!pendingName.empty()) {
                names.push_back(pendingName);
                pendingName.clear();
            } else {
                names.emplace_back();
            }
            types.push_back(vt);
        }
    }
    if (!pendingName.empty()) {
        throw runtime_error("declaration missing type for " + pendingName);
    }
}

static Op opFrom(const string& tok) {
    if (tok == "i32.const") return Op::I32Const;
    if (tok == "i32.add") return Op::I32Add;
    if (tok == "i32.sub") return Op::I32Sub;
    if (tok == "i32.mul") return Op::I32Mul;
    if (tok == "i32.div_s") return Op::I32DivS;
    if (tok == "i32.div_u") return Op::I32DivU;
    if (tok == "i32.rem_s") return Op::I32RemS;
    if (tok == "i32.rem_u") return Op::I32RemU;
    if (tok == "i32.and") return Op::I32And;
    if (tok == "i32.or") return Op::I32Or;
    if (tok == "i32.xor") return Op::I32Xor;
    if (tok == "i32.shl") return Op::I32Shl;
    if (tok == "i32.shr_s") return Op::I32ShrS;
    if (tok == "i32.shr_u") return Op::I32ShrU;
    if (tok == "i32.clz") return Op::I32Clz;
    if (tok == "i32.ctz") return Op::I32Ctz;
    if (tok == "i32.popcnt") return Op::I32Popcnt;
    if (tok == "i32.rotl") return Op::I32Rotl;
    if (tok == "i32.rotr") return Op::I32Rotr;

    if (tok == "i64.const") return Op::I64Const;
    if (tok == "i64.add") return Op::I64Add;
    if (tok == "i64.sub") return Op::I64Sub;
    if (tok == "i64.mul") return Op::I64Mul;
    if (tok == "i64.div_s") return Op::I64DivS;
    if (tok == "i64.div_u") return Op::I64DivU;
    if (tok == "i64.rem_s") return Op::I64RemS;
    if (tok == "i64.rem_u") return Op::I64RemU;
    if (tok == "i64.and") return Op::I64And;
    if (tok == "i64.or") return Op::I64Or;
    if (tok == "i64.xor") return Op::I64Xor;
    if (tok == "i64.shl") return Op::I64Shl;
    if (tok == "i64.shr_s") return Op::I64ShrS;
    if (tok == "i64.shr_u") return Op::I64ShrU;
    if (tok == "i64.clz") return Op::I64Clz;
    if (tok == "i64.ctz") return Op::I64Ctz;
    if (tok == "i64.popcnt") return Op::I64Popcnt;
    if (tok == "i64.rotl") return Op::I64Rotl;
    if (tok == "i64.rotr") return Op::I64Rotr;
    if (tok == "i64.eq") return Op::I64Eq;
    if (tok == "i64.ne") return Op::I64Ne;
    if (tok == "i64.lt_s") return Op::I64LtS;
    if (tok == "i64.lt_u") return Op::I64LtU;
    if (tok == "i64.gt_s") return Op::I64GtS;
    if (tok == "i64.gt_u") return Op::I64GtU;
    if (tok == "i64.le_s") return Op::I64LeS;
    if (tok == "i64.le_u") return Op::I64LeU;
    if (tok == "i64.ge_s") return Op::I64GeS;
    if (tok == "i64.ge_u") return Op::I64GeU;
    if (tok == "i64.eqz") return Op::I64Eqz;

    if (tok == "f32.const") return Op::F32Const;
    if (tok == "f32.add") return Op::F32Add;
    if (tok == "f32.sub") return Op::F32Sub;
    if (tok == "f32.mul") return Op::F32Mul;
    if (tok == "f32.div") return Op::F32Div;
    if (tok == "f32.min") return Op::F32Min;
    if (tok == "f32.max") return Op::F32Max;
    if (tok == "f32.abs") return Op::F32Abs;
    if (tok == "f32.neg") return Op::F32Neg;
    if (tok == "f32.sqrt") return Op::F32Sqrt;
    if (tok == "f32.ceil") return Op::F32Ceil;
    if (tok == "f32.floor") return Op::F32Floor;
    if (tok == "f32.trunc") return Op::F32Trunc;
    if (tok == "f32.nearest") return Op::F32Nearest;
    if (tok == "f32.copysign") return Op::F32Copysign;
    if (tok == "f32.eq") return Op::F32Eq;
    if (tok == "f32.ne") return Op::F32Ne;
    if (tok == "f32.lt") return Op::F32Lt;
    if (tok == "f32.gt") return Op::F32Gt;
    if (tok == "f32.le") return Op::F32Le;
    if (tok == "f32.ge") return Op::F32Ge;

    if (tok == "f64.const") return Op::F64Const;
    if (tok == "f64.add") return Op::F64Add;
    if (tok == "f64.sub") return Op::F64Sub;
    if (tok == "f64.mul") return Op::F64Mul;
    if (tok == "f64.div") return Op::F64Div;
    if (tok == "f64.min") return Op::F64Min;
    if (tok == "f64.max") return Op::F64Max;
    if (tok == "f64.abs") return Op::F64Abs;
    if (tok == "f64.neg") return Op::F64Neg;
    if (tok == "f64.sqrt") return Op::F64Sqrt;
    if (tok == "f64.ceil") return Op::F64Ceil;
    if (tok == "f64.floor") return Op::F64Floor;
    if (tok == "f64.trunc") return Op::F64Trunc;
    if (tok == "f64.nearest") return Op::F64Nearest;
    if (tok == "f64.copysign") return Op::F64Copysign;
    if (tok == "f64.eq") return Op::F64Eq;
    if (tok == "f64.ne") return Op::F64Ne;
    if (tok == "f64.lt") return Op::F64Lt;
    if (tok == "f64.gt") return Op::F64Gt;
    if (tok == "f64.le") return Op::F64Le;
    if (tok == "f64.ge") return Op::F64Ge;

    if (tok == "i32.load") return Op::I32Load;
    if (tok == "i32.store") return Op::I32Store;
    if (tok == "i32.load8_u") return Op::I32Load8U;
    if (tok == "i32.load8_s") return Op::I32Load8S;
    if (tok == "i32.store8") return Op::I32Store8;
    if (tok == "i32.load16_u") return Op::I32Load16U;
    if (tok == "i32.load16_s") return Op::I32Load16S;
    if (tok == "i32.store16") return Op::I32Store16;

    if (tok == "i64.load") return Op::I64Load;
    if (tok == "i64.store") return Op::I64Store;
    if (tok == "i64.load8_u") return Op::I64Load8U;
    if (tok == "i64.load8_s") return Op::I64Load8S;
    if (tok == "i64.load16_u") return Op::I64Load16U;
    if (tok == "i64.load16_s") return Op::I64Load16S;
    if (tok == "i64.load32_u") return Op::I64Load32U;
    if (tok == "i64.load32_s") return Op::I64Load32S;
    if (tok == "i64.store8") return Op::I64Store8;
    if (tok == "i64.store16") return Op::I64Store16;
    if (tok == "i64.store32") return Op::I64Store32;

    if (tok == "f32.load") return Op::F32Load;
    if (tok == "f32.store") return Op::F32Store;
    if (tok == "f64.load") return Op::F64Load;
    if (tok == "f64.store") return Op::F64Store;

    if (tok == "memory.size") return Op::MemorySize;
    if (tok == "memory.grow") return Op::MemoryGrow;

    if (tok == "local.get") return Op::LocalGet;
    if (tok == "local.set") return Op::LocalSet;
    if (tok == "local.tee") return Op::LocalTee;
    if (tok == "global.get") return Op::GlobalGet;
    if (tok == "global.set") return Op::GlobalSet;

    if (tok == "f32.convert_i32_s") return Op::F32ConvertI32S;
    if (tok == "f32.convert_i32_u") return Op::F32ConvertI32U;
    if (tok == "f32.convert_i64_s") return Op::F32ConvertI64S;
    if (tok == "f32.convert_i64_u") return Op::F32ConvertI64U;
    if (tok == "f64.convert_i32_s") return Op::F64ConvertI32S;
    if (tok == "f64.convert_i32_u") return Op::F64ConvertI32U;
    if (tok == "f64.convert_i64_s") return Op::F64ConvertI64S;
    if (tok == "f64.convert_i64_u") return Op::F64ConvertI64U;
    if (tok == "i32.trunc_f32_s") return Op::I32TruncF32S;
    if (tok == "i32.trunc_f32_u") return Op::I32TruncF32U;
    if (tok == "i32.trunc_f64_s") return Op::I32TruncF64S;
    if (tok == "i32.trunc_f64_u") return Op::I32TruncF64U;
    if (tok == "i64.trunc_f32_s") return Op::I64TruncF32S;
    if (tok == "i64.trunc_f32_u") return Op::I64TruncF32U;
    if (tok == "i64.trunc_f64_s") return Op::I64TruncF64S;
    if (tok == "i64.trunc_f64_u") return Op::I64TruncF64U;
    if (tok == "f64.promote_f32") return Op::F64PromoteF32;
    if (tok == "f32.demote_f64") return Op::F32DemoteF64;
    if (tok == "i32.wrap_i64") return Op::I32WrapI64;
    if (tok == "i64.extend_i32_s") return Op::I64ExtendI32S;
    if (tok == "i64.extend_i32_u") return Op::I64ExtendI32U;
    if (tok == "i32.reinterpret_f32") return Op::I32ReinterpretF32;
    if (tok == "f32.reinterpret_i32") return Op::F32ReinterpretI32;
    if (tok == "i64.reinterpret_f64") return Op::I64ReinterpretF64;
    if (tok == "f64.reinterpret_i64") return Op::F64ReinterpretI64;

    if (tok == "i32.eq") return Op::Eq;
    if (tok == "i32.ne") return Op::Ne;
    if (tok == "i32.lt_s") return Op::LtS;
    if (tok == "i32.lt_u") return Op::LtU;
    if (tok == "i32.gt_s") return Op::GtS;
    if (tok == "i32.gt_u") return Op::GtU;
    if (tok == "i32.le_s") return Op::LeS;
    if (tok == "i32.le_u") return Op::LeU;
    if (tok == "i32.ge_s") return Op::GeS;
    if (tok == "i32.ge_u") return Op::GeU;
    if (tok == "i32.eqz") return Op::Eqz;
    if (tok == "select") return Op::Select;

    if (tok == "block") return Op::Block;
    if (tok == "loop") return Op::Loop;
    if (tok == "if") return Op::If;
    if (tok == "else") return Op::Else;
    if (tok == "end") return Op::End;
    if (tok == "br") return Op::Br;
    if (tok == "br_if") return Op::BrIf;
    if (tok == "br_table") return Op::BrTable;
    if (tok == "return") return Op::Return;
    if (tok == "drop") return Op::Drop;
    if (tok == "call") return Op::Call;
    if (tok == "call_indirect") return Op::CallIndirect;
    if (tok == "unreachable") return Op::Unreachable;
    if (tok == "nop") return Op::Nop;

    return Op::Invalid;
}

static void parse_function_declarations(Function& fn,
                                        const vector<string>& tokens,
                                        Module& module) {
    size_t k = 0;
    while (k < tokens.size()) {
        const string& tok = tokens[k];
        if (!tok.empty() && tok[0] == ';') {
            k++;
            continue;
        }

        if (tok == "(" && k + 1 < tokens.size()) {
            const string& kind = tokens[k + 1];
            if (kind == "param") {
                size_t end = find_matching_paren(tokens, k);
                parse_decl_list(fn.param_names, fn.param_types, tokens, k + 2, end);
                k = end + 1;
                continue;
            } else if (kind == "local") {
                size_t end = find_matching_paren(tokens, k);
                parse_decl_list(fn.local_names, fn.local_types, tokens, k + 2, end);
                k = end + 1;
                continue;
            } else if (kind == "result") {
                size_t end = find_matching_paren(tokens, k);
                for (size_t idx = k + 2; idx < end; ++idx) {
                    if (tokens[idx].empty() || tokens[idx][0] == '$') continue;
                    fn.result_types.push_back(parse_valtype(tokens[idx]));
                }
                k = end + 1;
                continue;
            } else if (kind == "type") {
                size_t end = find_matching_paren(tokens, k);
                for (size_t idx = k + 2; idx < end; ++idx) {
                    const string& tt = tokens[idx];
                    if (tt.empty() || tt[0] == ';') continue;
                    if (is_int_token(tt)) {
                        fn.type_index = parse_i32(tt);
                    } else if (!tt.empty() && tt[0] == '$') {
                        auto it = module.type_by_name.find(tt);
                        if (it == module.type_by_name.end()) {
                            throw runtime_error("unknown type: " + tt);
                        }
                        fn.type_index = it->second;
                    }
                }
                k = end + 1;
                continue;
            }
        }

        if (opFrom(tok) != Op::Invalid) break;
        if (!tok.empty() && tok[0] == '$') {
            k++;
            continue;
        }
        k++;
    }

    if (fn.result_types.size() > 1) {
        throw runtime_error("multi-value results are not supported");
    }

    fn.local_lookup.clear();
    for (size_t i = 0; i < fn.param_names.size(); ++i) {
        if (!fn.param_names[i].empty()) fn.local_lookup[fn.param_names[i]] = static_cast<int>(i);
    }
    for (size_t i = 0; i < fn.local_names.size(); ++i) {
        if (!fn.local_names[i].empty()) {
            fn.local_lookup[fn.local_names[i]] = static_cast<int>(fn.param_types.size() + i);
        }
    }
}

static void parse_function_body(const vector<string>& bodyTokens, Function& fn) {
    fn.code.clear();

    for (size_t k = 0; k < bodyTokens.size();) {
        const string& t = bodyTokens[k];

        if (!t.empty() && t[0] == ';') {
            k++;
            continue;
        }

        if (t == "(" && k + 1 < bodyTokens.size()) {
            const string& t1 = bodyTokens[k + 1];
            if (t1 == "local" || t1 == "param" || t1 == "type" || t1 == "result") {
                k = find_matching_paren(bodyTokens, k) + 1;
                continue;
            }
        }

        Op op = opFrom(t);
        if (op == Op::Invalid) {
            k++;
            continue;
        }

        Instr ins;
        ins.op = op;

        switch (op) {
            case Op::I32Const:
                if (k + 1 < bodyTokens.size() && is_int_token(bodyTokens[k + 1])) {
                    ins.iarg = parse_i32(bodyTokens[k + 1]);
                    k += 2;
                } else {
                    k++;
                }
                break;

            case Op::I64Const:
                if (k + 1 < bodyTokens.size() && is_int_token(bodyTokens[k + 1])) {
                    ins.larg = parse_i64(bodyTokens[k + 1]);
                    k += 2;
                } else {
                    k++;
                }
                break;

            case Op::F32Const:
            case Op::F64Const:
                if (k + 1 < bodyTokens.size()) {
                    if (op == Op::F32Const) {
                        ins.farg = parse_f32(bodyTokens[k + 1]);
                    } else {
                        ins.farg = parse_f64(bodyTokens[k + 1]);
                    }
                    k += 2;
                } else {
                    k++;
                }
                break;

            case Op::LocalGet:
            case Op::LocalSet:
            case Op::LocalTee:
                if (k + 1 < bodyTokens.size()) {
                    ins.iarg = fn.resolve_local(bodyTokens[k + 1]);
                    k += 2;
                } else {
                    k++;
                }
                break;

            case Op::GlobalGet:
            case Op::GlobalSet:
                if (k + 1 < bodyTokens.size()) {
                    ins.sarg = bodyTokens[k + 1];
                    k += 2;
                } else {
                    k++;
                }
                break;

            case Op::Br:
            case Op::BrIf:
                if (k + 1 < bodyTokens.size() && is_int_token(bodyTokens[k + 1])) {
                    ins.iarg = parse_i32(bodyTokens[k + 1]);
                    k += 2;
                } else {
                    k++;
                }
                break;

            case Op::Call:
                if (k + 1 < bodyTokens.size()) {
                    const string& callee = bodyTokens[k + 1];
                    if (is_int_token(callee)) {
                        ins.iarg = parse_i32(callee);
                        ins.sarg.clear();
                    } else {
                        ins.iarg = -1;
                        ins.sarg = callee;
                    }
                    k += 2;
                } else {
                    k++;
                }
                break;

            case Op::CallIndirect: {
                ins.iarg = -1;
                ins.sarg.clear();
                k++;
                if (k < bodyTokens.size() && bodyTokens[k] == "(" && k + 1 < bodyTokens.size() && bodyTokens[k + 1] == "type") {
                    size_t end = find_matching_paren(bodyTokens, k);
                    if (k + 2 < end) {
                        const string& typ = bodyTokens[k + 2];
                        if (is_int_token(typ)) {
                            ins.iarg = parse_i32(typ);
                        } else {
                            ins.sarg = typ;
                        }
                    }
                    k = end + 1;
                }
                fn.code.push_back(ins);
                continue;
            }

            case Op::BrTable:
                k++;
                while (k < bodyTokens.size() && is_int_token(bodyTokens[k])) {
                    ins.table.push_back(parse_i32(bodyTokens[k]));
                    ++k;
                }
                break;

            case Op::Block:
            case Op::Loop:
            case Op::If:
                k++;
                skip_optional_result_sig(bodyTokens, k);
                fn.code.push_back(ins);
                continue;

            default:
                k++;
                break;
        }

        fn.code.push_back(ins);
    }

    vector<int> stackIdx;
    for (int i = 0; i < static_cast<int>(fn.code.size()); ++i) {
        Op op = fn.code[i].op;
        if (op == Op::Block || op == Op::Loop || op == Op::If) {
            stackIdx.push_back(i);
        } else if (op == Op::Else) {
            if (!stackIdx.empty()) {
                int j = stackIdx.back();
                if (fn.code[j].op == Op::If) {
                    fn.code[j].match_else = i;
                }
            }
        } else if (op == Op::End) {
            if (!stackIdx.empty()) {
                int j = stackIdx.back();
                fn.code[j].match_end = i;
                if (fn.code[j].op == Op::If && fn.code[j].match_else != -1) {
                    int e = fn.code[j].match_else;
                    fn.code[e].match_end = i;
                }
                stackIdx.pop_back();
            }
        }
    }
}

static FuncType parse_type_entry(const vector<string>& tokens, size_t start, size_t end) {
    FuncType ft;
    for (size_t i = start; i < end;) {
        if (tokens[i] == "(" && i + 1 < end) {
            const string& kind = tokens[i + 1];
            if (kind == "param") {
                size_t r = find_matching_paren(tokens, i);
                for (size_t idx = i + 2; idx < r; ++idx) {
                    const string& tok = tokens[idx];
                    if (tok.empty() || tok[0] == '$') continue;
                    ft.params.push_back(parse_valtype(tok));
                }
                i = r + 1;
                continue;
            } else if (kind == "result") {
                size_t r = find_matching_paren(tokens, i);
                for (size_t idx = i + 2; idx < r; ++idx) {
                    const string& tok = tokens[idx];
                    if (tok.empty() || tok[0] == '$') continue;
                    ft.results.push_back(parse_valtype(tok));
                }
                i = r + 1;
                continue;
            }
            i = find_matching_paren(tokens, i) + 1;
        } else {
            i++;
        }
    }
    if (ft.results.size() > 1) throw runtime_error("multi-value types unsupported");
    return ft;
}

static int ensure_type_index(Module& module,
                             const vector<ValueType>& params,
                             const vector<ValueType>& results) {
    for (size_t i = 0; i < module.types.size(); ++i) {
        if (module.types[i].params == params && module.types[i].results == results) {
            return static_cast<int>(i);
        }
    }
    FuncType ft;
    ft.params = params;
    ft.results = results;
    module.types.push_back(ft);
    return static_cast<int>(module.types.size() - 1);
}

static void apply_data_segments(Module& module) {
    const size_t page_size = 65536;
    if (module.memory.empty()) {
        module.memory.assign(page_size, 0);
    }
    for (const auto& seg : module.data_segments) {
        uint64_t end_offset = static_cast<uint64_t>(seg.offset) + seg.bytes.size();
        if (end_offset > module.memory.size()) {
            size_t pages = static_cast<size_t>((end_offset + page_size - 1) / page_size);
            module.memory.resize(pages * page_size, 0);
        }
        memcpy(module.memory.data() + seg.offset, seg.bytes.data(), seg.bytes.size());
    }
}

static void apply_elem_segments(Module& module) {
    for (const auto& seg : module.elem_segments) {
        uint64_t end_index = static_cast<uint64_t>(seg.offset) + seg.entries.size();
        if (end_index > module.table.size()) {
            module.table.resize(static_cast<size_t>(end_index), -1);
        }
        for (size_t i = 0; i < seg.entries.size(); ++i) {
            const ElemRef& ref = seg.entries[i];
            int func_index = -1;
            if (ref.is_index) {
                if (ref.index < 0 || ref.index >= static_cast<int>(module.funcs.size())) {
                    throw runtime_error("elem index out of range");
                }
                func_index = ref.index;
            } else {
                auto it = module.func_by_name.find(ref.name);
                if (it == module.func_by_name.end()) {
                    throw runtime_error("unknown function in elem: " + ref.name);
                }
                func_index = it->second;
            }
            module.table[seg.offset + i] = func_index;
        }
    }
}

static void resolve_indirect_types(Module& module) {
    for (auto& fn : module.funcs) {
        for (auto& ins : fn.code) {
            if (ins.op == Op::CallIndirect && ins.iarg < 0 && !ins.sarg.empty()) {
                auto it = module.type_by_name.find(ins.sarg);
                if (it == module.type_by_name.end()) {
                    throw runtime_error("unknown type in call_indirect: " + ins.sarg);
                }
                ins.iarg = it->second;
            }
        }
    }
}

Module parse_module(const string& path) {
    ifstream f(path);
    if (!f) throw runtime_error("cannot open file: " + path);
    stringstream buf;
    buf << f.rdbuf();
    string s = buf.str();

    Module M;
    M.memory.assign(64 * 1024, 0);
    M.globals["$counter"] = 0;
    M.globals["$constant"] = 100;

    auto T = tokenize(s);

    for (size_t i = 0; i < T.size();) {
        if (T[i] == "(" && i + 1 < T.size() && T[i + 1] == "type") {
            size_t j = find_matching_paren(T, i);
            size_t cursor = i + 2;
            string typeName;
            if (cursor < j && !T[cursor].empty() && T[cursor][0] == '$') {
                typeName = T[cursor];
                cursor++;
            }
            if (cursor < j && T[cursor] == "(" && cursor + 1 < j && T[cursor + 1] == "func") {
                size_t funcEnd = find_matching_paren(T, cursor);
                FuncType ft = parse_type_entry(T, cursor + 2, funcEnd);
                int idx = static_cast<int>(M.types.size());
                M.types.push_back(ft);
                if (!typeName.empty()) M.type_by_name[typeName] = idx;
            }
            i = j + 1;
        } else if (T[i] == "(" && i + 1 < T.size() && T[i + 1] == "import") {
            size_t j = find_matching_paren(T, i);
            size_t cursor = i + 2;
            string moduleName;
            string fieldName;
            if (cursor < j) {
                moduleName = strip_quotes(T[cursor]);
                cursor++;
            }
            if (cursor < j) {
                fieldName = strip_quotes(T[cursor]);
                cursor++;
            }

            while (cursor < j) {
                if (T[cursor] == "(" && cursor + 1 < j && T[cursor + 1] == "func") {
                    size_t funcEnd = find_matching_paren(T, cursor);
                    Function fn;
                    size_t inner = cursor + 2;
                    if (inner < funcEnd && !T[inner].empty() && T[inner][0] == '$') {
                        fn.name = T[inner];
                        inner++;
                    }
                    vector<string> funcTokens;
                    funcTokens.reserve(funcEnd - inner);
                    for (size_t k = inner; k < funcEnd; ++k) {
                        funcTokens.push_back(T[k]);
                    }
                    parse_function_declarations(fn, funcTokens, M);
                    if (fn.type_index < 0) {
                        fn.type_index = ensure_type_index(M, fn.param_types, fn.result_types);
                    }
                    fn.is_host = true;
                    if (moduleName == "wasi_snapshot_preview1" && fieldName == "fd_write") {
                        fn.host_impl = &wasi_fd_write;
                    } else {
                        throw runtime_error("unsupported import: " + moduleName + "::" + fieldName);
                    }
                    if (!fn.name.empty()) {
                        M.func_by_name[fn.name] = static_cast<int>(M.funcs.size());
                    }
                    M.funcs.push_back(std::move(fn));
                    break;
                }
                cursor++;
            }
            i = j + 1;
        } else if (T[i] == "(" && i + 1 < T.size() && T[i + 1] == "data") {
            size_t j = find_matching_paren(T, i);
            size_t cursor = i + 2;
            DataSegment seg;
            bool offsetSet = false;
            while (cursor < j) {
                if (T[cursor] == "(" && cursor + 1 < j && T[cursor + 1] == "i32.const") {
                    if (cursor + 2 >= j) throw runtime_error("data offset missing");
                    seg.offset = static_cast<uint32_t>(parse_i32(T[cursor + 2]));
                    offsetSet = true;
                    cursor = find_matching_paren(T, cursor) + 1;
                } else if (!T[cursor].empty() && T[cursor][0] == '"') {
                    vector<uint8_t> bytes = parse_string_literal(T[cursor]);
                    seg.bytes.insert(seg.bytes.end(), bytes.begin(), bytes.end());
                    cursor++;
                } else {
                    cursor++;
                }
            }
            if (!offsetSet) seg.offset = 0;
            M.data_segments.push_back(std::move(seg));
            i = j + 1;
        } else if (T[i] == "(" && i + 1 < T.size() && T[i + 1] == "table") {
            size_t j = find_matching_paren(T, i);
            size_t cursor = i + 2;
            if (cursor < j && !T[cursor].empty() && T[cursor][0] == '$') cursor++;
            int initial = 0;
            if (cursor < j && is_int_token(T[cursor])) {
                initial = parse_i32(T[cursor]);
                cursor++;
            }
            if (initial < 0) initial = 0;
            M.table.assign(static_cast<size_t>(initial), -1);
            i = j + 1;
        } else if (T[i] == "(" && i + 1 < T.size() && T[i + 1] == "elem") {
            size_t j = find_matching_paren(T, i);
            size_t cursor = i + 2;
            ElemSegment seg;
            if (cursor < j && T[cursor] == "(" && cursor + 1 < j && T[cursor + 1] == "i32.const") {
                if (cursor + 2 >= j) throw runtime_error("elem offset missing");
                seg.offset = static_cast<uint32_t>(parse_i32(T[cursor + 2]));
                cursor = find_matching_paren(T, cursor) + 1;
            }
            while (cursor < j) {
                const string& tok = T[cursor];
                if (!tok.empty() && tok[0] == '$') {
                    ElemRef ref;
                    ref.is_index = false;
                    ref.name = tok;
                    seg.entries.push_back(ref);
                } else if (is_int_token(tok)) {
                    ElemRef ref;
                    ref.is_index = true;
                    ref.index = parse_i32(tok);
                    seg.entries.push_back(ref);
                }
                cursor++;
            }
            M.elem_segments.push_back(std::move(seg));
            i = j + 1;
        } else if (T[i] == "(" && i + 1 < T.size() && T[i + 1] == "memory") {
            size_t j = find_matching_paren(T, i);
            size_t cursor = i + 2;
            if (cursor < j && !T[cursor].empty() && T[cursor][0] == '$') cursor++;
            int initial_pages = -1;
            while (cursor < j) {
                if (is_int_token(T[cursor])) {
                    if (initial_pages < 0) {
                        initial_pages = parse_i32(T[cursor]);
                    }
                }
                cursor++;
            }
            if (initial_pages < 0) initial_pages = 1;
            size_t page_size = 65536;
            M.memory.assign(static_cast<size_t>(initial_pages) * page_size, 0);
            i = j + 1;
        } else if (T[i] == "(" && i + 1 < T.size() && T[i + 1] == "func") {
            size_t j = find_matching_paren(T, i);
            size_t cursor = i + 2;
            string funcName;

            while (cursor < j) {
                const string& tok = T[cursor];
                if (tok == "(" && cursor + 1 < j && !T[cursor + 1].empty() && T[cursor + 1][0] == ';') {
                    cursor = find_matching_paren(T, cursor) + 1;
                    continue;
                }
                if (!tok.empty() && tok[0] == ';') {
                    cursor++;
                    continue;
                }
                if (!tok.empty() && tok[0] == '$') {
                    funcName = tok;
                    cursor++;
                }
                break;
            }

            vector<string> bodyTokens;
            bodyTokens.reserve(j - cursor);
            for (size_t k = cursor; k < j; ++k) {
                const string& tok = T[k];
                if (tok.size() >= 2 && tok[0] == ';' && tok[1] == ';') continue;
                bodyTokens.push_back(tok);
            }

            Function fn;
            fn.name = funcName;

            parse_function_declarations(fn, bodyTokens, M);
            parse_function_body(bodyTokens, fn);

            if (fn.type_index < 0) {
                fn.type_index = ensure_type_index(M, fn.param_types, fn.result_types);
            }

            int fnIndex = static_cast<int>(M.funcs.size());
            if (!fn.name.empty()) M.func_by_name[fn.name] = fnIndex;
            M.funcs.push_back(std::move(fn));

            i = j + 1;
        } else if (T[i] == "(" && i + 1 < T.size() && T[i + 1] == "export") {
            size_t j = find_matching_paren(T, i);
            i = (j < T.size() ? j + 1 : T.size());
        } else {
            ++i;
        }
    }

    for (auto& fn : M.funcs) {
        if (fn.type_index < 0) {
            fn.type_index = ensure_type_index(M, fn.param_types, fn.result_types);
        }
    }

    resolve_indirect_types(M);
    apply_data_segments(M);
    apply_elem_segments(M);

    return M;
}
