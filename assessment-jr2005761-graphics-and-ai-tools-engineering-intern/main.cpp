#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

enum class ValueType { I32, I64, F32, F64 };

struct Value {
    ValueType type{ValueType::I32};
    union {
        int32_t i32;
        int64_t i64;
        float f32;
        double f64;
    } data;

    Value() { data.i64 = 0; }

    static Value make_i32(int32_t v) {
        Value out;
        out.type = ValueType::I32;
        out.data.i32 = v;
        return out;
    }

    static Value make_i64(int64_t v) {
        Value out;
        out.type = ValueType::I64;
        out.data.i64 = v;
        return out;
    }

    static Value make_f32(float v) {
        Value out;
        out.type = ValueType::F32;
        out.data.f32 = v;
        return out;
    }

    static Value make_f64(double v) {
        Value out;
        out.type = ValueType::F64;
        out.data.f64 = v;
        return out;
    }
};

struct MaybeValue {
    bool has{false};
    Value value;
};

struct Module;

struct FuncType {
    vector<ValueType> params;
    vector<ValueType> results;
};

enum class Op {
    // numeric (i32)
    I32Const, I32Add, I32Sub, I32Mul, I32DivS, I32DivU, I32RemS, I32RemU,
    I32And, I32Or, I32Xor, I32Shl, I32ShrS, I32ShrU,
    I32Clz, I32Ctz, I32Popcnt, I32Rotl, I32Rotr,
    // numeric (i64)
    I64Const, I64Add, I64Sub, I64Mul, I64DivS, I64DivU, I64RemS, I64RemU,
    I64And, I64Or, I64Xor, I64Shl, I64ShrS, I64ShrU,
    I64Clz, I64Ctz, I64Popcnt, I64Rotl, I64Rotr,
    I64Eq, I64Ne, I64LtS, I64LtU, I64GtS, I64GtU, I64LeS, I64LeU, I64GeS, I64GeU, I64Eqz,
    // floating point
    F32Const, F32Add, F32Sub, F32Mul, F32Div, F32Min, F32Max,
    F32Abs, F32Neg, F32Sqrt, F32Ceil, F32Floor, F32Trunc, F32Nearest, F32Copysign,
    F32Eq, F32Ne, F32Lt, F32Gt, F32Le, F32Ge,
    F64Const, F64Add, F64Sub, F64Mul, F64Div, F64Min, F64Max,
    F64Abs, F64Neg, F64Sqrt, F64Ceil, F64Floor, F64Trunc, F64Nearest, F64Copysign,
    F64Eq, F64Ne, F64Lt, F64Gt, F64Le, F64Ge,
    // memory
    I32Load, I32Store, I32Load8U, I32Load8S, I32Store8, I32Load16U, I32Load16S, I32Store16,
    I64Load, I64Store, I64Load8U, I64Load8S, I64Load16U, I64Load16S, I64Load32U, I64Load32S,
    I64Store8, I64Store16, I64Store32,
    F32Load, F32Store, F64Load, F64Store,
    MemorySize, MemoryGrow,
    // vars
    LocalGet, LocalSet, LocalTee, GlobalGet, GlobalSet,
    // conversions / reinterpret
    F32ConvertI32S, F32ConvertI32U, F32ConvertI64S, F32ConvertI64U,
    F64ConvertI32S, F64ConvertI32U, F64ConvertI64S, F64ConvertI64U,
    I32TruncF32S, I32TruncF32U, I32TruncF64S, I32TruncF64U,
    I64TruncF32S, I64TruncF32U, I64TruncF64S, I64TruncF64U,
    F64PromoteF32, F32DemoteF64,
    I32WrapI64, I64ExtendI32S, I64ExtendI32U,
    I32ReinterpretF32, F32ReinterpretI32,
    I64ReinterpretF64, F64ReinterpretI64,
    // comparisons / logic (i32)
    Eq, Ne, LtS, LtU, GtS, GtU, LeS, LeU, GeS, GeU, Eqz, Select,
    // control
    Block, Loop, If, Else, End, Br, BrIf, BrTable,
    Return, Drop, Call, CallIndirect,
    // misc
    Unreachable, Nop, Invalid
};

struct Instr {
    Op op{Op::Invalid};
    int32_t iarg{0};
    int64_t larg{0};
    double farg{0.0};
    string sarg;
    vector<int32_t> table;
    int match_end{-1};
    int match_else{-1};
};

struct Function {
    string name;
    vector<Instr> code;
    vector<ValueType> param_types;
    vector<string> param_names;
    vector<ValueType> local_types;
    vector<string> local_names;
    vector<ValueType> result_types;
    unordered_map<string, int> local_lookup;
    int type_index{-1};
    bool is_host{false};
    MaybeValue (*host_impl)(Module&, vector<uint8_t>&, const vector<Value>& args){nullptr};

    int total_vars() const {
        return static_cast<int>(param_types.size() + local_types.size());
    }

    ValueType var_type(int idx) const {
        if (idx < 0 || idx >= total_vars()) {
            throw runtime_error("local index out of range");
        }
        if (idx < static_cast<int>(param_types.size())) {
            return param_types[idx];
        }
        return local_types[idx - static_cast<int>(param_types.size())];
    }

    int resolve_local(const string& tok) const;
};

struct ElemRef {
    bool is_index{false};
    int index{0};
    string name;
};

struct ElemSegment {
    uint32_t offset{0};
    vector<ElemRef> entries;
};

struct DataSegment {
    uint32_t offset{0};
    vector<uint8_t> bytes;
};

struct Module {
    vector<FuncType> types;
    unordered_map<string, int> type_by_name;
    vector<Function> funcs;
    unordered_map<string, int32_t> globals;
    vector<uint8_t> memory;
    unordered_map<string, int> func_by_name;
    vector<int> table;
    vector<DataSegment> data_segments;
    vector<ElemSegment> elem_segments;
};

// ---------------- Tokenizer ----------------
static vector<string> tokenize(const string& src) {
    vector<string> out;
    string cur;
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

        if (isspace(static_cast<unsigned char>(c)) || c == '(' || c == ')') {
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

static const char* type_name(ValueType t) {
    switch (t) {
        case ValueType::I32: return "i32";
        case ValueType::I64: return "i64";
        case ValueType::F32: return "f32";
        case ValueType::F64: return "f64";
    }
    return "?";
}

static Value make_zero(ValueType t) {
    switch (t) {
        case ValueType::I32: return Value::make_i32(0);
        case ValueType::I64: return Value::make_i64(0);
        case ValueType::F32: return Value::make_f32(0.0f);
        case ValueType::F64: return Value::make_f64(0.0);
    }
    throw runtime_error("unknown value type");
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

static MaybeValue wasi_fd_write(Module& module,
                                vector<uint8_t>& mem,
                                const vector<Value>& args);

// ---------------- Opcode table ----------------
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

static bool looks_like_func_definition(const vector<string>& body) {
    for (const auto& t : body) {
        if (!t.empty() && t[0] == ';') continue;
        if (opFrom(t) != Op::Invalid) return true;
    }
    return false;
}

static void skip_optional_result_sig(const vector<string>& T, size_t& k) {
    if (k < T.size() && T[k] == "(" && k + 1 < T.size() && T[k + 1] == "result") {
        size_t r = find_matching_paren(T, k);
        k = r + 1;
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

static MaybeValue wasi_fd_write(Module& /*module*/,
                                vector<uint8_t>& mem,
                                const vector<Value>& args) {
    if (args.size() != 4) throw runtime_error("fd_write requires 4 arguments");
    for (const auto& arg : args) {
        if (arg.type != ValueType::I32) throw runtime_error("fd_write expects i32 arguments");
    }

    auto check = [&](uint64_t addr, uint64_t size) {
        if (addr + size > mem.size()) throw runtime_error("fd_write out of bounds");
    };
    auto load32 = [&](uint32_t addr) -> uint32_t {
        check(addr, 4);
        return static_cast<uint32_t>(mem[addr]) |
               (static_cast<uint32_t>(mem[addr + 1]) << 8) |
               (static_cast<uint32_t>(mem[addr + 2]) << 16) |
               (static_cast<uint32_t>(mem[addr + 3]) << 24);
    };
    auto store32 = [&](uint32_t addr, uint32_t value) {
        check(addr, 4);
        mem[addr] = static_cast<uint8_t>(value & 0xFF);
        mem[addr + 1] = static_cast<uint8_t>((value >> 8) & 0xFF);
        mem[addr + 2] = static_cast<uint8_t>((value >> 16) & 0xFF);
        mem[addr + 3] = static_cast<uint8_t>((value >> 24) & 0xFF);
    };

    int32_t fd = args[0].data.i32;
    uint32_t iovs_ptr = static_cast<uint32_t>(args[1].data.i32);
    uint32_t iovs_len = static_cast<uint32_t>(args[2].data.i32);
    uint32_t nwritten_ptr = static_cast<uint32_t>(args[3].data.i32);

    string buffer;
    size_t total = 0;

    for (uint32_t idx = 0; idx < iovs_len; ++idx) {
        uint32_t entry_ptr = iovs_ptr + idx * 8u;
        check(entry_ptr, 8);
        uint32_t base = load32(entry_ptr);
        uint32_t len = load32(entry_ptr + 4);
        if (len == 0) continue;
        check(base, len);
        buffer.reserve(buffer.size() + len);
        for (uint32_t off = 0; off < len; ++off) {
            buffer.push_back(static_cast<char>(mem[base + off]));
        }
        total += len;
    }

    if (!buffer.empty() && (fd == 1 || fd == 2)) {
        cout.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        cout.flush();
    }

    if (nwritten_ptr) {
        store32(nwritten_ptr, static_cast<uint32_t>(total));
    }

    MaybeValue ret;
    ret.has = true;
    ret.value = Value::make_i32(0);
    return ret;
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

static Module parse_module(const string& path) {
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

            if (!looks_like_func_definition(bodyTokens)) {
                i = j + 1;
                continue;
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

// ---------------- Runtime helpers ----------------
static inline int clz32(uint32_t x) { return x ? __builtin_clz(x) : 32; }
static inline int ctz32(uint32_t x) { return x ? __builtin_ctz(x) : 32; }
static inline int popcnt32(uint32_t x) { return __builtin_popcount(x); }
static inline uint32_t rotl32(uint32_t x, uint32_t r) { r &= 31u; return (x << r) | (x >> (32 - r)); }
static inline uint32_t rotr32(uint32_t x, uint32_t r) { r &= 31u; return (x >> r) | (x << (32 - r)); }

static inline int clz64(uint64_t x) { return x ? __builtin_clzll(x) : 64; }
static inline int ctz64(uint64_t x) { return x ? __builtin_ctzll(x) : 64; }
static inline int popcnt64(uint64_t x) { return __builtin_popcountll(x); }
static inline uint64_t rotl64(uint64_t x, uint64_t r) { r &= 63u; return (x << r) | (x >> (64 - r)); }
static inline uint64_t rotr64(uint64_t x, uint64_t r) { r &= 63u; return (x >> r) | (x << (64 - r)); }

static int32_t trunc_f32_s_checked(float v) {
    if (isnan(v) || isinf(v)) throw runtime_error("i32.trunc_f32_s invalid");
    double d = trunc(static_cast<double>(v));
    if (d < numeric_limits<int32_t>::min() || d > numeric_limits<int32_t>::max()) {
        throw runtime_error("i32.trunc_f32_s overflow");
    }
    return static_cast<int32_t>(d);
}

static int32_t trunc_f32_u_checked(float v) {
    if (isnan(v) || isinf(v)) throw runtime_error("i32.trunc_f32_u invalid");
    double d = trunc(static_cast<double>(v));
    if (d < 0.0 || d > numeric_limits<uint32_t>::max()) {
        throw runtime_error("i32.trunc_f32_u overflow");
    }
    uint32_t u = static_cast<uint32_t>(d);
    return static_cast<int32_t>(u);
}

static int32_t trunc_f64_s_checked(double v) {
    if (isnan(v) || isinf(v)) throw runtime_error("i32.trunc_f64_s invalid");
    double d = trunc(v);
    if (d < numeric_limits<int32_t>::min() || d > numeric_limits<int32_t>::max()) {
        throw runtime_error("i32.trunc_f64_s overflow");
    }
    return static_cast<int32_t>(d);
}

static int32_t trunc_f64_u_checked(double v) {
    if (isnan(v) || isinf(v)) throw runtime_error("i32.trunc_f64_u invalid");
    double d = trunc(v);
    if (d < 0.0 || d > numeric_limits<uint32_t>::max()) {
        throw runtime_error("i32.trunc_f64_u overflow");
    }
    uint32_t u = static_cast<uint32_t>(d);
    return static_cast<int32_t>(u);
}

static int64_t trunc_f32_i64_s(float v) {
    if (isnan(v) || isinf(v)) throw runtime_error("i64.trunc_f32_s invalid");
    double d = trunc(static_cast<double>(v));
    if (d < static_cast<double>(numeric_limits<int64_t>::min()) ||
        d > static_cast<double>(numeric_limits<int64_t>::max())) {
        throw runtime_error("i64.trunc_f32_s overflow");
    }
    return static_cast<int64_t>(d);
}

static uint64_t trunc_f32_i64_u(float v) {
    if (isnan(v) || isinf(v)) throw runtime_error("i64.trunc_f32_u invalid");
    double d = trunc(static_cast<double>(v));
    if (d < 0.0 || d > static_cast<double>(numeric_limits<uint64_t>::max())) {
        throw runtime_error("i64.trunc_f32_u overflow");
    }
    return static_cast<uint64_t>(d);
}

static int64_t trunc_f64_i64_s(double v) {
    if (isnan(v) || isinf(v)) throw runtime_error("i64.trunc_f64_s invalid");
    double d = trunc(v);
    if (d < static_cast<double>(numeric_limits<int64_t>::min()) ||
        d > static_cast<double>(numeric_limits<int64_t>::max())) {
        throw runtime_error("i64.trunc_f64_s overflow");
    }
    return static_cast<int64_t>(d);
}

static uint64_t trunc_f64_i64_u(double v) {
    if (isnan(v) || isinf(v)) throw runtime_error("i64.trunc_f64_u invalid");
    double d = trunc(v);
    if (d < 0.0 || d > static_cast<double>(numeric_limits<uint64_t>::max())) {
        throw runtime_error("i64.trunc_f64_u overflow");
    }
    return static_cast<uint64_t>(d);
}

struct Label {
    int break_ip;
    int continue_ip;
    bool is_loop{false};
};

static MaybeValue exec_fn(Module& module,
                          int fn_index,
                          unordered_map<string, int32_t>& globals,
                          vector<uint8_t>& mem,
                          const vector<Value>& args);

static MaybeValue exec_fn(Module& module,
                          int fn_index,
                          unordered_map<string, int32_t>& globals,
                          vector<uint8_t>& mem) {
    static const vector<Value> empty_args;
    return exec_fn(module, fn_index, globals, mem, empty_args);
}

static MaybeValue exec_fn(Module& module,
                          int fn_index,
                          unordered_map<string, int32_t>& globals,
                          vector<uint8_t>& mem,
                          const vector<Value>& args) {
    if (fn_index < 0 || fn_index >= static_cast<int>(module.funcs.size())) {
        throw runtime_error("invalid function index");
    }
    const Function& fn = module.funcs[fn_index];

    if (args.size() != fn.param_types.size()) {
        throw runtime_error("argument count mismatch");
    }

    vector<Value> st;
    st.reserve(128);

    vector<Value> locals(fn.total_vars());
    for (size_t i = 0; i < fn.param_types.size(); ++i) {
        if (args[i].type != fn.param_types[i]) {
            throw runtime_error("argument type mismatch: expected " + string(type_name(fn.param_types[i])));
        }
        locals[i] = args[i];
    }
    for (size_t i = fn.param_types.size(); i < locals.size(); ++i) {
        locals[i] = make_zero(fn.var_type(static_cast<int>(i)));
    }

    auto get_local_ref = [&](int idx) -> Value& {
        if (idx < 0 || idx >= static_cast<int>(locals.size())) throw runtime_error("local index out of range");
        return locals[idx];
    };

    auto push = [&](const Value& v) { st.push_back(v); };
    auto pop = [&]() -> Value {
        if (st.empty()) throw runtime_error("stack underflow");
        Value v = st.back();
        st.pop_back();
        return v;
    };
    auto pop_expect = [&](ValueType expected) -> Value {
        Value v = pop();
        if (v.type != expected) throw runtime_error("type mismatch");
        return v;
    };
    auto pop_i32 = [&]() -> int32_t {
        return pop_expect(ValueType::I32).data.i32;
    };
    auto pop_i64 = [&]() -> int64_t {
        return pop_expect(ValueType::I64).data.i64;
    };
    auto pop_f32 = [&]() -> float {
        return pop_expect(ValueType::F32).data.f32;
    };
    auto pop_f64 = [&]() -> double {
        return pop_expect(ValueType::F64).data.f64;
    };

    auto check = [&](uint64_t a, uint64_t n) {
        if (a + n > mem.size()) throw runtime_error("mem OOB");
    };

    auto load32 = [&](uint32_t a) {
        check(a, 4);
        uint32_t v = static_cast<uint32_t>(mem[a]) |
                     (static_cast<uint32_t>(mem[a + 1]) << 8) |
                     (static_cast<uint32_t>(mem[a + 2]) << 16) |
                     (static_cast<uint32_t>(mem[a + 3]) << 24);
        return static_cast<int32_t>(v);
    };
    auto store32 = [&](uint32_t a, int32_t v) {
        check(a, 4);
        mem[a] = static_cast<uint8_t>(v & 0xFF);
        mem[a + 1] = static_cast<uint8_t>((v >> 8) & 0xFF);
        mem[a + 2] = static_cast<uint8_t>((v >> 16) & 0xFF);
        mem[a + 3] = static_cast<uint8_t>((v >> 24) & 0xFF);
    };
    auto load64 = [&](uint32_t a) {
        check(a, 8);
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= static_cast<uint64_t>(mem[a + i]) << (8 * i);
        }
        return static_cast<int64_t>(v);
    };
    auto store64 = [&](uint32_t a, int64_t v) {
        check(a, 8);
        uint64_t u = static_cast<uint64_t>(v);
        for (int i = 0; i < 8; ++i) {
            mem[a + i] = static_cast<uint8_t>((u >> (8 * i)) & 0xFF);
        }
    };
    auto load8u = [&](uint32_t a) {
        check(a, 1);
        return static_cast<int32_t>(mem[a]);
    };
    auto load8s = [&](uint32_t a) {
        check(a, 1);
        return static_cast<int32_t>(static_cast<int8_t>(mem[a]));
    };
    auto store8 = [&](uint32_t a, int64_t v) {
        check(a, 1);
        mem[a] = static_cast<uint8_t>(v & 0xFF);
    };
    auto load16u = [&](uint32_t a) {
        check(a, 2);
        uint32_t v = static_cast<uint32_t>(mem[a]) | (static_cast<uint32_t>(mem[a + 1]) << 8);
        return static_cast<int32_t>(v);
    };
    auto load16s = [&](uint32_t a) {
        check(a, 2);
        int16_t w = static_cast<int16_t>(static_cast<uint16_t>(mem[a]) | (static_cast<uint16_t>(mem[a + 1]) << 8));
        return static_cast<int32_t>(w);
    };
    auto store16 = [&](uint32_t a, int64_t v) {
        check(a, 2);
        mem[a] = static_cast<uint8_t>(v & 0xFF);
        mem[a + 1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    };

    vector<Label> labels;
    auto push_block = [&](int ip, bool is_loop) {
        Label L;
        L.is_loop = is_loop;
        L.continue_ip = ip + 1;
        int end_ip = fn.code[ip].match_end;
        L.break_ip = (end_ip < 0 ? static_cast<int>(fn.code.size()) : end_ip + 1);
        labels.push_back(L);
    };

    int ip = 0;
    while (ip >= 0 && ip < static_cast<int>(fn.code.size())) {
        const Instr& ins = fn.code[ip];

        switch (ins.op) {
            // ---------- numeric ----------
            case Op::I32Const: push(Value::make_i32(ins.iarg)); ip++; break;
            case Op::I32Add: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a + b)); ip++; } break;
            case Op::I32Sub: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a - b)); ip++; } break;
            case Op::I32Mul: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a * b)); ip++; } break;
            case Op::I32DivS: {
                int b = pop_i32(), a = pop_i32();
                if (b == 0) throw runtime_error("div0");
                if (a == numeric_limits<int32_t>::min() && b == -1) throw runtime_error("overflow");
                push(Value::make_i32(a / b)); ip++;
            } break;
            case Op::I32DivU: {
                uint32_t b = static_cast<uint32_t>(pop_i32());
                uint32_t a = static_cast<uint32_t>(pop_i32());
                if (!b) throw runtime_error("div0");
                push(Value::make_i32(static_cast<int32_t>(a / b))); ip++;
            } break;
            case Op::I32RemS: {
                int b = pop_i32(), a = pop_i32();
                if (!b) throw runtime_error("rem0");
                if (a == numeric_limits<int32_t>::min() && b == -1) { push(Value::make_i32(0)); ip++; break; }
                push(Value::make_i32(a % b)); ip++;
            } break;
            case Op::I32RemU: {
                uint32_t b = static_cast<uint32_t>(pop_i32());
                uint32_t a = static_cast<uint32_t>(pop_i32());
                if (!b) throw runtime_error("rem0");
                push(Value::make_i32(static_cast<int32_t>(a % b))); ip++;
            } break;
            case Op::I32And: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a & b)); ip++; } break;
            case Op::I32Or:  { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a | b)); ip++; } break;
            case Op::I32Xor: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a ^ b)); ip++; } break;
            case Op::I32Shl: { uint32_t b = static_cast<uint32_t>(pop_i32()), a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(static_cast<int32_t>(a << (b & 31u)))); ip++; } break;
            case Op::I32ShrS: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a >> (static_cast<uint32_t>(b) & 31u))); ip++; } break;
            case Op::I32ShrU: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(static_cast<int32_t>(static_cast<uint32_t>(a) >> (static_cast<uint32_t>(b) & 31u)))); ip++; } break;
            case Op::I32Clz: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(clz32(a))); ip++; } break;
            case Op::I32Ctz: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(ctz32(a))); ip++; } break;
            case Op::I32Popcnt: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(popcnt32(a))); ip++; } break;
            case Op::I32Rotl: { uint32_t b = static_cast<uint32_t>(pop_i32()), a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(static_cast<int32_t>(rotl32(a, b)))); ip++; } break;
            case Op::I32Rotr: { uint32_t b = static_cast<uint32_t>(pop_i32()), a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(static_cast<int32_t>(rotr32(a, b)))); ip++; } break;

            case Op::I64Const: push(Value::make_i64(ins.larg)); ip++; break;
            case Op::I64Add: { int64_t b = pop_i64(), a = pop_i64(); push(Value::make_i64(a + b)); ip++; } break;
            case Op::I64Sub: { int64_t b = pop_i64(), a = pop_i64(); push(Value::make_i64(a - b)); ip++; } break;
            case Op::I64Mul: { int64_t b = pop_i64(), a = pop_i64(); push(Value::make_i64(a * b)); ip++; } break;
            case Op::I64DivS: {
                int64_t b = pop_i64(), a = pop_i64();
                if (b == 0) throw runtime_error("div0");
                if (a == numeric_limits<int64_t>::min() && b == -1) throw runtime_error("overflow");
                push(Value::make_i64(a / b)); ip++;
            } break;
            case Op::I64DivU: {
                uint64_t b = static_cast<uint64_t>(pop_i64());
                uint64_t a = static_cast<uint64_t>(pop_i64());
                if (!b) throw runtime_error("div0");
                push(Value::make_i64(static_cast<int64_t>(a / b))); ip++;
            } break;
            case Op::I64RemS: {
                int64_t b = pop_i64(), a = pop_i64();
                if (!b) throw runtime_error("rem0");
                if (a == numeric_limits<int64_t>::min() && b == -1) { push(Value::make_i64(0)); ip++; break; }
                push(Value::make_i64(a % b)); ip++;
            } break;
            case Op::I64RemU: {
                uint64_t b = static_cast<uint64_t>(pop_i64());
                uint64_t a = static_cast<uint64_t>(pop_i64());
                if (!b) throw runtime_error("rem0");
                push(Value::make_i64(static_cast<int64_t>(a % b))); ip++;
            } break;
            case Op::I64And: { int64_t b = pop_i64(), a = pop_i64(); push(Value::make_i64(a & b)); ip++; } break;
            case Op::I64Or:  { int64_t b = pop_i64(), a = pop_i64(); push(Value::make_i64(a | b)); ip++; } break;
            case Op::I64Xor: { int64_t b = pop_i64(), a = pop_i64(); push(Value::make_i64(a ^ b)); ip++; } break;
            case Op::I64Shl: { uint64_t b = static_cast<uint64_t>(pop_i64()), a = static_cast<uint64_t>(pop_i64()); push(Value::make_i64(static_cast<int64_t>(a << (b & 63u)))); ip++; } break;
            case Op::I64ShrS: { int64_t b = pop_i64(), a = pop_i64(); push(Value::make_i64(a >> (static_cast<uint64_t>(b) & 63u))); ip++; } break;
            case Op::I64ShrU: { int64_t b = pop_i64(), a = pop_i64(); push(Value::make_i64(static_cast<int64_t>(static_cast<uint64_t>(a) >> (static_cast<uint64_t>(b) & 63u)))); ip++; } break;
            case Op::I64Clz: { uint64_t a = static_cast<uint64_t>(pop_i64()); push(Value::make_i64(clz64(a))); ip++; } break;
            case Op::I64Ctz: { uint64_t a = static_cast<uint64_t>(pop_i64()); push(Value::make_i64(ctz64(a))); ip++; } break;
            case Op::I64Popcnt: { uint64_t a = static_cast<uint64_t>(pop_i64()); push(Value::make_i64(popcnt64(a))); ip++; } break;
            case Op::I64Rotl: { uint64_t b = static_cast<uint64_t>(pop_i64()), a = static_cast<uint64_t>(pop_i64()); push(Value::make_i64(static_cast<int64_t>(rotl64(a, b)))); ip++; } break;
            case Op::I64Rotr: { uint64_t b = static_cast<uint64_t>(pop_i64()), a = static_cast<uint64_t>(pop_i64()); push(Value::make_i64(static_cast<int64_t>(rotr64(a, b)))); ip++; } break;
            case Op::I64Eq: { int64_t b = pop_i64(), a = pop_i64(); push(Value::make_i32(a == b)); ip++; } break;
            case Op::I64Ne: { int64_t b = pop_i64(), a = pop_i64(); push(Value::make_i32(a != b)); ip++; } break;
            case Op::I64LtS: { int64_t b = pop_i64(), a = pop_i64(); push(Value::make_i32(a < b)); ip++; } break;
            case Op::I64LtU: { uint64_t b = static_cast<uint64_t>(pop_i64()), a = static_cast<uint64_t>(pop_i64()); push(Value::make_i32(a < b)); ip++; } break;
            case Op::I64GtS: { int64_t b = pop_i64(), a = pop_i64(); push(Value::make_i32(a > b)); ip++; } break;
            case Op::I64GtU: { uint64_t b = static_cast<uint64_t>(pop_i64()), a = static_cast<uint64_t>(pop_i64()); push(Value::make_i32(a > b)); ip++; } break;
            case Op::I64LeS: { int64_t b = pop_i64(), a = pop_i64(); push(Value::make_i32(a <= b)); ip++; } break;
            case Op::I64LeU: { uint64_t b = static_cast<uint64_t>(pop_i64()), a = static_cast<uint64_t>(pop_i64()); push(Value::make_i32(a <= b)); ip++; } break;
            case Op::I64GeS: { int64_t b = pop_i64(), a = pop_i64(); push(Value::make_i32(a >= b)); ip++; } break;
            case Op::I64GeU: { uint64_t b = static_cast<uint64_t>(pop_i64()), a = static_cast<uint64_t>(pop_i64()); push(Value::make_i32(a >= b)); ip++; } break;
            case Op::I64Eqz: { int64_t a = pop_i64(); push(Value::make_i32(a == 0)); ip++; } break;

            case Op::F32Const: push(Value::make_f32(static_cast<float>(ins.farg))); ip++; break;
            case Op::F32Add: { float b = pop_f32(), a = pop_f32(); push(Value::make_f32(a + b)); ip++; } break;
            case Op::F32Sub: { float b = pop_f32(), a = pop_f32(); push(Value::make_f32(a - b)); ip++; } break;
            case Op::F32Mul: { float b = pop_f32(), a = pop_f32(); push(Value::make_f32(a * b)); ip++; } break;
            case Op::F32Div: { float b = pop_f32(), a = pop_f32(); push(Value::make_f32(a / b)); ip++; } break;
            case Op::F32Min: { float b = pop_f32(), a = pop_f32(); push(Value::make_f32(std::fmin(a, b))); ip++; } break;
            case Op::F32Max: { float b = pop_f32(), a = pop_f32(); push(Value::make_f32(std::fmax(a, b))); ip++; } break;
            case Op::F32Abs: { float a = pop_f32(); push(Value::make_f32(std::fabs(a))); ip++; } break;
            case Op::F32Neg: { float a = pop_f32(); push(Value::make_f32(-a)); ip++; } break;
            case Op::F32Sqrt: { float a = pop_f32(); push(Value::make_f32(std::sqrt(a))); ip++; } break;
            case Op::F32Ceil: { float a = pop_f32(); push(Value::make_f32(std::ceil(a))); ip++; } break;
            case Op::F32Floor: { float a = pop_f32(); push(Value::make_f32(std::floor(a))); ip++; } break;
            case Op::F32Trunc: { float a = pop_f32(); push(Value::make_f32(std::trunc(a))); ip++; } break;
            case Op::F32Nearest: { float a = pop_f32(); push(Value::make_f32(std::nearbyint(a))); ip++; } break;
            case Op::F32Copysign: { float b = pop_f32(), a = pop_f32(); push(Value::make_f32(std::copysign(a, b))); ip++; } break;
            case Op::F32Eq: { float b = pop_f32(), a = pop_f32(); push(Value::make_i32(a == b)); ip++; } break;
            case Op::F32Ne: { float b = pop_f32(), a = pop_f32(); push(Value::make_i32(a != b)); ip++; } break;
            case Op::F32Lt: { float b = pop_f32(), a = pop_f32(); push(Value::make_i32(a < b)); ip++; } break;
            case Op::F32Gt: { float b = pop_f32(), a = pop_f32(); push(Value::make_i32(a > b)); ip++; } break;
            case Op::F32Le: { float b = pop_f32(), a = pop_f32(); push(Value::make_i32(a <= b)); ip++; } break;
            case Op::F32Ge: { float b = pop_f32(), a = pop_f32(); push(Value::make_i32(a >= b)); ip++; } break;

            case Op::F64Const: push(Value::make_f64(ins.farg)); ip++; break;
            case Op::F64Add: { double b = pop_f64(), a = pop_f64(); push(Value::make_f64(a + b)); ip++; } break;
            case Op::F64Sub: { double b = pop_f64(), a = pop_f64(); push(Value::make_f64(a - b)); ip++; } break;
            case Op::F64Mul: { double b = pop_f64(), a = pop_f64(); push(Value::make_f64(a * b)); ip++; } break;
            case Op::F64Div: { double b = pop_f64(), a = pop_f64(); push(Value::make_f64(a / b)); ip++; } break;
            case Op::F64Min: { double b = pop_f64(), a = pop_f64(); push(Value::make_f64(std::fmin(a, b))); ip++; } break;
            case Op::F64Max: { double b = pop_f64(), a = pop_f64(); push(Value::make_f64(std::fmax(a, b))); ip++; } break;
            case Op::F64Abs: { double a = pop_f64(); push(Value::make_f64(std::fabs(a))); ip++; } break;
            case Op::F64Neg: { double a = pop_f64(); push(Value::make_f64(-a)); ip++; } break;
            case Op::F64Sqrt: { double a = pop_f64(); push(Value::make_f64(std::sqrt(a))); ip++; } break;
            case Op::F64Ceil: { double a = pop_f64(); push(Value::make_f64(std::ceil(a))); ip++; } break;
            case Op::F64Floor: { double a = pop_f64(); push(Value::make_f64(std::floor(a))); ip++; } break;
            case Op::F64Trunc: { double a = pop_f64(); push(Value::make_f64(std::trunc(a))); ip++; } break;
            case Op::F64Nearest: { double a = pop_f64(); push(Value::make_f64(std::nearbyint(a))); ip++; } break;
            case Op::F64Copysign: { double b = pop_f64(), a = pop_f64(); push(Value::make_f64(std::copysign(a, b))); ip++; } break;
            case Op::F64Eq: { double b = pop_f64(), a = pop_f64(); push(Value::make_i32(a == b)); ip++; } break;
            case Op::F64Ne: { double b = pop_f64(), a = pop_f64(); push(Value::make_i32(a != b)); ip++; } break;
            case Op::F64Lt: { double b = pop_f64(), a = pop_f64(); push(Value::make_i32(a < b)); ip++; } break;
            case Op::F64Gt: { double b = pop_f64(), a = pop_f64(); push(Value::make_i32(a > b)); ip++; } break;
            case Op::F64Le: { double b = pop_f64(), a = pop_f64(); push(Value::make_i32(a <= b)); ip++; } break;
            case Op::F64Ge: { double b = pop_f64(), a = pop_f64(); push(Value::make_i32(a >= b)); ip++; } break;

            // ---------- memory ----------
            case Op::I32Store: { int v = pop_i32(); uint32_t a = static_cast<uint32_t>(pop_i32()); store32(a, v); ip++; } break;
            case Op::I32Load: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(load32(a))); ip++; } break;
            case Op::I32Store8: { int v = pop_i32(); uint32_t a = static_cast<uint32_t>(pop_i32()); store8(a, v); ip++; } break;
            case Op::I32Load8U: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(load8u(a))); ip++; } break;
            case Op::I32Load8S: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(load8s(a))); ip++; } break;
            case Op::I32Store16: { int v = pop_i32(); uint32_t a = static_cast<uint32_t>(pop_i32()); store16(a, v); ip++; } break;
            case Op::I32Load16U: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(load16u(a))); ip++; } break;
            case Op::I32Load16S: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(load16s(a))); ip++; } break;
            case Op::I64Store: { int64_t v = pop_i64(); uint32_t a = static_cast<uint32_t>(pop_i32()); store64(a, v); ip++; } break;
            case Op::I64Load: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i64(load64(a))); ip++; } break;
            case Op::I64Store8: { int64_t v = pop_i64(); uint32_t a = static_cast<uint32_t>(pop_i32()); store8(a, v); ip++; } break;
            case Op::I64Store16: { int64_t v = pop_i64(); uint32_t a = static_cast<uint32_t>(pop_i32()); store16(a, v); ip++; } break;
            case Op::I64Store32: {
                int64_t v = pop_i64();
                uint32_t a = static_cast<uint32_t>(pop_i32());
                store32(a, static_cast<int32_t>(v & 0xFFFFFFFF));
                ip++;
            } break;
            case Op::I64Load8U: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i64(load8u(a))); ip++; } break;
            case Op::I64Load8S: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i64(load8s(a))); ip++; } break;
            case Op::I64Load16U: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i64(load16u(a))); ip++; } break;
            case Op::I64Load16S: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i64(load16s(a))); ip++; } break;
            case Op::I64Load32U: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i64(static_cast<uint64_t>(static_cast<uint32_t>(load32(a))))); ip++; } break;
            case Op::I64Load32S: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i64(load32(a))); ip++; } break;
            case Op::F32Store: {
                float v = pop_f32();
                uint32_t a = static_cast<uint32_t>(pop_i32());
                check(a, 4);
                uint32_t bits;
                memcpy(&bits, &v, sizeof(float));
                store32(a, static_cast<int32_t>(bits));
                ip++;
            } break;
            case Op::F32Load: {
                uint32_t a = static_cast<uint32_t>(pop_i32());
                int32_t bits = load32(a);
                float val;
                memcpy(&val, &bits, sizeof(float));
                push(Value::make_f32(val));
                ip++;
            } break;
            case Op::F64Store: {
                double v = pop_f64();
                uint32_t a = static_cast<uint32_t>(pop_i32());
                uint64_t bits;
                memcpy(&bits, &v, sizeof(double));
                store64(a, static_cast<int64_t>(bits));
                ip++;
            } break;
            case Op::F64Load: {
                uint32_t a = static_cast<uint32_t>(pop_i32());
                int64_t bits = load64(a);
                double val;
                uint64_t u = static_cast<uint64_t>(bits);
                memcpy(&val, &u, sizeof(double));
                push(Value::make_f64(val));
                ip++;
            } break;
            case Op::MemorySize: {
                int32_t pages = static_cast<int32_t>(mem.size() / 65536);
                push(Value::make_i32(pages));
                ip++;
            } break;
            case Op::MemoryGrow: {
                int32_t delta = pop_i32();
                if (delta < 0) { push(Value::make_i32(-1)); ip++; break; }
                size_t page_size = 65536;
                size_t old_pages = mem.size() / page_size;
                size_t new_pages = old_pages + static_cast<size_t>(delta);
                size_t new_size = new_pages * page_size;
                try {
                    mem.resize(new_size, 0);
                    push(Value::make_i32(static_cast<int32_t>(old_pages)));
                } catch (const bad_alloc&) {
                    push(Value::make_i32(-1));
                }
                ip++;
            } break;

            // ---------- variables ----------
            case Op::LocalGet: {
                Value v = get_local_ref(ins.iarg);
                push(v);
                ip++;
            } break;
            case Op::LocalSet: {
                Value v = pop();
                ValueType expected = fn.var_type(ins.iarg);
                if (v.type != expected) throw runtime_error("local.set type mismatch");
                get_local_ref(ins.iarg) = v;
                ip++;
            } break;
            case Op::LocalTee: {
                Value v = pop();
                ValueType expected = fn.var_type(ins.iarg);
                if (v.type != expected) throw runtime_error("local.tee type mismatch");
                get_local_ref(ins.iarg) = v;
                push(v);
                ip++;
            } break;
            case Op::GlobalGet: {
                int32_t v = globals[ins.sarg];
                push(Value::make_i32(v));
                ip++;
            } break;
            case Op::GlobalSet: {
                int32_t v = pop_i32();
                globals[ins.sarg] = v;
                ip++;
            } break;

            // ---------- conversions ----------
            case Op::F32ConvertI32S: { int32_t a = pop_i32(); push(Value::make_f32(static_cast<float>(a))); ip++; } break;
            case Op::F32ConvertI32U: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_f32(static_cast<float>(a))); ip++; } break;
            case Op::F64ConvertI32S: { int32_t a = pop_i32(); push(Value::make_f64(static_cast<double>(a))); ip++; } break;
            case Op::F64ConvertI32U: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_f64(static_cast<double>(a))); ip++; } break;
            case Op::F32ConvertI64S: { int64_t a = pop_i64(); push(Value::make_f32(static_cast<float>(a))); ip++; } break;
            case Op::F32ConvertI64U: { uint64_t a = static_cast<uint64_t>(pop_i64()); push(Value::make_f32(static_cast<float>(a))); ip++; } break;
            case Op::F64ConvertI64S: { int64_t a = pop_i64(); push(Value::make_f64(static_cast<double>(a))); ip++; } break;
            case Op::F64ConvertI64U: { uint64_t a = static_cast<uint64_t>(pop_i64()); push(Value::make_f64(static_cast<double>(a))); ip++; } break;
            case Op::I32TruncF32S: { float a = pop_f32(); push(Value::make_i32(trunc_f32_s_checked(a))); ip++; } break;
            case Op::I32TruncF32U: { float a = pop_f32(); push(Value::make_i32(trunc_f32_u_checked(a))); ip++; } break;
            case Op::I32TruncF64S: { double a = pop_f64(); push(Value::make_i32(trunc_f64_s_checked(a))); ip++; } break;
            case Op::I32TruncF64U: { double a = pop_f64(); push(Value::make_i32(trunc_f64_u_checked(a))); ip++; } break;
            case Op::I64TruncF32S: { float a = pop_f32(); push(Value::make_i64(trunc_f32_i64_s(a))); ip++; } break;
            case Op::I64TruncF32U: { float a = pop_f32(); push(Value::make_i64(static_cast<int64_t>(trunc_f32_i64_u(a)))); ip++; } break;
            case Op::I64TruncF64S: { double a = pop_f64(); push(Value::make_i64(trunc_f64_i64_s(a))); ip++; } break;
            case Op::I64TruncF64U: { double a = pop_f64(); push(Value::make_i64(static_cast<int64_t>(trunc_f64_i64_u(a)))); ip++; } break;
            case Op::F64PromoteF32: { float a = pop_f32(); push(Value::make_f64(static_cast<double>(a))); ip++; } break;
            case Op::F32DemoteF64: { double a = pop_f64(); push(Value::make_f32(static_cast<float>(a))); ip++; } break;
            case Op::I32WrapI64: {
                int64_t a = pop_i64();
                push(Value::make_i32(static_cast<int32_t>(a & 0xFFFFFFFF)));
                ip++;
            } break;
            case Op::I64ExtendI32S: { int32_t a = pop_i32(); push(Value::make_i64(static_cast<int64_t>(a))); ip++; } break;
            case Op::I64ExtendI32U: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i64(static_cast<int64_t>(a))); ip++; } break;
            case Op::I32ReinterpretF32: {
                float a = pop_f32();
                uint32_t bits;
                memcpy(&bits, &a, sizeof(float));
                push(Value::make_i32(static_cast<int32_t>(bits)));
                ip++;
            } break;
            case Op::F32ReinterpretI32: {
                int32_t bits = pop_i32();
                float val;
                memcpy(&val, &bits, sizeof(float));
                push(Value::make_f32(val));
                ip++;
            } break;
            case Op::I64ReinterpretF64: {
                double a = pop_f64();
                uint64_t bits;
                memcpy(&bits, &a, sizeof(double));
                push(Value::make_i64(static_cast<int64_t>(bits)));
                ip++;
            } break;
            case Op::F64ReinterpretI64: {
                int64_t bits = pop_i64();
                uint64_t u = static_cast<uint64_t>(bits);
                double val;
                memcpy(&val, &u, sizeof(double));
                push(Value::make_f64(val));
                ip++;
            } break;

            // ---------- comparisons (i32) ----------
            case Op::Eq: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a == b)); ip++; } break;
            case Op::Ne: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a != b)); ip++; } break;
            case Op::LtS: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a < b)); ip++; } break;
            case Op::LtU: { uint32_t b = static_cast<uint32_t>(pop_i32()), a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(a < b)); ip++; } break;
            case Op::GtS: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a > b)); ip++; } break;
            case Op::GtU: { uint32_t b = static_cast<uint32_t>(pop_i32()), a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(a > b)); ip++; } break;
            case Op::LeS: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a <= b)); ip++; } break;
            case Op::LeU: { uint32_t b = static_cast<uint32_t>(pop_i32()), a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(a <= b)); ip++; } break;
            case Op::GeS: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a >= b)); ip++; } break;
            case Op::GeU: { uint32_t b = static_cast<uint32_t>(pop_i32()), a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(a >= b)); ip++; } break;
            case Op::Eqz: { int a = pop_i32(); push(Value::make_i32(a == 0)); ip++; } break;
            case Op::Select: {
                Value condVal = pop();
                if (condVal.type != ValueType::I32) throw runtime_error("select condition must be i32");
                Value v2 = pop();
                Value v1 = pop();
                if (v1.type != v2.type) throw runtime_error("select type mismatch");
                push(condVal.data.i32 ? v1 : v2);
                ip++;
            } break;

            // ---------- control ----------
            case Op::Block:
                push_block(ip, false);
                ip++;
                break;
            case Op::Loop:
                push_block(ip, true);
                ip++;
                break;
            case Op::If: {
                push_block(ip, false);
                int cond = pop_i32();
                if (cond) {
                    ip++;
                } else {
                    if (ins.match_else != -1) ip = ins.match_else + 1;
                    else ip = (ins.match_end != -1 ? ins.match_end + 1 : ip + 1);
                }
            } break;
            case Op::Else: {
                int end_ip = ins.match_end;
                ip = (end_ip < 0 ? ip + 1 : end_ip + 1);
            } break;
            case Op::End:
                if (!labels.empty()) labels.pop_back();
                ip++;
                break;
            case Op::Br: {
                int depth = ins.iarg;
                if (depth < 0 || depth >= static_cast<int>(labels.size())) throw runtime_error("br depth");
                int idx = static_cast<int>(labels.size()) - 1 - depth;
                Label L = labels[idx];
                labels.resize(idx);
                ip = L.is_loop ? L.continue_ip : L.break_ip;
            } break;
            case Op::BrIf: {
                int cond = pop_i32();
                if (!cond) { ip++; break; }
                int depth = ins.iarg;
                if (depth < 0 || depth >= static_cast<int>(labels.size())) throw runtime_error("br_if depth");
                int idx = static_cast<int>(labels.size()) - 1 - depth;
                Label L = labels[idx];
                labels.resize(idx);
                ip = L.is_loop ? L.continue_ip : L.break_ip;
            } break;
            case Op::BrTable: {
                if (ins.table.empty()) { ip++; break; }
                int idxVal = pop_i32();
                if (idxVal < 0 || idxVal >= static_cast<int>(ins.table.size())) idxVal = static_cast<int>(ins.table.size()) - 1;
                int depth = ins.table[idxVal];
                if (depth < 0 || depth >= static_cast<int>(labels.size())) throw runtime_error("br_table depth");
                int idx = static_cast<int>(labels.size()) - 1 - depth;
                Label L = labels[idx];
                labels.resize(idx);
                ip = L.is_loop ? L.continue_ip : L.break_ip;
            } break;
            case Op::Return:
                ip = static_cast<int>(fn.code.size());
                break;
            case Op::Drop:
                pop();
                ip++;
                break;
            case Op::Call: {
                int target = ins.iarg;
                if (target < 0) {
                    auto it = module.func_by_name.find(ins.sarg);
                    if (it == module.func_by_name.end()) throw runtime_error("unknown function: " + ins.sarg);
                    target = it->second;
                }
                if (target < 0 || target >= static_cast<int>(module.funcs.size())) throw runtime_error("invalid call target");
                const Function& callee = module.funcs[target];
                vector<Value> callArgs(callee.param_types.size());
                for (size_t idx = 0; idx < callee.param_types.size(); ++idx) {
                    Value arg = pop_expect(callee.param_types[callee.param_types.size() - 1 - idx]);
                    callArgs[callee.param_types.size() - 1 - idx] = arg;
                }
                MaybeValue ret;
                if (callee.is_host) {
                    if (!callee.host_impl) throw runtime_error("host function missing implementation");
                    ret = callee.host_impl(module, mem, callArgs);
                } else {
                    ret = exec_fn(module, target, globals, mem, callArgs);
                }
                if (ret.has) push(ret.value);
                ip++;
            } break;
            case Op::CallIndirect: {
                uint32_t table_index = static_cast<uint32_t>(pop_i32());
                if (ins.iarg < 0 || ins.iarg >= static_cast<int>(module.types.size())) {
                    throw runtime_error("call_indirect bad type");
                }
                if (table_index >= module.table.size()) throw runtime_error("call_indirect OOB");
                int func_index = module.table[table_index];
                if (func_index < 0) throw runtime_error("call_indirect uninitialized");
                const Function& callee = module.funcs[func_index];
                const FuncType& expected = module.types[ins.iarg];
                if (callee.param_types != expected.params || callee.result_types != expected.results) {
                    throw runtime_error("call_indirect type mismatch");
                }
                vector<Value> callArgs(expected.params.size());
                for (size_t idx = 0; idx < expected.params.size(); ++idx) {
                    Value arg = pop_expect(expected.params[expected.params.size() - 1 - idx]);
                    callArgs[expected.params.size() - 1 - idx] = arg;
                }
                MaybeValue ret;
                if (callee.is_host) {
                    if (!callee.host_impl) throw runtime_error("host function missing implementation");
                    ret = callee.host_impl(module, mem, callArgs);
                } else {
                    ret = exec_fn(module, func_index, globals, mem, callArgs);
                }
                if (ret.has) push(ret.value);
                ip++;
            } break;

            case Op::Unreachable:
                throw runtime_error("unreachable");

            case Op::Nop:
                ip++;
                break;

            default:
                ip++;
                break;
        }
    }

    MaybeValue ret;
    if (!fn.result_types.empty()) {
        if (st.empty()) throw runtime_error("missing return value");
        Value v = st.back();
        if (v.type != fn.result_types[0]) {
            throw runtime_error("return type mismatch: expected " + string(type_name(fn.result_types[0])));
        }
        ret.has = true;
        ret.value = v;
    }
    return ret;
}

static string format_value(const Value& v) {
    ostringstream oss;
    oss << type_name(v.type) << '=';
    switch (v.type) {
        case ValueType::I32: oss << v.data.i32; break;
        case ValueType::I64: oss << v.data.i64; break;
        case ValueType::F32: oss << v.data.f32; break;
        case ValueType::F64: oss << v.data.f64; break;
    }
    return oss.str();
}

// ---------------- Main ----------------
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
