// main.cpp
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

enum class Op {
    // numeric (i32)
    I32Const, I32Add, I32Sub, I32Mul, I32DivS, I32DivU, I32RemS,
    I32And, I32Or, I32Xor, I32Shl, I32ShrS, I32ShrU,
    I32Clz, I32Ctz, I32Popcnt, I32Rotl, I32Rotr,
    // floating point
    F32Const, F32Add, F32Sub, F32Mul, F32Div, F32Min, F32Max,
    F32Abs, F32Neg, F32Sqrt, F32Ceil, F32Floor, F32Trunc, F32Nearest,
    F32Eq, F32Ne, F32Lt, F32Gt, F32Le, F32Ge,
    F64Const, F64Add, F64Mul, F64Sqrt, F64Gt,
    // memory
    I32Load, I32Store, I32Load8U, I32Load8S, I32Store8, I32Load16U, I32Load16S, I32Store16,
    MemorySize, MemoryGrow,
    // vars
    LocalGet, LocalSet, LocalTee, GlobalGet, GlobalSet,
    // conversions / reinterpret
    F32ConvertI32S, F32ConvertI32U, F64ConvertI32S,
    I32TruncF32S, I32TruncF32U, I32TruncF64S,
    F64PromoteF32, F32DemoteF64,
    I32ReinterpretF32, F32ReinterpretI32,
    // comparisons / logic (i32)
    Eq, Ne, LtS, LtU, GtS, GtU, LeS, GeS, Eqz, Select,
    // control
    Block, Loop, If, Else, End, Br, BrIf, BrTable,
    Return, Drop, Call,
    // misc
    Nop, Invalid
};

enum class ValueType { I32, F32, F64 };

struct Value {
    ValueType type{ValueType::I32};
    union {
        int32_t i32;
        float f32;
        double f64;
    } data;

    Value() { data.i32 = 0; }

    static Value make_i32(int32_t v) {
        Value out;
        out.type = ValueType::I32;
        out.data.i32 = v;
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

struct Instr {
    Op op{Op::Invalid};
    int32_t iarg{0};
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

struct Module {
    vector<Function> funcs;
    unordered_map<string, int32_t> globals;
    vector<uint8_t> memory;
    unordered_map<string, int> func_by_name;
};

// ---------------- Tokenizer ----------------
static vector<string> tokenize(const string& src) {
    vector<string> out;
    string cur;
    bool inLineComment = false;

    for (size_t i = 0; i < src.size(); ++i) {
        char c = src[i];

        if (!inLineComment && c == ';' && i + 1 < src.size() && src[i + 1] == ';') {
            inLineComment = true;
        }
        if (inLineComment) {
            if (c == '\n') inLineComment = false;
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
    if (!t.empty() && (t[0] == '-' || isdigit(static_cast<unsigned char>(t[0])))) return true;
    return false;
}

static int32_t parse_i32(const string& t) {
    if (t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X')) {
        uint32_t v = static_cast<uint32_t>(stoul(t, nullptr, 16));
        return static_cast<int32_t>(v);
    }
    return stoi(t);
}

static float parse_f32(const string& t) {
    if (t == "inf" || t == "infinity") return numeric_limits<float>::infinity();
    if (t == "-inf" || t == "-infinity") return -numeric_limits<float>::infinity();
    return stof(t);
}

static double parse_f64(const string& t) {
    if (t == "inf" || t == "infinity") return numeric_limits<double>::infinity();
    if (t == "-inf" || t == "-infinity") return -numeric_limits<double>::infinity();
    return stod(t);
}

static ValueType parse_valtype(const string& t) {
    if (t == "i32") return ValueType::I32;
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
        case ValueType::F32: return "f32";
        case ValueType::F64: return "f64";
    }
    return "?";
}

static Value make_zero(ValueType t) {
    switch (t) {
        case ValueType::I32: return Value::make_i32(0);
        case ValueType::F32: return Value::make_f32(0.0f);
        case ValueType::F64: return Value::make_f64(0.0);
    }
    throw runtime_error("unknown value type");
}

// ---------------- Opcode table ----------------
static Op opFrom(const string& tok) {
    if (tok == "i32.const") return Op::I32Const;
    if (tok == "i32.add") return Op::I32Add;
    if (tok == "i32.sub") return Op::I32Sub;
    if (tok == "i32.mul") return Op::I32Mul;
    if (tok == "i32.div_s") return Op::I32DivS;
    if (tok == "i32.div_u") return Op::I32DivU;
    if (tok == "i32.rem_s") return Op::I32RemS;
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
    if (tok == "f32.eq") return Op::F32Eq;
    if (tok == "f32.ne") return Op::F32Ne;
    if (tok == "f32.lt") return Op::F32Lt;
    if (tok == "f32.gt") return Op::F32Gt;
    if (tok == "f32.le") return Op::F32Le;
    if (tok == "f32.ge") return Op::F32Ge;

    if (tok == "f64.const") return Op::F64Const;
    if (tok == "f64.add") return Op::F64Add;
    if (tok == "f64.mul") return Op::F64Mul;
    if (tok == "f64.sqrt") return Op::F64Sqrt;
    if (tok == "f64.gt") return Op::F64Gt;

    if (tok == "i32.load") return Op::I32Load;
    if (tok == "i32.store") return Op::I32Store;
    if (tok == "i32.load8_u") return Op::I32Load8U;
    if (tok == "i32.load8_s") return Op::I32Load8S;
    if (tok == "i32.store8") return Op::I32Store8;
    if (tok == "i32.load16_u") return Op::I32Load16U;
    if (tok == "i32.load16_s") return Op::I32Load16S;
    if (tok == "i32.store16") return Op::I32Store16;
    if (tok == "memory.size") return Op::MemorySize;
    if (tok == "memory.grow") return Op::MemoryGrow;

    if (tok == "local.get") return Op::LocalGet;
    if (tok == "local.set") return Op::LocalSet;
    if (tok == "local.tee") return Op::LocalTee;
    if (tok == "global.get") return Op::GlobalGet;
    if (tok == "global.set") return Op::GlobalSet;

    if (tok == "f32.convert_i32_s") return Op::F32ConvertI32S;
    if (tok == "f32.convert_i32_u") return Op::F32ConvertI32U;
    if (tok == "f64.convert_i32_s") return Op::F64ConvertI32S;
    if (tok == "i32.trunc_f32_s") return Op::I32TruncF32S;
    if (tok == "i32.trunc_f32_u") return Op::I32TruncF32U;
    if (tok == "i32.trunc_f64_s") return Op::I32TruncF64S;
    if (tok == "f64.promote_f32") return Op::F64PromoteF32;
    if (tok == "f32.demote_f64") return Op::F32DemoteF64;
    if (tok == "i32.reinterpret_f32") return Op::I32ReinterpretF32;
    if (tok == "f32.reinterpret_i32") return Op::F32ReinterpretI32;

    if (tok == "i32.eq") return Op::Eq;
    if (tok == "i32.ne") return Op::Ne;
    if (tok == "i32.lt_s") return Op::LtS;
    if (tok == "i32.lt_u") return Op::LtU;
    if (tok == "i32.gt_s") return Op::GtS;
    if (tok == "i32.gt_u") return Op::GtU;
    if (tok == "i32.le_s") return Op::LeS;
    if (tok == "i32.ge_s") return Op::GeS;
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

static void parse_function_declarations(Function& fn, const vector<string>& tokens) {
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

            case Op::F32Const:
                if (k + 1 < bodyTokens.size()) {
                    ins.farg = parse_f32(bodyTokens[k + 1]);
                    k += 2;
                } else {
                    k++;
                }
                break;

            case Op::F64Const:
                if (k + 1 < bodyTokens.size()) {
                    ins.farg = parse_f64(bodyTokens[k + 1]);
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

// ---------------- Module parsing ----------------
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
        if (T[i] == "(" && i + 1 < T.size() && T[i + 1] == "func") {
            size_t j = find_matching_paren(T, i);
            if (j >= T.size()) break;

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

            parse_function_declarations(fn, bodyTokens);
            parse_function_body(bodyTokens, fn);

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

    return M;
}

// ---------------- Runtime helpers ----------------
static inline int clz32(uint32_t x) { return x ? __builtin_clz(x) : 32; }
static inline int ctz32(uint32_t x) { return x ? __builtin_ctz(x) : 32; }
static inline int popcnt32(uint32_t x) { return __builtin_popcount(x); }
static inline uint32_t rotl32(uint32_t x, uint32_t r) { r &= 31u; return (x << r) | (x >> (32 - r)); }
static inline uint32_t rotr32(uint32_t x, uint32_t r) { r &= 31u; return (x >> r) | (x << (32 - r)); }

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
    st.reserve(64);

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
    auto pop_i32 = [&]() -> int32_t {
        Value v = pop();
        if (v.type != ValueType::I32) throw runtime_error("expected i32");
        return v.data.i32;
    };
    auto pop_f32 = [&]() -> float {
        Value v = pop();
        if (v.type != ValueType::F32) throw runtime_error("expected f32");
        return v.data.f32;
    };
    auto pop_f64 = [&]() -> double {
        Value v = pop();
        if (v.type != ValueType::F64) throw runtime_error("expected f64");
        return v.data.f64;
    };

    auto check = [&](size_t a, size_t n) {
        if (a + n > mem.size()) throw runtime_error("mem OOB");
    };

    auto load32 = [&](uint32_t a) {
        check(a, 4);
        uint32_t v = mem[a] | (mem[a + 1] << 8) | (mem[a + 2] << 16) | (mem[a + 3] << 24);
        return static_cast<int32_t>(v);
    };
    auto store32 = [&](uint32_t a, int32_t v) {
        check(a, 4);
        mem[a] = v & 0xFF;
        mem[a + 1] = (v >> 8) & 0xFF;
        mem[a + 2] = (v >> 16) & 0xFF;
        mem[a + 3] = (v >> 24) & 0xFF;
    };
    auto load8u = [&](uint32_t a) {
        check(a, 1);
        return static_cast<int32_t>(mem[a]);
    };
    auto load8s = [&](uint32_t a) {
        check(a, 1);
        return static_cast<int32_t>(static_cast<int8_t>(mem[a]));
    };
    auto store8 = [&](uint32_t a, int32_t v) {
        check(a, 1);
        mem[a] = v & 0xFF;
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
    auto store16 = [&](uint32_t a, int32_t v) {
        check(a, 2);
        mem[a] = v & 0xFF;
        mem[a + 1] = (v >> 8) & 0xFF;
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
            case Op::I32DivS: { int b = pop_i32(), a = pop_i32(); if (!b) throw runtime_error("div0"); push(Value::make_i32(a / b)); ip++; } break;
            case Op::I32DivU: { uint32_t b = static_cast<uint32_t>(pop_i32()), a = static_cast<uint32_t>(pop_i32()); if (!b) throw runtime_error("div0"); push(Value::make_i32(static_cast<int32_t>(a / b))); ip++; } break;
            case Op::I32RemS: { int b = pop_i32(), a = pop_i32(); if (!b) throw runtime_error("rem0"); push(Value::make_i32(a % b)); ip++; } break;
            case Op::I32And: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a & b)); ip++; } break;
            case Op::I32Or:  { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a | b)); ip++; } break;
            case Op::I32Xor: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a ^ b)); ip++; } break;
            case Op::I32Shl: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(static_cast<int32_t>(static_cast<uint32_t>(a) << (static_cast<uint32_t>(b) & 31u)))); ip++; } break;
            case Op::I32ShrS: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a >> (static_cast<uint32_t>(b) & 31u))); ip++; } break;
            case Op::I32ShrU: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(static_cast<int32_t>(static_cast<uint32_t>(a) >> (static_cast<uint32_t>(b) & 31u)))); ip++; } break;
            case Op::I32Clz: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(clz32(a))); ip++; } break;
            case Op::I32Ctz: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(ctz32(a))); ip++; } break;
            case Op::I32Popcnt: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(popcnt32(a))); ip++; } break;
            case Op::I32Rotl: { uint32_t b = static_cast<uint32_t>(pop_i32()), a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(static_cast<int32_t>(rotl32(a, b)))); ip++; } break;
            case Op::I32Rotr: { uint32_t b = static_cast<uint32_t>(pop_i32()), a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(static_cast<int32_t>(rotr32(a, b)))); ip++; } break;

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
            case Op::F32Eq: { float b = pop_f32(), a = pop_f32(); push(Value::make_i32(a == b)); ip++; } break;
            case Op::F32Ne: { float b = pop_f32(), a = pop_f32(); push(Value::make_i32(a != b)); ip++; } break;
            case Op::F32Lt: { float b = pop_f32(), a = pop_f32(); push(Value::make_i32(a < b)); ip++; } break;
            case Op::F32Gt: { float b = pop_f32(), a = pop_f32(); push(Value::make_i32(a > b)); ip++; } break;
            case Op::F32Le: { float b = pop_f32(), a = pop_f32(); push(Value::make_i32(a <= b)); ip++; } break;
            case Op::F32Ge: { float b = pop_f32(), a = pop_f32(); push(Value::make_i32(a >= b)); ip++; } break;

            case Op::F64Const: push(Value::make_f64(ins.farg)); ip++; break;
            case Op::F64Add: { double b = pop_f64(), a = pop_f64(); push(Value::make_f64(a + b)); ip++; } break;
            case Op::F64Mul: { double b = pop_f64(), a = pop_f64(); push(Value::make_f64(a * b)); ip++; } break;
            case Op::F64Sqrt: { double a = pop_f64(); push(Value::make_f64(std::sqrt(a))); ip++; } break;
            case Op::F64Gt: { double b = pop_f64(), a = pop_f64(); push(Value::make_i32(a > b)); ip++; } break;

            // ---------- memory ----------
            case Op::I32Store: { int v = pop_i32(); uint32_t a = static_cast<uint32_t>(pop_i32()); store32(a, v); ip++; } break;
            case Op::I32Load: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(load32(a))); ip++; } break;
            case Op::I32Store8: { int v = pop_i32(); uint32_t a = static_cast<uint32_t>(pop_i32()); store8(a, v); ip++; } break;
            case Op::I32Load8U: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(load8u(a))); ip++; } break;
            case Op::I32Load8S: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(load8s(a))); ip++; } break;
            case Op::I32Store16: { int v = pop_i32(); uint32_t a = static_cast<uint32_t>(pop_i32()); store16(a, v); ip++; } break;
            case Op::I32Load16U: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(load16u(a))); ip++; } break;
            case Op::I32Load16S: { uint32_t a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(load16s(a))); ip++; } break;
            case Op::MemorySize: { int32_t pages = static_cast<int32_t>(mem.size() / 65536); push(Value::make_i32(pages)); ip++; } break;
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
            case Op::LocalGet: { Value v = get_local_ref(ins.iarg); push(v); ip++; } break;
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
            case Op::I32TruncF32S: { float a = pop_f32(); push(Value::make_i32(trunc_f32_s_checked(a))); ip++; } break;
            case Op::I32TruncF32U: { float a = pop_f32(); push(Value::make_i32(trunc_f32_u_checked(a))); ip++; } break;
            case Op::I32TruncF64S: { double a = pop_f64(); push(Value::make_i32(trunc_f64_s_checked(a))); ip++; } break;
            case Op::F64PromoteF32: { float a = pop_f32(); push(Value::make_f64(static_cast<double>(a))); ip++; } break;
            case Op::F32DemoteF64: { double a = pop_f64(); push(Value::make_f32(static_cast<float>(a))); ip++; } break;
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

            // ---------- comparisons (i32) ----------
            case Op::Eq: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a == b)); ip++; } break;
            case Op::Ne: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a != b)); ip++; } break;
            case Op::LtS: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a < b)); ip++; } break;
            case Op::LtU: { uint32_t b = static_cast<uint32_t>(pop_i32()), a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(a < b)); ip++; } break;
            case Op::GtS: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a > b)); ip++; } break;
            case Op::GtU: { uint32_t b = static_cast<uint32_t>(pop_i32()), a = static_cast<uint32_t>(pop_i32()); push(Value::make_i32(a > b)); ip++; } break;
            case Op::LeS: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a <= b)); ip++; } break;
            case Op::GeS: { int b = pop_i32(), a = pop_i32(); push(Value::make_i32(a >= b)); ip++; } break;
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
                    callArgs[callee.param_types.size() - 1 - idx] = pop();
                }
                MaybeValue ret = exec_fn(module, target, globals, mem, callArgs);
                if (ret.has) push(ret.value);
                ip++;
            } break;

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
