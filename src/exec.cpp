/**
 * @file exec.cpp
 * @brief Implements the WebAssembly runtime interpreter.
 *
 * This file contains the execution loop and helper routines for evaluating
 * WebAssembly instructions. It manages:
 *   - The operand stack and local variables
 *   - Memory access and bounds checking
 *   - Numeric, control flow, and conversion operations
 *   - Calls to user-defined and host (WASI) functions
 *
 * Each instruction is dispatched sequentially using an instruction pointer.
 * Control flow is managed via labeled blocks and structured branching.
 */


#include "exec.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace std;

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

MaybeValue wasi_fd_write(Module& /*module*/,
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

MaybeValue exec_fn(Module& module,
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

string format_value(const Value& v) {
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

