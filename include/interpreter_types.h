#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
    std::vector<ValueType> params;
    std::vector<ValueType> results;
};

enum class Op {
    I32Const,
    I32Add,
    I32Sub,
    I32Mul,
    I32DivS,
    I32DivU,
    I32RemS,
    I32RemU,
    I32And,
    I32Or,
    I32Xor,
    I32Shl,
    I32ShrS,
    I32ShrU,
    I32Clz,
    I32Ctz,
    I32Popcnt,
    I32Rotl,
    I32Rotr,
    I64Const,
    I64Add,
    I64Sub,
    I64Mul,
    I64DivS,
    I64DivU,
    I64RemS,
    I64RemU,
    I64And,
    I64Or,
    I64Xor,
    I64Shl,
    I64ShrS,
    I64ShrU,
    I64Clz,
    I64Ctz,
    I64Popcnt,
    I64Rotl,
    I64Rotr,
    I64Eq,
    I64Ne,
    I64LtS,
    I64LtU,
    I64GtS,
    I64GtU,
    I64LeS,
    I64LeU,
    I64GeS,
    I64GeU,
    I64Eqz,
    F32Const,
    F32Add,
    F32Sub,
    F32Mul,
    F32Div,
    F32Min,
    F32Max,
    F32Abs,
    F32Neg,
    F32Sqrt,
    F32Ceil,
    F32Floor,
    F32Trunc,
    F32Nearest,
    F32Copysign,
    F32Eq,
    F32Ne,
    F32Lt,
    F32Gt,
    F32Le,
    F32Ge,
    F64Const,
    F64Add,
    F64Sub,
    F64Mul,
    F64Div,
    F64Min,
    F64Max,
    F64Abs,
    F64Neg,
    F64Sqrt,
    F64Ceil,
    F64Floor,
    F64Trunc,
    F64Nearest,
    F64Copysign,
    F64Eq,
    F64Ne,
    F64Lt,
    F64Gt,
    F64Le,
    F64Ge,
    I32Load,
    I32Store,
    I32Load8U,
    I32Load8S,
    I32Store8,
    I32Load16U,
    I32Load16S,
    I32Store16,
    I64Load,
    I64Store,
    I64Load8U,
    I64Load8S,
    I64Load16U,
    I64Load16S,
    I64Load32U,
    I64Load32S,
    I64Store8,
    I64Store16,
    I64Store32,
    F32Load,
    F32Store,
    F64Load,
    F64Store,
    MemorySize,
    MemoryGrow,
    LocalGet,
    LocalSet,
    LocalTee,
    GlobalGet,
    GlobalSet,
    F32ConvertI32S,
    F32ConvertI32U,
    F32ConvertI64S,
    F32ConvertI64U,
    F64ConvertI32S,
    F64ConvertI32U,
    F64ConvertI64S,
    F64ConvertI64U,
    I32TruncF32S,
    I32TruncF32U,
    I32TruncF64S,
    I32TruncF64U,
    I64TruncF32S,
    I64TruncF32U,
    I64TruncF64S,
    I64TruncF64U,
    F64PromoteF32,
    F32DemoteF64,
    I32WrapI64,
    I64ExtendI32S,
    I64ExtendI32U,
    I32ReinterpretF32,
    F32ReinterpretI32,
    I64ReinterpretF64,
    F64ReinterpretI64,
    Eq,
    Ne,
    LtS,
    LtU,
    GtS,
    GtU,
    LeS,
    LeU,
    GeS,
    GeU,
    Eqz,
    Select,
    Block,
    Loop,
    If,
    Else,
    End,
    Br,
    BrIf,
    BrTable,
    Return,
    Drop,
    Call,
    CallIndirect,
    Unreachable,
    Nop,
    Invalid
};

struct Instr {
    Op op{Op::Invalid};
    int32_t iarg{0};
    int64_t larg{0};
    double farg{0.0};
    std::string sarg;
    std::vector<int32_t> table;
    int match_end{-1};
    int match_else{-1};
};

struct Function {
    std::string name;
    std::vector<Instr> code;
    std::vector<ValueType> param_types;
    std::vector<std::string> param_names;
    std::vector<ValueType> local_types;
    std::vector<std::string> local_names;
    std::vector<ValueType> result_types;
    std::unordered_map<std::string, int> local_lookup;
    int type_index{-1};
    bool is_host{false};
    MaybeValue (*host_impl)(Module&, std::vector<uint8_t>&, const std::vector<Value>& args){nullptr};

    int total_vars() const {
        return static_cast<int>(param_types.size() + local_types.size());
    }

    ValueType var_type(int idx) const {
        if (idx < 0 || idx >= total_vars()) {
            throw std::runtime_error("local index out of range");
        }
        if (idx < static_cast<int>(param_types.size())) {
            return param_types[idx];
        }
        return local_types[idx - static_cast<int>(param_types.size())];
    }

    int resolve_local(const std::string& tok) const;
};

struct ElemRef {
    bool is_index{false};
    int index{0};
    std::string name;
};

struct ElemSegment {
    uint32_t offset{0};
    std::vector<ElemRef> entries;
};

struct DataSegment {
    uint32_t offset{0};
    std::vector<uint8_t> bytes;
};

struct Module {
    std::vector<FuncType> types;
    std::unordered_map<std::string, int> type_by_name;
    std::vector<Function> funcs;
    std::unordered_map<std::string, int32_t> globals;
    std::vector<uint8_t> memory;
    std::unordered_map<std::string, int> func_by_name;
    std::vector<int> table;
    std::vector<DataSegment> data_segments;
    std::vector<ElemSegment> elem_segments;
};
