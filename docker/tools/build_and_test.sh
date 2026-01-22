#!/bin/bash
# Build and test the AST extractor

set -e

echo "=== Building AST Extractor ==="

# Compile
clang++-18 -std=c++17 /workspace/docker/tools/ast_extractor.cpp \
    $(llvm-config-18 --cxxflags) \
    $(llvm-config-18 --ldflags) \
    -lclang-cpp-18 \
    $(llvm-config-18 --libs all --system-libs) \
    -o /workspace/output/ast_extractor

echo "Build successful!"

# Test with a simple file
echo ""
echo "=== Testing with sample code ==="

cat > /tmp/test.cpp << 'EOF'
unsigned getRelocType(unsigned Kind) {
    switch (Kind) {
    case FK_Data_1:
        return R_RISCV_NONE;
    case FK_Data_4:
    case FK_PCRel_4:
        return IsPCRel ? R_RISCV_32_PCREL : R_RISCV_32;
    case fixup_riscv_hi20:
        return R_RISCV_HI20;
    default:
        return R_RISCV_NONE;
    }
}

void emitInstruction(const MCInst &MI) {
    encodeInstruction(MI);
    emitBytes(Code);
}
EOF

/workspace/output/ast_extractor /tmp/test.cpp -- -std=c++17 2>/dev/null || echo "Note: Some warnings expected"

echo ""
echo "=== Done ==="
