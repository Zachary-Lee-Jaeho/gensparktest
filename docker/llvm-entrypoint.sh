#!/bin/bash
# LLVM Analysis Environment Entrypoint
# Provides easy access to LLVM tools and analysis scripts

set -e

# Print LLVM version info
echo "=== VEGA-Verified LLVM Analysis Environment ==="
echo "LLVM Version: $(llvm-config --version 2>/dev/null || echo 'checking...')"
echo "Clang Version: $(clang --version 2>/dev/null | head -1 || echo 'checking...')"
echo ""

# If command provided, execute it
if [ "$#" -gt 0 ]; then
    exec "$@"
else
    # Interactive mode
    exec /bin/bash
fi
