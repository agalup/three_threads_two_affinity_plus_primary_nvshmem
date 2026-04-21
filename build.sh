#!/usr/bin/env bash
set -euo pipefail
MPI_INC="/usr/lib/x86_64-linux-gnu/openmpi/include"
MPI_LIB="/usr/lib/x86_64-linux-gnu/openmpi/lib"
SRC="$(dirname "$0")/three_threads_two_affinity_plus_primary_nvshmem_fixed.cu"
OUT="$(dirname "$0")/three_threads_two_affinity_plus_primary_nvshmem_fixed"
LOG="$(dirname "$0")/build.log"
rm -f "$OUT"
nvcc -std=c++17 -rdc=true -arch=sm_80 -O2 -ccbin mpicxx \
    -I"$NVSHMEM_HOME/include" \
    -I"$MPI_INC" \
    -o "$OUT" \
    "$SRC" \
    "$NVSHMEM_HOME/lib/libnvshmem_device.a" \
    -L"$NVSHMEM_HOME/lib" -lnvshmem_host \
    -L"$MPI_LIB" -lmpi \
    -Xlinker -rpath -Xlinker "$NVSHMEM_HOME/lib" \
    -Xlinker -rpath -Xlinker "$MPI_LIB" \
    -lcuda > "$LOG" 2>&1
echo "built $OUT"
