#!/usr/bin/env bash
set -u
NVSHMEM_HOME="/home/agnes/nvshmem-3.6.5-cuda13"
export NVSHMEM_HOME
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RDMAV_DRIVERS="${RDMAV_DRIVERS:-mlx5}"
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-none}"

# Attach as client to the user's canonical MPS daemon (kept always-on).
# Affinity contexts (CU_EXEC_AFFINITY_TYPE_SM_COUNT) require MPS.
export CUDA_MPS_PIPE_DIRECTORY=/home/agnes/mps/mps
export CUDA_MPS_LOG_DIRECTORY=/home/agnes/mps/log

DIR="$(dirname "$0")"
BIN="$DIR/three_threads_two_affinity_plus_primary_nvshmem_fixed"
if [ ! -x "$BIN" ]; then "$DIR/build.sh"; fi

# env prefix forces mpirun to see the corrected LD_LIBRARY_PATH; `export` alone
# doesn't reach orted-spawned children reliably.
env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    mpirun -x LD_LIBRARY_PATH -x NVSHMEM_REMOTE_TRANSPORT -x CUDA_MPS_PIPE_DIRECTORY -x CUDA_MPS_LOG_DIRECTORY -n 2 "$BIN"
