#!/usr/bin/env bash
set -u
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RDMAV_DRIVERS="${RDMAV_DRIVERS:-mlx5}"
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-none}"


# Warn if MPS is not running, or if this process cannot reach its pipe
# directory. Affinity contexts (CU_EXEC_AFFINITY_TYPE_SM_COUNT) fail with
# CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY if the client can't talk to MPS.
# If the daemon uses a non-default pipe dir, set CUDA_MPS_PIPE_DIRECTORY
# in your environment before running.
if ! pgrep -f nvidia-cuda-mps-control >/dev/null; then
    echo "WARNING: nvidia-cuda-mps-control daemon not found. Start MPS before running this test; affinity contexts will fail otherwise." >&2
else
    mps_pipe_dir="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
    if [ ! -S "$mps_pipe_dir/control" ]; then
        echo "WARNING: CUDA_MPS_PIPE_DIRECTORY=$mps_pipe_dir has no control socket. Export CUDA_MPS_PIPE_DIRECTORY pointing to your MPS daemon's pipe dir, or affinity contexts will fail with CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY." >&2
    fi
fi

# Warn if NVSHMEM_HOME does not point at a version that has the symbols the
# binary needs. Both 3.3.x and 3.6.x ship libnvshmem_host.so.3, so the SONAME
# alone does not tell you. Probe for nvshmem_selected_device_transport, which
# is present in 3.6+ and missing in 3.3.x. Without this the program dies with
# "undefined symbol nvshmem_selected_device_transport, version NVSHMEM".
if [ -z "${NVSHMEM_HOME:-}" ]; then
    echo "WARNING: NVSHMEM_HOME is not set. Export it to your NVSHMEM 3.6.x install path." >&2
elif ! nm -D "$NVSHMEM_HOME/lib/libnvshmem_host.so.3" 2>/dev/null | grep -q nvshmem_selected_device_transport; then
    echo "WARNING: $NVSHMEM_HOME/lib/libnvshmem_host.so.3 lacks nvshmem_selected_device_transport; probably NVSHMEM < 3.6. Point NVSHMEM_HOME at a 3.6+ install." >&2
fi

# Private tmpdir silences "PMIX ERROR: NO-PERMISSIONS ... dstore_base.c:238".
# Default /tmp is shared + other users' PMIx stores block OpenMPI's cleanup.
OMPI_TMPDIR="/tmp/$USER-ompi"
mkdir -p "$OMPI_TMPDIR"
chmod 700 "$OMPI_TMPDIR"
export TMPDIR="$OMPI_TMPDIR"

DIR="$(dirname "$0")"
BIN="$DIR/three_threads_two_affinity_plus_primary_nvshmem_fixed"
if [ ! -x "$BIN" ]; then "$DIR/build.sh"; fi

# env prefix forces mpirun to see the corrected LD_LIBRARY_PATH; `export` alone
# doesn't reach orted-spawned children reliably.
env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    mpirun --mca orte_tmpdir_base "$OMPI_TMPDIR" -x LD_LIBRARY_PATH -x NVSHMEM_REMOTE_TRANSPORT -x TMPDIR -n 2 "$BIN"
