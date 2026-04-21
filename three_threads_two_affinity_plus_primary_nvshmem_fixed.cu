// three_threads_two_affinity_plus_primary_nvshmem_fixed.cu
//
// Run with 2 MPI ranks, one rank per GPU.
//
// Thread layout per process:
//   main thread      -> only launches 3 threads
//   app thread       -> app kernel in affinity context
//   service thread   -> service kernel in another affinity context
//   nvshmem thread   -> NVSHMEM in primary/runtime context
//
// This is a structural smoke test. It does NOT make NVSHMEM usable
// from the affinity contexts.

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define CK(call) do {                                                     \
    cudaError_t e_ = (call);                                              \
    if (e_ != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,           \
                cudaGetErrorString(e_));                                  \
        std::abort();                                                     \
    }                                                                     \
} while (0)

#define CKU(call) do {                                                    \
    CUresult e_ = (call);                                                 \
    if (e_ != CUDA_SUCCESS) {                                             \
        const char *name_ = nullptr, *msg_ = nullptr;                     \
        cuGetErrorName(e_, &name_);                                       \
        cuGetErrorString(e_, &msg_);                                      \
        fprintf(stderr, "CUDA driver %s:%d: %s : %s\n",                   \
                __FILE__, __LINE__,                                       \
                name_ ? name_ : "unknown", msg_ ? msg_ : "unknown");      \
        std::abort();                                                     \
    }                                                                     \
} while (0)

#define CM(call) do {                                                     \
    int e_ = (call);                                                      \
    if (e_ != MPI_SUCCESS) {                                              \
        fprintf(stderr, "MPI %s:%d: code=%d\n", __FILE__, __LINE__, e_);  \
        std::abort();                                                     \
    }                                                                     \
} while (0)

#define NS(call) do {                                                     \
    int e_ = (call);                                                      \
    if (e_ != 0) {                                                        \
        fprintf(stderr, "NVSHMEM %s:%d: code=%d\n", __FILE__, __LINE__, e_); \
        std::abort();                                                     \
    }                                                                     \
} while (0)

struct Shared {
    std::mutex m;
    std::condition_variable cv;
    bool ready = false;
    int rank = -1;
    int nranks = -1;
    int gpu = -1;
};

static std::mutex g_ctx_create_mutex;

__global__ void app_kernel(int *out, int rank) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *out = 1000 + rank;
    }
}

__global__ void service_kernel(int *out, int rank) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *out = 2000 + rank;
    }
}

__global__ void nvshmem_put_kernel(int *mailbox, int value, int peer) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        nvshmem_int_p(mailbox, value, peer);
        nvshmem_quiet();
    }
}

static CUcontext make_affinity_ctx_serialized(CUdevice dev, int sm_count, const char *tag, int rank) {
    std::lock_guard<std::mutex> lock(g_ctx_create_mutex);

    CUexecAffinityParam affinity{};
    affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
    affinity.param.smCount.val = sm_count;

    // CUDA 13 API: affinity lives in CUctxCreateParams, flags is a separate arg.
    CUctxCreateParams create_params{};
    create_params.execAffinityParams = &affinity;
    create_params.numExecAffinityParams = 1;

    CUcontext ctx = nullptr;
    fprintf(stdout, "[rank %d] %s: before cuCtxCreate request_sms=%d\n", rank, tag, sm_count);
    fflush(stdout);

    CKU(cuCtxCreate(&ctx, &create_params, CU_CTX_SCHED_AUTO, dev));

    fprintf(stdout, "[rank %d] %s: created ctx=%p\n", rank, tag, (void*)ctx);
    fflush(stdout);

    // Detach it from this creator thread; worker thread will set it current again.
    CKU(cuCtxSetCurrent(nullptr));
    return ctx;
}

static void app_thread_fn(Shared *s, int app_sms) {
    int rank, gpu;
    {
        std::unique_lock<std::mutex> lock(s->m);
        s->cv.wait(lock, [&]{ return s->ready; });
        rank = s->rank;
        gpu  = s->gpu;
    }

    CKU(cuInit(0));
    CUdevice dev;
    CKU(cuDeviceGet(&dev, gpu));

    CUcontext app_ctx = make_affinity_ctx_serialized(dev, app_sms, "app", rank);
    CKU(cuCtxSetCurrent(app_ctx));

    int *d_out = nullptr;
    CK(cudaMalloc(&d_out, sizeof(int)));
    app_kernel<<<1,1>>>(d_out, rank);
    CK(cudaDeviceSynchronize());

    int h = -1;
    CK(cudaMemcpy(&h, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    fprintf(stdout, "[rank %d] app thread: ctx=%p result=%d\n",
            rank, (void*)app_ctx, h);
    fflush(stdout);

    CK(cudaFree(d_out));
    CKU(cuCtxSetCurrent(nullptr));
    CKU(cuCtxDestroy(app_ctx));
}

static void service_thread_fn(Shared *s, int service_sms) {
    int rank, gpu;
    {
        std::unique_lock<std::mutex> lock(s->m);
        s->cv.wait(lock, [&]{ return s->ready; });
        rank = s->rank;
        gpu  = s->gpu;
    }

    CKU(cuInit(0));
    CUdevice dev;
    CKU(cuDeviceGet(&dev, gpu));

    CUcontext service_ctx = make_affinity_ctx_serialized(dev, service_sms, "service", rank);
    CKU(cuCtxSetCurrent(service_ctx));

    int *d_out = nullptr;
    CK(cudaMalloc(&d_out, sizeof(int)));
    service_kernel<<<1,1>>>(d_out, rank);
    CK(cudaDeviceSynchronize());

    int h = -1;
    CK(cudaMemcpy(&h, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    fprintf(stdout, "[rank %d] service thread: ctx=%p result=%d\n",
            rank, (void*)service_ctx, h);
    fflush(stdout);

    CK(cudaFree(d_out));
    CKU(cuCtxSetCurrent(nullptr));
    CKU(cuCtxDestroy(service_ctx));
}

static void nvshmem_thread_fn(Shared *s, int argc, char **argv) {
    int provided = -1;
    CM(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "Need MPI_THREAD_FUNNELED or better, got %d\n", provided);
        std::abort();
    }

    int rank = -1, nranks = -1;
    CM(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CM(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

    int ngpus = 0;
    CK(cudaGetDeviceCount(&ngpus));
    if (ngpus < 2) {
        if (rank == 0) fprintf(stderr, "Need at least 2 GPUs.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
        return;
    }

    int gpu = rank % ngpus;

    {
        std::lock_guard<std::mutex> lock(s->m);
        s->rank = rank;
        s->nranks = nranks;
        s->gpu = gpu;
        s->ready = true;
    }
    s->cv.notify_all();

    if (nranks != 2) {
        if (rank == 0) fprintf(stderr, "Run with exactly 2 MPI ranks.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    // Primary/runtime context path on this thread.
    CK(cudaSetDevice(gpu));

    MPI_Comm comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    attr.mpi_comm = &comm;
    NS(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr));

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = 1 - mype;

    fprintf(stdout,
            "[rank %d] nvshmem thread: primary/runtime path gpu=%d mype=%d npes=%d\n",
            rank, gpu, mype, npes);
    fflush(stdout);

    int *mailbox = (int*)nvshmem_malloc(sizeof(int));
    if (!mailbox) {
        fprintf(stderr, "[rank %d] nvshmem_malloc failed\n", rank);
        std::abort();
    }

    CK(cudaMemset(mailbox, 0, sizeof(int)));
    nvshmem_barrier_all();

    int my_value = 3000 + mype;
    nvshmem_put_kernel<<<1,1>>>(mailbox, my_value, peer);
    CK(cudaDeviceSynchronize());

    nvshmem_barrier_all();

    int host_val = -1;
    CK(cudaMemcpy(&host_val, mailbox, sizeof(int), cudaMemcpyDeviceToHost));
    fprintf(stdout,
            "[rank %d] nvshmem thread: mailbox=%d expected=%d\n",
            rank, host_val, 3000 + peer);
    fflush(stdout);

    nvshmem_free(mailbox);
    nvshmem_finalize();
    CM(MPI_Finalize());
}

int main(int argc, char **argv) {
    Shared shared;

    // Adjust for your GPU. Keep conservative first.
    int app_sms = 20;
    int service_sms = 20;

    std::thread nvshmem_thread(nvshmem_thread_fn, &shared, argc, argv);
    std::thread app_thread(app_thread_fn, &shared, app_sms);
    std::thread service_thread(service_thread_fn, &shared, service_sms);

    nvshmem_thread.join();
    app_thread.join();
    service_thread.join();
    return 0;
}
