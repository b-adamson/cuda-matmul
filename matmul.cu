// Tiled matrix multiply — naive vs shared-memory, measured.
//
// Active T4 harness: baseline.cuh. Historical A6000 harness retained below.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

constexpr int TILE = 16;

// ---------------------------------------------------------------- kernels

// TODO 1. One thread per element of C. Work out row and col from blockIdx,
// blockDim and threadIdx, then loop over k accumulating A[row][k]*B[k][col].
// Guard against row/col running past N.
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row*N + k] * B[k*N + col];
        C[row*N + col] = sum;
    }
}


// TODO 2. Same result, but each block cooperatively loads a TILE x TILE patch
// of A and of B into __shared__ memory, then every thread in the block reads
// its operands from there. Loop over N/TILE phases. You will need two
// __syncthreads() per phase — think about why, and what breaks without each.
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {


    float sum = 0.0f;
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;                     
    int ty = threadIdx.y;                      

    int col = blockIdx.x * TILE + tx;           
    int row = blockIdx.y * TILE + ty;              

    for (int t = 0; t < (N + TILE - 1)/TILE; ++t) {
        As[ty][tx] = (row < N && t*TILE + tx < N) ? A[row*N + t*TILE + tx] : 0.0f;
        Bs[ty][tx] = (t*TILE + ty < N && col < N) ? B[(t*TILE + ty)*N + col] : 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) C[row*N + col] = sum;
}

// Register tiling. Each block computes a BM x BN tile of C with the same 16x16
// threads, so each thread owns a TM x TN patch of that tile and keeps its
// TM*TN partial sums in registers. Per k-step a thread loads TM values of A and
// TN of B, then does the TM x TN outer product: (TM+TN) shared loads for
// TM*TN FMAs, instead of 2 loads per FMA.
//
// Partial tiles are zero-filled; output writes are bounds-checked.
constexpr int BM = 64, BN = 64, BK = 8;
constexpr int TM = 4,  TN = 4;
static_assert(BM / TM == TILE && BN / TN == TILE, "thread block must be TILE x TILE");

__global__ void matmul_regtiled(const float* A, const float* B, float* C, int N) {
    // +1 pads the row of As so the two rows of threads in a warp (ty, ty+1)
    // hit different banks when reading As[ty*TM + i][k].
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN];

    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int tid = ty * TILE + tx;                  // 0 .. TILE*TILE-1

    int blockRow = blockIdx.y * BM;            // top row of this block's C tile
    int blockCol = blockIdx.x * BN;            // left column of this block's C tile

    float acc[TM][TN] = {};                    // this thread's partial sums

    for (int t = 0; t < (N + BK - 1) / BK; ++t) {
        // Cooperative load: BM*BK floats of A and BK*BN floats of B, spread
        // over TILE*TILE threads, so each thread loads several elements.
        for (int e = tid; e < BM * BK; e += TILE * TILE) {
            int r = e / BK, c = e % BK;
            As[r][c] = (blockRow + r < N && t * BK + c < N)
                ? A[(blockRow + r) * N + t * BK + c] : 0.0f;
        }
        for (int e = tid; e < BK * BN; e += TILE * TILE) {
            int r = e / BN, c = e % BN;
            Bs[r][c] = (t * BK + r < N && blockCol + c < N)
                ? B[(t * BK + r) * N + blockCol + c] : 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            for (int i = 0; i < TM; ++i) a[i] = As[ty * TM + i][k];
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][j * TILE + tx];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    // Columns are strided by TILE (col = j*TILE + tx) so threads with
    // consecutive tx read consecutive Bs words and write consecutive C words.
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            if (blockRow + ty * TM + i < N && blockCol + j * TILE + tx < N)
                C[(blockRow + ty * TM + i) * N + blockCol + j * TILE + tx] = acc[i][j];
}

#ifdef LEGACY_A6000_HARNESS
// Historical harness retained verbatim apart from the main entry-point comment.
// ------------------------------------------------------------------ bytes

// TODO 3. How many bytes does each kernel actually read from global memory?
// Not how many it *needs* — how many it issues. Think about how many times a
// single element of A is re-read across all threads. This number is the whole
// point of the exercise; derive it before you measure anything.
static double bytes_naive(int N) { return 2.0 * N * N * N * 4; }
static double bytes_tiled(int N) { return 2.0 * N * N * N / TILE * 4; }
// (N/BM)(N/BN) blocks x N/BK phases x (BM*BK + BK*BN) floats = N^3 (BM+BN)/(BM*BN) floats.
static double bytes_regtiled(int N) { return 1.0 * N * N * N * (BM + BN) / (BM * BN) * 4; }

// ----------------------------------------------------------------- harness

static void cpu_reference(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < N; ++k) s += A[i*N+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

static bool verify(const float* got, const float* want, int N) {
    for (int i = 0; i < N*N; ++i) {
        float tol = 1e-3f * fmaxf(1.0f, fabsf(want[i]));
        if (fabsf(got[i] - want[i]) > tol) {
            printf("  MISMATCH at %d: got %f want %f\n", i, got[i], want[i]);
            return false;
        }
    }
    return true;
}

typedef void (*Kernel)(const float*, const float*, float*, int);

// outTile: edge of the C tile each block produces (TILE for one element per
// thread, BM/BN for the register-tiled kernel). The block is always TILE x TILE.
static float time_kernel(Kernel k, const float* dA, const float* dB, float* dC,
                         int N, int iters, int outTile) {
    dim3 block(TILE, TILE);
    dim3 grid((N + outTile - 1) / outTile, (N + outTile - 1) / outTile);

    k<<<grid, block>>>(dA, dB, dC, N);        // warm up
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) k<<<grid, block>>>(dA, dB, dC, N);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    return ms / iters;
}

// cuBLAS baseline. cuBLAS is column-major and our matrices are row-major, so
// we ask for C^T = B^T * A^T: passing B first and A second gives a column-major
// result that is exactly our row-major C.
static float time_cublas(cublasHandle_t h, const float* dA, const float* dB,
                         float* dC, int N, int iters) {
    const float alpha = 1.0f, beta = 0.0f;
    auto gemm = [&]() {
        cublasStatus_t s = cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                       &alpha, dB, N, dA, N, &beta, dC, N);
        if (s != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS error %d\n", (int)s); exit(1); }
    };

    gemm();                                   // warm up
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) gemm();
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    return ms / iters;
}

static void report(const char* name, float ms, double bytes, int N) {
    double secs   = ms / 1e3;
    double gflops = (2.0 * N * N * N / secs) / 1e9;
    double gbs    = (bytes / secs) / 1e9;
    printf("  %-14s %8.3f ms   %8.1f GFLOP/s   %8.1f GB/s\n",
           name, ms, gflops, gbs);
}

// Retained for comparison with the original A6000 experiment. The active
// T4 harness below provides validation at all sizes and machine-readable timing.
int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    printf("N = %d, TILE = %d\n\n", N, TILE);

    size_t bytes = (size_t)N * N * sizeof(float);
    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes), *hRef = (float*)malloc(bytes);

    srand(1);
    for (int i = 0; i < N*N; ++i) {
        hA[i] = (float)rand() / RAND_MAX;
        hB[i] = (float)rand() / RAND_MAX;
    }

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));
    CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    bool check = (N <= 512);               // CPU reference is O(N^3), keep it small
    if (check) cpu_reference(hA, hB, hRef, N);

    struct { const char* name; Kernel k; double (*bytes)(int); int outTile; } runs[] = {
        { "naive",     matmul_naive,     bytes_naive,     TILE },
        { "tiled",     matmul_tiled,     bytes_tiled,     TILE },
        { "regtiled",  matmul_regtiled,  bytes_regtiled,  BM   },
    };

    for (auto& r : runs) {
        CHECK(cudaMemset(dC, 0, bytes));
        float ms = time_kernel(r.k, dA, dB, dC, N, 10, r.outTile);
        CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
        if (check && !verify(hC, hRef, N)) { printf("  %s: WRONG\n", r.name); continue; }
        report(r.name, ms, r.bytes(N), N);
    }

    // cuBLAS baseline. Its GB/s uses the minimum traffic (A and B read once),
    // since we don't know what it actually issues.
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { printf("cublasCreate failed\n"); return 1; }
    CHECK(cudaMemset(dC, 0, bytes));
    float ms = time_cublas(handle, dA, dB, dC, N, 10);
    CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
    if (check && !verify(hC, hRef, N)) printf("  cublas: WRONG\n");
    else report("cublas", ms, 2.0 * N * N * sizeof(float), N);
    cublasDestroy(handle);

    printf("\nA6000 peak: 38.7 TFLOP/s FP32, 768 GB/s (see README).\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hRef);
    return 0;
}

#else
#include "baseline.cuh"
#endif
