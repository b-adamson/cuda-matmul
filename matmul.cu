// Single-precision square matrix multiply, C = A * B, row-major.
//
// Three hand-written kernels of increasing sophistication, benchmarked against
// cuBLAS by the harness in baseline.cuh:
//   matmul_naive     one thread per output element, operands from global memory
//   matmul_tiled     16x16 shared-memory tiles, one output element per thread
//   matmul_regtiled  64x64x8 shared-memory tiles, 4x4 outputs per thread in registers
//
// All kernels handle any N (partial tiles are zero-filled or bounds-checked).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

constexpr int TILE = 16;   // thread block is TILE x TILE for every kernel

// ---------------------------------------------------------------- kernels

// One thread per element of C; every operand comes straight from global memory.
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

// Shared-memory tiling. Each block loads a TILE x TILE tile of A and of B into
// shared memory once per phase (one element per thread), and every thread then
// reads its operands from there. Phase t covers k = t*TILE .. t*TILE+TILE-1.
// The first barrier makes the tiles complete before use; the second stops a
// fast thread overwriting a tile that a slow thread is still reading.
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
// TN of B, then does the TM x TN outer product: TM+TN shared loads for TM*TN FMAs.
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

#include "baseline.cuh"
