// Tiled matrix multiply — naive vs shared-memory, measured.
//
// The harness is done. You write the two kernels and the byte count.

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
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

    for (int t = 0; t < N/TILE; ++t) {
        As[ty][tx] = A[row*N + t*TILE + tx];
        Bs[ty][tx] = B[t*TILE*N + ty*N + col];

        __syncthreads();
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    C[row*N + col] = sum;   
}

// ------------------------------------------------------------------ bytes

// TODO 3. How many bytes does each kernel actually read from global memory?
// Not how many it *needs* — how many it issues. Think about how many times a
// single element of A is re-read across all threads. This number is the whole
// point of the exercise; derive it before you measure anything.
static double bytes_naive(int N) { return 2.0 * N * N * N * 4; }
static double bytes_tiled(int N) { return 2.0 * N * N * N / TILE * 4; }

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

static float time_kernel(Kernel k, const float* dA, const float* dB, float* dC,
                         int N, int iters) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

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

static void report(const char* name, float ms, double bytes, int N) {
    double secs   = ms / 1e3;
    double gflops = (2.0 * N * N * N / secs) / 1e9;
    double gbs    = (bytes / secs) / 1e9;
    printf("  %-14s %8.3f ms   %8.1f GFLOP/s   %8.1f GB/s\n",
           name, ms, gflops, gbs);
}

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

    struct { const char* name; Kernel k; double (*bytes)(int); } runs[] = {
        { "naive", matmul_naive, bytes_naive },
        { "tiled", matmul_tiled, bytes_tiled },
    };

    for (auto& r : runs) {
        CHECK(cudaMemset(dC, 0, bytes));
        float ms = time_kernel(r.k, dA, dB, dC, N, 10);
        CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
        if (check && !verify(hC, hRef, N)) { printf("  %s: WRONG\n", r.name); continue; }
        report(r.name, ms, r.bytes(N), N);
    }

    printf("\nA6000 peak: check the spec sheet and fill it into the README.\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hRef);
    return 0;
}
