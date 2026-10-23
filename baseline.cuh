// Benchmark and validation harness for the kernels in matmul.cu (included from there).
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <random>
#include <vector>

#define BLAS(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(s), __FILE__, __LINE__); \
    exit(1); } } while (0)

static int positive(const char* text) {
    char* end = nullptr;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || end == text || *end || value < 1 || value > INT_MAX) {
        fprintf(stderr, "Invalid positive integer: %s\n", text);
        exit(2);
    }
    return int(value);
}

static void launch(const char* name, const float* a, const float* b, float* c,
                   int n, cublasHandle_t handle) {
    dim3 block(TILE, TILE);
    if (!strcmp(name, "cublas")) {
        const float alpha = 1, beta = 0;
        // Row-major C = AB is column-major C^T = B^T A^T.
        BLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                        &alpha, b, n, a, n, &beta, c, n));
    } else if (!strcmp(name, "naive")) {
        dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
        matmul_naive<<<grid, block>>>(a, b, c, n);
    } else if (!strcmp(name, "tiled")) {
        dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
        matmul_tiled<<<grid, block>>>(a, b, c, n);
    } else {
        dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
        matmul_regtiled<<<grid, block>>>(a, b, c, n);
    }
    CHECK(cudaGetLastError());
}

// Independent double-precision CPU oracle: all outputs for n <= 128,
// otherwise 32 deterministic outputs including corners and the final edge.
static bool cpu_check(const std::vector<float>& a, const std::vector<float>& b,
                      const std::vector<float>& c, int n) {
    const size_t count = size_t(n) * n;
    const size_t checks = n <= 128 ? count : 32;
    double max_error = 0;
    for (size_t q = 0; q < checks; ++q) {
        size_t idx = n <= 128 ? q : (q == 0 ? 0 : q == 1 ? count - 1 :
            q == 2 ? size_t(n - 1) : q == 3 ? size_t(n - 1)*n :
            (q * 2654435761ULL) % count);
        size_t row = idx / n, col = idx % n;
        double ref = 0, scale = 0;
        for (int k = 0; k < n; ++k) {
            double term = double(a[row*n + k]) * b[size_t(k)*n + col];
            ref += term;
            scale += fabs(term);
        }
        double error = fabs(double(c[idx]) - ref);
        max_error = std::max(max_error, error);
        // Scale by sum |a*b| to handle cancellation in signed inputs.
        if (!std::isfinite(c[idx]) || error > 1e-5 + 2e-5 * scale) {
            fprintf(stderr, "CPU mismatch at %zu: got %.9g, reference %.17g\n",
                    idx, c[idx], ref);
            return false;
        }
    }
    fprintf(stderr, "CPU oracle: %zu outputs, max absolute error %.6g\n", checks, max_error);
    return true;
}

static bool compare(const std::vector<float>& got, const std::vector<float>& ref) {
    double max_error = 0, squared_error = 0, squared_ref = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) return false;
        double error = double(got[i]) - ref[i];
        max_error = std::max(max_error, fabs(error));
        squared_error += error * error;
        squared_ref += double(ref[i]) * ref[i];
        if (fabs(error) > 1e-3 + 2e-3 * fabs(double(ref[i]))) {
            fprintf(stderr, "cuBLAS mismatch at %zu: got %.9g, reference %.9g\n",
                    i, got[i], ref[i]);
            return false;
        }
    }
    double relative_l2 = sqrt(squared_error / std::max(squared_ref, 1e-30));
    fprintf(stderr, "cuBLAS comparison: max absolute error %.6g, relative L2 %.6g\n",
            max_error, relative_l2);
    return relative_l2 <= 2e-4;
}

int main(int argc, char** argv) {
    bool profile = argc > 1 && !strcmp(argv[1], "--profile");
    if (argc > (profile ? 4 : 5) || (profile && argc != 4)) {
        fprintf(stderr, "Usage: %s [N [all|naive|tiled|regtiled|cublas [iterations [samples]]]]\n"
                        "       %s --profile naive|tiled|regtiled N\n", argv[0], argv[0]);
        return 2;
    }
    int n = profile ? positive(argv[3]) : argc > 1 ? positive(argv[1]) : 1024;
    const char* selected = profile ? argv[2] : argc > 2 ? argv[2] : "all";
    int iterations = !profile && argc > 3 ? positive(argv[3]) : 10;
    int samples = !profile && argc > 4 ? positive(argv[4]) : 5;
    const char* names[] = {"naive", "tiled", "regtiled", "cublas"};
    bool valid = !profile && !strcmp(selected, "all");
    for (const char* name : names)
        if (!strcmp(selected, name) && (!profile || strcmp(name, "cublas"))) valid = true;
    // int indexing is used in the educational kernels.
    if (!valid || n > 32768) {
        fprintf(stderr, "Unknown kernel or N exceeds supported maximum 32768\n");
        return 2;
    }
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int driver = 0, runtime = 0;
    CHECK(cudaDriverGetVersion(&driver)); CHECK(cudaRuntimeGetVersion(&runtime));
    fprintf(stderr, "GPU=%s sm_%d%d SMs=%d L2=%d bytes driver=%d runtime=%d\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount,
            prop.l2CacheSize, driver, runtime);
    size_t count = size_t(n) * n, bytes = count * sizeof(float);
    size_t free_bytes = 0, total_bytes = 0;
    CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (bytes > free_bytes / (profile ? 3 : 4)) {
        fprintf(stderr, "Insufficient free GPU memory\n");
        return 2;
    }
    std::vector<float> a(count), b(count), got(count), ref(profile ? 0 : count);
    std::mt19937 rng(1);
    // Explicit conversion makes inputs reproducible across standard libraries.
    for (size_t i = 0; i < count; ++i) {
        a[i] = float(rng() >> 8) * (2.0f / 16777216.0f) - 1.0f;
        b[i] = float(rng() >> 8) * (2.0f / 16777216.0f) - 1.0f;
    }
    float *da, *db, *dc, *dr = nullptr;
    CHECK(cudaMalloc(&da, bytes)); CHECK(cudaMalloc(&db, bytes));
    CHECK(cudaMalloc(&dc, bytes));
    CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    cublasHandle_t handle = nullptr;
    if (profile) {
        // Exactly one target launch; no reference/warmup kernels to filter out.
        launch(selected, da, db, dc, n, handle);
        CHECK(cudaDeviceSynchronize());
    } else {
        CHECK(cudaMalloc(&dr, bytes));
        BLAS(cublasCreate(&handle));
        BLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
        int blas_version = 0;
        BLAS(cublasGetVersion(handle, &blas_version));
        fprintf(stderr, "cuBLAS=%d math=pedantic FP32 seed=1\n", blas_version);
        launch("cublas", da, db, dr, n, handle);
        CHECK(cudaMemcpy(ref.data(), dr, bytes, cudaMemcpyDeviceToHost));
        if (!cpu_check(a, b, ref, n)) return 1;
        printf("kernel,n,sample,iterations,ms,tflops,correct\n");
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
        for (const char* name : names) {
            if (strcmp(selected, "all") && strcmp(selected, name)) continue;
            CHECK(cudaMemset(dc, 0xff, bytes)); // NaNs expose missing writes.
            launch(name, da, db, dc, n, handle);
            CHECK(cudaMemcpy(got.data(), dc, bytes, cudaMemcpyDeviceToHost));
            if (!compare(got, ref) || !cpu_check(a, b, got, n)) return 1;
            for (int w = 0; w < 5; ++w) launch(name, da, db, dc, n, handle);
            CHECK(cudaDeviceSynchronize());
            for (int sample = 0; sample < samples; ++sample) {
                CHECK(cudaEventRecord(start));
                for (int i = 0; i < iterations; ++i) launch(name, da, db, dc, n, handle);
                CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
                float elapsed = 0;
                CHECK(cudaEventElapsedTime(&elapsed, start, stop));
                double ms = elapsed / iterations;
                if (!(ms > 0) || !std::isfinite(ms)) return 1;
                printf("%s,%d,%d,%d,%.9g,%.9g,true\n", name, n, sample,
                       iterations, ms, 2.0*n*n*n / (ms * 1e9));
            }
            CHECK(cudaMemcpy(got.data(), dc, bytes, cudaMemcpyDeviceToHost));
            if (!compare(got, ref)) return 1;
        }
        CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
        BLAS(cublasDestroy(handle)); CHECK(cudaFree(dr));
    }
    CHECK(cudaFree(da)); CHECK(cudaFree(db)); CHECK(cudaFree(dc));
    return 0;
}
