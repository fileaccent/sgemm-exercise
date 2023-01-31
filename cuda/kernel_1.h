__global__ void gemm_kernel1(float *d_A, float *d_B, float *d_C, int M, int N, int K) {
    int idx = threadIdx.x  + blockIdx.x * blockDim.x;
    int ix = idx % N; // n
    int iy = idx / N; // m
    // printf("ix: %d \n", ix);
    // printf("iy: %d \n", iy);
    if (idx >= M * N) return;
    float total = 0;
    for (int i = 0; i < K; i++) {
        total += d_A[iy * K + i] * d_B[i * N + ix];
    }
    // printf("total: %lf \n", total);
    d_C[iy * N + ix] = total;
}
float test1 () {
    test_start();
    int thread_size = 32;
    gemm_kernel1<<<dim3((C_size + thread_size - 1) / thread_size), dim3(thread_size)>>>(d_A, d_B, d_C, M, N, K);
    KernelErrChk();
    cudaEventRecord(start, 0);
    for (int i = 0; i < iteration; i++) {
        gemm_kernel1<<<dim3((C_size + thread_size - 1) / thread_size), dim3(thread_size)>>>(d_A, d_B, d_C, M, N, K);
    }
    test_end();
    return elapsedTime / iteration;
}