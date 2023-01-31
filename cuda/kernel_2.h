__global__ void gemm_kernel2(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m, int n, int k) {
    extern __shared__ float sh[];
    float *A_sh = sh; // 
    float *B_sh = sh + m * k;
    // int N_index = idx % N; // C矩阵元素列号
    // int M_index = idx / N; // C矩阵元素行号
    int N_tile_index = blockIdx.x % ((N + n - 1)/ n); // tile的列号
    int M_tile_index = blockIdx.x / ((N + n - 1)/ n); // tile的行号
    int n_index = threadIdx.x % (n); // tile内的4 * 4列号
    int m_index = threadIdx.x / (n); // tile内的4 * 4行号
    float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < K; K_tile_index += k) {
        if ((M_tile_index * m + m_index) * K + K_tile_index + n_index < M * K) {
            A_sh[m_index * k + n_index] = d_A[(M_tile_index * m + m_index) * K + K_tile_index + n_index];
        } else {
            A_sh[m_index * k + n_index] = 0;
        }
        if ((K_tile_index + m_index) * N + N_tile_index * n + n_index < K * N) {
            B_sh[m_index * n + n_index] = d_B[(K_tile_index + m_index) * N + N_tile_index * n + n_index];
        } else {
            B_sh[m_index * n + n_index] = 0;
        }
        __syncthreads();
        for (int k_index = 0; k_index < k; k_index++) {
            total += A_sh[m_index * k + k_index] * B_sh[k_index * n + n_index];
        }
         __syncthreads();
    }
    // if (threadIdx.x == 0)
    //     printf("C: %d total: %lf \n", M_tile_index * m * N + N_tile_index * n + m_index * N + n_index, total);
    d_C[(M_tile_index * m + m_index) * N + N_tile_index * n + n_index] = total;
}
float test2 () {
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((C_size + thread_size - 1) / thread_size);
    dim3 thread(thread_size);
    int shared_size = sizeof(float) * (m * k + k * n);
    gemm_kernel2<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(cudaEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel2<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}