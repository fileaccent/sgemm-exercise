#define prefetch9()\
    pre_thread_num = (m * k + blockDim.x - 1)/ blockDim.x;\
    ix = threadIdx.x * pre_thread_num;\
    A_m_index = ix / k;\
    A_n_index = ix % k;\
    int d_A_index = (M_tile_index * m + A_m_index) * K + K_tile_index * k + A_n_index;\
    ix = A_m_index * (k + padding) + A_n_index;\
    FLOAT4(A_sh[ix]) = FLOAT4(d_A[d_A_index]);\
    pre_thread_num = (k * n + blockDim.x - 1) / blockDim.x;\
    ix = threadIdx.x * pre_thread_num;\
    B_m_index = ix / n;\
    B_n_index = ix % n;\
    ix = B_m_index * (n + padding) + B_n_index;\
    int d_B_index = (K_tile_index * k + B_m_index) * N + N_tile_index * n + B_n_index;\
    FLOAT4(B_sh[ix]) = FLOAT4(d_B[d_B_index]);
    
__global__ void gemm_kernel9(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m, int n, int k) {
    int padding = 4; 
    // 16 * 16 = 256
    // m * k = 1024 一个线程读 4 个
    // m = 64;
    // n = 64;
    // k = 16;
    const int TM = 4;
    const int TN = 4;
    const int TK = 2;
    extern __shared__ float sh[];
    float *A_sh = sh;
    float *B_sh = sh + m * (k + padding);
    int N_tile_index = blockIdx.x; // tile的列号
    int M_tile_index = blockIdx.y; // tile的行号
    int A_m_index;
    int A_n_index;
    int B_m_index;
    int B_n_index;
    int C_m_index = threadIdx.x / ((n + TN - 1) / TN); // tile内的4 * 4行号
    int C_n_index = threadIdx.x % ((n + TN - 1) / TN); // tile内的4 * 4列号
    int pre_thread_num;
    int ix;
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[TM][TK];
    float reg_B[TK][TN];
    float reg_C[TM][TN] = {0.0f};
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        prefetch9();
        __syncthreads();
        for (int k_reg_index = 0; k_reg_index < k; k_reg_index+= TK) {

            for (int i = 0; i < TM / 2; i++) {
                int A_index = C_m_index * TM / 2 * (k + padding) + k_reg_index +  i * (k + padding);
                reg_A[i][0] = A_sh[A_index];
                reg_A[i][1] = A_sh[A_index + 1];
                A_index = (C_m_index * TM / 2 + m / 2) * (k + padding) + k_reg_index +  i * (k + padding);
                reg_A[i + 2][0] = A_sh[A_index];
                reg_A[i + 2][1] = A_sh[A_index + 1];
            }
            for (int i = 0; i < TK; i++) {
                int B_index = k_reg_index * (n + padding) + C_n_index * TN / 2 + i * (n + padding);
                reg_B[i][0] = B_sh[B_index];
                reg_B[i][1] = B_sh[B_index + 1];
                B_index = k_reg_index * (n + padding) + C_n_index * TN / 2 + n / 2 + i * (n + padding);
                reg_B[i][2] = B_sh[B_index];
                reg_B[i][3] = B_sh[B_index + 1];
            }
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    for (int k_index = 0; k_index < TK; k_index++) {
                        reg_C[i][j] += reg_A[i][k_index] * reg_B[k_index][j];
                    }
                }
            }
        }
         __syncthreads();
    }

    for (int i = 0; i < TM / 2; i++) {
       int C_index = (M_tile_index * m + C_m_index * TM / 2) * N + N_tile_index * n + C_n_index * TN / 2 + i * N;
       d_C[C_index] = reg_C[i][0];
       d_C[C_index + 1] = reg_C[i][1];
       C_index = (M_tile_index * m + C_m_index * TM / 2 + m / 2) * N + N_tile_index * n + C_n_index * TN / 2 + i * N;
       d_C[C_index] = reg_C[i + 2][0];
       d_C[C_index + 1] = reg_C[i + 2][1];
       C_index = (M_tile_index * m + C_m_index * TM / 2) * N + N_tile_index * n + C_n_index * TN / 2 + n / 2 + i * N;
       d_C[C_index] = reg_C[i][2];
       d_C[C_index + 1] = reg_C[i][3];
       C_index = (M_tile_index * m + C_m_index * TM / 2 + m / 2) * N + N_tile_index * n + C_n_index * TN / 2 + n / 2 + i * N;
       d_C[C_index] = reg_C[i + 2][2];
       d_C[C_index + 1] = reg_C[i + 2][3];
    }
}
float test9 () {
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int TM = 4;
    const int TN = 4;
    const int TK = 2;
    const int padding = 4;
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((m * n + TM * TN - 1) / (TM * TN));
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding));
    gemm_kernel9<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(cudaEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel9<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}