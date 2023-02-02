#define prefetch6()\
    int d_A_index = (M_tile_index * m + A_m_index) * K + K_tile_index * k + A_n_index;\
    ix = A_m_index * (k + padding) + A_n_index;\
    FLOAT4(A_sh[A_sh_offset + ix])\
    mid_A = FLOAT4(d_A[d_A_index]);\
    int d_B_index = (K_tile_index * k + B_m_index) * N + N_tile_index * n + B_n_index;\
    ix = B_m_index * (n + padding) + B_n_index;\
    FLOAT4(B_sh[B_sh_offset + ix])\
    mid_B = FLOAT4(d_B[d_B_index]);
__global__ void gemm_kernel6(float *d_A, float *d_B, float *d_C, int M, int N, int K) {
    const int padding = 4; 
    // 16 * 16 = 256
    // m * k = 1024 一个线程读 4 个
    const int m = 64;
    const int n = 64;
    const int k = 16;
    // const int reg_size = 4;
    const int TM = 4;
    const int TN = 4;
    const int TK = 4;
    __shared__ float A_sh[2][m][k + padding];
    __shared__ float B_sh[2][k][n + padding];
    int N_tile_index = blockIdx.x; // tile的列号
    int M_tile_index = blockIdx.y; // tile的行号
    int A_m_index;
    int A_n_index;
    int B_m_index;
    int B_n_index;
    int A_sh_offset = 0;
    int B_sh_offset = 0;
    int reg_offset = 0;
    int C_m_index = threadIdx.x / ((n + TN - 1) / TN); // tile内的4 * 4行号
    int C_n_index = threadIdx.x % ((n + TN - 1) / TN); // tile内的4 * 4列号
    int d_A_index;
    int d_B_index;
    int pre_thread_num;
    int ix;
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[2][TM][TK];
    float reg_B[2][TK][TN];
    float reg_C[TM][TN] = {0.0f};
    float4 mid_A;
    float4 mid_B;
    // float total = 0.0f;
    pre_thread_num = (m * k + blockDim.x - 1)/ blockDim.x;\
    ix = threadIdx.x * pre_thread_num;\
    A_m_index = ix / k;\
    A_n_index = ix % k;\
    pre_thread_num = (k * n + blockDim.x - 1) / blockDim.x;\
    ix = threadIdx.x * pre_thread_num;\
    B_m_index = ix / n;\
    B_n_index = ix % n;\
    d_A_index = (M_tile_index * m + A_m_index) * K + A_n_index;
    d_B_index = (B_m_index) * N + N_tile_index * n + B_n_index;
    int K_tile_index = 0;	
    {
      mid_A = FLOAT4(d_A[d_A_index + K_tile_index * k]);
      mid_B = FLOAT4(d_B[d_B_index + K_tile_index * k * N]);
      FLOAT4(A_sh[A_sh_offset][A_m_index][A_n_index]) = mid_A;
      FLOAT4(B_sh[A_sh_offset][B_m_index][B_n_index]) = mid_B;
    }
    do {
        __syncthreads();
        int k_reg_index = 0;
        {
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TK; j++) {
                    reg_A[reg_offset][i][j] = A_sh[A_sh_offset][C_m_index * TM + i][k_reg_index + j];
                }
            }
            for (int i = 0; i < TK; i++) {
                for (int j = 0; j < TN; j++) {
                    reg_B[reg_offset][i][j] = B_sh[B_sh_offset][k_reg_index + i][C_n_index * TN + j];
                }
            }
        }
        do {
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TK; j++) {
                    reg_A[reg_offset][i][j] = A_sh[A_sh_offset][C_m_index * TM + i][k_reg_index + j];
                }
            }
            for (int i = 0; i < TK; i++) {
                for (int j = 0; j < TN; j++) {
                    reg_B[reg_offset][i][j] = B_sh[B_sh_offset][k_reg_index + i][C_n_index * TN + j];
                }
            }
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    for (int k_index = 0; k_index < TK; k_index++) {
                    reg_C[i][j] += reg_A[reg_offset][i][k_index] * reg_B[reg_offset][k_index][j];
                    }
                }
            }
            reg_offset ^= 1;
            k_reg_index += TK;
            if (k_reg_index < k) {
                for (int i = 0; i < TM; i++) {
                    for (int j = 0; j < TK; j++) {
                    reg_A[reg_offset][i][j] = A_sh[A_sh_offset][C_m_index * TM + i][k_reg_index + j];
                    }
                }
                for (int i = 0; i < TK; i++) {
                    for (int j = 0; j < TN; j++) {
                    reg_B[reg_offset][i][j] = B_sh[B_sh_offset][k_reg_index + i][C_n_index * TN + j];
                    }
                }
            }
        } while (k_reg_index < k);
        A_sh_offset ^= 1;
        B_sh_offset ^= 1;
        K_tile_index++;
        if (K_tile_index < int((K + k - 1) / k)) {
            mid_A = FLOAT4(d_A[d_A_index + K_tile_index * k]);
            mid_B = FLOAT4(d_B[d_B_index + K_tile_index * k * N]);
            FLOAT4(A_sh[A_sh_offset][A_m_index][A_n_index]) = mid_A;
            FLOAT4(B_sh[A_sh_offset][B_m_index][B_n_index]) = mid_B;
        }
    } while (K_tile_index < int((K + k - 1) / k));
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int C_index = (M_tile_index * m + C_m_index * TM) * N + N_tile_index * n + C_n_index * TN + i * N + j;
            d_C[C_index] = reg_C[i][j];
        }
    }
}
float test6 () {
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int TM = 4;
    const int TN = 4;
    const int reg_size = 4;
    const int padding = 4;
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((thread_size + TM * TN - 1) / (TM * TN));
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding)) * 2;
    gemm_kernel6<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel6<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K);
    }
    test_end();
    return elapsedTime / iteration;
}
