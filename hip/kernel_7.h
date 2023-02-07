__global__ void gemm_kernel7(float *d_A, float *d_B, float *d_C, int M, int N, int K) {
    const int padding = 4; 
    // 16 * 16 = 256
    // m * k = 1024 一个线程读 4 个
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int TM = 4;
    const int TN = 4;
    const int TK = 2;
    __shared__ float A_sh[m][k + padding];
    __shared__ float B_sh[k][n + padding];
    const int N_tile_index = blockIdx.y; // tile的列号
    const int M_tile_index = blockIdx.x; // tile的行号
    const int idx = threadIdx.x;
    const int A_m_index = idx >> 2;
    const int A_n_index = (idx & 3) << 2;
    const int B_m_index = idx >> 4;
    const int B_n_index = (idx & 15) << 2;
    const int C_m_index = idx >> 4; // tile内的4 * 4行号
    const int C_n_index = idx & 15; // tile内的4 * 4列号
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[TM][TK];
    float reg_B[TK][TN];
    float reg_C[TM][TN] = {0.0f};
    const int d_A_index = (M_tile_index * m + A_m_index) * K + A_n_index;
    const int d_B_index = (B_m_index) * N + N_tile_index * n + B_n_index;
    
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        FLOAT4(A_sh[A_m_index][A_n_index]) = FLOAT4(d_A[d_A_index + K_tile_index * k]);
        FLOAT4(B_sh[B_m_index][B_n_index]) = FLOAT4(d_B[d_B_index + K_tile_index * k * N]);
        __syncthreads();
        for (int k_reg_index = 0; k_reg_index < k; k_reg_index+= TK) {
            for (int i = 0; i < TM; i++) {
                FLOAT2(reg_A[i][0]) = FLOAT2(A_sh[C_m_index * TM + i][k_reg_index]);
            }
            for (int i = 0; i < TK; i++) {
                FLOAT4(reg_B[i][0]) = FLOAT4(B_sh[k_reg_index + i][C_n_index * TN]);
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

    for (int i = 0; i < TM; i++) {
        int C_index = (M_tile_index * m + C_m_index * TM) * N + N_tile_index * n + C_n_index * TN + i * N;
        FLOAT4(d_C[C_index]) = FLOAT4(reg_C[i][0]);
    }
}
float test7 () {
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int TM = 4;
    const int TN = 4;
    const int padding = 4;
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((m * n + TM * TN - 1) / (TM * TN));
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding));
    gemm_kernel7<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel7<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K);
    }
    test_end();
    return elapsedTime / iteration;
}
#define prefetch7_1()\
    pre_thread_num = (m * k + blockDim.x - 1)/ blockDim.x;\
    ix = threadIdx.x * pre_thread_num;\
    int pre_thread_m = (pre_thread_num + k - 1) / k;\
    int pre_thread_n = pre_thread_num % k;\
    A_m_index = ix / k;\
    A_n_index = ix % k;\
    int d_A_index = (M_tile_index * m + A_m_index) * K + K_tile_index * k + A_n_index;\
    ix = A_m_index * (k + padding) + A_n_index;\
    set_value_matrix(&A_sh[ix], &d_A[d_A_index], pre_thread_m, pre_thread_n, k + padding, K);\
    pre_thread_num = (k * n + blockDim.x - 1) / blockDim.x;\
    ix = threadIdx.x * pre_thread_num;\
    pre_thread_m = (pre_thread_num + n - 1) / n;\
    pre_thread_n = pre_thread_num % n;\
    B_m_index = ix / n;\
    B_n_index = ix % n;\
    ix = B_m_index * (n + padding) + B_n_index;\
    int d_B_index = (K_tile_index * k + B_m_index) * N + N_tile_index * n + B_n_index;\
    set_value_matrix(&B_sh[ix], &d_B[d_B_index], pre_thread_m, pre_thread_n, n + padding, N);
__global__ void gemm_kernel7_1(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m, int n, int k) {
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
        prefetch7_1();
	    __syncthreads();
        for (int k_reg_index = 0; k_reg_index < k; k_reg_index += TK) {
	    int A_index = C_m_index * TM * (k + padding) + k_reg_index;
	    set_value_matrix((float *)reg_A[0], &A_sh[A_index], TM, TK, TK, k + padding);
            int B_index = k_reg_index * (n + padding) + C_n_index * TN;
	    set_value_matrix((float *)reg_B[0], &B_sh[B_index], TK, TN, TN, n + padding);
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
    
    int C_index = (M_tile_index * m + C_m_index * TM) * N + N_tile_index * n + C_n_index * TN;
    set_value_matrix(&d_C[C_index], (float *)reg_C[0], TM, TN, N, TN);
    // for (int i = 0; i < TM; i++) {
    //    int C_index = (M_tile_index * m + C_m_index * TM) * N + N_tile_index * n + C_n_index * TN + i * N;
    //    set_value(&d_C[C_index], reg_C[i], TN);
    // }
}
float test7_1 () {
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int TM = 4;
    const int TN = 4;
    const int padding = 4;
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((m * n + TM * TN - 1) / (TM * TN));
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding));
    gemm_kernel7_1<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel7_1<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}

#define prefetch7_2()\
    pre_thread_num = (m * k + blockDim.x - 1)/ blockDim.x;\
    ix = threadIdx.x * pre_thread_num;\
    int pre_thread_m = (pre_thread_num + k - 1) / k;\
    int pre_thread_n = pre_thread_num % k;\
    A_m_index = ix / k;\
    A_n_index = ix % k;\
    int d_A_index = (M_tile_index * m + A_m_index) * K + K_tile_index * k + A_n_index;\
    ix = A_m_index * (k + padding) + A_n_index;\
    set_value_matrix(&A_sh[ix], &d_A[d_A_index], pre_thread_m, pre_thread_n, k + padding, K);\
    pre_thread_num = (k * n + blockDim.x - 1) / blockDim.x;\
    ix = threadIdx.x * pre_thread_num;\
    pre_thread_m = (pre_thread_num + n - 1) / n;\
    pre_thread_n = pre_thread_num % n;\
    B_m_index = ix / n;\
    B_n_index = ix % n;\
    ix = B_m_index * (n + padding) + B_n_index;\
    int d_B_index = (K_tile_index * k + B_m_index) * N + N_tile_index * n + B_n_index;\
    set_value_matrix(&B_sh[ix], &d_B[d_B_index], pre_thread_m, pre_thread_n, n + padding, N);
__global__ void gemm_kernel7_2(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m, int n, int k) {
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
        prefetch7_2();
	__syncthreads();
        for (int k_reg_index = 0; k_reg_index < k; k_reg_index += TK) {
	    int A_index = C_m_index * TM * (k + padding) + k_reg_index;
	    set_value_matrix((float *)reg_A[0], &A_sh[A_index], TM, TK, TK, k + padding);
            int B_index = k_reg_index * (n + padding) + C_n_index * TN;
	    set_value_matrix((float *)reg_B[0], &B_sh[B_index], TK, TN, TN, n + padding);
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
    
    int C_index = (M_tile_index * m + C_m_index * TM) * N + N_tile_index * n + C_n_index * TN;
    set_value_matrix(&d_C[C_index], (float *)reg_C[0], TM, TN, N, TN);
}
float test7_2 () {
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int TM = 4;
    const int TN = 4;
    const int padding = 4;
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((m * n + TM * TN - 1) / (TM * TN));
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding));
    gemm_kernel7_2<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel7_2<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}

// #define prefetch7_3()\
//     ix = threadIdx.x * A_pre_thread_num;\
//     A_m_index = ix / k;\
//     A_n_index = ix % k;\
//     int d_A_index = (M_tile_index * m + A_m_index) * K + K_tile_index * k + A_n_index;\
//     ix = A_m_index * (k + padding) + A_n_index;\
//     FLOAT4(A_sh[ix]) = FLOAT4(d_A[d_A_index]);\
//     ix = threadIdx.x * B_pre_thread_num;\
//     B_m_index = ix / n;\
//     B_n_index = ix % n;\
//     ix = B_m_index * (n + padding) + B_n_index;\
//     int d_B_index = (K_tile_index * k + B_m_index) * N + N_tile_index * n + B_n_index;\
//     FLOAT4(B_sh[ix]) = FLOAT4(d_B[d_B_index]);

#define calc_k_fma1()\
    reg_C[0][0] += reg_A[0][0] * reg_B[0][0];\
    reg_C[0][1] += reg_A[0][0] * reg_B[0][1];\
    reg_C[0][2] += reg_A[0][0] * reg_B[0][2];\
    reg_C[0][3] += reg_A[0][0] * reg_B[0][3];\
    \
    FLOAT2(reg_A[1][0]) = FLOAT2(A_sh[A_index + (k + padding)]);\
    \
    reg_C[1][0] += reg_A[1][0] * reg_B[0][0];\
    reg_C[1][1] += reg_A[1][0] * reg_B[0][1];\
    reg_C[1][2] += reg_A[1][0] * reg_B[0][2];\
    reg_C[1][3] += reg_A[1][0] * reg_B[0][3];\
    \
    FLOAT2(reg_A[2][0]) = FLOAT2(A_sh[A_index + 2 * (k + padding)]);\
    \
    reg_C[2][0] += reg_A[2][0] * reg_B[0][0];\
    reg_C[2][1] += reg_A[2][0] * reg_B[0][1];\
    reg_C[2][2] += reg_A[2][0] * reg_B[0][2];\
    reg_C[2][3] += reg_A[2][0] * reg_B[0][3];\
    \
    FLOAT2(reg_A[3][0]) = FLOAT2(A_sh[A_index + 3 * (k + padding)]);\
    \
    reg_C[3][0] += reg_A[3][0] * reg_B[0][0];\
    reg_C[3][1] += reg_A[3][0] * reg_B[0][1];\
    reg_C[3][2] += reg_A[3][0] * reg_B[0][2];\
    reg_C[3][3] += reg_A[3][0] * reg_B[0][3];\
    \
    FLOAT4(reg_B[1][0]) = FLOAT4(B_sh[B_index + (n + padding)]);\
    \
    reg_C[0][0] += reg_A[0][1] * reg_B[1][0];\
    reg_C[0][1] += reg_A[0][1] * reg_B[1][1];\
    reg_C[0][2] += reg_A[0][1] * reg_B[1][2];\
    reg_C[0][3] += reg_A[0][1] * reg_B[1][3];

#define calc_k_fma2()\
    reg_C[1][0] += reg_A[1][1] * reg_B[1][0];\
    reg_C[1][1] += reg_A[1][1] * reg_B[1][1];\
    reg_C[1][2] += reg_A[1][1] * reg_B[1][2];\
    reg_C[1][3] += reg_A[1][1] * reg_B[1][3];\
    \
    FLOAT2(reg_A[0][0]) = FLOAT2(A_sh[A_index + TK]);\
    \
    reg_C[2][0] += reg_A[2][1] * reg_B[1][0];\
    reg_C[2][1] += reg_A[2][1] * reg_B[1][1];\
    \
    reg_C[2][2] += reg_A[2][1] * reg_B[1][2];\
    FLOAT4(reg_B[0][0]) = FLOAT4(B_sh[B_index + TK * (n + padding)]);\
    reg_C[2][3] += reg_A[2][1] * reg_B[1][3];\
    \
    reg_C[3][0] += reg_A[3][1] * reg_B[1][0];\
    reg_C[3][1] += reg_A[3][1] * reg_B[1][1];\
    reg_C[3][2] += reg_A[3][1] * reg_B[1][2];\
    reg_C[3][3] += reg_A[3][1] * reg_B[1][3];
__global__ void gemm_kernel7_3(float *d_A, float *d_B, float *d_C, int M, int N, int K) { 
    // 16 * 16 = 256
    // m * k = 1024 一个线程读 4 个
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int TM = 4;
    const int TN = 4;
    const int TK = 2;
    const int padding = 4;
    extern __shared__ float sh[]; 
    float *A_sh = sh;
    float *B_sh = sh + m * (k + padding);
    const int N_tile_index = blockIdx.x; // tile的列号
    const int M_tile_index = blockIdx.y; // tile的行号
    const int idx = threadIdx.x;
    const int A_m_index = idx >> 2;
    const int A_n_index = (idx & 3) << 2;
    const int B_m_index = idx >> 4;
    const int B_n_index = (idx & 15) << 2;
    const int C_m_index = idx >> 4; // tile内的4 * 4行号
    const int C_n_index = idx & 15; // tile内的4 * 4列号
    const int A_pre_thread_num = 4;
    const int B_pre_thread_num = 4;
    const int d_A_index = (M_tile_index * m + A_m_index) * K + A_n_index;
    const int d_B_index = (B_m_index) * N + N_tile_index * n + B_n_index;
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[TM][TK];
    float reg_B[TK][TN];
    float reg_C[TM][TN] = {0.0f};
    int K_tile_index = 0;
    // prefetch7_3();
    for (; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        FLOAT4(A_sh[A_m_index * (k + padding) + A_n_index]) = FLOAT4(d_A[d_A_index + K_tile_index * k]);
        FLOAT4(B_sh[B_m_index * (n + padding) + B_n_index]) = FLOAT4(d_B[d_B_index + K_tile_index * k * N]);
	__syncthreads();
        int A_index = C_m_index * TM * (k + padding);
        int B_index = C_n_index * TN;
        FLOAT2(reg_A[0][0]) = FLOAT2(A_sh[A_index]);
        FLOAT4(reg_B[0][0]) = FLOAT4(B_sh[B_index]);
        calc_k_fma1();// 0
        calc_k_fma2();
        A_index += TK;
        B_index += TK * (n + padding);
        calc_k_fma1();// 2
        calc_k_fma2();
        A_index += TK;
        B_index += TK * (n + padding);
        calc_k_fma1();// 4
        calc_k_fma2();
        A_index += TK;
        B_index += TK * (n + padding);
        calc_k_fma1();// 6
        calc_k_fma2();
        A_index += TK;
        B_index += TK * (n + padding);
        calc_k_fma1();// 8
        calc_k_fma2();
        A_index += TK;
        B_index += TK * (n + padding);
        calc_k_fma1();// 10
        calc_k_fma2();
        A_index += TK;
        B_index += TK * (n + padding);
        calc_k_fma1();// 12
        calc_k_fma2();
        A_index += TK;
        B_index += TK * (n + padding);
        calc_k_fma1();// 14
        reg_C[1][0] += reg_A[1][1] * reg_B[1][0];
        reg_C[1][1] += reg_A[1][1] * reg_B[1][1];
        reg_C[1][2] += reg_A[1][1] * reg_B[1][2];
        reg_C[1][3] += reg_A[1][1] * reg_B[1][3];   
        reg_C[2][0] += reg_A[2][1] * reg_B[1][0];
        reg_C[2][1] += reg_A[2][1] * reg_B[1][1];
        
	    reg_C[2][2] += reg_A[2][1] * reg_B[1][2];
        reg_C[2][3] += reg_A[2][1] * reg_B[1][3];
        reg_C[3][0] += reg_A[3][1] * reg_B[1][0];
        reg_C[3][1] += reg_A[3][1] * reg_B[1][1];
        reg_C[3][2] += reg_A[3][1] * reg_B[1][2];
        reg_C[3][3] += reg_A[3][1] * reg_B[1][3];
	    __syncthreads();

    }
    int C_index = (M_tile_index * m + C_m_index * TM) * N + N_tile_index * n + C_n_index * TN;
    FLOAT4(d_C[C_index]) = FLOAT4(reg_C[0][0]);
    FLOAT4(d_C[C_index + 1 * N]) = FLOAT4(reg_C[1][0]);
    FLOAT4(d_C[C_index + 2 * N]) = FLOAT4(reg_C[2][0]);
    FLOAT4(d_C[C_index + 3 * N]) = FLOAT4(reg_C[3][0]);
    // for (int i = 0; i < TM; i++) {
    //    int C_index = (M_tile_index * m + C_m_index * TM) * N + N_tile_index * n + C_n_index * TN + i * N;
    //    FLOAT4(d_C[C_index]) = FLOAT4(reg_C[i][0]);
    // }
}
float test7_3 () {
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
    gemm_kernel7_3<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel7_3<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K);
    }
    test_end();
    return elapsedTime / iteration;
}


#define calc_k_fma1_7_4()\
    reg_C[0][0] += reg_A[0][0] * reg_B[0][0];\
    reg_C[0][1] += reg_A[0][0] * reg_B[0][1];\
    reg_C[0][2] += reg_A[0][0] * reg_B[0][2];\
    reg_C[0][3] += reg_A[0][0] * reg_B[0][3];\
    \
    FLOAT2(reg_A[1][0]) = FLOAT2(A_sh[A_sh_offset + A_index + (k + padding)]);\
    \
    reg_C[1][0] += reg_A[1][0] * reg_B[0][0];\
    reg_C[1][1] += reg_A[1][0] * reg_B[0][1];\
    reg_C[1][2] += reg_A[1][0] * reg_B[0][2];\
    reg_C[1][3] += reg_A[1][0] * reg_B[0][3];\
    \
    FLOAT2(reg_A[2][0]) = FLOAT2(A_sh[A_sh_offset + A_index + 2 * (k + padding)]);\
    \
    reg_C[2][0] += reg_A[2][0] * reg_B[0][0];\
    reg_C[2][1] += reg_A[2][0] * reg_B[0][1];\
    reg_C[2][2] += reg_A[2][0] * reg_B[0][2];\
    reg_C[2][3] += reg_A[2][0] * reg_B[0][3];\
    \
    FLOAT2(reg_A[3][0]) = FLOAT2(A_sh[A_sh_offset + A_index + 3 * (k + padding)]);\
    \
    reg_C[3][0] += reg_A[3][0] * reg_B[0][0];\
    reg_C[3][1] += reg_A[3][0] * reg_B[0][1];\
    reg_C[3][2] += reg_A[3][0] * reg_B[0][2];\
    reg_C[3][3] += reg_A[3][0] * reg_B[0][3];\
    \
    FLOAT4(reg_B[1][0]) = FLOAT4(B_sh[B_sh_offset + B_index + (n + padding)]);\
    \
    reg_C[0][0] += reg_A[0][1] * reg_B[1][0];\
    reg_C[0][1] += reg_A[0][1] * reg_B[1][1];\
    reg_C[0][2] += reg_A[0][1] * reg_B[1][2];\
    reg_C[0][3] += reg_A[0][1] * reg_B[1][3];

#define calc_k_fma2_7_4()\
    reg_C[1][0] += reg_A[1][1] * reg_B[1][0];\
    reg_C[1][1] += reg_A[1][1] * reg_B[1][1];\
    reg_C[1][2] += reg_A[1][1] * reg_B[1][2];\
    reg_C[1][3] += reg_A[1][1] * reg_B[1][3];\
    \
    FLOAT2(reg_A[0][0]) = FLOAT2(A_sh[A_sh_offset + A_index + TK]);\
    \
    reg_C[2][0] += reg_A[2][1] * reg_B[1][0];\
    reg_C[2][1] += reg_A[2][1] * reg_B[1][1];\
    \
    reg_C[2][2] += reg_A[2][1] * reg_B[1][2];\
    FLOAT4(reg_B[0][0]) = FLOAT4(B_sh[B_sh_offset + B_index + TK * (n + padding)]);\
    reg_C[2][3] += reg_A[2][1] * reg_B[1][3];\
    \
    reg_C[3][0] += reg_A[3][1] * reg_B[1][0];\
    reg_C[3][1] += reg_A[3][1] * reg_B[1][1];\
    reg_C[3][2] += reg_A[3][1] * reg_B[1][2];\
    reg_C[3][3] += reg_A[3][1] * reg_B[1][3];
__global__ void gemm_kernel7_4(float *d_A, float *d_B, float *d_C, int M, int N, int K) {
    // 16 * 16 = 256
    // m * k = 1024 一个线程读 4 个
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int TM = 4;
    const int TN = 4;
    const int TK = 2;
    extern __shared__ float sh[];
    const int padding = 8; 
    float *A_sh = sh;
    float *B_sh = sh + 2 * m * (k + padding);
    const int N_tile_index = blockIdx.x; // tile的列号
    const int M_tile_index = blockIdx.y; // tile的行号
    const int idx = threadIdx.x;
    const int A_m_index = idx >> 2;
    const int A_n_index = (idx & 3) << 2;
    const int B_m_index = idx >> 4;
    const int B_n_index = (idx & 15) << 2;
    const int C_m_index = idx >> 4; // tile内的4 * 4行号
    const int C_n_index = idx & 15; // tile内的4 * 4列号
    const int A_pre_thread_num = (m * k + blockDim.x - 1)/ blockDim.x;
    const int B_pre_thread_num = (k * n + blockDim.x - 1)/ blockDim.x;
    const int d_A_index = (M_tile_index * m + A_m_index) * K + A_n_index;
    const int d_B_index = (B_m_index) * N + N_tile_index * n + B_n_index;
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[TM][TK];
    float reg_B[TK][TN];
    float reg_C[TM][TN] = {0.0f};
    int A_sh_offset = 0;
    int B_sh_offset = 0;
    const int A_sh_size = m * (k + padding);
    const int B_sh_size = k * (n + padding);
    int K_tile_index = 0;
    float4 readA;
    float4 readB;
    lgkmcnt<0>();
    // FLOAT4(A_sh[A_sh_offset + A_m_index * (k + padding) + A_n_index]) = FLOAT4(d_A[d_A_index + K_tile_index * k]);
    // FLOAT4(B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index]) = FLOAT4(d_B[d_B_index + K_tile_index * k * N]);
    global_load<0>(&d_A[d_A_index + K_tile_index * k], readA);
    global_load<0>(&d_B[d_B_index + K_tile_index * k * N], readB);
    vmcnt<0>();

    FLOAT4(A_sh[A_sh_offset + A_m_index * (k + padding) + A_n_index]) = readA;
    FLOAT4(B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index]) = readB;

    do {
        __syncthreads();
        if (K_tile_index + 1 < int((K + k - 1) / k)) {
            global_load<0>(&d_A[d_A_index + (K_tile_index + 1) * k], readA);
            global_load<0>(&d_B[d_B_index + (K_tile_index + 1) * k * N], readB);
	    // printf("readA: %lf %lf %lf %lf\n", readA.x, readA.y, readA.z, readA.w);
            // FLOAT4(A_sh[(A_sh_offset^A_sh_size) + A_m_index * (k + padding) + A_n_index]) = FLOAT4(d_A[d_A_index + (K_tile_index + 1) * k]);
            // FLOAT4(B_sh[(B_sh_offset^B_sh_size) + B_m_index * (n + padding) + B_n_index]) = FLOAT4(d_B[d_B_index + (K_tile_index + 1) * k * N]);
        }
        
        lgkmcnt<0>();
        
        {
            int A_index = C_m_index * TM * (k + padding);
            int B_index = C_n_index * TN;
            FLOAT2(reg_A[0][0]) = FLOAT2(A_sh[A_sh_offset + A_index]);
            FLOAT4(reg_B[0][0]) = FLOAT4(B_sh[B_sh_offset + B_index]);
            calc_k_fma1_7_4();// 0
            calc_k_fma2_7_4();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_4();// 2
            calc_k_fma2_7_4();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_4();// 4
            calc_k_fma2_7_4();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_4();// 6
            calc_k_fma2_7_4();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_4();// 8
            calc_k_fma2_7_4();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_4();// 10
            calc_k_fma2_7_4();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_4();// 12
            calc_k_fma2_7_4();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_4();// 14
            reg_C[1][0] += reg_A[1][1] * reg_B[1][0];
            reg_C[1][1] += reg_A[1][1] * reg_B[1][1];
            reg_C[1][2] += reg_A[1][1] * reg_B[1][2];
            reg_C[1][3] += reg_A[1][1] * reg_B[1][3];   
            reg_C[2][0] += reg_A[2][1] * reg_B[1][0];
            reg_C[2][1] += reg_A[2][1] * reg_B[1][1];

            reg_C[2][2] += reg_A[2][1] * reg_B[1][2];
            reg_C[2][3] += reg_A[2][1] * reg_B[1][3];
            reg_C[3][0] += reg_A[3][1] * reg_B[1][0];
            reg_C[3][1] += reg_A[3][1] * reg_B[1][1];
            reg_C[3][2] += reg_A[3][1] * reg_B[1][2];
            reg_C[3][3] += reg_A[3][1] * reg_B[1][3];
        }
        A_sh_offset ^= A_sh_size;
        B_sh_offset ^= B_sh_size;
        K_tile_index++;
        vmcnt<0>();
        FLOAT4(A_sh[A_sh_offset + A_m_index * (k + padding) + A_n_index]) = readA;
        FLOAT4(B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index]) = readB;
    } while (K_tile_index < int((K + k - 1) / k));
    int C_index = (M_tile_index * m + C_m_index * TM) * N + N_tile_index * n + C_n_index * TN;
    lgkmcnt<0>();
    global_store<0>(&d_C[C_index], FLOAT4(reg_C[0][0]));
    global_store<0>(&d_C[C_index + 1 * N], FLOAT4(reg_C[1][0]));
    global_store<0>(&d_C[C_index + 2 * N], FLOAT4(reg_C[2][0]));
    global_store<0>(&d_C[C_index + 3 * N], FLOAT4(reg_C[3][0]));
    vmcnt<0>();
    // FLOAT4(d_C[C_index]) = FLOAT4(reg_C[0][0]);
    // FLOAT4(d_C[C_index + 1 * N]) = FLOAT4(reg_C[1][0]);
    // FLOAT4(d_C[C_index + 2 * N]) = FLOAT4(reg_C[2][0]);
    // FLOAT4(d_C[C_index + 3 * N]) = FLOAT4(reg_C[3][0]);
}
float test7_4 () {
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int TM = 4;
    const int TN = 4;
    const int TK = 2;
    const int padding = 8;
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((m * n + TM * TN - 1) / (TM * TN));
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding)) * 2;
    gemm_kernel7_4<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel7_4<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K);
    }
    test_end();
    return elapsedTime / iteration;
}

#define calc_k_fma1_7_5()\
    reg_C[0][0] += reg_A[reg_offset][0][0] * reg_B[reg_offset][0][0];\
    reg_C[0][1] += reg_A[reg_offset][0][0] * reg_B[reg_offset][0][1];\
    reg_C[0][2] += reg_A[reg_offset][0][0] * reg_B[reg_offset][0][2];\
    reg_C[0][3] += reg_A[reg_offset][0][0] * reg_B[reg_offset][0][3];\
    \
    FLOAT2(reg_A[reg_offset^1][1][0]) = FLOAT2(A_sh[A_sh_offset + A_index + (k + padding) + TK]);\
    \
    reg_C[1][0] += reg_A[reg_offset][1][0] * reg_B[reg_offset][0][0];\
    reg_C[1][1] += reg_A[reg_offset][1][0] * reg_B[reg_offset][0][1];\
    reg_C[1][2] += reg_A[reg_offset][1][0] * reg_B[reg_offset][0][2];\
    reg_C[1][3] += reg_A[reg_offset][1][0] * reg_B[reg_offset][0][3];\
    \
    FLOAT2(reg_A[reg_offset^1][2][0]) = FLOAT2(A_sh[A_sh_offset + A_index + 2 * (k + padding) + TK]);\
    \
    reg_C[2][0] += reg_A[reg_offset][2][0] * reg_B[reg_offset][0][0];\
    reg_C[2][1] += reg_A[reg_offset][2][0] * reg_B[reg_offset][0][1];\
    reg_C[2][2] += reg_A[reg_offset][2][0] * reg_B[reg_offset][0][2];\
    reg_C[2][3] += reg_A[reg_offset][2][0] * reg_B[reg_offset][0][3];\
    \
    FLOAT2(reg_A[reg_offset^1][3][0]) = FLOAT2(A_sh[A_sh_offset + A_index + 3 * (k + padding) + TK]);\
    \
    reg_C[3][0] += reg_A[reg_offset][3][0] * reg_B[reg_offset][0][0];\
    reg_C[3][1] += reg_A[reg_offset][3][0] * reg_B[reg_offset][0][1];\
    reg_C[3][2] += reg_A[reg_offset][3][0] * reg_B[reg_offset][0][2];\
    reg_C[3][3] += reg_A[reg_offset][3][0] * reg_B[reg_offset][0][3];\
    \
    FLOAT4(reg_B[reg_offset^1][1][0]) = FLOAT4(B_sh[B_sh_offset + B_index + (n + padding) + TK * (n + padding)]);\
    \
    reg_C[0][0] += reg_A[reg_offset][0][1] * reg_B[reg_offset][1][0];\
    reg_C[0][1] += reg_A[reg_offset][0][1] * reg_B[reg_offset][1][1];\
    reg_C[0][2] += reg_A[reg_offset][0][1] * reg_B[reg_offset][1][2];\
    reg_C[0][3] += reg_A[reg_offset][0][1] * reg_B[reg_offset][1][3];

#define calc_k_fma2_7_5()\
    reg_C[1][0] += reg_A[reg_offset][1][1] * reg_B[reg_offset][1][0];\
    reg_C[1][1] += reg_A[reg_offset][1][1] * reg_B[reg_offset][1][1];\
    reg_C[1][2] += reg_A[reg_offset][1][1] * reg_B[reg_offset][1][2];\
    reg_C[1][3] += reg_A[reg_offset][1][1] * reg_B[reg_offset][1][3];\
    \
    FLOAT2(reg_A[reg_offset][0][0]) = FLOAT2(A_sh[A_sh_offset + A_index + 2 * TK]);\
    \
    reg_C[2][0] += reg_A[reg_offset][2][1] * reg_B[reg_offset][1][0];\
    reg_C[2][1] += reg_A[reg_offset][2][1] * reg_B[reg_offset][1][1];\
    \
    reg_C[2][2] += reg_A[reg_offset][2][1] * reg_B[reg_offset][1][2];\
    FLOAT4(reg_B[reg_offset][0][0]) = FLOAT4(B_sh[B_sh_offset + B_index + 2 * TK * (n + padding)]);\
    reg_C[2][3] += reg_A[reg_offset][2][1] * reg_B[reg_offset][1][3];\
    \
    reg_C[3][0] += reg_A[reg_offset][3][1] * reg_B[reg_offset][1][0];\
    reg_C[3][1] += reg_A[reg_offset][3][1] * reg_B[reg_offset][1][1];\
    reg_C[3][2] += reg_A[reg_offset][3][1] * reg_B[reg_offset][1][2];\
    reg_C[3][3] += reg_A[reg_offset][3][1] * reg_B[reg_offset][1][3];
__global__ __launch_bounds__(256) void gemm_kernel7_5(float *d_A, float *d_B, float *d_C, int M, int N, int K) {
    // 16 * 16 = 256
    // m * k = 1024 一个线程读 4 个
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int TM = 4;
    const int TN = 4;
    const int TK = 2;
    extern __shared__ __align__(16 * 1024) float sh[];
    const int padding = 8; 
    float *A_sh = sh;
    float *B_sh = sh + 2 * m * (k + padding);
    const int N_tile_index = blockIdx.x; // tile的列号
    const int M_tile_index = blockIdx.y; // tile的行号
    const int idx = threadIdx.x;
    const int A_m_index = idx >> 2;
    const int A_n_index = (idx & 3) << 2;
    const int B_m_index = idx >> 4;
    const int B_n_index = (idx & 15) << 2;
    const int C_m_index = idx >> 4; // tile内的4 * 4行号
    const int C_n_index = idx & 15; // tile内的4 * 4列号
    const int A_pre_thread_num = (m * k + blockDim.x - 1)/ blockDim.x;
    const int B_pre_thread_num = (k * n + blockDim.x - 1)/ blockDim.x;
    const int d_A_index = (M_tile_index * m + A_m_index) * K + A_n_index;
    const int d_B_index = (B_m_index) * N + N_tile_index * n + B_n_index;
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[2][TM][TK];
    float reg_B[2][TK][TN];
    float reg_C[TM][TN] = {0.0f};
    int A_sh_offset = 0;
    int B_sh_offset = 0;
    const int A_sh_size = m * (k + padding);
    const int B_sh_size = k * (n + padding);
    int K_tile_index = 0;
    int reg_offset = 0;
    float4 readA;
    float4 readB;
    lgkmcnt<0>();
    // FLOAT4(A_sh[A_sh_offset + A_m_index * (k + padding) + A_n_index]) = FLOAT4(d_A[d_A_index + K_tile_index * k]);
    // FLOAT4(B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index]) = FLOAT4(d_B[d_B_index + K_tile_index * k * N]);
    global_load<0>(&d_A[d_A_index + K_tile_index * k], readA);
    global_load<0>(&d_B[d_B_index + K_tile_index * k * N], readB);
    vmcnt<0>();

    FLOAT4(A_sh[A_sh_offset + A_m_index * (k + padding) + A_n_index]) = readA;
    FLOAT4(B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index]) = readB;
    do {
        __syncthreads();
        if (K_tile_index + 1 < int((K + k - 1) / k)) {
            global_load<0>(&d_A[d_A_index + (K_tile_index + 1) * k], readA);
            global_load<0>(&d_B[d_B_index + (K_tile_index + 1) * k * N], readB);
	    // printf("readA: %lf %lf %lf %lf\n", readA.x, readA.y, readA.z, readA.w);
            // FLOAT4(A_sh[(A_sh_offset^A_sh_size) + A_m_index * (k + padding) + A_n_index]) = FLOAT4(d_A[d_A_index + (K_tile_index + 1) * k]);
            // FLOAT4(B_sh[(B_sh_offset^B_sh_size) + B_m_index * (n + padding) + B_n_index]) = FLOAT4(d_B[d_B_index + (K_tile_index + 1) * k * N]);
        }
        lgkmcnt<0>();
        {
            int A_index = C_m_index * TM * (k + padding);
            int B_index = C_n_index * TN;
            int k_reg_index = 0;
            FLOAT2(reg_A[reg_offset][0][0]) = FLOAT2(A_sh[A_sh_offset + A_index]);
            FLOAT2(reg_A[reg_offset][1][0]) = FLOAT2(A_sh[A_sh_offset + A_index + (k + padding)]);
            FLOAT2(reg_A[reg_offset][2][0]) = FLOAT2(A_sh[A_sh_offset + A_index + 2 * (k + padding)]);
            FLOAT2(reg_A[reg_offset][3][0]) = FLOAT2(A_sh[A_sh_offset + A_index + 3 * (k + padding)]);
            FLOAT4(reg_B[reg_offset][0][0]) = FLOAT4(B_sh[B_sh_offset + B_index]);
            FLOAT4(reg_B[reg_offset][1][0]) = FLOAT4(B_sh[B_sh_offset + B_index + (n + padding)]);

            FLOAT2(reg_A[reg_offset^1][0][0]) = FLOAT2(A_sh[A_sh_offset + A_index + TK]);
            FLOAT4(reg_B[reg_offset^1][0][0]) = FLOAT4(B_sh[B_sh_offset + B_index + TK * (n + padding)]);
            do {
                if (k_reg_index + TK < k) {
                   calc_k_fma1_7_5();// 0
                } else {
                    reg_C[0][0] += reg_A[reg_offset][0][0] * reg_B[reg_offset][0][0];\
                    reg_C[0][1] += reg_A[reg_offset][0][0] * reg_B[reg_offset][0][1];\
                    reg_C[0][2] += reg_A[reg_offset][0][0] * reg_B[reg_offset][0][2];\
                    reg_C[0][3] += reg_A[reg_offset][0][0] * reg_B[reg_offset][0][3];\
                    
                    reg_C[1][0] += reg_A[reg_offset][1][0] * reg_B[reg_offset][0][0];\
                    reg_C[1][1] += reg_A[reg_offset][1][0] * reg_B[reg_offset][0][1];\
                    reg_C[1][2] += reg_A[reg_offset][1][0] * reg_B[reg_offset][0][2];\
                    reg_C[1][3] += reg_A[reg_offset][1][0] * reg_B[reg_offset][0][3];\
                    
                    reg_C[2][0] += reg_A[reg_offset][2][0] * reg_B[reg_offset][0][0];\
                    reg_C[2][1] += reg_A[reg_offset][2][0] * reg_B[reg_offset][0][1];\
                    reg_C[2][2] += reg_A[reg_offset][2][0] * reg_B[reg_offset][0][2];\
                    reg_C[2][3] += reg_A[reg_offset][2][0] * reg_B[reg_offset][0][3];\
                    
                    reg_C[3][0] += reg_A[reg_offset][3][0] * reg_B[reg_offset][0][0];\
                    reg_C[3][1] += reg_A[reg_offset][3][0] * reg_B[reg_offset][0][1];\
                    reg_C[3][2] += reg_A[reg_offset][3][0] * reg_B[reg_offset][0][2];\
                    reg_C[3][3] += reg_A[reg_offset][3][0] * reg_B[reg_offset][0][3];\
                    
                    reg_C[0][0] += reg_A[reg_offset][0][1] * reg_B[reg_offset][1][0];\
                    reg_C[0][1] += reg_A[reg_offset][0][1] * reg_B[reg_offset][1][1];\
                    reg_C[0][2] += reg_A[reg_offset][0][1] * reg_B[reg_offset][1][2];\
                    reg_C[0][3] += reg_A[reg_offset][0][1] * reg_B[reg_offset][1][3];
                }
                if (k_reg_index + 2 * TK < k) {
                   calc_k_fma2_7_5();
                }  else {
                    reg_C[1][0] += reg_A[reg_offset][1][1] * reg_B[reg_offset][1][0];
                    reg_C[1][1] += reg_A[reg_offset][1][1] * reg_B[reg_offset][1][1];
                    reg_C[1][2] += reg_A[reg_offset][1][1] * reg_B[reg_offset][1][2];
                    reg_C[1][3] += reg_A[reg_offset][1][1] * reg_B[reg_offset][1][3];   
                    
		    reg_C[2][0] += reg_A[reg_offset][2][1] * reg_B[reg_offset][1][0];
                    reg_C[2][1] += reg_A[reg_offset][2][1] * reg_B[reg_offset][1][1];
                    reg_C[2][2] += reg_A[reg_offset][2][1] * reg_B[reg_offset][1][2];
                    reg_C[2][3] += reg_A[reg_offset][2][1] * reg_B[reg_offset][1][3];
                    
		    reg_C[3][0] += reg_A[reg_offset][3][1] * reg_B[reg_offset][1][0];
                    reg_C[3][1] += reg_A[reg_offset][3][1] * reg_B[reg_offset][1][1];
                    reg_C[3][2] += reg_A[reg_offset][3][1] * reg_B[reg_offset][1][2];
                    reg_C[3][3] += reg_A[reg_offset][3][1] * reg_B[reg_offset][1][3];
                }
                reg_offset ^= 1;
                A_index += TK;
                B_index += TK * (n + padding);
                k_reg_index += TK;
            } while (k_reg_index < k);

            // calc_k_fma1_7_5();// 0
            // calc_k_fma2_7_5();
            // A_index += TK;
            // B_index += TK * (n + padding);
            // calc_k_fma1_7_5();// 2
            // calc_k_fma2_7_5();
            // A_index += TK;
            // B_index += TK * (n + padding);
            // calc_k_fma1_7_5();// 4
            // calc_k_fma2_7_5();
            // A_index += TK;
            // B_index += TK * (n + padding);
            // calc_k_fma1_7_5();// 6
            // calc_k_fma2_7_5();
            // A_index += TK;
            // B_index += TK * (n + padding);
            // calc_k_fma1_7_5();// 8
            // calc_k_fma2_7_5();
            // A_index += TK;
            // B_index += TK * (n + padding);
            // calc_k_fma1_7_5();// 10
            // calc_k_fma2_7_5();
            // A_index += TK;
            // B_index += TK * (n + padding);
            // calc_k_fma1_7_5();// 12
            // calc_k_fma2_7_5();
            // A_index += TK;
            // B_index += TK * (n + padding);
            // calc_k_fma1_7_4();// 14
            // reg_C[1][0] += reg_A[1][1] * reg_B[1][0];
            // reg_C[1][1] += reg_A[1][1] * reg_B[1][1];
            // reg_C[1][2] += reg_A[1][1] * reg_B[1][2];
            // reg_C[1][3] += reg_A[1][1] * reg_B[1][3];   
            // reg_C[2][0] += reg_A[2][1] * reg_B[1][0];
            // reg_C[2][1] += reg_A[2][1] * reg_B[1][1];

            // reg_C[2][2] += reg_A[2][1] * reg_B[1][2];
            // reg_C[2][3] += reg_A[2][1] * reg_B[1][3];
            // reg_C[3][0] += reg_A[3][1] * reg_B[1][0];
            // reg_C[3][1] += reg_A[3][1] * reg_B[1][1];
            // reg_C[3][2] += reg_A[3][1] * reg_B[1][2];
            // reg_C[3][3] += reg_A[3][1] * reg_B[1][3];
        }
        A_sh_offset ^= A_sh_size;
        B_sh_offset ^= B_sh_size;
        K_tile_index++;
        vmcnt<0>();
        FLOAT4(A_sh[A_sh_offset + A_m_index * (k + padding) + A_n_index]) = readA;
        FLOAT4(B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index]) = readB;
    } while (K_tile_index < int((K + k - 1) / k));
    int C_index = (M_tile_index * m + C_m_index * TM) * N + N_tile_index * n + C_n_index * TN;
    lgkmcnt<0>();
    global_store<0>(&d_C[C_index], FLOAT4(reg_C[0][0]));
    global_store<0>(&d_C[C_index + 1 * N], FLOAT4(reg_C[1][0]));
    global_store<0>(&d_C[C_index + 2 * N], FLOAT4(reg_C[2][0]));
    global_store<0>(&d_C[C_index + 3 * N], FLOAT4(reg_C[3][0]));
    vmcnt<0>();
    // FLOAT4(d_C[C_index]) = FLOAT4(reg_C[0][0]);
    // FLOAT4(d_C[C_index + 1 * N]) = FLOAT4(reg_C[1][0]);
    // FLOAT4(d_C[C_index + 2 * N]) = FLOAT4(reg_C[2][0]);
    // FLOAT4(d_C[C_index + 3 * N]) = FLOAT4(reg_C[3][0]);
}
float test7_5 () {
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int TM = 4;
    const int TN = 4;
    const int TK = 2;
    const int padding = 8;
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((m * n + TM * TN - 1) / (TM * TN));
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding)) * 2;
    gemm_kernel7_5<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel7_5<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K);
    }
    test_end();
    return elapsedTime / iteration;
}


__global__ void gemm_kernel7_6(float *d_A, float *d_B, float *d_C, int M, int N, int K) {
    const int padding = 4; 
    // 16 * 16 = 256
    // m * k = 1024 一个线程读 4 个
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int TM = 4;
    const int TN = 4;
    const int TK = 4;
    __shared__ float A_sh[2][m][k + padding];
    __shared__ float B_sh[2][k][n + padding];
    const int N_tile_index = blockIdx.y; // tile的列号
    const int M_tile_index = blockIdx.x; // tile的行号
    const int idx = threadIdx.x;
    const int A_m_index = idx >> 2;
    const int A_n_index = (idx & 3) << 2;
    const int B_m_index = idx >> 4;
    const int B_n_index = (idx & 15) << 2;
    const int C_m_index = idx >> 4; // tile内的4 * 4行号
    const int C_n_index = idx & 15; // tile内的4 * 4列号
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[TM][TK];
    float reg_B[TK][TN];
    float reg_C[TM][TN] = {0.0f};
    const int d_A_index = (M_tile_index * m + A_m_index) * K + A_n_index;
    const int d_B_index = (B_m_index) * N + N_tile_index * n + B_n_index;
    int sh_offset = 0;
    int K_tile_index = 0;
    FLOAT4(A_sh[sh_offset][A_m_index][A_n_index]) = FLOAT4(d_A[d_A_index + K_tile_index * k]);
    FLOAT4(B_sh[sh_offset][B_m_index][B_n_index]) = FLOAT4(d_B[d_B_index + K_tile_index * k * N]);
    do {
        __syncthreads();
        if (threadIdx.y == 1 && K_tile_index + 1 < int((K + k - 1) / k)) {
            FLOAT4(A_sh[sh_offset^1][A_m_index][A_n_index]) = FLOAT4(d_A[d_A_index + (K_tile_index + 1) * k]);
            FLOAT4(B_sh[sh_offset^1][B_m_index][B_n_index]) = FLOAT4(d_B[d_B_index + (K_tile_index + 1) * k * N]);  
        }
        if (threadIdx.y == 0) {
            for (int k_reg_index = 0; k_reg_index < k; k_reg_index+= TK) {
                for (int i = 0; i < TM; i++) {
                    FLOAT4(reg_A[i][0]) = FLOAT4(A_sh[sh_offset][C_m_index * TM + i][k_reg_index]);
                }
                for (int i = 0; i < TK; i++) {
                    FLOAT4(reg_B[i][0]) = FLOAT4(B_sh[sh_offset][k_reg_index + i][C_n_index * TN]);
                }
                for (int i = 0; i < TM; i++) {
                    for (int j = 0; j < TN; j++) {
                        for (int k_index = 0; k_index < TK; k_index++) {
                            reg_C[i][j] += reg_A[i][k_index] * reg_B[k_index][j];
                        }
                    }
                }
            }
        }
        sh_offset ^= 1;
        K_tile_index++;
    } while (K_tile_index < int((K + k - 1) / k));
    if (threadIdx.y == 0) {
        for (int i = 0; i < TM; i++) {
            int C_index = (M_tile_index * m + C_m_index * TM) * N + N_tile_index * n + C_n_index * TN + i * N;
            FLOAT4(d_C[C_index]) = FLOAT4(reg_C[i][0]);
        }
    }
}
float test7_6 () {
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int TM = 4;
    const int TN = 4;
    const int padding = 4;
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((m * n + TM * TN - 1) / (TM * TN), 2);
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding));
    gemm_kernel7_6<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel7_6<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K);
    }
    test_end();
    return elapsedTime / iteration;
}
#define calc_k_fma1_7_7()\
    reg_C[0][0] += A_sh[A_sh_offset + A_index] * B_sh[B_sh_offset + B_index];\
    reg_C[0][1] += A_sh[A_sh_offset + A_index] * B_sh[B_sh_offset + B_index + 1];\
    reg_C[0][2] += A_sh[A_sh_offset + A_index] * B_sh[B_sh_offset + B_index + 2];\
    reg_C[0][3] += A_sh[A_sh_offset + A_index] * B_sh[B_sh_offset + B_index + 3];\
    \
    reg_C[1][0] += A_sh[A_sh_offset + A_index + (k + padding)] * B_sh[B_sh_offset + B_index];\
    reg_C[1][1] += A_sh[A_sh_offset + A_index + (k + padding)] * B_sh[B_sh_offset + B_index + 1];\
    reg_C[1][2] += A_sh[A_sh_offset + A_index + (k + padding)] * B_sh[B_sh_offset + B_index + 2];\
    reg_C[1][3] += A_sh[A_sh_offset + A_index + (k + padding)] * B_sh[B_sh_offset + B_index + 3];\
    \
    reg_C[2][0] += A_sh[A_sh_offset + A_index + 2 * (k + padding)] * B_sh[B_sh_offset + B_index];\
    reg_C[2][1] += A_sh[A_sh_offset + A_index + 2 * (k + padding)] * B_sh[B_sh_offset + B_index + 1];\
    reg_C[2][2] += A_sh[A_sh_offset + A_index + 2 * (k + padding)] * B_sh[B_sh_offset + B_index + 2];\
    reg_C[2][3] += A_sh[A_sh_offset + A_index + 2 * (k + padding)] * B_sh[B_sh_offset + B_index + 3];\
    \
    reg_C[3][0] += A_sh[A_sh_offset + A_index + 3 * (k + padding)] * B_sh[B_sh_offset + B_index];\
    reg_C[3][1] += A_sh[A_sh_offset + A_index + 3 * (k + padding)] * B_sh[B_sh_offset + B_index + 1];\
    reg_C[3][2] += A_sh[A_sh_offset + A_index + 3 * (k + padding)] * B_sh[B_sh_offset + B_index + 2];\
    reg_C[3][3] += A_sh[A_sh_offset + A_index + 3 * (k + padding)] * B_sh[B_sh_offset + B_index + 3];\
    \
    reg_C[0][0] += A_sh[A_sh_offset + A_index + 1] * B_sh[B_sh_offset + B_index + (n + padding)];\
    reg_C[0][1] += A_sh[A_sh_offset + A_index + 1] * B_sh[B_sh_offset + B_index + (n + padding) + 1];\
    reg_C[0][2] += A_sh[A_sh_offset + A_index + 1] * B_sh[B_sh_offset + B_index + (n + padding) + 2];\
    reg_C[0][3] += A_sh[A_sh_offset + A_index + 1] * B_sh[B_sh_offset + B_index + (n + padding) + 3];
#define calc_k_fma2_7_7()\
    reg_C[1][0] += A_sh[A_sh_offset + A_index + (k + padding) + 1] * B_sh[B_sh_offset + B_index + (n + padding)];\
    reg_C[1][1] += A_sh[A_sh_offset + A_index + (k + padding) + 1] * B_sh[B_sh_offset + B_index + (n + padding) + 1];\
    reg_C[1][2] += A_sh[A_sh_offset + A_index + (k + padding) + 1] * B_sh[B_sh_offset + B_index + (n + padding) + 2];\
    reg_C[1][3] += A_sh[A_sh_offset + A_index + (k + padding) + 1] * B_sh[B_sh_offset + B_index + (n + padding) + 3];\
    \
    reg_C[2][0] += A_sh[A_sh_offset + A_index + 2 * (k + padding) + 1] * B_sh[B_sh_offset + B_index + (n + padding)];\
    reg_C[2][1] += A_sh[A_sh_offset + A_index + 2 * (k + padding) + 1] * B_sh[B_sh_offset + B_index + (n + padding) + 1];\
    reg_C[2][2] += A_sh[A_sh_offset + A_index + 2 * (k + padding) + 1] * B_sh[B_sh_offset + B_index + (n + padding) + 2];\
    reg_C[2][3] += A_sh[A_sh_offset + A_index + 2 * (k + padding) + 1] * B_sh[B_sh_offset + B_index + (n + padding) + 3];\
    \
    reg_C[3][0] += A_sh[A_sh_offset + A_index + 3 * (k + padding) + 1] * B_sh[B_sh_offset + B_index + (n + padding)];\
    reg_C[3][1] += A_sh[A_sh_offset + A_index + 3 * (k + padding) + 1] * B_sh[B_sh_offset + B_index + (n + padding) + 1];\
    reg_C[3][2] += A_sh[A_sh_offset + A_index + 3 * (k + padding) + 1] * B_sh[B_sh_offset + B_index + (n + padding) + 2];\
    reg_C[3][3] += A_sh[A_sh_offset + A_index + 3 * (k + padding) + 1] * B_sh[B_sh_offset + B_index + (n + padding) + 3];
__global__ void gemm_kernel7_7(float *d_A, float *d_B, float *d_C, int M, int N, int K) {
    // 16 * 16 = 256
    // m * k = 1024 一个线程读 4 个
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int TM = 4;
    const int TN = 4;
    const int TK = 2;
    extern __shared__ __align__(16 * 1024) float sh[];
    const int padding = 4; 
    float *A_sh = sh;
    float *B_sh = sh + 2 * m * (k + padding);
    const int N_tile_index = blockIdx.x; // tile的列号
    const int M_tile_index = blockIdx.y; // tile的行号
    const int idx = threadIdx.x;
    const int A_m_index = idx >> 2;
    const int A_n_index = (idx & 3) << 2;
    const int B_m_index = idx >> 4;
    const int B_n_index = (idx & 15) << 2;
    const int warp_id = idx / WAVE_SIZE;
    const int warp_id_m = warp_id / 1; // 4 * 1 的格子
    const int warp_id_n = warp_id % 1;
    const int lane_id = idx % WAVE_SIZE;
    const int lane_id_m = lane_id / 16; // 4 * 16 的块
    const int lane_id_n = lane_id % 16;
    const int C_m_index = warp_id_m * 4 + lane_id_m; // tile内的4 * 4行号
    const int C_n_index = warp_id_n * 16 + lane_id_n; // tile内的4 * 4列号
    // const int C_m_index = idx >> 4; // tile内的4 * 4行号
    // const int C_n_index = idx & 15; // tile内的4 * 4列号
    const int d_A_index = (M_tile_index * m + A_m_index) * K + A_n_index;
    const int d_B_index = (B_m_index) * N + N_tile_index * n + B_n_index;
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[TM][TK];
    float reg_B[TK][TN];
    float reg_C[TM][TN] = {0.0f};
    int A_sh_offset = 0;
    int B_sh_offset = 0;
    const int A_sh_size = m * (k + padding);
    const int B_sh_size = k * (n + padding);
    int K_tile_index = 0;
    float4 readA;
    float4 readB;
    if (threadIdx.y == 0) {
       FLOAT4(A_sh[A_sh_offset + A_m_index * (k + padding) + A_n_index]) = FLOAT4(d_A[d_A_index + K_tile_index * k]);
       FLOAT4(B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index]) = FLOAT4(d_B[d_B_index + K_tile_index * k * N]);
    }
    do {
        __syncthreads();
        if (threadIdx.y == 1) {
            if (K_tile_index + 1 < int((K + k - 1) / k)) {
                FLOAT4(A_sh[(A_sh_offset^A_sh_size) + A_m_index * (k + padding) + A_n_index]) = FLOAT4(d_A[d_A_index + (K_tile_index + 1) * k]);
                FLOAT4(B_sh[(B_sh_offset^B_sh_size) + B_m_index * (n + padding) + B_n_index]) = FLOAT4(d_B[d_B_index + (K_tile_index + 1) * k * N]);
            }
        }
        if (threadIdx.y == 0) {
            int A_index = C_m_index * TM * (k + padding);
            int B_index = C_n_index * TN;
            calc_k_fma1_7_7();// 0
            calc_k_fma2_7_7();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_7();// 2
            calc_k_fma2_7_7();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_7();// 4
            calc_k_fma2_7_7();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_7();// 6
            calc_k_fma2_7_7();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_7();// 8
            calc_k_fma2_7_7();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_7();// 10
            calc_k_fma2_7_7();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_7();// 12
            calc_k_fma2_7_7();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_7();// 14
            calc_k_fma2_7_7();
            
        }
        A_sh_offset ^= A_sh_size;
        B_sh_offset ^= B_sh_size;
        K_tile_index++;
    } while (K_tile_index < int((K + k - 1) / k));
    if (threadIdx.y == 0) {
        const int C_index = (M_tile_index * m + C_m_index * TM) * N + N_tile_index * n + C_n_index * TN;
        FLOAT4(d_C[C_index]) = FLOAT4(reg_C[0][0]);
        FLOAT4(d_C[C_index + 1 * N]) = FLOAT4(reg_C[1][0]);
        FLOAT4(d_C[C_index + 2 * N]) = FLOAT4(reg_C[2][0]);
        FLOAT4(d_C[C_index + 3 * N]) = FLOAT4(reg_C[3][0]);
    }
}
float test7_7 () {
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
    dim3 thread((m * n + TM * TN - 1) / (TM * TN), 2);
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding)) * 2;
    gemm_kernel7_7<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel7_7<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K);
    }
    test_end();
    return elapsedTime / iteration;
}
#define calc_k_fma1_7_8()\
    reg_C[0][0] += reg_A[0][0] * reg_B[0][0];\
    reg_C[0][1] += reg_A[0][0] * reg_B[0][1];\
    reg_C[0][2] += reg_A[0][0] * reg_B[0][2];\
    reg_C[0][3] += reg_A[0][0] * reg_B[0][3];\
    \
    FLOAT2(reg_A[1][0]) = FLOAT2(A_sh[A_sh_offset + A_index + (k + padding)]);\
    \
    reg_C[1][0] += reg_A[1][0] * reg_B[0][0];\
    reg_C[1][1] += reg_A[1][0] * reg_B[0][1];\
    reg_C[1][2] += reg_A[1][0] * reg_B[0][2];\
    reg_C[1][3] += reg_A[1][0] * reg_B[0][3];\
    \
    FLOAT2(reg_A[2][0]) = FLOAT2(A_sh[A_sh_offset + A_index + 2 * (k + padding)]);\
    \
    reg_C[2][0] += reg_A[2][0] * reg_B[0][0];\
    reg_C[2][1] += reg_A[2][0] * reg_B[0][1];\
    reg_C[2][2] += reg_A[2][0] * reg_B[0][2];\
    reg_C[2][3] += reg_A[2][0] * reg_B[0][3];\
    \
    FLOAT2(reg_A[3][0]) = FLOAT2(A_sh[A_sh_offset + A_index + 3 * (k + padding)]);\
    \
    reg_C[3][0] += reg_A[3][0] * reg_B[0][0];\
    reg_C[3][1] += reg_A[3][0] * reg_B[0][1];\
    reg_C[3][2] += reg_A[3][0] * reg_B[0][2];\
    reg_C[3][3] += reg_A[3][0] * reg_B[0][3];\
    \
    FLOAT4(reg_B[1][0]) = FLOAT4(B_sh[B_sh_offset + B_index + (n + padding)]);\
    \
    reg_C[0][0] += reg_A[0][1] * reg_B[1][0];\
    reg_C[0][1] += reg_A[0][1] * reg_B[1][1];\
    reg_C[0][2] += reg_A[0][1] * reg_B[1][2];\
    reg_C[0][3] += reg_A[0][1] * reg_B[1][3];

#define calc_k_fma2_7_8()\
    reg_C[1][0] += reg_A[1][1] * reg_B[1][0];\
    reg_C[1][1] += reg_A[1][1] * reg_B[1][1];\
    reg_C[1][2] += reg_A[1][1] * reg_B[1][2];\
    reg_C[1][3] += reg_A[1][1] * reg_B[1][3];\
    \
    FLOAT2(reg_A[0][0]) = FLOAT2(A_sh[A_sh_offset + A_index + TK]);\
    \
    reg_C[2][0] += reg_A[2][1] * reg_B[1][0];\
    reg_C[2][1] += reg_A[2][1] * reg_B[1][1];\
    \
    reg_C[2][2] += reg_A[2][1] * reg_B[1][2];\
    FLOAT4(reg_B[0][0]) = FLOAT4(B_sh[B_sh_offset + B_index + TK * (n + padding)]);\
    reg_C[2][3] += reg_A[2][1] * reg_B[1][3];\
    \
    reg_C[3][0] += reg_A[3][1] * reg_B[1][0];\
    reg_C[3][1] += reg_A[3][1] * reg_B[1][1];\
    reg_C[3][2] += reg_A[3][1] * reg_B[1][2];\
    reg_C[3][3] += reg_A[3][1] * reg_B[1][3];
__global__ void gemm_kernel7_8(float *d_A, float *d_B, float *d_C, int M, int N, int K) {
    // 16 * 16 = 256
    // m * k = 1024 一个线程读 4 个
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int TM = 4;
    const int TN = 4;
    const int TK = 2;
    extern __shared__ float sh[];
    const int padding = 4; 
    float *A_sh = sh;
    float *B_sh = sh + 2 * m * (k + padding);
    const int N_tile_index = blockIdx.x; // tile的列号
    const int M_tile_index = blockIdx.y; // tile的行号
    const int idx = threadIdx.x;
    const int A_m_index = idx >> 2;
    const int A_n_index = (idx & 3) << 2;
    const int B_m_index = idx >> 4;
    const int B_n_index = (idx & 15) << 2;
    const int C_m_index = idx >> 4; // tile内的4 * 4行号
    const int C_n_index = idx & 15; // tile内的4 * 4列号
    const int A_pre_thread_num = (m * k + blockDim.x - 1)/ blockDim.x;
    const int B_pre_thread_num = (k * n + blockDim.x - 1)/ blockDim.x;
    const int d_A_index = (M_tile_index * m + A_m_index) * K + A_n_index;
    const int d_B_index = (B_m_index) * N + N_tile_index * n + B_n_index;
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[TM][TK];
    float reg_B[TK][TN];
    float reg_C[TM][TN] = {0.0f};
    int A_sh_offset = 0;
    int B_sh_offset = 0;
    const int A_sh_size = m * (k + padding);
    const int B_sh_size = k * (n + padding);
    int K_tile_index = 0;
    float4 readA;
    float4 readB;

    lgkmcnt<0>();
    // FLOAT4(A_sh[A_sh_offset + A_m_index * (k + padding) + A_n_index]) = FLOAT4(d_A[d_A_index + K_tile_index * k]);
    // FLOAT4(B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index]) = FLOAT4(d_B[d_B_index + K_tile_index * k * N]);
    global_load<0>(&d_A[d_A_index + K_tile_index * k], readA);
    global_load<0>(&d_B[d_B_index + K_tile_index * k * N], readB);
    vmcnt<0>();

    FLOAT4(A_sh[A_sh_offset + A_m_index * (k + padding) + A_n_index]) = readA;
    FLOAT4(B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index]) = readB;

    do {
        __syncthreads();
        if (K_tile_index + 1 < int((K + k - 1) / k)) {
            global_load<0>(&d_A[d_A_index + (K_tile_index + 1) * k], readA);
            global_load<0>(&d_B[d_B_index + (K_tile_index + 1) * k * N], readB);
            // FLOAT4(A_sh[(A_sh_offset^A_sh_size) + A_m_index * (k + padding) + A_n_index]) = FLOAT4(d_A[d_A_index + (K_tile_index + 1) * k]);
            // FLOAT4(B_sh[(B_sh_offset^B_sh_size) + B_m_index * (n + padding) + B_n_index]) = FLOAT4(d_B[d_B_index + (K_tile_index + 1) * k * N]);
        }
        
        lgkmcnt<0>();
        
        {
            int A_index = C_m_index * TM * (k + padding);
            int B_index = C_n_index * TN;
            FLOAT2(reg_A[0][0]) = FLOAT2(A_sh[A_sh_offset + A_index]);
            FLOAT4(reg_B[0][0]) = FLOAT4(B_sh[B_sh_offset + B_index]);
            calc_k_fma1_7_8();// 0
            calc_k_fma2_7_8();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_8();// 2
            calc_k_fma2_7_8();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_8();// 4
            calc_k_fma2_7_8();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_8();// 6
            calc_k_fma2_7_8();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_8();// 8
            calc_k_fma2_7_8();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_8();// 10
            calc_k_fma2_7_8();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_8();// 12
            calc_k_fma2_7_8();
            A_index += TK;
            B_index += TK * (n + padding);
            calc_k_fma1_7_8();// 14
            reg_C[1][0] += reg_A[1][1] * reg_B[1][0];
            reg_C[1][1] += reg_A[1][1] * reg_B[1][1];
            reg_C[1][2] += reg_A[1][1] * reg_B[1][2];
            reg_C[1][3] += reg_A[1][1] * reg_B[1][3];   
            reg_C[2][0] += reg_A[2][1] * reg_B[1][0];
            reg_C[2][1] += reg_A[2][1] * reg_B[1][1];

            reg_C[2][2] += reg_A[2][1] * reg_B[1][2];
            reg_C[2][3] += reg_A[2][1] * reg_B[1][3];
            reg_C[3][0] += reg_A[3][1] * reg_B[1][0];
            reg_C[3][1] += reg_A[3][1] * reg_B[1][1];
            reg_C[3][2] += reg_A[3][1] * reg_B[1][2];
            reg_C[3][3] += reg_A[3][1] * reg_B[1][3];
        }
        A_sh_offset ^= A_sh_size;
        B_sh_offset ^= B_sh_size;
        K_tile_index++;
        vmcnt<0>();
        FLOAT4(A_sh[A_sh_offset + A_m_index * (k + padding) + A_n_index]) = readA;
        FLOAT4(B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index]) = readB;
    } while (K_tile_index < int((K + k - 1) / k));
    const int C_index = (M_tile_index * m + C_m_index * TM) * N + N_tile_index * n + C_n_index * TN;
    // global_load<0>(&d_C[C_index], *((float4 *)&reg_C[0][0]));
    // global_load<0>(&d_C[C_index + 1 * N], *((float4*)&reg_C[1][0]));
    // global_load<0>(&d_C[C_index + 2 * N], *((float4*)&reg_C[2][0]));
    // global_load<0>(&d_C[C_index + 3 * N], *((float4*)&reg_C[3][0]));
    FLOAT4(d_C[C_index]) = FLOAT4(reg_C[0][0]);
    FLOAT4(d_C[C_index + 1 * N]) = FLOAT4(reg_C[1][0]);
    FLOAT4(d_C[C_index + 2 * N]) = FLOAT4(reg_C[2][0]);
    FLOAT4(d_C[C_index + 3 * N]) = FLOAT4(reg_C[3][0]);
    // vmcnt<0>();
}
float test7_8 () {
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
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding)) * 2;
    gemm_kernel7_8<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel7_8<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K);
    }
    test_end();
    return elapsedTime / iteration;
}
