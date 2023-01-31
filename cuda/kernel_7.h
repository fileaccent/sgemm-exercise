#define prefetch7()\
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
__global__ void gemm_kernel7(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m, int n, int k) {
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
        prefetch7();
	    __syncthreads();
        for (int k_reg_index = 0; k_reg_index < k; k_reg_index+= TK) {
            for (int i = 0; i < TM; i++) {
                int A_index = C_m_index * TM * (k + padding) + k_reg_index +  i * (k + padding);
                reg_A[i][0] = A_sh[A_index];
                reg_A[i][1] = A_sh[A_index + 1];
            }
            for (int i = 0; i < TK; i++) {
                int B_index = k_reg_index * (n + padding) + C_n_index * TN + i * (n + padding);
                FLOAT4(reg_B[i][0]) = FLOAT4(B_sh[B_index]);
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
    const int TK = 2;
    const int padding = 4;
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((m * n + TM * TN - 1) / (TM * TN));
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding));
    gemm_kernel7<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(cudaEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel7<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
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
    ErrChk(cudaEventRecord(start, 0));
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
    // for (int i = 0; i < TM; i++) {
    //    int C_index = (M_tile_index * m + C_m_index * TM) * N + N_tile_index * n + C_n_index * TN + i * N;
    //    set_value(&d_C[C_index], reg_C[i], TN);
    // }
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
    ErrChk(cudaEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel7_2<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}