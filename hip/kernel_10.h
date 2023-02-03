__global__ void gemm_kernel10(float *d_A, float *d_B, float *d_C, int M, int N, int K) {
    const int padding = 4;
    const int BM = 64;
    const int BN = 64;
    const int BK = 16;
    const int TM = 4;
    const int TN = 4;
    const int TK = 2;
    extern __shared__ float sh[];
    float *A_sh = sh;
    float *B_sh = sh + 2 * BM * (BK + padding);
    const int bl_x = blockIdx.x + blockIdx.y * gridDim.x; 
    // const int N_tile_index = bl_x / ((N + BN - 1) / BN); // tile的列号
    // const int M_tile_index = bl_x % ((N + BN - 1) / BN); // tile的列号
    const int N_tile_index = blockIdx.y; // tile的行号
    const int M_tile_index = blockIdx.x; // tile的行号
    const int C_BM_index = threadIdx.x / ((BN + TN - 1) / TN); // tile内的4 * 4行号
    const int C_BN_index = threadIdx.x % ((BN + TN - 1) / TN); // tile内的4 * 4列号
    const int A_pre_thread_num = (BM * BK + blockDim.x - 1)/ blockDim.x;
    const int B_pre_thread_num = (BK * BN + blockDim.x - 1) / blockDim.x;
    int ix;
    ix = threadIdx.x * A_pre_thread_num;
    const int A_pre_thread_m = (A_pre_thread_num + BK - 1) / BK;
    const int A_pre_thread_n = A_pre_thread_num % BK;
    const int A_BM_index = ix / BK;
    const int A_BN_index = ix % BK;
    ix = threadIdx.x * B_pre_thread_num;
    const int B_pre_thread_m = (B_pre_thread_num + BN - 1) / BN;
    const int B_pre_thread_n = B_pre_thread_num % BN;
    const int B_BM_index = ix / BN;
    const int B_BN_index = ix % BN;
    const int d_A_index = (M_tile_index * BM + A_BM_index) * K + A_BN_index;
    const int d_B_index = (B_BM_index) * N + N_tile_index * BN + B_BN_index;
    
    // printf("m_index: %d, n_index: %d\BN", m_index, n_index);
    float reg_A[TM][TK];
    float reg_B[TK][TN];
    float reg_C[TM][TN] = {0.0f};
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + BK - 1) / BK); K_tile_index+= 2) {
        set_value_matrix(&A_sh[A_BM_index * (BK + padding) + A_BN_index], &d_A[d_A_index + K_tile_index * BK], A_pre_thread_m, A_pre_thread_n, BK + padding, K);
        set_value_matrix(&B_sh[B_BM_index * (BN + padding) + B_BN_index], &d_B[d_B_index + K_tile_index * BK * N], B_pre_thread_m, B_pre_thread_n, BN + padding, N);
	    __syncthreads();
        for (int k_reg_index = 0; k_reg_index < BK; k_reg_index += TK) {
            int A_index = C_BM_index * TM * (BK + padding) + k_reg_index;
            set_value_matrix((float *)reg_A[0], &A_sh[A_index], TM, TK, TK, BK + padding);
            int B_index = k_reg_index * (BN + padding) + C_BN_index * TN;
	        set_value_matrix((float *)reg_B[0], &B_sh[B_index], TK, TN, TN, BN + padding);
            for (int k_index = 0; k_index < TK; k_index++) {
                for (int i = 0; i < TM; i++) {
                     for (int j = 0; j < TN; j++) {
                         reg_C[i][j] += reg_A[i][k_index] * reg_B[k_index][j];
		     }
                }
            }
        }
         __syncthreads();

        set_value_matrix(&A_sh[BM * (BK + padding) + A_BM_index * (BK + padding) + A_BN_index], &d_A[d_A_index + (K_tile_index + 1) * BK], A_pre_thread_m, A_pre_thread_n, BK + padding, K);
        set_value_matrix(&B_sh[BK * (BN + padding) + B_BM_index * (BN + padding) + B_BN_index], &d_B[d_B_index + (K_tile_index + 1) * BK * N], B_pre_thread_m, B_pre_thread_n, BN + padding, N);
	    __syncthreads();
        for (int k_reg_index = 0; k_reg_index < BK; k_reg_index += TK) {
            int A_index = C_BM_index * TM * (BK + padding) + k_reg_index;
            set_value_matrix((float *)reg_A[0], &A_sh[A_index], TM, TK, TK, BK + padding);
            int B_index = k_reg_index * (BN + padding) + C_BN_index * TN;
	        set_value_matrix((float *)reg_B[0], &B_sh[B_index], TK, TN, TN, BN + padding);
            for (int k_index = 0; k_index < TK; k_index++) {
                for (int i = 0; i < TM; i++) {
                     for (int j = 0; j < TN; j++) {
                         reg_C[i][j] += reg_A[i][k_index] * reg_B[k_index][j];
		     }
                }
            }
        }
         __syncthreads();
    }
    
    int C_index = (M_tile_index * BM + C_BM_index * TM) * N + N_tile_index * BN + C_BN_index * TN;
    set_value_matrix(&d_C[C_index], (float *)reg_C[0], TM, TN, N, TN);
}
float test10 () {
    const int BM = 64;
    const int BN = 64;
    const int BK = 16;
    const int TM = 4;
    const int TN = 4;
    const int padding = 4;
    const int WGM = 8;
    // int thread_size = (BM * BN + reg_size * reg_size - 1) / (reg_size * reg_size);
    test_start();
    int thread_size = min(BM * BN, C_size);
    // dim3 block((M + BM - 1) / BM * (N + BN - 1) / BN / WGM, WGM);
    dim3 block((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 thread((BM * BN + TM * TN - 1) / (TM * TN));
    int shared_size = sizeof(float) * (BM * (BK + padding) + BK * (BN + padding)) * 2;
    gemm_kernel10<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel10<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K);
    }
    test_end();
    return elapsedTime / iteration;
}

