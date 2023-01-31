#define prefetch8()\
    pre_thread_num = (m * k + blockDim.x * blockDim.y - 1)/ (blockDim.x * blockDim.y);\
    ix = idx * pre_thread_num;\
    A_m_index = ix / k;\
    A_n_index = ix % k;\
    int d_A_index = (M_tile_index * m + A_m_index) * K + K_tile_index * k + A_n_index;\
    ix = A_m_index * (k + padding) + A_n_index;\
    FLOAT4(A_sh[ix]) = FLOAT4(d_A[d_A_index]);\
    pre_thread_num = (k * n + blockDim.x * blockDim.y - 1) / (blockDim.x * blockDim.y);\
    ix = idx * pre_thread_num;\
    B_m_index = ix / n;\
    B_n_index = ix % n;\
    ix = B_m_index * (n + padding) + B_n_index;\
    int d_B_index = (K_tile_index * k + B_m_index) * N + N_tile_index * n + B_n_index;\
    FLOAT4(B_sh[ix]) = FLOAT4(d_B[d_B_index]);
__global__ void gemm_kernel8(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m , int n, int k) {
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
    int idx = threadIdx.x + threadIdx.y * blockDim.x;
    const int N_tile_index = blockIdx.x; // tile的列号
    const int M_tile_index = blockIdx.y; // tile的行号
    const int warp_id = idx / WAVE_SIZE;
    const int warp_id_m = warp_id / 2; // 2 * 2 的格子
    const int warp_id_n = warp_id % 2;
    const int lane_id = idx % WAVE_SIZE;
    const int lane_id_m = lane_id / 8; // 8 * 8 的块
    const int lane_id_n = lane_id % 8;
    const int C_m_index = warp_id_m * 8 + lane_id_m; // tile内的4 * 4行号
    const int C_n_index = warp_id_n * 8 + lane_id_n; // tile内的4 * 4列号
    int A_m_index;
    int A_n_index;
    int B_m_index;
    int B_n_index;
    int pre_thread_num;
    int ix;
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[TM][TK];
    float reg_B[TK][TN];
    float reg_C[TM][TN] = {0.0f};
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        prefetch8();
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
float test8 () {
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
    dim3 thread((m + TM - 1) / TM, (n + TN - 1) / TN);
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding));
    gemm_kernel8<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel8<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}

#define prefetch8_1()\
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
__global__ void gemm_kernel8_1(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m , int n, int k) {
    int padding = 4; 
    // 16 * 16 = 256
    // m * k = 1024 一个线程读 4 个
    // m = 64;
    // n = 64;
    // k = 16;
    const int TM = 4;
    const int TN = 4;
    const int TK = 2;
    extern __shared__ __align__(16 * 1024) float sh[];
    float *A_sh = sh;
    float *B_sh = sh + m * (k + padding);
    const int N_tile_index = blockIdx.x; // tile的列号
    const int M_tile_index = blockIdx.y; // tile的行号
    const int warp_id = threadIdx.x / WAVE_SIZE;
    const int warp_id_m = warp_id / 2; // 2 * 2 的格子
    const int warp_id_n = warp_id % 2;
    const int lane_id = threadIdx.x % WAVE_SIZE;
    const int lane_id_m = lane_id / 8; // 8 * 8 的块
    const int lane_id_n = lane_id % 8;
    const int C_m_index = warp_id_m * 8 + lane_id_m; // tile内的4 * 4行号
    const int C_n_index = warp_id_n * 8 + lane_id_n; // tile内的4 * 4列号
    int A_m_index;
    int A_n_index;
    int B_m_index;
    int B_n_index;
    int pre_thread_num;
    int ix;
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[TM][TK];
    float reg_B[TK][TN];
    float reg_C[TM][TN] = {0.0f};
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        prefetch8_1();
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
float test8_1 () {
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
    gemm_kernel8_1<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel8_1<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}
#define prefetch8_2()\
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
__global__ void gemm_kernel8_2(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m , int n, int k) {
    int padding = 4; 
    // 16 * 16 = 256
    // m * k = 1024 一个线程读 4 个
    // m = 64;
    // n = 64;
    // k = 16;
    const int TM = 4;
    const int TN = 4;
    const int TK = 2;
    extern __shared__ __align__(16 * 1024) float sh[];
    float *A_sh = sh;
    float *B_sh = sh + m * (k + padding);
    const int N_tile_index = blockIdx.x; // tile的列号
    const int M_tile_index = blockIdx.y; // tile的行号
    const int warp_id = threadIdx.x / WAVE_SIZE;
    const int warp_id_m = warp_id / 1; // 4 * 1 的格子
    const int warp_id_n = warp_id % 1;
    const int lane_id = threadIdx.x % WAVE_SIZE;
    const int lane_id_m = lane_id / 16; // 4 * 16 的块
    const int lane_id_n = lane_id % 16;
    const int C_m_index = warp_id_m * 4 + lane_id_m; // tile内的4 * 4行号
    const int C_n_index = warp_id_n * 16 + lane_id_n; // tile内的4 * 4列号
    int A_m_index;
    int A_n_index;
    int B_m_index;
    int B_n_index;
    int pre_thread_num;
    int ix;
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[TM][TK];
    float reg_B[TK][TN];
    float reg_C[TM][TN] = {0.0f};
    float mid_B[TN];
    float mid_A;
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        prefetch8_2();
	    __syncthreads();
        for (int k_reg_index = 0; k_reg_index < k; k_reg_index+= TK) {
            for (int k_index = 0; k_index < TK; k_index++) {
                int B_index = k_reg_index * (n + padding) + C_n_index * TN + k_index * (n + padding);
                FLOAT4(mid_B[0]) = FLOAT4(B_sh[B_index]);
                for (int i = 0; i < TM; i++) {
                    int A_index = C_m_index * TM * (k + padding) + k_reg_index +  i * (k + padding);
                    mid_A = A_sh[A_index + k_index];
                    for (int j = 0; j < TN; j++) {
                        reg_C[i][j] += mid_A * mid_B[j];
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
float test8_2 () {
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
    gemm_kernel8_2<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel8_2<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}

#define prefetch8_3()\
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
    ix = B_m_index + B_n_index * (k + padding);\
    int d_B_index = (K_tile_index * k + B_m_index) * N + (N_tile_index * n + B_n_index);\
    mid_B = FLOAT4(d_B[d_B_index]);\
    B_sh[ix] = mid_B.x;\
    B_sh[ix + (k + padding)] = mid_B.y;\
    B_sh[ix + 2 * (k + padding)] = mid_B.z;\
    B_sh[ix + 3 * (k + padding)] = mid_B.w;

__global__ void gemm_kernel8_3(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m , int n, int k) {
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
    const int N_tile_index = blockIdx.x; // tile的列号
    const int M_tile_index = blockIdx.y; // tile的行号
    const int warp_id = threadIdx.x / WAVE_SIZE;
    const int warp_id_m = warp_id / 2; // 2 * 2 的格子
    const int warp_id_n = warp_id % 2;
    const int lane_id = threadIdx.x % WAVE_SIZE;
    const int lane_id_m = lane_id / 8; // 8 * 8 的块
    const int lane_id_n = lane_id % 8;
    const int C_m_index = warp_id_m * 8 + lane_id_m; // tile内的4 * 4行号
    const int C_n_index = warp_id_n * 8 + lane_id_n; // tile内的4 * 4列号
    int A_m_index;
    int A_n_index;
    int B_m_index;
    int B_n_index;
    int pre_thread_num;
    int ix;
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float4 mid_B;
    float reg_A[TM][TK];
    float reg_B[TN][TK];
    float reg_C[TM][TN] = {0.0f};
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        prefetch8_3();
	__syncthreads();
        for (int k_reg_index = 0; k_reg_index < k; k_reg_index+= TK) {
            for (int i = 0; i < TM; i++) {
                int A_index = C_m_index * TM * (k + padding) + k_reg_index +  i * (k + padding);
                // int A_index = (C_m_index * TM + i) + k_reg_index * (m + padding);
                // FLOAT4(reg_A[i][0]) = FLOAT4(A_sh[A_index]);
                reg_A[i][0] = A_sh[A_index];
                reg_A[i][1] = A_sh[A_index + 1];
            }
            for (int i = 0; i < TN; i++) {
                // int B_index = k_reg_index * (n + padding) + C_n_index * TN + i * (n + padding);
                int B_index = k_reg_index + (C_n_index * TN + i) * (k + padding);
                // FLOAT4(reg_B[i][0]) = FLOAT4(B_sh[B_index]);
		reg_B[i][0] = B_sh[B_index];
		reg_B[i][1] = B_sh[B_index + 1];
            }
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    for (int k_index = 0; k_index < TK; k_index++) {
                        reg_C[i][j] += reg_A[i][k_index] * reg_B[j][k_index];
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
float test8_3 () {
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
    int shared_size = sizeof(float) * (m * (k + padding) + (k + padding) * n);
    gemm_kernel8_3<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel8_3<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}
#define prefetch8_4()\
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
__global__ void gemm_kernel8_4(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m , int n, int k) {
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
    const int N_tile_index = blockIdx.x; // tile的列号
    const int M_tile_index = blockIdx.y; // tile的行号
    const int warp_id = threadIdx.x / WAVE_SIZE;
    const int warp_id_m = warp_id / 1; // 4 * 1 的格子
    const int warp_id_n = warp_id % 1;
    const int lane_id = threadIdx.x % WAVE_SIZE;
    const int lane_id_m = lane_id / 16; // 4 * 16 的块
    const int lane_id_n = lane_id % 16;
    const int C_m_index = warp_id_m * 4 + lane_id_m; // tile内的4 * 4行号
    const int C_n_index = warp_id_n * 16 + lane_id_n; // tile内的4 * 4列号
    int A_m_index;
    int A_n_index;
    int B_m_index;
    int B_n_index;
    int pre_thread_num;
    int ix;
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[TM][TK];
    float reg_B[TK][TN];
    float reg_C[TM][TN] = {0.0f};
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        prefetch8_4();
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
float test8_4 () {
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
    gemm_kernel8_4<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel8_4<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}
#define prefetch8_5()\
    pre_thread_num = (m * k + blockDim.x * blockDim.y - 1)/ (blockDim.x * blockDim.y);\
    ix = idx * pre_thread_num;\
    A_m_index = ix / k;\
    A_n_index = ix % k;\
    int d_A_index = (M_tile_index * m + A_m_index) * K + K_tile_index * k + A_n_index;\
    ix = A_m_index * (k + padding) + A_n_index;\
    FLOAT4(A_sh[ix]) = FLOAT4(d_A[d_A_index]);\
    pre_thread_num = (k * n + blockDim.x * blockDim.y - 1) / (blockDim.x * blockDim.y);\
    ix = idx * pre_thread_num;\
    B_m_index = ix / n;\
    B_n_index = ix % n;\
    ix = B_m_index * (n + padding) + B_n_index;\
    int d_B_index = (K_tile_index * k + B_m_index) * N + N_tile_index * n + B_n_index;\
    FLOAT4(B_sh[ix]) = FLOAT4(d_B[d_B_index]);
__global__ void gemm_kernel8_5(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m , int n, int k) {
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
    const int N_tile_index = blockIdx.x; // tile的列号
    const int M_tile_index = blockIdx.y; // tile的行号
    int idx = threadIdx.x + blockDim.x * threadIdx.y;
    const int warp_id = idx / WAVE_SIZE;
    const int warp_id_m = warp_id / 1; // 4 * 1 的格子
    const int warp_id_n = warp_id % 1;
    const int lane_id = idx % WAVE_SIZE;
    const int lane_id_m = lane_id / 32 * 2 + lane_id % 2; // 4 * 16 的块
    const int lane_id_n = lane_id % 32 / 2;
    const int C_m_index = warp_id_m * 4 + lane_id_m; // tile内的4 * 4行号
    const int C_n_index = warp_id_n * 16 + lane_id_n; // tile内的4 * 4列号
    int A_m_index;
    int A_n_index;
    int B_m_index;
    int B_n_index;
    int pre_thread_num;
    int ix;
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[TM][TK];
    float reg_B[TK][TN];
    float reg_C[TM][TN] = {0.0f};
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        prefetch8_5();
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
float test8_5 () {
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
    dim3 thread((m + TM - 1) / TM, (n + TN - 1) / TN);
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding));
    gemm_kernel8_5<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel8_5<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}
#define prefetch8_6()\
    pre_thread_num = (m * k + blockDim.x * blockDim.y - 1)/ (blockDim.x * blockDim.y);\
    for (int i = 0; i < pre_thread_num; i++) {\
        int ix = idx * pre_thread_num  + i;\
        int n_index = ix % k;\
        int m_index = ix / k;\
        ix = m_index * (k + padding) + n_index;\
        int d_A_index = (M_tile_index * m + m_index) * K + K_tile_index * k + n_index;\
        if (d_A_index < M * K) {\
            A_sh[ix] = d_A[d_A_index];\
        } else {\
            A_sh[ix] = 0;\
        }\
    }\
    pre_thread_num = (k * n + blockDim.x * blockDim.y - 1) / (blockDim.x * blockDim.y);\
    for (int i = 0; i < pre_thread_num; i++) {\
        int ix = idx * pre_thread_num + i;\
        int n_index = ix % n;\
        int m_index = ix / n;\
        ix = m_index * (n + padding) + n_index;\
        int d_B_index = (K_tile_index * k + m_index) * N + N_tile_index * n + n_index;\
        if (d_B_index < K * N) {\
            B_sh[ix] = d_B[d_B_index];\
        } else {\
            B_sh[ix] = 0;\
        }\
    }

__global__ void gemm_kernel8_6(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m , int n, int k) {
    int padding = 4; 
    // 16 * 16 = 256
    // m * k = 1024 一个线程读 4 个
    // m = 16;
    // n = 16;
    // k = 64;
    const int TM = 4;
    const int TN = 4;
    const int TK = 2;
    extern __shared__ float sh[];
    float *A_sh = sh;
    float *B_sh = sh + m * (k + padding);
    const int N_tile_index = blockIdx.x; // tile的列号
    const int M_tile_index = blockIdx.y; // tile的行号
    int idx = threadIdx.x + blockDim.x * threadIdx.y;
    const int warp_id = idx / WAVE_SIZE;
    const int warp_id_m = warp_id / 1; // 2 * 2 的格子
    const int warp_id_n = warp_id % 1;
    const int lane_id = idx % WAVE_SIZE;
    const int lane_id_m = lane_id / 32 * 2 + lane_id % 2; // 4 * 16 的块
    const int lane_id_n = lane_id % 32 / 2;
    const int C_m_index = warp_id_m * 4 + lane_id_m; // tile内的4 * 4行号
    const int C_n_index = warp_id_n * 16 + lane_id_n; // tile内的4 * 4列号
    // const int C_m_index = idx  % ((n + TN - 1) / TN);
    // const int C_n_index = idx  / ((n + TN - 1) / TN);
    int A_m_index;
    int A_n_index;
    int B_m_index;
    int B_n_index;
    int pre_thread_num;
    int ix;
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[TM][TK];
    float reg_B[TK][TN];
    float reg_C[TM][TN] = {0.0f};
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        prefetch8_6();
	__syncthreads();
        for (int k_reg_index = 0; k_reg_index < k; k_reg_index+= TK) {
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TK; j++) {
                    int A_index = C_m_index * TM * (k + padding) + k_reg_index +  i * (k + padding) + j;
                    reg_A[i][j] = A_sh[A_index];
                }
            }
            for (int i = 0; i < TK; i++) {
                for (int j = 0; j < TN; j++) {
                    int B_index = k_reg_index * (n + padding) + C_n_index * TN + i * (n + padding) + j;    
                    reg_B[i][j] = B_sh[B_index];
                }
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
       for (int j = 0; j < TN; j++) {
           int C_index = (M_tile_index * m + C_m_index * TM) * N + N_tile_index * n + C_n_index * TN + i * N + j;
           d_C[C_index] = reg_C[i][j];
        }
    }
}
float test8_6 () {
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
    dim3 thread((m + TM - 1) / TM,  (n + TN - 1) / TN);
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding));
    gemm_kernel8_6<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel8_6<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}