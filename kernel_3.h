__global__ void gemm_kernel3(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m, int n, int k) {
    const int reg_size = 4;
    extern __shared__ float sh[];
    float *A_sh = sh;
    float *B_sh = sh + m * k;
    int N_tile_index = blockIdx.x; // tile的列号
    int M_tile_index = blockIdx.y; // tile的行号
    int n_index = threadIdx.x % ((n + reg_size - 1) / reg_size); // tile内的4 * 4列号
    int m_index = threadIdx.x / ((n + reg_size - 1) / reg_size); // tile内的4 * 4行号
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    // float reg_A[reg_size][reg_size];
    // float reg_B[reg_size][reg_size];
    float reg_C[reg_size][reg_size] = {0.0f};
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        int pre_thread_num = (m * k + blockDim.x - 1)/ blockDim.x;
	for (int i = 0; i < pre_thread_num; i++) {
           int ix = threadIdx.x * pre_thread_num  + i;
           int n_index = ix % k;
	   int m_index = ix / k;
	   if ((M_tile_index * m + m_index) * K + K_tile_index * k + n_index < M * K) {
		A_sh[ix] = d_A[(M_tile_index * m + m_index) * K + K_tile_index * k + n_index];
	   } else {
		A_sh[ix] = 0;
	   }
	}
	pre_thread_num = (k * n + blockDim.x - 1) / blockDim.x;
	for (int i = 0; i < pre_thread_num; i++) {
	   int ix = threadIdx.x * pre_thread_num + i;
	   int n_index = ix % n;
	   int m_index = ix / n;
	   if ((K_tile_index * k + m_index) * N + N_tile_index * n + n_index < K * N) {
	        B_sh[ix] = d_B[(K_tile_index * k + m_index) * N + N_tile_index * n + n_index];
	   } else {
		B_sh[ix] = 0;
           }
	}
        __syncthreads();
        for (int k_reg_index = 0; k_reg_index < k; k_reg_index+= reg_size) {
            for (int i = 0; i < reg_size; i++) {
                for (int j = 0; j < reg_size; j++) {
                    for (int k_index = 0; k_index < reg_size; k_index++) {
                        reg_C[i][j] += A_sh[m_index * reg_size * k + k_reg_index +  i * k + k_index] * B_sh[k_reg_index * n + n_index * reg_size + k_index * n + j];
                    }
                }
            }
        }
         __syncthreads();
    }
    for (int i = 0; i < reg_size; i++) {
        for (int j = 0; j < reg_size; j++) {
            int C_index = (M_tile_index * m + m_index * reg_size) * N + N_tile_index * n + n_index * reg_size + i * N + j;
            if (C_index < M * N) {
                // printf("C_index: %d \n", C_index);
                d_C[C_index] = reg_C[i][j];
            }
        }
    }
}
float test3 () {
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int reg_size = 4;
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((thread_size + reg_size * reg_size - 1) / (reg_size * reg_size));
    // printf("block: %d, thread: %d \n", block.x, thread.x);
    int shared_size = sizeof(float) * (m * k + k * n);
    gemm_kernel3<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel3<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}

__global__ void gemm_kernel3_1(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m, int n, int k) {
    const int reg_size = 4;
    extern __shared__ float sh[];
    float *A_sh = sh;
    float *B_sh = sh + m * k;
    int N_tile_index = blockIdx.x; // tile的列号
    int M_tile_index = blockIdx.y; // tile的行号
    int n_index = threadIdx.x % ((n + reg_size - 1) / reg_size); // tile内的4 * 4列号
    int m_index = threadIdx.x / ((n + reg_size - 1) / reg_size); // tile内的4 * 4行号
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[reg_size];
    float reg_B[reg_size];
    float reg_C[reg_size][reg_size] = {0.0f};
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        int pre_thread_num = (m * k + blockDim.x - 1)/ blockDim.x;
        for (int i = 0; i < pre_thread_num; i++) {
            int ix = threadIdx.x * pre_thread_num  + i;
            int n_index = ix % k;
            int m_index = ix / k;
            if ((M_tile_index * m + m_index) * K + K_tile_index * k + n_index < M * K) {
                A_sh[ix] = d_A[(M_tile_index * m + m_index) * K + K_tile_index * k + n_index];
            } else {
                A_sh[ix] = 0;
            }
        }
        pre_thread_num = (k * n + blockDim.x - 1) / blockDim.x;
        for (int i = 0; i < pre_thread_num; i++) {
            int ix = threadIdx.x * pre_thread_num + i;
            int n_index = ix % n;
            int m_index = ix / n;
            if ((K_tile_index * k + m_index) * N + N_tile_index * n + n_index < K * N) {
                B_sh[ix] = d_B[(K_tile_index * k + m_index) * N + N_tile_index * n + n_index];
            } else {
                B_sh[ix] = 0;
            }
        }
        __syncthreads();
        for (int k_reg_index = 0; k_reg_index < k; k_reg_index++) {
            for (int i = 0; i < reg_size; i++) {
                reg_A[i] = A_sh[m_index * reg_size * k + k_reg_index + i * k];
                reg_B[i] = B_sh[k_reg_index * n + n_index * reg_size + i];
            }
            for (int i = 0; i < reg_size; i++) {
                for (int j = 0; j < reg_size; j++) {
                    reg_C[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
         __syncthreads();
    }
    for (int i = 0; i < reg_size; i++) {
        for (int j = 0; j < reg_size; j++) {
            int C_index = (M_tile_index * m + m_index * reg_size) * N + N_tile_index * n + n_index * reg_size + i * N + j;
            if (C_index < M * N) {
                // printf("C_index: %d \n", C_index);
                d_C[C_index] = reg_C[i][j];
            }
        }
    }
}
float test3_1 () {
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int reg_size = 4;
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((thread_size + reg_size * reg_size - 1) / (reg_size * reg_size));
    // printf("block: %d, thread: %d \n", block.x, thread.x);
    int shared_size = sizeof(float) * (m * k + k * n);
    gemm_kernel3<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel3<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}
__global__ void gemm_kernel3_2(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m, int n, int k) {
    const int reg_size = 8;
    extern __shared__ float sh[];
    float *A_sh = sh;
    float *B_sh = sh + m * k;
    int N_tile_index = blockIdx.x; // tile的列号
    int M_tile_index = blockIdx.y; // tile的行号
    int n_index = threadIdx.x % ((n + reg_size - 1) / reg_size); // tile内的4 * 4列号
    int m_index = threadIdx.x / ((n + reg_size - 1) / reg_size); // tile内的4 * 4行号
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    // float reg_A[reg_size];
    // float reg_B[reg_size];
    float reg_C[reg_size][reg_size] = {0.0f};
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        int pre_thread_num = (m * k + blockDim.x - 1)/ blockDim.x;
        for (int i = 0; i < pre_thread_num; i++) {
            int ix = threadIdx.x * pre_thread_num  + i;
            int n_index = ix % k;
            int m_index = ix / k;
            if ((M_tile_index * m + m_index) * K + K_tile_index * k + n_index < M * K) {
                A_sh[ix] = d_A[(M_tile_index * m + m_index) * K + K_tile_index * k + n_index];
            } else {
                A_sh[ix] = 0;
            }
        }
        pre_thread_num = (k * n + blockDim.x - 1) / blockDim.x;
        for (int i = 0; i < pre_thread_num; i++) {
            int ix = threadIdx.x * pre_thread_num + i;
            int n_index = ix % n;
            int m_index = ix / n;
            if ((K_tile_index * k + m_index) * N + N_tile_index * n + n_index < K * N) {
                B_sh[ix] = d_B[(K_tile_index * k + m_index) * N + N_tile_index * n + n_index];
            } else {
                B_sh[ix] = 0;
            }
        }
        __syncthreads();

        for (int k_reg_index = 0; k_reg_index < k; k_reg_index++) {
            for (int i = 0; i < reg_size; i++) {
                for (int j = 0; j < reg_size; j++) {
                        reg_C[i][j] += A_sh[m_index * reg_size * k + k_reg_index +  i * k] * B_sh[k_reg_index * n + n_index * reg_size + j];
                }
            }
        }
         __syncthreads();
    }
    for (int i = 0; i < reg_size; i++) {
        for (int j = 0; j < reg_size; j++) {
            int C_index = (M_tile_index * m + m_index * reg_size) * N + N_tile_index * n + n_index * reg_size + i * N + j;
            if (C_index < M * N) {
                // printf("C_index: %d \n", C_index);
                d_C[C_index] = reg_C[i][j];
            }
        }
    }
}
float test3_2 () {
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    const int m = 128;
    const int n = 128;
    const int k = 8;
    const int reg_size = 8;
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((thread_size + reg_size * reg_size - 1) / (reg_size * reg_size));
    // printf("block: %d, thread: %d \n", block.x, thread.x);
    int shared_size = sizeof(float) * (m * k + k * n);
    gemm_kernel3_2<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel3_2<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}
__global__ void gemm_kernel3_3(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m, int n, int k) {
    const int reg_size = 8;
    extern __shared__ float sh[];
    float *A_sh = sh;
    float *B_sh = sh + m * k;
    int N_tile_index = blockIdx.x; // tile的列号
    int M_tile_index = blockIdx.y; // tile的行号
    int n_index = threadIdx.x % ((n + reg_size - 1) / reg_size); // tile内的4 * 4列号
    int m_index = threadIdx.x / ((n + reg_size - 1) / reg_size); // tile内的4 * 4行号
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[reg_size];
    float reg_B[reg_size];
    float reg_C[reg_size][reg_size] = {0.0f};
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        int pre_thread_num = (m * k + blockDim.x - 1)/ blockDim.x;
        for (int i = 0; i < pre_thread_num; i++) {
            int ix = threadIdx.x * pre_thread_num  + i;
            int n_index = ix % k;
            int m_index = ix / k;
            if ((M_tile_index * m + m_index) * K + K_tile_index * k + n_index < M * K) {
                A_sh[ix] = d_A[(M_tile_index * m + m_index) * K + K_tile_index * k + n_index];
            } else {
                A_sh[ix] = 0;
            }
        }
        pre_thread_num = (k * n + blockDim.x - 1) / blockDim.x;
        for (int i = 0; i < pre_thread_num; i++) {
            int ix = threadIdx.x * pre_thread_num + i;
            int n_index = ix % n;
            int m_index = ix / n;
            if ((K_tile_index * k + m_index) * N + N_tile_index * n + n_index < K * N) {
                B_sh[ix] = d_B[(K_tile_index * k + m_index) * N + N_tile_index * n + n_index];
            } else {
                B_sh[ix] = 0;
            }
        }
        __syncthreads();

        for (int k_reg_index = 0; k_reg_index < k; k_reg_index++) {
            for (int i = 0; i < reg_size; i++) {
                reg_A[i] = A_sh[m_index * reg_size * k + k_reg_index +  i * k];
                reg_B[i] = B_sh[k_reg_index * n + n_index * reg_size + i];
            }
            for (int i = 0; i < reg_size; i++) {
                for (int j = 0; j < reg_size; j++) {
                    reg_C[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
         __syncthreads();
    }
    for (int i = 0; i < reg_size; i++) {
        for (int j = 0; j < reg_size; j++) {
            int C_index = (M_tile_index * m + m_index * reg_size) * N + N_tile_index * n + n_index * reg_size + i * N + j;
            if (C_index < M * N) {
                // printf("C_index: %d \n", C_index);
                d_C[C_index] = reg_C[i][j];
            }
        }
    }
}
float test3_3 () {
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    const int m = 128;
    const int n = 128;
    const int k = 8;
    const int reg_size = 8;
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((thread_size + reg_size * reg_size - 1) / (reg_size * reg_size));
    // printf("block: %d, thread: %d \n", block.x, thread.x);
    int shared_size = sizeof(float) * (m * k + k * n);
    gemm_kernel3_3<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel3_3<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}

__global__ void gemm_kernel3_4(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m, int n, int k) {
    const int TM = 8;
    const int TN = 8;
    // const int TK = 4;
    const int padding = 4;
    extern __shared__ float sh[];
    float *A_sh = sh;
    float *B_sh = sh + (m + padding) * k;
    int N_tile_index = blockIdx.x; // tile的列号
    int M_tile_index = blockIdx.y; // tile的行号
    int n_index = threadIdx.x % ((n + TN - 1) / TN); // tile内的4 * 4列号
    int m_index = threadIdx.x / ((n + TN - 1) / TN); // tile内的4 * 4行号
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[TM];
    float reg_B[TN];
    float reg_C[TM][TN] = {0.0f};
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        int pre_thread_num = (m * k + blockDim.x - 1)/ blockDim.x;
        for (int i = 0; i < pre_thread_num; i++) {
            int ix = threadIdx.x * pre_thread_num  + i;
            int n_index = ix % k;
            int m_index = ix / k;
            // ix = m_index * (k + padding) + n_index;
            ix = n_index * (m + padding) + m_index; // 转置
            int d_A_index = (M_tile_index * m + m_index) * K + K_tile_index * k + n_index;
            if (d_A_index < M * K) {
                A_sh[ix] = d_A[d_A_index];
            } else {
                A_sh[ix] = 0;
            }
        }
        pre_thread_num = (k * n + blockDim.x - 1) / blockDim.x;
        for (int i = 0; i < pre_thread_num; i++) {
            int ix = threadIdx.x * pre_thread_num + i;
            int n_index = ix % n;
            int m_index = ix / n;
            ix = m_index * (n + padding) + n_index;
            int d_B_index = (K_tile_index * k + m_index) * N + N_tile_index * n + n_index;
            if (d_B_index < K * N) {
                B_sh[ix] = d_B[d_B_index];
            } else {
                B_sh[ix] = 0;
            }
        }
        __syncthreads();

        for (int k_reg_index = 0; k_reg_index < k; k_reg_index++) {
            for (int i = 0; i < TM; i++) {
                // reg_A[i] = A_sh[m_index * TM * k + k_reg_index +  i * k];
                reg_A[i] = A_sh[(m_index * TM + i) + k_reg_index * (m + padding)];
            }
            for (int i = 0; i < TN; i++) {
                reg_B[i] = B_sh[k_reg_index * (n + padding) + n_index * TN + i];
            }
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    reg_C[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
         __syncthreads();
    }
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int C_index = (M_tile_index * m + m_index * TM) * N + N_tile_index * n + n_index * TN + i * N + j;
            if (C_index < M * N) {
                // printf("C_index: %d \n", C_index);
                d_C[C_index] = reg_C[i][j];
            }
        }
    }
}
float test3_4 () {
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    const int m = 128;
    const int n = 128;
    const int k = 8;
    const int TM = 8;
    const int TN = 8;
    const int TK = 8;
    const int padding = 4;
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((thread_size + TM * TN - 1) / (TM * TN));
    // printf("block: %d, thread: %d \n", block.x, thread.x);
    int shared_size = sizeof(float) * ((m + padding) * k + k * (n + padding));
    gemm_kernel3_4<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel3_4<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}

__global__ void gemm_kernel3_5(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m, int n, int k) {
    const int TM = 8;
    const int TN = 8;
    // const int TK = 4;
    const int padding = 4;
    extern __shared__ float sh[];
    float *A_sh = sh;
    float *B_sh = sh + (m + padding) * k;
    int N_tile_index = blockIdx.x; // tile的列号
    int M_tile_index = blockIdx.y; // tile的行号
    int warp_id = threadIdx.x / WAVE_SIZE; // 4
    int warp_id_m = warp_id / 2; // 2 * 2
    int warp_id_n = warp_id % 2;
    int lane_id = threadIdx.x % WAVE_SIZE;
    int lane_id_m = lane_id / 8; // 8 * 8
    int lane_id_n = lane_id % 8;
    // int n_index = threadIdx.x % ((n + TN - 1) / TN); // tile内的4 * 4列号
    // int m_index = threadIdx.x / ((n + TN - 1) / TN); // tile内的4 * 4行号
    int m_index = warp_id_m * 8 + lane_id_m;
    int n_index = warp_id_n * 8 + lane_id_n;
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[TM];
    float reg_B[TN];
    float reg_C[TM][TN] = {0.0f};
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        int pre_thread_num = (m * k + blockDim.x - 1)/ blockDim.x;
        for (int i = 0; i < pre_thread_num; i++) {
            int ix = threadIdx.x * pre_thread_num  + i;
            int n_index = ix % k;
            int m_index = ix / k;
            // ix = m_index * (k + padding) + n_index;
            ix = n_index * (m + padding) + m_index; // 转置
            int d_A_index = (M_tile_index * m + m_index) * K + K_tile_index * k + n_index;
            if (d_A_index < M * K) {
                A_sh[ix] = d_A[d_A_index];
            } else {
                A_sh[ix] = 0;
            }
        }
        pre_thread_num = (k * n + blockDim.x - 1) / blockDim.x;
        for (int i = 0; i < pre_thread_num; i++) {
            int ix = threadIdx.x * pre_thread_num + i;
            int n_index = ix % n;
            int m_index = ix / n;
            ix = m_index * (n + padding) + n_index;
            int d_B_index = (K_tile_index * k + m_index) * N + N_tile_index * n + n_index;
            if (d_B_index < K * N) {
                B_sh[ix] = d_B[d_B_index];
            } else {
                B_sh[ix] = 0;
            }
        }
        __syncthreads();

        for (int k_reg_index = 0; k_reg_index < k; k_reg_index++) {
            for (int i = 0; i < TM; i++) {
                // reg_A[i] = A_sh[m_index * TM * k + k_reg_index +  i * k];
                reg_A[i] = A_sh[(m_index * TM + i) + k_reg_index * (m + padding)];
            }
            for (int i = 0; i < TN; i++) {
                reg_B[i] = B_sh[k_reg_index * (n + padding) + n_index * TN + i];
            }
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    reg_C[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
         __syncthreads();
    }
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int C_index = (M_tile_index * m + m_index * TM) * N + N_tile_index * n + n_index * TN + i * N + j;
            if (C_index < M * N) {
                // printf("C_index: %d \n", C_index);
                d_C[C_index] = reg_C[i][j];
            }
        }
    }
}
float test3_5 () {
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    const int m = 128;
    const int n = 128;
    const int k = 8;
    const int TM = 8;
    const int TN = 8;
    const int TK = 8;
    const int padding = 4;
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((thread_size + TM * TN - 1) / (TM * TN));
    // printf("block: %d, thread: %d \n", block.x, thread.x);
    int shared_size = sizeof(float) * ((m + padding) * k + k * (n + padding));
    gemm_kernel3_5<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel3_5<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}

__global__ void gemm_kernel3_6(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m, int n, int k) {
    const int TM = 8;
    const int TN = 8;
    const int TK = 4;
    const int padding = 4;
    extern __shared__ float sh[];
    float *A_sh = sh;
    float *B_sh = sh + (m + padding) * k;
    int N_tile_index = blockIdx.x; // tile的列号
    int M_tile_index = blockIdx.y; // tile的行号
    int warp_id = threadIdx.x / WAVE_SIZE; // 4
    int warp_id_m = warp_id / 2; // 2 * 2
    int warp_id_n = warp_id % 2;
    int lane_id = threadIdx.x % WAVE_SIZE;
    int lane_id_m = lane_id / 8; // 8 * 8
    int lane_id_n = lane_id % 8;
    // int n_index = threadIdx.x % ((n + TN - 1) / TN); // tile内的4 * 4列号
    // int m_index = threadIdx.x / ((n + TN - 1) / TN); // tile内的4 * 4行号
    int m_index = warp_id_m * 8 + lane_id_m;
    int n_index = warp_id_n * 8 + lane_id_n;
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[TM][TK];
    float reg_B[TN][TK];
    float reg_C[TM][TN] = {0.0f};
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        int pre_thread_num = (m * k + blockDim.x - 1)/ blockDim.x;
        for (int i = 0; i < pre_thread_num; i++) {
            int ix = threadIdx.x * pre_thread_num  + i;
            int n_index = ix % k;
            int m_index = ix / k;
            // ix = m_index * (k + padding) + n_index;
            ix = n_index * (m + padding) + m_index; // 转置
            int d_A_index = (M_tile_index * m + m_index) * K + K_tile_index * k + n_index;
            if (d_A_index < M * K) {
                A_sh[ix] = d_A[d_A_index];
            } else {
                A_sh[ix] = 0;
            }
        }
        pre_thread_num = (k * n + blockDim.x - 1) / blockDim.x;
        for (int i = 0; i < pre_thread_num; i++) {
            int ix = threadIdx.x * pre_thread_num + i;
            int n_index = ix % n;
            int m_index = ix / n;
            ix = m_index * (n + padding) + n_index;
            int d_B_index = (K_tile_index * k + m_index) * N + N_tile_index * n + n_index;
            if (d_B_index < K * N) {
                B_sh[ix] = d_B[d_B_index];
            } else {
                B_sh[ix] = 0;
            }
        }
        __syncthreads();

        for (int k_reg_index = 0; k_reg_index < k; k_reg_index += TK) {
            for (int i = 0; i < TM; i++) {
                for (int k_index = 0; k_index < TK; k_index++) {
                    reg_A[i][k_index] = A_sh[(m_index * TM + i) + (k_reg_index + k_index) * (m + padding)];
                }
                // reg_A[i] = A_sh[m_index * TM * k + k_reg_index +  i * k];
            }
            for (int i = 0; i < TN; i++) {
                for (int k_index = 0; k_index < TK; k_index++) {
                    reg_B[i][k_index] = B_sh[(k_reg_index + k_index) * (n + padding) + n_index * TN + i];
                }
            }
            for (int k_index = 0; k_index < TK; k_index++) {
                for (int i = 0; i < TM; i++) {
                    for (int j = 0; j < TN; j++) {
                        reg_C[i][j] += reg_A[i][k_index] * reg_B[j][k_index];
                    }
                }
            }
        }
         __syncthreads();
    }
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int C_index = (M_tile_index * m + m_index * TM) * N + N_tile_index * n + n_index * TN + i * N + j;
            if (C_index < M * N) {
                // printf("C_index: %d \n", C_index);
                d_C[C_index] = reg_C[i][j];
            }
        }
    }
}
float test3_6 () {
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    const int m = 128;
    const int n = 128;
    const int k = 16;
    const int TM = 8;
    const int TN = 8;
    const int padding = 4;
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((thread_size + TM * TN - 1) / (TM * TN));
    // printf("block: %d, thread: %d \n", block.x, thread.x);
    int shared_size = sizeof(float) * ((m + padding) * k + k * (n + padding));
    gemm_kernel3_6<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel3_6<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}