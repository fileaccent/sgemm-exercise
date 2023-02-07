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
    gemm_kernel3_1<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel3_1<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}
__global__ void gemm_kernel3_2(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m, int n, int k) {
    const int reg_size = 8;
    const int padding = 4;
    extern __shared__ float sh[];
    float *A_sh = sh;
    float *B_sh = sh + m * (k + padding);
    const int N_tile_index = blockIdx.y; // tile的列号
    const int M_tile_index = blockIdx.x; // tile的行号
    const int idx = threadIdx.x;
    const int A_m_index = idx >> 1; // 128
    const int A_n_index = (idx & 1) << 2; // 8
    const int B_m_index = idx >> 5; // 8
    const int B_n_index = (idx & 31) << 2; // 128
    const int C_m_index = idx >> 4; // tile内的8 * 8行号
    const int C_n_index = idx & 15; // tile内的8 * 8列号
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A;
    float reg_B[reg_size];
    float reg_C[reg_size][reg_size] = {0.0f};
    const int d_A_index = (M_tile_index * m + A_m_index) * K + A_n_index;
    const int d_B_index = (B_m_index) * N + N_tile_index * n + B_n_index;
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        FLOAT4(A_sh[A_m_index * (k  + padding) + A_n_index]) = FLOAT4(d_A[d_A_index + K_tile_index * k]);
        FLOAT4(B_sh[B_m_index * (n + padding) + B_n_index]) = FLOAT4(d_B[d_B_index + K_tile_index * k * N]);
        __syncthreads();
        for (int k_reg_index = 0; k_reg_index < k; k_reg_index++) {
            FLOAT4(reg_B[0]) = FLOAT4(B_sh[k_reg_index * (n + padding) + C_n_index * reg_size]);
            FLOAT4(reg_B[4]) = FLOAT4(B_sh[k_reg_index * (n + padding) + C_n_index * reg_size + 4]);
            for (int i = 0; i < reg_size; i++) {
                reg_A = A_sh[C_m_index * reg_size * (k + padding) + k_reg_index +  i * (k + padding)];
                for (int j = 0; j < reg_size; j++) {
                        reg_C[i][j] += reg_B[j] * reg_A;
                }
            }
        }
         __syncthreads();
    }
    const int C_index = (M_tile_index * m + C_m_index * reg_size) * N + N_tile_index * n + C_n_index * reg_size;
    FLOAT4(d_C[C_index]) = FLOAT4(reg_C[0][0]);
    FLOAT4(d_C[C_index + 4]) = FLOAT4(reg_C[0][4]);
    FLOAT4(d_C[C_index + 1 * N]) = FLOAT4(reg_C[1][0]);
    FLOAT4(d_C[C_index + 1 * N + 4]) = FLOAT4(reg_C[1][4]);
    FLOAT4(d_C[C_index + 2 * N]) = FLOAT4(reg_C[2][0]);
    FLOAT4(d_C[C_index + 2 * N + 4]) = FLOAT4(reg_C[2][4]);
    FLOAT4(d_C[C_index + 3 * N]) = FLOAT4(reg_C[3][0]);
    FLOAT4(d_C[C_index + 3 * N + 4]) = FLOAT4(reg_C[3][4]);
    FLOAT4(d_C[C_index + 4 * N]) = FLOAT4(reg_C[4][0]);
    FLOAT4(d_C[C_index + 4 * N + 4]) = FLOAT4(reg_C[4][4]);
    FLOAT4(d_C[C_index + 5 * N]) = FLOAT4(reg_C[5][0]);
    FLOAT4(d_C[C_index + 5 * N + 4]) = FLOAT4(reg_C[5][4]);
    FLOAT4(d_C[C_index + 6 * N]) = FLOAT4(reg_C[6][0]);
    FLOAT4(d_C[C_index + 6 * N + 4]) = FLOAT4(reg_C[6][4]);
    FLOAT4(d_C[C_index + 7 * N]) = FLOAT4(reg_C[7][0]);
    FLOAT4(d_C[C_index + 7 * N + 4]) = FLOAT4(reg_C[7][4]);
}
float test3_2 () {
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    const int m = 128;
    const int n = 128;
    const int k = 8;
    const int reg_size = 8;
    const int padding = 4;
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((thread_size + reg_size * reg_size - 1) / (reg_size * reg_size));
    // printf("block: %d, thread: %d \n", block.x, thread.x);
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding));
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
    const int padding = 4;
    extern __shared__ float sh[];
    float *A_sh = sh;
    float *B_sh = sh + m * (k + padding);
    const int N_tile_index = blockIdx.y; // tile的列号
    const int M_tile_index = blockIdx.x; // tile的行号
    const int idx = threadIdx.x;
    const int A_m_index = idx >> 1; // 128
    const int A_n_index = (idx & 1) << 2; // 8
    const int B_m_index = idx >> 5; // 8
    const int B_n_index = (idx & 31) << 2; // 128
    const int C_m_index = idx >> 4; // tile内的8 * 8行号
    const int C_n_index = idx & 15; // tile内的8 * 8列号
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[reg_size];
    float reg_B[reg_size];
    float reg_C[reg_size][reg_size] = {0.0f};
    float4 readA;
    const int d_A_index = (M_tile_index * m + A_m_index) * K + A_n_index;
    const int d_B_index = (B_m_index) * N + N_tile_index * n + B_n_index;
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        readA = FLOAT4(d_A[d_A_index + K_tile_index * k]);
        A_sh[A_n_index * (m + padding) + A_m_index] = readA.x;
        A_sh[(A_n_index + 1) * (m + padding) + A_m_index] = readA.y;
        A_sh[(A_n_index + 2) * (m + padding) + A_m_index] = readA.z;
        A_sh[(A_n_index + 3) * (m + padding) + A_m_index] = readA.w;
        FLOAT4(B_sh[B_m_index * (n + padding) + B_n_index]) = FLOAT4(d_B[d_B_index + K_tile_index * k * N]);
        __syncthreads();
        for (int k_reg_index = 0; k_reg_index < k; k_reg_index++) {
            FLOAT4(reg_A[0]) = FLOAT4(A_sh[(C_m_index * reg_size) + k_reg_index * (m + padding)]);
            FLOAT4(reg_A[4]) = FLOAT4(A_sh[(C_m_index * reg_size + 4) + k_reg_index * (m + padding)]);
            FLOAT4(reg_B[0]) = FLOAT4(B_sh[k_reg_index * (n + padding) + C_n_index * reg_size]);
            FLOAT4(reg_B[4]) = FLOAT4(B_sh[k_reg_index * (n + padding) + C_n_index * reg_size + 4]);
            for (int i = 0; i < reg_size; i++) {
                for (int j = 0; j < reg_size; j++) {
                        reg_C[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
         __syncthreads();
    }
    const int C_index = (M_tile_index * m + C_m_index * reg_size) * N + N_tile_index * n + C_n_index * reg_size;
    FLOAT4(d_C[C_index]) = FLOAT4(reg_C[0][0]);
    FLOAT4(d_C[C_index + 4]) = FLOAT4(reg_C[0][4]);
    FLOAT4(d_C[C_index + 1 * N]) = FLOAT4(reg_C[1][0]);
    FLOAT4(d_C[C_index + 1 * N + 4]) = FLOAT4(reg_C[1][4]);
    FLOAT4(d_C[C_index + 2 * N]) = FLOAT4(reg_C[2][0]);
    FLOAT4(d_C[C_index + 2 * N + 4]) = FLOAT4(reg_C[2][4]);
    FLOAT4(d_C[C_index + 3 * N]) = FLOAT4(reg_C[3][0]);
    FLOAT4(d_C[C_index + 3 * N + 4]) = FLOAT4(reg_C[3][4]);
    FLOAT4(d_C[C_index + 4 * N]) = FLOAT4(reg_C[4][0]);
    FLOAT4(d_C[C_index + 4 * N + 4]) = FLOAT4(reg_C[4][4]);
    FLOAT4(d_C[C_index + 5 * N]) = FLOAT4(reg_C[5][0]);
    FLOAT4(d_C[C_index + 5 * N + 4]) = FLOAT4(reg_C[5][4]);
    FLOAT4(d_C[C_index + 6 * N]) = FLOAT4(reg_C[6][0]);
    FLOAT4(d_C[C_index + 6 * N + 4]) = FLOAT4(reg_C[6][4]);
    FLOAT4(d_C[C_index + 7 * N]) = FLOAT4(reg_C[7][0]);
    FLOAT4(d_C[C_index + 7 * N + 4]) = FLOAT4(reg_C[7][4]);
}
float test3_3 () {
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    const int m = 128;
    const int n = 128;
    const int k = 8;
    const int reg_size = 8;
    const int padding = 4;
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((thread_size + reg_size * reg_size - 1) / (reg_size * reg_size));
    // printf("block: %d, thread: %d \n", block.x, thread.x);
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding));
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
    const int reg_size = 8;
    const int padding = 4;
    extern __shared__ float sh[];
    float *A_sh = sh;
    float *B_sh = sh + (m + padding) * k;
    const int N_tile_index = blockIdx.y; // tile的列号
    const int M_tile_index = blockIdx.x; // tile的行号
    const int idx = threadIdx.x;
    const int A_m_index = (idx & 127); // 128
    const int A_n_index = 0; // 8
    const int B_m_index = (idx & 127) >> 4; // 8
    const int B_n_index = ((idx & 127) & 15) << 3; // 128
    const int C_m_index = idx >> 4; // tile内的8 * 8行号
    const int C_n_index = idx & 15; // tile内的8 * 8列号
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[reg_size];
    float reg_B[reg_size];
    float reg_C[reg_size][reg_size] = {0.0f};
    float4 readA1, readA2;
    const int d_A_index = (M_tile_index * m + A_m_index) * K + A_n_index;
    const int d_B_index = (B_m_index) * N + N_tile_index * n + B_n_index;
    // float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
        if (idx < 128) {
            readA1 = FLOAT4(d_A[d_A_index + K_tile_index * k]);
            A_sh[A_n_index * (m + padding) + A_m_index] = readA1.x;
            A_sh[(A_n_index + 1) * (m + padding) + A_m_index] = readA1.y;
            A_sh[(A_n_index + 2) * (m + padding) + A_m_index] = readA1.z;
            A_sh[(A_n_index + 3) * (m + padding) + A_m_index] = readA1.w;
            readA2 = FLOAT4(d_A[d_A_index + K_tile_index * k + 4]);
            A_sh[(A_n_index + 4) * (m + padding) + A_m_index] = readA2.x;
            A_sh[(A_n_index + 5) * (m + padding) + A_m_index] = readA2.y;
            A_sh[(A_n_index + 6) * (m + padding) + A_m_index] = readA2.z;
            A_sh[(A_n_index + 7) * (m + padding) + A_m_index] = readA2.w;
        } else {
            FLOAT4(B_sh[B_m_index * (n + padding) + B_n_index]) = FLOAT4(d_B[d_B_index + K_tile_index * k * N]);
            FLOAT4(B_sh[B_m_index * (n + padding) + B_n_index + 4]) = FLOAT4(d_B[d_B_index + K_tile_index * k * N + 4]);
        }
        __syncthreads();
        for (int k_reg_index = 0; k_reg_index < k; k_reg_index++) {
            FLOAT4(reg_A[0]) = FLOAT4(A_sh[(C_m_index * reg_size) + k_reg_index * (m + padding)]);
            FLOAT4(reg_A[4]) = FLOAT4(A_sh[(C_m_index * reg_size + 4) + k_reg_index * (m + padding)]);
            FLOAT4(reg_B[0]) = FLOAT4(B_sh[k_reg_index * (n + padding) + C_n_index * reg_size]);
            FLOAT4(reg_B[4]) = FLOAT4(B_sh[k_reg_index * (n + padding) + C_n_index * reg_size + 4]);
            for (int i = 0; i < reg_size; i++) {
                for (int j = 0; j < reg_size; j++) {
                        reg_C[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
         __syncthreads();
    }
    const int C_index = (M_tile_index * m + C_m_index * reg_size) * N + N_tile_index * n + C_n_index * reg_size;
    FLOAT4(d_C[C_index]) = FLOAT4(reg_C[0][0]);
    FLOAT4(d_C[C_index + 4]) = FLOAT4(reg_C[0][4]);
    FLOAT4(d_C[C_index + 1 * N]) = FLOAT4(reg_C[1][0]);
    FLOAT4(d_C[C_index + 1 * N + 4]) = FLOAT4(reg_C[1][4]);
    FLOAT4(d_C[C_index + 2 * N]) = FLOAT4(reg_C[2][0]);
    FLOAT4(d_C[C_index + 2 * N + 4]) = FLOAT4(reg_C[2][4]);
    FLOAT4(d_C[C_index + 3 * N]) = FLOAT4(reg_C[3][0]);
    FLOAT4(d_C[C_index + 3 * N + 4]) = FLOAT4(reg_C[3][4]);
    FLOAT4(d_C[C_index + 4 * N]) = FLOAT4(reg_C[4][0]);
    FLOAT4(d_C[C_index + 4 * N + 4]) = FLOAT4(reg_C[4][4]);
    FLOAT4(d_C[C_index + 5 * N]) = FLOAT4(reg_C[5][0]);
    FLOAT4(d_C[C_index + 5 * N + 4]) = FLOAT4(reg_C[5][4]);
    FLOAT4(d_C[C_index + 6 * N]) = FLOAT4(reg_C[6][0]);
    FLOAT4(d_C[C_index + 6 * N + 4]) = FLOAT4(reg_C[6][4]);
    FLOAT4(d_C[C_index + 7 * N]) = FLOAT4(reg_C[7][0]);
    FLOAT4(d_C[C_index + 7 * N + 4]) = FLOAT4(reg_C[7][4]);
}
float test3_4 () {
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    const int m = 128;
    const int n = 128;
    const int k = 8;
    const int reg_size = 8;
    const int padding = 4;
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((thread_size + reg_size * reg_size - 1) / (reg_size * reg_size));
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
    const int reg_size = 8;
    const int padding = 4;
    extern __shared__ float sh[];
    float *A_sh = sh;
    float *B_sh = sh + 2 * (m + padding) * k;
    const int N_tile_index = blockIdx.y; // tile的列号
    const int M_tile_index = blockIdx.x; // tile的行号
    const int idx = threadIdx.x;
    const int A_m_index = idx >> 1; // 128
    const int A_n_index = (idx & 1) << 2; // 8
    const int B_m_index = idx >> 5; // 8
    const int B_n_index = (idx & 31) << 2; // 128
    const int C_m_index = idx >> 4; // tile内的8 * 8行号
    const int C_n_index = idx & 15; // tile内的8 * 8列号
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    float reg_A[reg_size];
    float reg_B[reg_size];
    float reg_C[reg_size][reg_size] = {0.0f};
    Float4 readA;
    Float4 readB;
    int A_sh_offset = 0;
    int B_sh_offset = 0;
    const int A_sh_size = (m + padding) * k;
    const int B_sh_size = k * (n + padding);
    const int d_A_index = (M_tile_index * m + A_m_index) * K + A_n_index;
    const int d_B_index = (B_m_index) * N + N_tile_index * n + B_n_index;
    // float total = 0.0f;
    int K_tile_index = 0;
    lgkmcnt<0>();
    vmcnt<0>();
    
    readA.x = d_A[d_A_index + K_tile_index * k];
    readA.y = d_A[d_A_index + K_tile_index * k + 1];
    readA.z = d_A[d_A_index + K_tile_index * k + 2];
    readA.w = d_A[d_A_index + K_tile_index * k + 3];
    readB.x = d_B[d_B_index + K_tile_index * k * N];
    readB.y = d_B[d_B_index + K_tile_index * k * N + 1];
    readB.z = d_B[d_B_index + K_tile_index * k * N + 2];
    readB.w = d_B[d_B_index + K_tile_index * k * N + 3];
    // global_load<0>(&d_A[d_A_index], readA);
    // global_load<0>(&d_B[d_B_index], readB);
    // vmcnt<0>();
    A_sh[A_sh_offset + A_n_index * (m + padding) + A_m_index] = readA.x;
    A_sh[A_sh_offset + (A_n_index + 1) * (m + padding) + A_m_index] = readA.y;
    A_sh[A_sh_offset + (A_n_index + 2) * (m + padding) + A_m_index] = readA.z;
    A_sh[A_sh_offset + (A_n_index + 3) * (m + padding) + A_m_index] = readA.w;
    B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index] = readB.x;
    B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index + 1] = readB.y;
    B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index + 2] = readB.z;
    B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index + 3] = readB.w;
    lgkmcnt<0>();
    
    // for(;K_tile_index < int((K + k - 1) / k); K_tile_index++) {
    do {
        __syncthreads();
        if (K_tile_index + 1 < int((K + k - 1) / k)){
            // readA = FLOAT4(d_A[d_A_index + K_tile_index * k]);
	    // readB = FLOAT4(d_B[d_B_index + K_tile_index * k * N]);
            global_load<0>(&d_A[d_A_index + (K_tile_index + 1) * k], readA);
            global_load<0>(&d_B[d_B_index + (K_tile_index + 1) * k * N], readB);
            A_sh[A_sh_offset + A_n_index * (m + padding) + A_m_index] = readA.x;
        }
        lgkmcnt<0>();
        for (int k_reg_index = 0; k_reg_index < k; k_reg_index++) {
            FLOAT4(reg_A[0]) = FLOAT4(A_sh[A_sh_offset + (C_m_index * reg_size) + k_reg_index * (m + padding)]);
            FLOAT4(reg_A[4]) = FLOAT4(A_sh[A_sh_offset + (C_m_index * reg_size + 4) + k_reg_index * (m + padding)]);
            FLOAT4(reg_B[0]) = FLOAT4(B_sh[B_sh_offset + k_reg_index * (n + padding) + C_n_index * reg_size]);
            FLOAT4(reg_B[4]) = FLOAT4(B_sh[B_sh_offset + k_reg_index * (n + padding) + C_n_index * reg_size + 4]);
            for (int i = 0; i < reg_size; i++) {
                for (int j = 0; j < reg_size; j++) {
                        reg_C[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
	A_sh_offset ^= A_sh_size;
	B_sh_offset ^= B_sh_size;
	K_tile_index++;
        vmcnt<0>();
        lgkmcnt<0>();
	if (K_tile_index < int((K + k - 1) / k)) {
            A_sh[A_sh_offset + A_n_index * (m + padding) + A_m_index] = readA.x;
            A_sh[A_sh_offset + (A_n_index + 1) * (m + padding) + A_m_index] = readA.y;
            A_sh[A_sh_offset + (A_n_index + 2) * (m + padding) + A_m_index] = readA.z;
            A_sh[A_sh_offset + (A_n_index + 3) * (m + padding) + A_m_index] = readA.w;
            B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index] = readB.x;
            B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index + 1] = readB.y;
            B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index + 2] = readB.z;
            B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index + 3] = readB.w;
	}
        lgkmcnt<0>();
    // }
    } while (K_tile_index < int((K + k - 1) / k));
    // const int C_index = (M_tile_index * m + C_m_index * reg_size) * N + N_tile_index * n + C_n_index * reg_size;
    // lgkmcnt<0>();
    // global_store<0>(&d_C[C_index], FLOAT4(reg_C[0][0]));
    // global_store<0>(&d_C[C_index + 4], FLOAT4(reg_C[0][4]));
    // global_store<0>(&d_C[C_index + 1 * N], FLOAT4(reg_C[1][0]));
    // global_store<0>(&d_C[C_index + 1 * N + 4], FLOAT4(reg_C[1][4]));
    // global_store<0>(&d_C[C_index + 2 * N], FLOAT4(reg_C[2][0]));
    // global_store<0>(&d_C[C_index + 2 * N + 4], FLOAT4(reg_C[2][4]));
    // global_store<0>(&d_C[C_index + 3 * N], FLOAT4(reg_C[3][0]));
    // global_store<0>(&d_C[C_index + 3 * N + 4], FLOAT4(reg_C[3][4]));
    // global_store<0>(&d_C[C_index + 4 * N], FLOAT4(reg_C[4][0]));
    // global_store<0>(&d_C[C_index + 4 * N + 4], FLOAT4(reg_C[4][4]));
    // global_store<0>(&d_C[C_index + 5 * N], FLOAT4(reg_C[5][0]));
    // global_store<0>(&d_C[C_index + 5 * N + 4], FLOAT4(reg_C[5][4]));
    // global_store<0>(&d_C[C_index + 6 * N], FLOAT4(reg_C[6][0]));
    // global_store<0>(&d_C[C_index + 6 * N + 4], FLOAT4(reg_C[6][4]));
    // global_store<0>(&d_C[C_index + 7 * N], FLOAT4(reg_C[7][0]));
    // global_store<0>(&d_C[C_index + 7 * N + 4], FLOAT4(reg_C[7][4]));
    // vmcnt<0>();
    const int C_index = (M_tile_index * m + C_m_index * reg_size) * N + N_tile_index * n + C_n_index * reg_size;
    FLOAT4(d_C[C_index]) = FLOAT4(reg_C[0][0]);
    FLOAT4(d_C[C_index + 4]) = FLOAT4(reg_C[0][4]);
    FLOAT4(d_C[C_index + 1 * N]) = FLOAT4(reg_C[1][0]);
    FLOAT4(d_C[C_index + 1 * N + 4]) = FLOAT4(reg_C[1][4]);
    FLOAT4(d_C[C_index + 2 * N]) = FLOAT4(reg_C[2][0]);
    FLOAT4(d_C[C_index + 2 * N + 4]) = FLOAT4(reg_C[2][4]);
    FLOAT4(d_C[C_index + 3 * N]) = FLOAT4(reg_C[3][0]);
    FLOAT4(d_C[C_index + 3 * N + 4]) = FLOAT4(reg_C[3][4]);
    FLOAT4(d_C[C_index + 4 * N]) = FLOAT4(reg_C[4][0]);
    FLOAT4(d_C[C_index + 4 * N + 4]) = FLOAT4(reg_C[4][4]);
    FLOAT4(d_C[C_index + 5 * N]) = FLOAT4(reg_C[5][0]);
    FLOAT4(d_C[C_index + 5 * N + 4]) = FLOAT4(reg_C[5][4]);
    FLOAT4(d_C[C_index + 6 * N]) = FLOAT4(reg_C[6][0]);
    FLOAT4(d_C[C_index + 6 * N + 4]) = FLOAT4(reg_C[6][4]);
    FLOAT4(d_C[C_index + 7 * N]) = FLOAT4(reg_C[7][0]);
    FLOAT4(d_C[C_index + 7 * N + 4]) = FLOAT4(reg_C[7][4]);
    // const int reg_size = 8;
    // const int padding = 4;
    // extern __shared__ float sh[];
    // float *A_sh = sh;
    // float *B_sh = sh + 2 * (m + padding) * k;
    // const int N_tile_index = blockIdx.y; // tile的列号
    // const int M_tile_index = blockIdx.x; // tile的行号
    // const int idx = threadIdx.x;
    // const int A_m_index = (idx & 127); // 128
    // const int A_n_index = 0; // 8
    // const int B_m_index = (idx & 127) >> 4; // 8
    // const int B_n_index = ((idx & 127) & 15) << 3; // 128
    // const int C_m_index = idx >> 4; // tile内的8 * 8行号
    // const int C_n_index = idx & 15; // tile内的8 * 8列号
    // // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    // float reg_A[reg_size];
    // float reg_B[reg_size];
    // float reg_C[reg_size][reg_size] = {0.0f};
    // const int d_A_index = (M_tile_index * m + A_m_index) * K + A_n_index;
    // const int d_B_index = (B_m_index) * N + N_tile_index * n + B_n_index;
    // // float total = 0.0f;
    // int A_sh_offset = 0;
    // int B_sh_offset = 0;
    // const int A_sh_size = k * (m + padding);
    // const int B_sh_size = k * (n + padding);
    // float4 read[2];
    // int K_tile_index = 0;
    // lgkmcnt<0>();
    // if (idx < 128) {
    //     global_load<0>(&d_A[d_A_index + K_tile_index * k], read[0]);
    //     vmcnt<0>();
    //     A_sh[A_sh_offset + A_n_index * (m + padding) + A_m_index] = read[0].x;
    //     A_sh[A_sh_offset + (A_n_index + 1) * (m + padding) + A_m_index] = read[0].y;
    //     A_sh[A_sh_offset + (A_n_index + 2) * (m + padding) + A_m_index] = read[0].z;
    //     A_sh[A_sh_offset + (A_n_index + 3) * (m + padding) + A_m_index] = read[0].w;
    //     // printf("readA1: %lf %lf %lf %lf , readA2: %lf %lf %lf %lf ,,, %lf\n", read1.x, read1.y, read1.z, read1.w, read2.x, read2.y, read2.z, read2.w, d_A[d_A_index + K_tile_index * k]);
    //     global_load<0>(&d_A[d_A_index + (K_tile_index + 1) * k + 4], read[1]);
    //     vmcnt<0>();
    //     // printf("readA1: %lf %lf %lf %lf , readA2: %lf %lf %lf %lf ,,, %lf\n", read1.x, read1.y, read1.z, read1.w, read2.x, read2.y, read2.z, read2.w, d_A[d_A_index + K_tile_index * k]);
    //     A_sh[A_sh_offset + (A_n_index + 4) * (m + padding) + A_m_index] = read[1].x;
    //     A_sh[A_sh_offset + (A_n_index + 5) * (m + padding) + A_m_index] = read[1].y;
    //     A_sh[A_sh_offset + (A_n_index + 6) * (m + padding) + A_m_index] = read[1].z;
    //     A_sh[A_sh_offset + (A_n_index + 7) * (m + padding) + A_m_index] = read[1].w;
    // } else {
    //     global_load<0>(&d_B[d_B_index + K_tile_index * k * N], read[0]);
    //     vmcnt<0>();
    //     FLOAT4(B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index]) = read[0];
    //     global_load<0>(&d_B[d_B_index + K_tile_index * k * N + 4], read[1]);
    //     vmcnt<0>();
    //     FLOAT4(B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index + 4]) = read[1];
	// // printf("readB1: %lf %lf %lf %lf , readB2: %lf %lf %lf %lf ,,, %lf\n", read1.x, read1.y, read1.z, read1.w, read2.x, read2.y, read2.z, read2.w, d_B[d_B_index + K_tile_index * k * N]);
    // }

    // do {
    //     __syncthreads();
    //     if (K_tile_index + 1 < int((K + k - 1) / k)) {
    //         if (idx < 128) {
    //             global_load<0>(&d_A[d_A_index + (K_tile_index + 1) * k], read[0]);
    //             global_load<0>(&d_A[d_A_index + (K_tile_index + 1) * k + 4], read[0]);
    //         } else {
    //             global_load<0>(&d_B[d_B_index + (K_tile_index + 1) * k * N], read[1]);
    //             global_load<0>(&d_B[d_B_index + (K_tile_index + 1) * k * N + 4], read[1]);
    //         }
    //     }
    //     lgkmcnt<0>();
    //     for (int k_reg_index = 0; k_reg_index < k; k_reg_index++) {
    //         FLOAT4(reg_A[0]) = FLOAT4(A_sh[A_sh_offset + (C_m_index * reg_size) + k_reg_index * (m + padding)]);
    //         FLOAT4(reg_A[4]) = FLOAT4(A_sh[A_sh_offset + (C_m_index * reg_size + 4) + k_reg_index * (m + padding)]);
    //         FLOAT4(reg_B[0]) = FLOAT4(B_sh[B_sh_offset + k_reg_index * (n + padding) + C_n_index * reg_size]);
    //         FLOAT4(reg_B[4]) = FLOAT4(B_sh[B_sh_offset + k_reg_index * (n + padding) + C_n_index * reg_size + 4]);
    //         for (int i = 0; i < reg_size; i++) {
    //             for (int j = 0; j < reg_size; j++) {
    //                     reg_C[i][j] += reg_A[i] * reg_B[j];
    //             }
    //         }
    //     }
    //     A_sh_offset ^= A_sh_size;
    //     B_sh_offset ^= B_sh_size;
    //     K_tile_index++;
	// if (K_tile_index < int((K + k - 1) / k)) {
    //         if (idx < 128) {
    //             vmcnt<0>();
    //             A_sh[A_sh_offset + A_n_index * (m + padding) + A_m_index] = read[0].x;
    //             A_sh[A_sh_offset + (A_n_index + 1) * (m + padding) + A_m_index] = read[0].y;
    //             A_sh[A_sh_offset + (A_n_index + 2) * (m + padding) + A_m_index] = read[0].z;
    //             A_sh[A_sh_offset + (A_n_index + 3) * (m + padding) + A_m_index] = read[0].w;
    //             vmcnt<0>();
    //             A_sh[A_sh_offset + (A_n_index + 4) * (m + padding) + A_m_index] = read[1].x;
    //             A_sh[A_sh_offset + (A_n_index + 5) * (m + padding) + A_m_index] = read[1].y;
    //             A_sh[A_sh_offset + (A_n_index + 6) * (m + padding) + A_m_index] = read[1].z;
    //             A_sh[A_sh_offset + (A_n_index + 7) * (m + padding) + A_m_index] = read[1].w;
    //             printf("readA1: %lf %lf %lf %lf , readA2: %lf %lf %lf %lf ,,, %lf\n", read[0].x, read[0].y, read[0].z, read[0].w, read[1].x, read[1].y, read[1].z, read[1].w, d_A[d_A_index + K_tile_index * k]);
    //         } else {
    //             vmcnt<0>();
    //             FLOAT4(B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index]) = read[0];
    //             vmcnt<0>();
    //             FLOAT4(B_sh[B_sh_offset + B_m_index * (n + padding) + B_n_index + 4]) = read[1];
    //             printf("readB1: %lf %lf %lf %lf , readB2: %lf %lf %lf %lf ,,, %lf\n", read[0].x, read[0].y, read[0].z, read[0].w, read[1].x, read[1].y, read[1].z, read[1].w, d_B[d_B_index + (K_tile_index + 1) * k * N]);
    //         }
	// }
    // } while (K_tile_index < int((K + k - 1) / k));
    // const int C_index = (M_tile_index * m + C_m_index * reg_size) * N + N_tile_index * n + C_n_index * reg_size;
    // FLOAT4(d_C[C_index]) = FLOAT4(reg_C[0][0]);
    // FLOAT4(d_C[C_index + 4]) = FLOAT4(reg_C[0][4]);
    // FLOAT4(d_C[C_index + 1 * N]) = FLOAT4(reg_C[1][0]);
    // FLOAT4(d_C[C_index + 1 * N + 4]) = FLOAT4(reg_C[1][4]);
    // FLOAT4(d_C[C_index + 2 * N]) = FLOAT4(reg_C[2][0]);
    // FLOAT4(d_C[C_index + 2 * N + 4]) = FLOAT4(reg_C[2][4]);
    // FLOAT4(d_C[C_index + 3 * N]) = FLOAT4(reg_C[3][0]);
    // FLOAT4(d_C[C_index + 3 * N + 4]) = FLOAT4(reg_C[3][4]);
    // FLOAT4(d_C[C_index + 4 * N]) = FLOAT4(reg_C[4][0]);
    // FLOAT4(d_C[C_index + 4 * N + 4]) = FLOAT4(reg_C[4][4]);
    // FLOAT4(d_C[C_index + 5 * N]) = FLOAT4(reg_C[5][0]);
    // FLOAT4(d_C[C_index + 5 * N + 4]) = FLOAT4(reg_C[5][4]);
    // FLOAT4(d_C[C_index + 6 * N]) = FLOAT4(reg_C[6][0]);
    // FLOAT4(d_C[C_index + 6 * N + 4]) = FLOAT4(reg_C[6][4]);
    // FLOAT4(d_C[C_index + 7 * N]) = FLOAT4(reg_C[7][0]);
    // FLOAT4(d_C[C_index + 7 * N + 4]) = FLOAT4(reg_C[7][4]);
    // const int C_index = (M_tile_index * m + C_m_index * reg_size) * N + N_tile_index * n + C_n_index * reg_size;
    // lgkmcnt<0>();
    // global_store<0>(&d_C[C_index], FLOAT4(reg_C[0][0]));
    // global_store<0>(&d_C[C_index + 4], FLOAT4(reg_C[0][4]));
    // global_store<0>(&d_C[C_index + 1 * N], FLOAT4(reg_C[1][0]));
    // global_store<0>(&d_C[C_index + 1 * N + 4], FLOAT4(reg_C[1][4]));
    // global_store<0>(&d_C[C_index + 2 * N], FLOAT4(reg_C[2][0]));
    // global_store<0>(&d_C[C_index + 2 * N + 4], FLOAT4(reg_C[2][4]));
    // global_store<0>(&d_C[C_index + 3 * N], FLOAT4(reg_C[3][0]));
    // global_store<0>(&d_C[C_index + 3 * N + 4], FLOAT4(reg_C[3][4]));
    // global_store<0>(&d_C[C_index + 4 * N], FLOAT4(reg_C[4][0]));
    // global_store<0>(&d_C[C_index + 4 * N + 4], FLOAT4(reg_C[4][4]));
    // global_store<0>(&d_C[C_index + 5 * N], FLOAT4(reg_C[5][0]));
    // global_store<0>(&d_C[C_index + 5 * N + 4], FLOAT4(reg_C[5][4]));
    // global_store<0>(&d_C[C_index + 6 * N], FLOAT4(reg_C[6][0]));
    // global_store<0>(&d_C[C_index + 6 * N + 4], FLOAT4(reg_C[6][4]));
    // global_store<0>(&d_C[C_index + 7 * N], FLOAT4(reg_C[7][0]));
    // global_store<0>(&d_C[C_index + 7 * N + 4], FLOAT4(reg_C[7][4]));
    // vmcnt<0>();
}
float test3_5 () {
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    const int m = 128;
    const int n = 128;
    const int k = 8;
    const int reg_size = 8;
    const int padding = 4;
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((thread_size + reg_size * reg_size - 1) / (reg_size * reg_size));
    // printf("block: %d, thread: %d \n", block.x, thread.x);
    int shared_size = sizeof(float) * ((m + padding) * k + k * (n + padding)) * 2;
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
