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
    const int idx = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = idx / WAVE_SIZE;
    const int warp_id_m = warp_id / 2; // 2 * 2 的格子
    const int warp_id_n = warp_id % 2;
    const int lane_id = idx % WAVE_SIZE;
    const int lane_id_m = lane_id / 16 * 2 + lane_id % 2; // 8 * 8 的块
    const int lane_id_n = lane_id % 16 / 2;
    const int C_m_index = warp_id_m * 8 + lane_id_m; // tile内的4 * 4行号
    const int C_n_index = warp_id_n * 8 + lane_id_n; // tile内的4 * 4列号
    // int C_m_index = threadIdx.x / ((n + TN - 1) / TN); // tile内的4 * 4行号
    // int C_n_index = threadIdx.x % ((n + TN - 1) / TN); // tile内的4 * 4列号
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
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel9<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}
__global__ void gemm_kernel9_1(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    int comp_a_smem_m = ty * TM + m;
                    int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
        }
    }
}
float test9_1 () {
    const int m = 128;
    const int n = 128;
    const int k = 8;
    const int TM = 8;
    const int TN = 8;
    const int TK = 2;
    const int padding = 4;
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((m + TM - 1) / TM, (n + TN - 1) / TN);
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding));
    gemm_kernel9_1<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel9_1<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K);
    }
    test_end();
    return elapsedTime / iteration;
}
__global__ void gemm_kernel9_2(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BK][BM];
    __shared__ float s_b[BK][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        s_a[load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();

        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty * TM / 2         ]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx * TN / 2         ]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx * TN / 2 + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}
float test9_2 () {
    const int m = 128;
    const int n = 128;
    const int k = 8;
    const int TM = 8;
    const int TN = 8;
    const int TK = 2;
    const int padding = 4;
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((m + TM - 1) / TM, (n + TN - 1) / TN);
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding));
    gemm_kernel9_2<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel9_2<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K);
    }
    test_end();
    return elapsedTime / iteration;
}
__global__ void gemm_kernel9_3(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM];
    __shared__ float s_b[2][BK][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        s_a[0][load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
    }

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {

        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2         ]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2         ]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        s_a[smem_sel_next][load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();
    }

    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2         ]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2         ]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

        #pragma unroll
        for (int tm = 0; tm < TM; tm++) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}
float test9_3 () {
    const int m = 128;
    const int n = 128;
    const int k = 8;
    const int TM = 8;
    const int TN = 8;
    const int TK = 2;
    const int padding = 4;
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    test_start();
    int thread_size = min(m * n, C_size);
    dim3 block((M + m - 1) / m, (N + n - 1) / n);
    dim3 thread((m + TM - 1) / TM, (n + TN - 1) / TN);
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding));
    gemm_kernel9_3<<<block, thread, shared_size>>>(d_A, d_B, d_C, M, N, K);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        gemm_kernel9_3<<<block, thread,  shared_size>>>(d_A, d_B, d_C, M, N, K);
    }
    test_end();
    return elapsedTime / iteration;
}
