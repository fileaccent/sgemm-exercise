#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <omp.h>
#include <cublas_v2.h>
using namespace std;
#define iteration 1
const int reg_size = 2;
int M = 1 << 10;
int K = 1 << 10;
int N = 1 << 10;
const int  m = 16;
const int  n = 16;
const int  k = 16;
const int WAVE_SIZE = 32;
// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define ErrChk(code) { Assert((code), __FILE__, __LINE__); }
static inline void Assert(cudaError_t  code, const char *file, int line){
	if(code!=cudaSuccess) {
		printf("CUDA Runtime Error: %s:%d:'%s'\n", file, line,cudaGetErrorString(code));
		exit(EXIT_FAILURE);
	}
}
// static inline void Assert(miopenStatus_t code, const char *file, int line){
//     if (code!=miopenStatusSuccess){
// 		printf("cuDNN API Error: %s:%d:'%s'\n", file, line, miopenGetErrorString(code));
//         exit(EXIT_FAILURE);
//     }
// }
// static inline void Assert(cudablasStatus_t code, const char *file, int line){
//     if (code!=cudaBLAS_STATUS_SUCCESS){
// 		printf("cuBLAS API Error: %s:%d:'%s'\n", file, line, cudablasGetErrorString(code));
//         exit(EXIT_FAILURE);
//     }
// }

#define KernelErrChk(){\
		cudaError_t errSync  = cudaGetLastError();\
		cudaError_t errAsync = cudaDeviceSynchronize();\
		if (errSync != cudaSuccess) {\
			  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));\
			  exit(EXIT_FAILURE);\
		}\
		if (errAsync != cudaSuccess){\
			printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));\
			exit(EXIT_FAILURE);\
		}\
}


#define test_start()\
    float * h_A; \
    float * h_B; \
    float * h_C; \
    float * test_C;\
    int A_size = M * K; \
    int B_size = K * N; \
    int C_size = M * N; \
    int A_bytes = sizeof(float) * A_size; \
    int B_bytes = sizeof(float) * B_size; \
    int C_bytes = sizeof(float) * C_size; \
    h_A = (float *) malloc(A_bytes); \
    h_B = (float *) malloc(B_bytes); \
    h_C = (float *) malloc(C_bytes); \
    test_C = (float *) malloc(C_bytes); \
    for (int i = 0; i < A_size; i++) { \
        h_A[i] = rand() % 5; \
    } \
    for (int i = 0; i < B_size; i++) { \
        h_B[i] = rand() % 5; \
    } \
    float * d_A; \
    float * d_B; \
    float * d_C; \
    ErrChk(cudaMalloc(&d_A, A_bytes));\
    ErrChk(cudaMalloc(&d_B, B_bytes));\
    ErrChk(cudaMalloc(&d_C, C_bytes));\
    ErrChk(cudaMemcpy(d_A, h_A, A_bytes, cudaMemcpyHostToDevice)); \
    ErrChk(cudaMemcpy(d_B, h_B, B_bytes, cudaMemcpyHostToDevice)); \
    cudaEvent_t start, stop; \
    float elapsedTime = 0.0f; \
    ErrChk(cudaEventCreate(&start)); \
    ErrChk(cudaEventCreate(&stop));


#define test_end()\
    ErrChk(cudaEventRecord(stop, 0)); \
    ErrChk(cudaEventSynchronize(stop)); \
    ErrChk(cudaEventElapsedTime(&elapsedTime, start, stop)); \
    ErrChk(cudaMemcpy(h_C, d_C, C_bytes, cudaMemcpyDeviceToHost));\
    /*for (int i = 0; i < M; i++) {\
        for (int j = 0; j < N; j++) {\
            int total = 0;\
            for (int k = 0; k < K; k++) {\
                total += h_A[i * K + k] * h_B[k * N + j];\
            }\
            test_C[i * N + j] = total;\
        }\
    }\
    bool isSame = true;\
    for (int i = 0; i < C_size; i++) {\
        if (abs(test_C[i] - h_C[i]) > 0.01) {\
            cout << "error: i: " << i << " test_C: " << test_C[i] << " h_C[i]: " << h_C[i] << endl;\
            isSame = false;\
            break;\
        }\
    }*/\
    ErrChk(cudaFree(d_A));\
    ErrChk(cudaFree(d_B));\
    ErrChk(cudaFree(d_C));\
    ErrChk(cudaEventDestroy(start)); \
    ErrChk(cudaEventDestroy(stop));\
    free(h_A);\
    free(h_B);\
    free(h_C);

__device__ void set_value(float* dst, float* source, int n) {
    int i = 0;
    while (i < n) {
       if (i + 3 < n) {
          FLOAT4(dst[i]) = FLOAT4(source[i]);
          i += 4;
       } else if (i + 1 < n) {
          FLOAT2(dst[i]) = FLOAT2(source[i]);
          i += 2;
       } else if (i < n) {
          dst[i] = source[i];
          i++;
       }
    }
}

__device__ void set_value_matrix(float* dst, float* source, int dst_m, int dst_n, int dst_lda, int source_lda) {
    for (int i = 0; i < dst_m; i++) {
        set_value(&dst[i * dst_lda], &source[i * source_lda], dst_n);
    }
}

float cublas_result() {
    test_start();
    int thread_size = 32;
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C, N);
    KernelErrChk();
    cudaEventRecord(start, 0);
    for (int i = 0; i < iteration; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C, N);
    }
    test_end();
    // rocblas_destroy_handle(handle);
    return elapsedTime / iteration;
}

#include"kernel_1.h"
#include"kernel_2.h"
#include"kernel_3.h"
#include"kernel_4.h"
#include"kernel_5.h"
#include"kernel_6.h"
#include"kernel_7.h"
#include"kernel_8.h"
#include"kernel_9.h"

int main () {
    float baseTime;
    float preTime;
    float nowTime;
    float Tflops;
    // 一些记录:
    // 一个块最多由1024个线程
    
    cout << "warp: " << WAVE_SIZE << endl;
    // cublas版本
    baseTime = cublas_result();
    Tflops = 2 * ((float)M * N * K) / (baseTime / 1000) / 1e12;
    cout << "cublas: " << baseTime << "ms" << ", Tflops: " << Tflops << endl;

    // baseTime = test1();
    // 1. 无优化版本
    // cout << "test1: " << baseTime  << "ms"<< endl;
    // 2. 共享内存分块
    preTime = baseTime;
    nowTime = test2();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test2: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime  <<  ", Tflops: " << Tflops << endl;
    preTime = nowTime;
    // 3. 线程分块, 一个线程加好几个值
    nowTime = test3();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime  << ", Tflops: " << Tflops  << endl;
    preTime = nowTime;
    // 3.1 寄存器缓存
    nowTime = test3_1();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3_1: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops  << endl;
    preTime = nowTime;
    // 3.2 128 * 8 * 128
    nowTime = test3_2();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3_2: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops  << endl;
    preTime = nowTime;
    // 3.3 128 * 8 * 128 用寄存器缓存
    nowTime = test3_3();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3_3: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops  << endl;
    preTime = nowTime;
    // 3.4 128 * 8 * 128 padding + A 转置
    nowTime = test3_4();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3_4: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops  << endl;
    preTime = nowTime;
    // 3.5 128 * 8 * 128 warp分块
    nowTime = test3_5();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3_5: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops  << endl;
    preTime = nowTime;
    // 3.6 128 * 8 * 128 
    nowTime = test3_6();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3_6: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops  << endl;
    preTime = nowTime;

    // 4. 共享内存冲突处理 padding:4
    nowTime = test4();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test4: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << endl;
    preTime = nowTime;
    // 5. 寄存器缓存
    nowTime = test5();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test5: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << endl;
    preTime = nowTime;
    // 6. 数据预取或双缓存: 负优化
    nowTime = test6();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test6: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << endl;
    preTime = nowTime;
    // 7. 调整寄存器计算时的块
    nowTime = test7();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << endl;
    preTime = nowTime;
    // 7.1 调整参数
    nowTime = test7_1();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7_1: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << endl;
    preTime = nowTime;
    // 7.2 融合8的所有优化
    nowTime = test7_2();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7_2: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << endl;
    preTime = nowTime;
    // 8. 分为 warp 块执行, 无效果
    nowTime = test8();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << endl;
    preTime = nowTime;
    // 8.1 对齐共享内存地址
    nowTime = test8_1();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8_1: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << endl;
    preTime = nowTime;
    // 8.2 使用向量外积
    nowTime = test8_2();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8_2: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << endl;
    preTime = nowTime;
    // 8.3 sh_B 转置(负优化)
    nowTime = test8_3();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8_3: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << endl;
    preTime = nowTime;
    // 8.4  warp 修改尺寸
    nowTime = test8_4();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8_4: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << endl;
    preTime = nowTime;
    // 8.5 warp z字布局(不能说有没有优化)
    nowTime = test8_5();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8_5: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << endl;
    preTime = nowTime;
    // 8.6
    nowTime = test8_6();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8_6: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << endl;
    preTime = nowTime;

    // 9. 将4 * 4 分开拆成 2 * 2 的块执行
    nowTime = test9();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test9: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << endl;
    preTime = nowTime;
    // // 10.1  
    // nowTime = test10_1();
    // Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    // cout << "test10_1: " << nowTime  << "ms speedup: " << preTime / nowTime << ", Glops: " << Tflops << endl;
    // preTime = nowTime;
    // // 10.2
    // nowTime = test10_2();
    // Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    // cout << "test10_2: " << nowTime  << "ms speedup: " << preTime / nowTime << ", Glops: " << Tflops << endl;
    // preTime = nowTime;

    return 0;
}

