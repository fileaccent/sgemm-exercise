#include <iostream>
#include <cstdlib>
#include <cmath>
#include <hip/hip_runtime.h>
#include <omp.h>
#include <rocblas.h>
#include <rocwmma/rocwmma.hpp>
using namespace std;
#define iteration 1000
const int reg_size = 2;
int M = 1 << 12;
int K = 1 << 12;
int N = 1 << 12;
const int  m = 16;
const int  n = 16;
const int  k = 16;
const int WAVE_SIZE = rocwmma::AMDGCN_WAVE_SIZE;
typedef float Float4 __attribute__((ext_vector_type(4)));
typedef float Float2 __attribute__((ext_vector_type(2)));
// #define __local __attribute__((address_space(3)))
// __device__ inline static __local void* __to_local(unsigned x) { return (__local void*)x; }
__device__ inline static __local void* __to_local(float* x) { return (__local void*)x; }
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define ErrChk(code) { Assert((code), __FILE__, __LINE__); }
static inline void Assert(hipError_t  code, const char *file, int line){
	if(code!=hipSuccess) {
		printf("CUDA Runtime Error: %s:%d:'%s'\n", file, line,hipGetErrorString(code));
		exit(EXIT_FAILURE);
	}
}
// static inline void Assert(miopenStatus_t code, const char *file, int line){
//     if (code!=miopenStatusSuccess){
// 		printf("cuDNN API Error: %s:%d:'%s'\n", file, line, miopenGetErrorString(code));
//         exit(EXIT_FAILURE);
//     }
// }
// static inline void Assert(hipblasStatus_t code, const char *file, int line){
//     if (code!=HIPBLAS_STATUS_SUCCESS){
// 		printf("cuBLAS API Error: %s:%d:'%s'\n", file, line, hipblasGetErrorString(code));
//         exit(EXIT_FAILURE);
//     }
// }

#define KernelErrChk(){\
		hipError_t errSync  = hipGetLastError();\
		hipError_t errAsync = hipDeviceSynchronize();\
		if (errSync != hipSuccess) {\
			  printf("Sync kernel error: %s\n", hipGetErrorString(errSync));\
			  exit(EXIT_FAILURE);\
		}\
		if (errAsync != hipSuccess){\
			printf("Async kernel error: %s\n", hipGetErrorString(errAsync));\
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
        h_A[i] = rand() % 3 * 0.1; \
    } \
    for (int i = 0; i < B_size; i++) { \
        h_B[i] = rand() % 4 * 0.01; \
    } \
    float * d_A; \
    float * d_B; \
    float * d_C; \
    ErrChk(hipMalloc(&d_A, A_bytes));\
    ErrChk(hipMalloc(&d_B, B_bytes));\
    ErrChk(hipMalloc(&d_C, C_bytes));\
    ErrChk(hipMemcpy(d_A, h_A, A_bytes, hipMemcpyHostToDevice)); \
    ErrChk(hipMemcpy(d_B, h_B, B_bytes, hipMemcpyHostToDevice)); \
    hipEvent_t start, stop; \
    float elapsedTime = 0.0f; \
    ErrChk(hipEventCreate(&start)); \
    ErrChk(hipEventCreate(&stop));


#define test_end()\
    ErrChk(hipEventRecord(stop, 0)); \
    ErrChk(hipEventSynchronize(stop)); \
    ErrChk(hipEventElapsedTime(&elapsedTime, start, stop)); \
    ErrChk(hipMemcpy(h_C, d_C, C_bytes, hipMemcpyDeviceToHost));\
    /*for (int i = 0; i < M; i++) {\
        for (int j = 0; j < N; j++) {\
            float total = 0;\
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
    ErrChk(hipFree(d_A));\
    ErrChk(hipFree(d_B));\
    ErrChk(hipFree(d_C));\
    ErrChk(hipEventDestroy(start)); \
    ErrChk(hipEventDestroy(stop));\
    free(h_A);\
    free(h_B);\
    free(h_C);

__device__ void set_value(float* dst, float* source,const int n) {
    int i = 0;
    if (n == 1) {
       dst[0] = source[0];
    } else if (n == 2) {
       FLOAT2(dst[0]) = FLOAT2(source[0]);
    } else if (n == 4) {
       FLOAT4(dst[0]) = FLOAT4(source[0]);
    } else {
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
}

__device__ void set_value_matrix(float* dst, float* source, int dst_m, int dst_n, int dst_lda, int source_lda) {
    for (int i = 0; i < dst_m; i++) {
        set_value(&dst[i * dst_lda], &source[i * source_lda], dst_n);
    }
}
template<uint32_t offset>
inline __device__ void global_load(float* ptr, float4 &val) {
    if(offset == 0) {
    asm volatile("\n \
    global_load_dwordx4 %0, %1, off \n \
    "
    :"=v"(val)
    :"v"(ptr)
    );
    return;
    }
    if(offset == 8) {
    asm volatile("\n \
    global_load_dwordx4 %0, %1, off offset:32 \n \
    "
    :"=v"(val)
    :"v"(ptr));
    }
}

template<uint32_t offset>
inline __device__ void global_load(float* ptr, Float4 &val) {
    if(offset == 0) {
    asm volatile("\n \
    global_load_dwordx4 %0, %1, off \n \
    "
    :"=v"(val)
    :"v"(ptr)
    );
    return;
    }
    if(offset == 8) {
    asm volatile("\n \
    global_load_dwordx4 %0, %1, off offset:32 \n \
    "
    :"=v"(val)
    :"v"(ptr));
    }
}
template<uint32_t offset>
inline __device__ void global_store(float* ptr, float4 val) {
    Float4 mid;
    mid.x = val.x;
    mid.y = val.y;
    mid.z = val.z;
    mid.w = val.w;
    if(offset == 0*32) {
    asm volatile("\n \
    global_store_dwordx4 %1, %0, off \n \
    "
    :
    :"v"(mid), "v"(ptr));
    return;
    }
    if(offset == 16) {
    asm volatile("\n \
    global_store_dwordx4 %1, %0, off offset:16*4*4 \n \
    "
    :
    :"v"(mid), "v"(ptr));
    }
}


template<uint32_t cnt>
inline __device__ void lgkmcnt(){
  if(cnt == 0) {
    asm volatile("\n \
    s_waitcnt lgkmcnt(0) \n \
    "::);
  }
  if(cnt == 1) {
    asm volatile("\n \
    s_waitcnt lgkmcnt(1) \n \
    "::);
  }
  if(cnt == 2) {
    asm volatile("\n \
    s_waitcnt lgkmcnt(2) \n \
    "::);
  }
  if(cnt == 3) {
    asm volatile("\n \
    s_waitcnt lgkmcnt(3) \n \
    "::);
  }
  if(cnt == 4) {
    asm volatile("\n \
    s_waitcnt lgkmcnt(4) \n \
    "::);
  }
  if(cnt == 5) {
    asm volatile("\n \
    s_waitcnt lgkmcnt(5) \n \
    "::);
  }
  if(cnt == 6) {
    asm volatile("\n \
    s_waitcnt lgkmcnt(6) \n \
    "::);
  }

/**
* Disabling as 16 is to high to fit in 4bits (15 max)
  if(cnt == 16) {
    asm volatile("\n \
    s_waitcnt lgkmcnt(16) \n \
    "::);
  }
*/
}

template<uint32_t cnt>
inline __device__ void vmcnt() {
    if(cnt == 0) {
      asm volatile ("\n \
      s_waitcnt vmcnt(0) \n \
      "::);
    }
    if(cnt == 1) {
      asm volatile ("\n \
      s_waitcnt vmcnt(1) \n \
      "::);
    }
    if(cnt == 2) {
      asm volatile ("\n \
      s_waitcnt vmcnt(2) \n \
      "::);
    }
    if(cnt == 4) {
      asm volatile ("\n \
      s_waitcnt vmcnt(2) \n \
      "::);
    }
}
template<uint32_t cnt>
inline __device__ void vscnt() {
    if(cnt == 0) {
      asm volatile ("\n \
      s_waitcnt vscnt(0) \n \
      "::);
    }
    if(cnt == 1) {
      asm volatile ("\n \
      s_waitcnt vscnt(1) \n \
      "::);
    }
    if(cnt == 2) {
      asm volatile ("\n \
      s_waitcnt vscnt(2) \n \
      "::);
    }
    if(cnt == 4) {
      asm volatile ("\n \
      s_waitcnt vscnt(2) \n \
      "::);
    }
}
inline __device__ void fma_op(float &c, float &a, float &b) {
    asm volatile("\n \
          v_fma_f32 %0, %1, %2, %0  \n \
          "
          :
          :"v"(c), "v"(a), "v"(b)
          );
}
float rocblas_result() {
    test_start();
    int thread_size = 32;
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
    KernelErrChk();
    hipEventRecord(start, 0);
    for (int i = 0; i < iteration; i++) {
        rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
    }
    test_end();
    rocblas_destroy_handle(handle);
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
#include"kernel_10.h"

int main () {
    float baseTime;
    float preTime;
    float nowTime;
    float Tflops;
    // 一些记录:
    // 一个块最多由1024个线程
    
    cout << "warp: " << WAVE_SIZE << endl;
    // rocblas版本
    baseTime = rocblas_result();
    Tflops = 2 * ((float)M * N * K) / (baseTime / 1000) / 1e12;
    cout << "rocblas: " << baseTime << "ms" << ", Tflops: " << Tflops << endl;
    preTime = test1();
    // 1. 无优化版本
    cout << "test1: " << preTime  << "ms"<< endl;
    // 2. 共享内存分块
    preTime = baseTime;
    nowTime = test2();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test2: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime  <<  ", Tflops: " << Tflops  << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 3. 线程分块, 一个线程加好几个值
    nowTime = test3();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime  << ", Tflops: " << Tflops  << ", Tflops_ratio: " << Tflops / 23.1  << endl;
    preTime = nowTime;
    // 3.1 寄存器缓存
    nowTime = test3_1();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3_1: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops  << ", Tflops_ratio: " << Tflops / 23.1  << endl;
    preTime = nowTime;
    // 3.2 128 * 8 * 128
    nowTime = test3_2();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3_2: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops  << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 3.3 128 * 8 * 128 用寄存器缓存
    nowTime = test3_3();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3_3: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 3.4 128 * 8 * 128 padding + A 转置
    nowTime = test3_4();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3_4: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 3.5 128 * 8 * 128 warp分块
    nowTime = test3_5();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3_5: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 3.6 128 * 8 * 128 
    nowTime = test3_6();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3_6: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;

    // 4. 共享内存冲突处理 padding:4
    nowTime = test4();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test4: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 5. 寄存器缓存
    nowTime = test5();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test5: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 6. 数据预取或双缓存: 负优化
    nowTime = test6();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test6: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 7. 调整寄存器计算时的块
    nowTime = test7();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 7.1 调整参数
    nowTime = test7_1();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7_1: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 7.2 调整参数
    nowTime = test7_2();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7_2: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 7.3
    nowTime = test7_3();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7_3: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 7.4
    nowTime = test7_4();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7_4: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 7.5
    nowTime = test7_5();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7_5: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 7.6
    nowTime = test7_6();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7_6: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 7.7
    nowTime = test7_7();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7_7: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 7.8
    nowTime = test7_8();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7_8: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 8. 分为 warp 块执行, 无效果
    nowTime = test8();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 8.1 对齐共享内存地址
    nowTime = test8_1();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8_1: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 8.2 使用向量外积
    nowTime = test8_2();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8_2: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 8.3 sh_B 转置(负优化)
    nowTime = test8_3();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8_3: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 8.4  warp 修改尺寸
    nowTime = test8_4();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8_4: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 8.5 warp z字布局(不能说有没有优化)
    nowTime = test8_5();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8_5: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 8.6
    nowTime = test8_6();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8_6: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;

    // 9. 将4 * 4 分开拆成 2 * 2 的块执行
    nowTime = test9();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test9: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 9.1
    nowTime = test9_1();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test9_1: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 9.2
    nowTime = test9_2();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test9_2: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 9.3
    nowTime = test9_3();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test9_3: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;

    // 10  
    nowTime = test10();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test10: " << nowTime  << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // // 10.2
    // nowTime = test10_2();
    // Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    // cout << "test10_2: " << nowTime  << "ms speedup: " << preTime / nowTime << ", Glops: " << Tflops << endl;
    // preTime = nowTime;

    return 0;
}

