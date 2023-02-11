
本项目属于 gemm 优化(支持 cuda 和 hip)
========

运行
---------
hip

``` shell
hipcc main.cpp -fopenmp -lrocblas --std=c++14 && ./a.out
```

cuda

``` shell
nvcc main.cpp -lcublas && ./a.out
```

性能
---------
1. 在 Vega 10 XTX [Radeon Vega Frontier Edition] 中有 rocblas 80% 的性能, 是峰值性能的50%

2. 在 MI100 中大概只有rocblas 30% 的性能(rocblas的性能不正常, 怀疑使用了matrix core), 是峰值性能的50%

3. cuda 部分 属于 hip 的代码转过去的, 并未对性能作特殊优化

4. 综合最快的kernel函数为7.4


参数解释
---------
1. 下面的数据均为M = 4096, N = 4096, K = 4096 的结果

2. 数据所用硬件为MI100, SCLK 1472Mhz, MCLK 1200Mhz

3. speedup: 针对上一版 kernel 的加速比

4. rocblas_ratio: 和rocblas执行时间的比例

5. Tflops: Tflops

6. Tflops_ratio: 和MI100峰值flops的比例

存在问题
---------
1. 如何用理论解释优化结果

2. 如何用rocprof 分析出性能瓶颈

3. cuda gemm的优化方法大部分都已经试过了, 有无效果和负效果的, 是自己问题, 还是 hip 本身不支持

4. hip gemm的代码大部分都是用汇编写的, 是否意味着 hipcc 的优化差

5. 共享内存汇编指令, 存在着部分数据无法读取的问题, 如何解决

6. 性能未达预期, 如何继续优化?

7. 为什么rocblas的Tflops 值比显卡的峰值性能还高, 但低于matrix core的峰值性能, 是使用了 matrix core 没说, 还是我算错了?

8. tensile和cuda gemm 都使用了 M = 128, N = 128, K = 8 的矩阵分块, 无论我如何尝试都无法超过 M = 64, N = 64, K = 16 的分块, 是我的实现方式有问题?


优化记录:
=========================

1 最初始版本 从全局内存读取一行一列计算输出
---------

    test1: 387.927ms

  ``` c++
    __global__ void gemm_kernel1(float *d_A, float *d_B, float *d_C, int M, int N, int K) {
        int idx = threadIdx.x  + blockIdx.x * blockDim.x;
        int ix = idx % N; // n
        int iy = idx / N; // m
        // printf("ix: %d \n", ix);
        // printf("iy: %d \n", iy);
        if (idx >= M * N) return;
        float total = 0;
        for (int i = 0; i < K; i++) {
            total += d_A[iy * K + i] * d_B[i * N + ix];
        }
        // printf("total: %lf \n", total);
        d_C[iy * N + ix] = total;
    }
  ```

2 共享内存读取块计算, 一个block从全局内存读取一行块和一列块计算输出
---------

    test2: 50.1172ms speedup: 0.0892845, rocblas_ratio: 0.0892845, Tflops: 2.74235, Tflops_ratio: 0.118717

  ``` c++
  __global__ void gemm_kernel2(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m, int n, int k) {
    extern __shared__ float sh[];
    float *A_sh = sh; // 
    float *B_sh = sh + m * k;
    int N_tile_index = blockIdx.x % ((N + n - 1)/ n); // tile的列号
    int M_tile_index = blockIdx.x / ((N + n - 1)/ n); // tile的行号
    int n_index = threadIdx.x % (n); // tile内的4 * 4列号
    int m_index = threadIdx.x / (n); // tile内的4 * 4行号
    float total = 0.0f;
    for (int K_tile_index = 0; K_tile_index < K; K_tile_index += k) {
        // 共享内存读取数据
        if ((M_tile_index * m + m_index) * K + K_tile_index + n_index < M * K) {
            A_sh[m_index * k + n_index] = d_A[(M_tile_index * m + m_index) * K + K_tile_index + n_index];
        } else {
            A_sh[m_index * k + n_index] = 0;
        }
        if ((K_tile_index + m_index) * N + N_tile_index * n + n_index < K * N) {
            B_sh[m_index * n + n_index] = d_B[(K_tile_index + m_index) * N + N_tile_index * n + n_index];
        } else {
            B_sh[m_index * n + n_index] = 0;
        }
        __syncthreads();
        // 一个线程计算一个输出元素
        for (int k_index = 0; k_index < k; k_index++) {
            total += A_sh[m_index * k + k_index] * B_sh[k_index * n + n_index];
        }
         __syncthreads();
    }
    d_C[(M_tile_index * m + m_index) * N + N_tile_index * n + n_index] = total;
  }
  ```

3 一个线程计算多个输出元素
---------

    test3: 25.2284ms speedup: 1.98654, rocblas_ratio: 0.177367, Tflops: 5.44778, Tflops_ratio: 0.235835

  ``` c++
    __global__ void gemm_kernel3(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m, int n, int k) {
        const int reg_size = 4;
        extern __shared__ float sh[];
        float *A_sh = sh;
        float *B_sh = sh + m * k;
        int N_tile_index = blockIdx.x; // tile的列号
        int M_tile_index = blockIdx.y; // tile的行号
        int n_index = threadIdx.x % ((n + reg_size - 1) / reg_size); // tile内的4 * 4列号
        int m_index = threadIdx.x / ((n + reg_size - 1) / reg_size); // tile内的4 * 4行号
        float reg_C[reg_size][reg_size] = {0.0f};
        // float total = 0.0f;
        for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++) {
            // 计算一个线程应该从全局内存读取多个元素
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
  ```

同时在kernel_3.h中, 实验了cuda常用的gemm优化方法, 效果并不理想
---------

### 3.1 计算线程局部矩阵乘时, 每个计算一个k值, 存储A和B元素的寄存器仅仅需要一维

        test3_1: 25.5791ms speedup: 0.986289, rocblas_ratio: 0.174935, Tflops: 5.37309, Tflops_ratio: 0.232601

### 3.2 因为计算局部输出时, 读取一列A和一行B, 所以每个只需记录一个A元素, B还是需要一行

        test3_2: 17.7575ms speedup: 1.44047, rocblas_ratio: 0.251988, Tflops: 7.73976, Tflops_ratio: 0.335055

### 3.3 将A转置然后使用向量加载

        test3_3: 17.6317ms speedup: 1.00714, rocblas_ratio: 0.253787, Tflops: 7.79499, Tflops_ratio: 0.337446

### 3.4 前128线程读取A矩阵块, 后128线程读取B矩阵块

        test3_4: 17.9752ms speedup: 0.980892, rocblas_ratio: 0.248937, Tflops: 7.64604, Tflops_ratio: 0.330998

### 3.5 使用汇编指令辅助实现的共享内存双缓存(不使用汇编效果很差)

        test3_5: 17.7971ms speedup: 1.01, rocblas_ratio: 0.251427, Tflops: 7.72253, Tflops_ratio: 0.334309

### 3.6 使用汇编指令实现寄存器双缓存(没有实现)

        test3_6: 22.8915ms speedup: 0.777455, rocblas_ratio: 0.195474, Tflops: 6.00392, Tflops_ratio: 0.25991

4 填充共享内存, 减少共享内存的冲突 并且使用向量读取
---------
    test4: 14.3921ms speedup: 1.59056, rocblas_ratio: 0.310912, Tflops: 9.5496, Tflops_ratio: 0.413402

5 使用寄存器缓存数据, 之前虽然线程又分块, 但是直接使用共享内存. 现在会先读取到寄存器再完成计算
---------
    test5: 13.553ms speedup: 1.06192, rocblas_ratio: 0.330163, Tflops: 10.1409, Tflops_ratio: 0.438999

6 使用双缓存, 该kernel不使用汇编, 导致性能很差(如果要实现双缓存, 需要编译器配合, 显然hipcc没有配合)
---------
    test6: 164.493ms speedup: 0.0823921, rocblas_ratio: 0.0272028, Tflops: 0.835529, Tflops_ratio: 0.036170

7 线程分块改为 TM = 4, TN = 4, TK = 2
---------
    test7: 12.9557ms speedup: 12.6966, rocblas_ratio: 0.345385, Tflops: 10.6084, Tflops_ratio: 0.459238

### 7.1 使用多维矩阵的共享内存, 性能有所下降

    test7_1: 14.2125ms speedup: 0.91157, rocblas_ratio: 0.314842, Tflops: 9.6703, Tflops_ratio: 0.418628

### 7.2 使用辅助函数去读取元素, 为了方便调整分块, 性能有所下降

    test7_2: 14.1483ms speedup: 1.00454, rocblas_ratio: 0.316271, Tflops: 9.71418, Tflops_ratio: 0.420527

### 7.3 计算布局计算输出时, 将使用寄存器计算局部积的循环完全展开, 分析数据依赖, 将存储指令插入到计算指令中(效果不错)

    test7_3: 11.992ms speedup: 1.17981, rocblas_ratio: 0.373138, Tflops: 11.4608, Tflops_ratio: 0.49614

### 7.4 使用汇编指令支持的共享内存双循环

    test7_4: 11.1994ms speedup: 1.07077, rocblas_ratio: 0.399545, Tflops: 12.2719, Tflops_ratio: 0.531253

### 7.5 使用普通的寄存器双循环(不使用汇编指令, 用于和7.6对比)

    test7_5: 12.0331ms speedup: 0.930723, rocblas_ratio: 0.371866, Tflops: 11.4218, Tflops_ratio: 0.494449

### 7.6 使用汇编指令的寄存器双循环(没有完成, ds_read_d32指令, 某些位置读取不到, 暂时解决不了)

    test7_6: 13.1542ms speedup: 0.914768, rocblas_ratio: 0.340171, Tflops: 10.4483, Tflops_ratio: 0.452307

### 7.7 使用warp分块

    test7_7: 15.514ms speedup: 0.847894, rocblas_ratio: 0.288429, Tflops: 8.85904, Tflops_ratio: 0.383508

### 7.8 7.4 版本修改padding, 进行对比

    test7_8: 11.2175ms speedup: 1.38302, rocblas_ratio: 0.398904, Tflops: 12.2522, Tflops_ratio: 0.5304

8 warp分块研究
---------
### 8.1 warp 8 * 8 的块

    test8_1: 14.2521ms speedup: 1.00704, rocblas_ratio: 0.313967, Tflops: 9.64343, Tflops_ratio: 0.417465

### 8.2 warp 4 * 16 的块

    test8_2: 13.474ms speedup: 1.05774, rocblas_ratio: 0.332097, Tflops: 10.2003, Tflops_ratio: 0.441571

### 8.3 无优化

    test8_3: 22.2234ms speedup: 0.606299, rocblas_ratio: 0.20135, Tflops: 6.18442, Tflops_ratio: 0.267724

### 8.4 使用z型读取, 具体可查看[10]

    test8_4: 13.0805ms speedup: 1.69897, rocblas_ratio: 0.342088, Tflops: 10.5072, Tflops_ratio: 0.454855

9 将4 * 4 的矩阵切成2 * 2的块, 将warp分块也切成4部分, 2 * 2 的块去计算, warp分块的每个部分
---------  
    test9: 14.2987ms speedup: 1.58132, rocblas_ratio: 0.312943, Tflops: 9.61198, Tflops_ratio: 0.416103

### 9.1 其他现成的cuda kernel进行略微改动
    
    test9_1: 16.2988ms speedup: 1.05635, rocblas_ratio: 0.274541, Tflops: 8.43245, Tflops_ratio: 0.365041

10 其他现成的 cuda kernel(可参考[7], 性能很差, 证明cuda的代码不能直接用于hip)
---------
    test10: 186.349ms speedup: 0.0874637, rocblas_ratio: 0.0240123, Tflops: 0.737533, Tflops_ratio: 0.031927

## 汇编指令的使用
 
- 全局内存的读取和写入其实都是异步的. 但是如果只用 hip 全部会变成同步的指令

    举例:

    ``` c++
    global_load<0>(ptr, register);
    ```

    参数1是地址, 参数2是register


    register 要求是 Float4 不同于 float4, 可参考[2]

    注意不要直接全局内存写入到共享内存, 要用寄存器做传递

- 全局内存的同步

    该指令表示等待所有的全局内存读取指令完成, 再继续执行, 可以作为同步指令(注意: 不同访存指令, 乱序发射)

    ``` c++
    vmcnt<0>();
    ```

- 共享内存的同步

    该指令表示等待共享内存的读取

    ``` c++
    lgkmcnt<0>();
    ```

- 共享内存的读取指令有点问题, 后面搞清楚补充


## 参考
   [1] [HIP-Performance-Optmization-on-VEGA64: hip 性能分析](https://github.com/fsword73/HIP-Performance-Optmization-on-VEGA64.git)


   [2] [全局内存读取的解释](https://github.com/RadeonOpenCompute/ROCm/issues/341)


   [3] [内联汇编语言的使用方法](https://github.com/adityaatluri/gemm-vega64.git) 

   [4] [hip gemm的编写方法](https://www.patentguru.com/cn/CN110147248B#:~:text=%E7%AC%AC6%E6%AC%A1%E5%B1%95%E5%BC%80%E7%9A%8424%E4%BD%8D%E7%BD%AEs_waitcnt,vmcnt%20%283%29%E6%8C%87%E4%BB%A4%E8%A1%A8%E7%A4%BA3%E6%9D%A1%E5%85%A8%E5%B1%80%E8%AE%BF%E5%AD%98%E6%8C%87%E4%BB%A4%E8%BF%98%E6%9C%AA%E5%AE%8C%E6%88%90%EF%BC%8C%E5%8D%B3%E6%8C%89%E9%A1%BA%E5%BA%8F%E5%8F%91%E5%B0%84%E4%BA%864%E6%9D%A1%E8%AE%BF%E5%AD%98%E6%8C%87%E4%BB%A4%EF%BC%8C%E5%85%B6%E4%B8%AD%E7%AC%AC1%E6%9D%A1%E8%AE%BF%E5%AD%98%E6%8C%87%E4%BB%A4%E7%BB%93%E6%9E%9C%E5%B7%B2%E7%BB%8F%E8%BF%94%E5%9B%9E%EF%BC%8C%E5%89%A9%E4%B8%8B3%E6%9D%A1%E6%8C%87%E4%BB%A4%E7%BB%93%E6%9E%9C%E8%BF%98%E6%9C%AA%E8%BF%94%E5%9B%9E%EF%BC%8C%E8%BF%99%E6%A0%B7%E5%B0%B1%E6%98%AF%E5%9C%A8%E7%AD%89%E5%BE%85%E7%AC%AC1%E6%9D%A1%E6%8C%87%E4%BB%A4%E7%BB%93%E6%9E%9C%E8%BF%94%E5%9B%9E%E7%84%B6%E5%90%8E%E7%BB%A7%E7%BB%AD%E6%89%A7%E8%A1%8C%EF%BC%8C%E5%90%A6%E5%88%99%E5%B0%B1%E7%AD%89%E5%BE%85%E3%80%82)

   [5] [gemm 优化 使用更多的汇编, 可读性较差](https://github.com/adityaatluri/CopyRXVega64.git)

   [6] [gemm 优化 也用的汇编](https://github.com/fsword73/SGEMM_on_VEGA.git)

   [7] [CUDA SGEMM矩阵乘法优化笔记——从入门到cublas](https://zhuanlan.zhihu.com/p/518857175)

   [8] [如何高效实现矩阵乘？万文长字带你从CUDA初学者的角度入门](https://www.eet-china.com/mp/a178966.html)

   [9] [传统 CUDA GEMM 不完全指北](https://zhuanlan.zhihu.com/p/584236348)

   [10] [A full walk through of the SGEMM implementation](https://github.com/nervanasystems/maxas/wiki/sgemm#reading-from-shared)