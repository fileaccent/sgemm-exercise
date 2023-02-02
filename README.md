# 本项目属于 gemm 优化(支持 cuda 和 hip)

## 运行

- hip
``` shell
hipcc main.cpp -fopenmp -lrocblas --std=c++14 && ./a.out
```