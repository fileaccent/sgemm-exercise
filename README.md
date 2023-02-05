# 本项目属于 gemm 优化(支持 cuda 和 hip)

## 运行

- hip
``` shell
hipcc main.cpp -fopenmp -lrocblas --amdgpu-target=gfx908  --std=c++14 && ./a.out
```

- cuda
``` shell
nvcc main.cpp -lcublas && ./a.out
```

- 性能
  - 在 MI100 中大概只有rocblas 30% 的性能, 希望后面能通过其他方式优化
  - 在性能在 70% 以上再给出分析(性能差给出分析也没啥用)
  - cuda 部分 属于 hip 的代码转过去的, 并未对性能作特殊优化
  - 综合最快的kernel函数为7.3