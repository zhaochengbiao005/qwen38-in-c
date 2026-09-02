# #30: --threads CLI 参数接通

Label: wayfinder:task
Claim: 未认领
Blocked by: —

## Resolution

`--threads N` 参数接通 `omp_set_num_threads(N)`。不给参数时用 `OMP_NUM_THREADS` 环境变量（默认行为不变），两者都没给时用全部核心。启动时打印实际线程数到 stderr。验证：`--threads 4` prefill 19s、`--threads 12` prefill 8.8s、ctest 9/9 无回归。

Status: CLOSED
