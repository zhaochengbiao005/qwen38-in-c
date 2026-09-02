# Windows 平台层映射与工具链决策 (Ticket-02)

研究对象: kimi-k3-in-c 本地副本 `F:\project\测试PPT\kimi-k3-in-c` (include/ + src/ + benchmarks/ 全文 grep 确认, 非凭票证臆测)。

## 0. 事实基线: 蓝本实际用到的平台面

结论先行: 蓝本的平台面比预想小。**没有 mmap/munmap** (sys/mman.h 只为 `madvise(MADV_HUGEPAGE)`)，**没有 pthread** (并行全靠 OpenMP)，**没有 sched_getaffinity/CPU affinity**，**没有 popen**，**没有 sysconf**，**没有 syslog**。真正的 POSIX 原语集中在四个文件:

| 文件 | 用到的非 ISO C 内容 |
|---|---|
| src/io/k3_st.c | `open(O_RDONLY)`, `open(O_RDONLY\|O_DIRECT)`, `pread` (6 处循环读), `close`, `opendir/readdir`<dirent.h> |
| src/io/k3_trunk.c | `open/pread/close` + O_DIRECT, `posix_memalign`, `madvise(MADV_HUGEPAGE)`, `clock_gettime(CLOCK_MONOTONIC)` |
| src/cache/k3_cache.c | `posix_memalign`, `madvise(MADV_HUGEPAGE)`, `clock_gettime(CLOCK_MONOTONIC)` |
| src/cli/k3_run.c | `getrusage(RUSAGE_SELF).ru_maxrss`, 读 `/proc/meminfo` (MemAvailable), `clock_gettime` |
| src/io/k3_portable_io.h | `posix_fadvise(POSIX_FADV_WILLNEED)` 封装, `fcntl(F_NOCACHE)` (Darwin 分支) |
| src/core/k3_ops.c | `<immintrin.h>` (文件中部、`#if defined(__AVX2__)` 包裹), `_mm256_*` FMA intrinsic, `#pragma omp` (5 处, 全部 `parallel for schedule(static) [if (n>64)]`) |
| benchmarks/bench_kernels.c | `clock_gettime` |

C 语言特性核对 (grep 确认): **无 VLA**, **无 `_Generic`**, **无 `__builtin_*`**, **无 `alloca`**；用到 `static inline`、C99 for-loop 声明、`restrict` 未见使用。大部分文件首行 `#define _POSIX_C_SOURCE 200809L` / `_GNU_SOURCE` —— Windows 端口里这两个宏对 MSVC 无害可保留 (建议统一移到平台头)。

## 1. API → Windows 映射清单

| POSIX (出处) | Windows 等价物 | 难度 |
|---|---|---|
| `open(O_RDONLY)` / `close` (k3_st.c:199,347,435-436) | `CreateFileW(GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS)` | 直白 |
| `pread(fd,buf,n,off)` 循环 (k3_st.c:203,221,468,486,503,543; k3_trunk.c:393) | `ReadFile` + `OVERLAPPED.Offset/OffsetHigh` (hEvent=NULL 即同步带偏移读)。注意 OVERLAPPED 同步读要求每次调用构造 OVERLAPPED —— 这正是 pread 语义 | 直白 |
| `O_DIRECT` (k3_st.c:347) | `CreateFile` 加 `FILE_FLAG_NO_BUFFERING`。失败回退逻辑必须保留 (蓝本 dfd=-1 回退普通读已写好) | 有中坑, 见 §2 |
| `posix_fadvise(WILLNEED)` (k3_portable_io.h:41) | **无 direct 等价**。近似: 无缓冲读天然跳过缓存；预取可开低优先级线程先摸一遍。建议实现为 no-op + 注释 | **需重新设计(退化)** |
| `fcntl(F_NOCACHE)` (Darwin 分支) | Win32 不需要 (NO_BUFFERING 在 open 时给定) | n/a |
| `posix_memalign(al, want)` (k3_cache.c:320; k3_trunk.c:380) | `_aligned_malloc(want, al)` / `_aligned_free`；若要对标 2MB "hugepage" 用 `VirtualAlloc(MEM_LARGE_PAGES)` (需 SeLockMemoryPrivilege) | 直白 |
| `madvise(MADV_HUGEPAGE)` (k3_cache.c:325; k3_trunk.c:382) | **无 direct 等价**。`MEM_LARGE_PAGES` 必须在 VirtualAlloc 时决定，不能事后 advise。移植方案: 分配前就按大页策略分配 | **需重新设计(分配路径预处理)** |
| `clock_gettime(CLOCK_MONOTONIC)` (4 处) | UCRT 的 `clock_gettime(CLOCK_MONOTONIC)` 自 VS2015 起可用；最稳妥用 `QueryPerformanceCounter` 封装 (纳秒单调, 无跨 epoch 问题) | 直白 |
| `getrusage(RUSAGE_SELF).ru_maxrss` (k3_run.c:424) | `GetProcessMemoryInfo` 的 `PeakWorkingSetSize` (字节, 直接对应 ru_maxrss 语义) | 直白 |
| `/proc/meminfo` MemAvailable (k3_run.c:436,700) | `GlobalMemoryStatusEx` 的 `ullAvailPhys` | 直白 |
| `opendir` / `readdir` 列 shard 文件 (k3_st.c:368-386) | `FindFirstFileW/FindNextFileW` 模式匹配 `*.safetensors`；排序逻辑 (qsort) 原样保留 | 直白 |
| `qsort` / `fopen/fread/fwrite/fgets/fclose` | ISO C, MSVC 全支持 | 无 |
| OpenMP `#pragma omp parallel for` (k3_ops.c 5 处) | 蓝本只用了 OpenMP 2.0 子集，`schedule(static)`+`if()` 子句都在 2.0 内 → **MSVC /openmp 原生吃得下**; clang-cl 用 /openmp:llvm 更佳 | 直白 |
| `<immintrin.h>` + `_mm256_fmadd_*` | MSVC/clang-cl 同名同头, 见 §4 | 直白 |

## 2. O_DIRECT → FILE_FLAG_NO_BUFFERING 全部约束

蓝本 `K3_ST_ALIGN = 4096` 并对 offset/length/buffer 三向对齐 (k3_st.h:48)。Windows 语义一致但更严格 — 对齐单位是**卷上存储设备的物理扇区大小**, 不一定是逻辑 512/4096:

1. **buffer 地址对齐**: 用户态缓冲区起始地址必须对齐到物理扇区大小 (`_aligned_malloc(n, sector)`)。
2. **文件偏移对齐**: 每次 `ReadFile` 的 OVERLAPPED 偏移必须是物理扇区大小的整数倍。蓝本把读窗口外扩到 4096 边界 (`lo = off & ~4095`)，该逻辑在扇区=4096 的设备上直接等价；对 512 扇区盘它也满足 (4096 是 512 的倍数)。
3. **长度对齐**: 读取字节数必须是物理扇区大小整数倍 (蓝本同样外扩 `hi`)。
4. **扇区大小查询**: `CreateFile(\\.\X:)` 打开卷 + `DeviceIoControl(IOCTL_STORAGE_QUERY_PROPERTY, StorageAccessAlignmentProperty)` 得 `STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR.BytesPerPhysicalSector`；工程上可用 `GetDiskFreeSpaceW` 的 `lpBytesPerSector` (逻辑扇区) 兜底，实际直接固定 4096 对齐已覆盖全部常见设备 (4Kn、512e)，推荐 **sector := max(4096, 查询值)**。
5. 语义差异: Linux O_DIRECT 只"建议"绕过 page cache；Windows NO_BUFFERING 是强制的且读失败返回 ERROR_INVALID_PARAMETER (87)。**打开失败要回退**，与蓝本 dfd=-1 回退路径一致。
6. NO_BUFFERING + RANDOM_ACCESS 可同时给，无副作用。

最小可用示范:

```c
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <malloc.h>
#include <stdio.h>

static HANDLE k3_open_direct(const wchar_t *path) {
    HANDLE h = CreateFileW(path, GENERIC_READ,
        FILE_SHARE_READ, NULL, OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS, NULL);
    if (h == INVALID_HANDLE_VALUE)
        h = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, NULL,
            OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, NULL); /* 回退 */
    return h;
}

/* pread 等价: 同步、无共享文件指针干扰、天然带偏移 */
static long long k3_pread(HANDLE h, void *buf, size_t n, long long off) {
    OVERLAPPED ov = {0};
    ov.Offset = (DWORD)(off & 0xFFFFFFFF);
    ov.OffsetHigh = (DWORD)((unsigned long long)off >> 32);
    DWORD got = 0;
    if (!ReadFile(h, buf, (DWORD)n, &got, &ov)) {
        if (GetLastError() != ERROR_HANDLE_EOF) return -1;
    }
    return got;
}

/* buffer 必须扇区对齐; len=sector 的倍数; off=sector 的倍数 */
int read_demo(const wchar_t *p) {
    const size_t SECTOR = 4096;
    HANDLE h = k3_open_direct(p);
    if (h == INVALID_HANDLE_VALUE) return -1;
    void *buf = _aligned_malloc(SECTOR * 8, SECTOR);   /* 约束 1 */
    long long got = k3_pread(h, buf, SECTOR * 8, 0);   /* 约束 2/3 */
    _aligned_free(buf);
    CloseHandle(h);
    return got > 0 ? 0 : -1;
}
```

## 3. 工具链对比: MSVC cl vs clang-cl vs MinGW

C 特性审计 (基于本蓝本, 非泛论):

| 特性 | 蓝本用否 | MSVC | clang-cl | MinGW-w64 |
|---|---|---|---|---|
| C99 for 内声明 / `static inline` / `long long` | 大量 | ✔ (VS2015+) | ✔ | ✔ |
| 变长数组 VLA | **没有用** | ✘ (MSVC 从不支持) — 无影响 | ✔ | ✔ |
| `_Generic` | **没有用** | ✘ (连 /std:c11 也无) — 无影响 | ✔ | ✔ |
| `restrict` | 未用 | ✔ /std:c11 | ✔ | ✔ |
| `clock_gettime` | 用 4 处 | ✔ UCRT (CLOCK_MONOTONIC 可用) | ✔ | ✔ |
| `<immintrin.h>` AVX2+FMA | 用 | ✔ (/arch:AVX2 含 FMA) | ✔ | ✔ |
| OpenMP | 用 (2.0 子集: parallel for / schedule(static) / if) | ✔ /openmp (实现冻结在 2.0, **恰好够用**) | ✔ /openmp:llvm (libomp) | ✔ -fopenmp |
| **`__AVX2__` 预定义宏** | **守卫 AVX2 路径的关键** | ✘ **MSVC 即使在 /arch:AVX2 下也不定义 `__AVX2__`** —— 这是最大陷阱: 直接用 cl 会静默回退纯标量路径, 性能塌一半以上 | ✔ (clang 语义: -mavx2 定义 `__AVX2__`+`__FMA__`; 用 `/clang:-mavx2` 传) | ✔ |
| `pread`/`O_DIRECT`/`madvise` 等 POSIX | 用 | ✘ 均需 §1 平台层 | ✘ 同左 | 部分 (MinGW 有 POSIX 残层: 无 `O_DIRECT`、无 `madvise`、`pread` 视发行版不齐) —— 光靠 MinGW 解决不了 I/O 主线 |
| 数值对齐敏感性 | 高 (mult-add vs fma, -ffp-contract=off) | 需显式: cl 默认 fp:precise 基本不收缩, 但应加 `/fp:precise` 锁定; 标量路径务必与 AVX2 位一致 (test_ops 断言) | ✔ -ffp-contract=off 原样复用 | ✔ |

**推荐组合: clang-cl (LLVM 18+, 随 VS 安装) + CMake + Ninja + libomp (/openmp:llvm)。** 理由:
1. 唯一同时给出 `__AVX2__/__FMA__` 宏 + 完整 immintrin + 现代 OpenMP + `-ffp-contract=off` 语义 的官方 Windows 工具链, 蓝本 AVX2 宏守卫 **零改动可复用**。
2. MSVC cl 作为备选 CI 目标: 需补一个特性探测头 (见 §5 的 `K3_ENABLE_AVX2` 重定义), OpenMP 2.0 够用, 但要接受次优代码生成。
3. MinGW 排除为主线: 无 O_DIRECT 等价、POSIX 层半截, 反而要多写一套 shim; 且 gcc on Windows 的 SEH/CRT 组合与最终分发兼容性差。

## 4. AVX2/AVX-512 intrinsic 头文件差异

- `<immintrin.h>` 在 MSVC、clang-cl、gcc/MinGW 下都是**同一个头文件名、同一组 intrinsic 名**, 蓝本用到 `_mm256_fmadd_pd/_ps`、`_mm256_cvtepu16_epi32`、`_mm_slli_epi32` 等在三家全部可用。
- 真正的差异是**宏定义时机**: gcc/clang 在 `-mavx2 -mfma` 下定义 `__AVX2__`、`__FMA__`；MSVC cl **任何 /arch 下都不定义** (这是 MSVC 的著名行为)。蓝本 `#if defined(__AVX2__)` 守卫 + 文件中部 `#include <immintrin.h>` 的写法:
  - clang-cl: 零改动直接工作 (文件作用域 include, 不在函数内, 合法)。
  - MSVC: 必须改守卫。建议抽象为自建宏: `#if defined(__AVX2__) || defined(K3_FORCE_AVX2)`，CMake 用 `check_c_compiler_flag`/`try_compile` 探测后定 `K3_ENABLE_AVX2`。注意 MSVC 无运行时/编译时分发 —— 要么总体 /arch:AVX2 要么 dispatch；蓝本是单函数守卫, 建议保持 `/arch:AVX2` 整体编译 + 启动时 cpuid 检查 (clang-cl 同 Ferrari)。
- AVX-512: 蓝本**没有使用**任何 AVX-512 (grep 无 `_mm512`), 无需处理。
- 数值提示: AVX2 路径使用 `_mm256_fmadd_*` 需要 FMA —— clang 下 `__FMA__` 随 `-mavx2 -mfma` 一起出现, 启用在 CMake 里显式 `-mavx2 -mfma` 双给，别只给 avx2 依赖隐式 FMA。

## 5. 平台抽象层接口草案

放在 `src/platform/k3_plat.h` + `k3_plat_win32.c` / `k3_plat_posix.c`。只抽蓝本真正用到的东西, 不做大而全:

```c
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <wchar.h>

/* ---- 文件 I/O: fd 抽象为不透明句柄 ---- */
typedef uintptr_t   k3_file_t;          /* POSIX fd / Win32 HANDLE */
#define K3_FILE_INVALID ((k3_file_t)-1)

/* flags: K3_OPEN_DIRECT = 绕过页缓存 (O_DIRECT / FILE_FLAG_NO_BUFFERING).
   direct 失败由打开方内部回退, 调用者通过 out 参数知道是不是 direct. */
int         k3_file_open(k3_file_t *out, const char *path, int direct);
void        k3_file_close(k3_file_t f);
int64_t     k3_file_pread(k3_file_t f, void *buf, size_t n, int64_t off); /* 语义同 pread, 调用方循环 */
int64_t     k3_file_size(k3_file_t f);    /* 用于 shard 长度校验(替代 fstat) */

/* 扇区/页大小查询: direct 读对齐单位 = max(4096, 查询值) */
uint32_t    k3_sector_size(void);

/* ---- 目录列举 (替代 opendir/readdir+qsort; 返回已排序名单) ---- */
int         k3_list_dir(const char *dir, const char *suffix,
                        char ***names_out, int *count_out); /* 调用方 k3_free */
void        k3_free_names(char **names, int count);

/* ---- 对齐/大页内存 (posix_memalign + madvise(HUGEPAGE) 合一) ---- */
/* hint: K3_MEM_HUGEPAGE 请求大页; 失败内部分配照常, 仅是提示 */
void       *k3_mem_alloc(size_t n, size_t align, int huge_hint);
void        k3_mem_free(void *p);

/* ---- 时间与系统信息 ---- */
uint64_t    k3_now_ns(void);              /* CLOCK_MONOTONIC / QPC, 单调纳秒 */
uint64_t    k3_peak_rss_bytes(void);      /* ru_maxrss / PeakWorkingSetSize */
uint64_t    k3_mem_available_bytes(void); /* /proc/meminfo / GlobalMemoryStatusEx */

/* ---- 预取提示 (posix_fadvise 语义; Win32 先 no-op) ---- */
void        k3_prefetch_hint(k3_file_t f, int64_t off, int64_t len);

/* ---- SIMD 特性 ---- */
#if defined(__AVX2__) || defined(K3_ENABLE_AVX2)
#  define K3_HAVE_AVX2 1
#  include <immintrin.h>
#endif
```

迁移时的替换表 (机械): `posix_memalign` → `k3_mem_alloc`; `madvise(_,HUGEPAGE)` → 传播到 `k3_mem_alloc(huge_hint=1)`; `pread` 循环 → `k3_file_pread` 循环; `getrusage` → `k3_peak_rss_bytes`; `/proc/meminfo` → `k3_mem_available_bytes`; `opendir/readdir` → `k3_list_dir`; `clock_gettime` → `k3_now_ns`。`k3_portable_io.h` 在 Windows 树直接废弃, 由 k3_plat 取代。

## 6. 遗留风险

- CMake Makefile 平台块按 uname 分支 —— Windows 树新建 CMake 为主, Makefile 移植不划算。
- MSVC 备选路径下 AVX2 数值可见性须用 test_ops 双向 (scalar vs AVX2) 断言守住, 蓝本注释已声明两条路径位一致, 这是验收门。
- `MEM_LARGE_PAGES` 需要 SeLockMemoryPrivilege, 普通用户默认没有, 必须设计成优雅降级。