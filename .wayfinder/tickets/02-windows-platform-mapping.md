# Ticket-02: Windows 平台层映射与工具链决策（research, AFK）

Label: wayfinder:research
Claim: 已自动声明（charting 会话分发）
Blocked by: 无（frontier）

## Question

把 kimi-k3-in-c 依赖的 Linux 原语映射到 Windows：pread → ReadFile/FILE_FLAG_RANDOM_ACCESS 或可写层封装；O_DIRECT → FILE_FLAG_NO_BUFFERING 的对齐约束（sector 对齐 buffer/offset/长度）；mmap → CreateFileMapping/MapViewOfFile 在流式场景是否值得；线程亲和/pinning API；CMAKE 与 MSVC 对 C99 的支持边界，是否用 clang-cl；AVX2 intrinsics 在两个编译器下的一致性。产出：平台抽象层接口草案 + 工具链结论（.wayfinder/research/windows-platform.md）。

## Resolution
研究文档: .wayfinder/research/windows-platform.md。要点:

- **平台面小于预想**: 实测只用 open/pread/close + O_DIRECT + posix_memalign + madvise(HUGEPAGE) + clock_gettime + getrusage + /proc/meminfo + opendir/readdir + OpenMP(2.0 子集) + immintrin AVX2/FMA。**无 mmap、无 pthread、无 sched_affinity、无 popen**；语言特性上无 VLA/_Generic/__builtin。
- **映射**: pread→ReadFile+OVERLAPPED 同步带偏移读; open→CreateFileW; O_DIRECT→FILE_FLAG_NO_BUFFERING(4096 三向对齐: buffer/_aligned_malloc、offset、length；扇区大小经 IOCTL_STORAGE_QUERY_PROPERTY 查询，工程取 max(4096,查询值)，失败必须回退，与蓝本 dfd=-1 路径一致); madvise(HUGEPAGE)→VirtualAlloc(MEM_LARGE_PAGES) 须在分配时决定，属"无 direct 等价，改分配路径"; posix_fadvise→no-op; getrusage→GetProcessMemoryInfo; /proc/meminfo→GlobalMemoryStatusEx。
- **工具链: 推荐 clang-cl + CMake + Ninja + /openmp:llvm**。核心原因: MSVC 在 /arch:AVX2 下也**不定义 `__AVX2__` 宏**，蓝本守卫会静默退化为标量路径; clang-cl 下 `-mavx2 -mfma` 宏守卫零改动复用。MSVC 作备选(需补 K3_ENABLE_AVX2 探测宏，OpenMP 2.0 恰好够用); MinGW 排除为主线(无 O_DIRECT、半截 POSIX 层)。蓝本无 AVX-512; 数值一致性靠 -ffp-contract=off / /fp:precise + test_ops 位一致断言兜底。
- **PAL 草案**: src/platform/k3_plat.h —— k3_file_open/pread/close/size、k3_sector_size、k3_list_dir(已排序)、k3_mem_alloc(大页 hint 折叠 madvise)、k3_now_ns/k3_peak_rss_bytes/k3_mem_available_bytes、k3_prefetch_hint(no-op)。k3_portable_io.h 在 Windows 树废弃。