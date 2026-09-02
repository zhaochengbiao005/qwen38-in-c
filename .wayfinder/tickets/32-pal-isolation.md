# #32: PAL 隔离——算法文件零 #ifdef _WIN32

Label: wayfinder:task
Claim: 海鸥
Blocked by: —

## Problem

spec.md 决策 10 / US-13 要求"算法文件零 `#ifdef _WIN32`"，但两个算法文件
违反此规则，直接调 Win32 API：

- `q35_st.c`：`#include <windows.h>`、`WIN32_FIND_DATAA`、`FindFirstFileA`/
  `Next`/`Close`（目录枚举）、`_fseeki64`/`_ftelli64`（64-bit seek）。
- `q35_model.c`：`#include <windows.h>`、`HANDLE`/`CreateFileA`/
  `CreateFileMappingA`/`MapViewOfFile`/`UnmapViewOfFile`/`CloseHandle`
  （只读 mmap）、`WIN32_FIND_DATAA`/`FindFirstFileA`/`Next`/`Close`
  （目录枚举）、`GlobalMemoryStatusEx`/`MEMORYSTATUSEX`（可用内存查询）。

## Plan

在 PAL 中新增以下接口，实现全部放 `q35_plat_win.c`：

```c
/* 目录枚举 */
typedef struct Q35PlatDir Q35PlatDir;
Q35PlatDir *q35_plat_dir_open(const char *dir);
const char *q35_plat_dir_next(Q35PlatDir *d);  /* NULL = done */
void q35_plat_dir_close(Q35PlatDir *d);

/* 64-bit 文件 seek/tell（替代 _fseeki64/_ftelli64） */
int    q35_plat_fseek64(FILE *f, int64_t offset, int whence);
int64_t q35_plat_ftell64(FILE *f);

/* 只读内存映射 */
typedef struct Q35PlatMmap Q35PlatMmap;
Q35PlatMmap *q35_plat_mmap_ro(const char *path, const uint8_t **out_base);
void q35_plat_munmap(Q35PlatMmap *m);

/* 可用物理内存（字节），0 = 查询失败 */
uint64_t q35_plat_avail_phys(void);
```

改造点：
1. `q35_st.c`：删 `#ifdef _WIN32`/`<windows.h>`；`q35_st_open_dir` 用
   `q35_plat_dir_*` 替换 `FindFirstFile` 系列；`index_file` 和
   `q35_st_read` 用 `q35_plat_fseek64`/`ftell64` 替换 `_fseeki64`/`_ftelli64`。
2. `q35_model.c`：删 `#ifdef _WIN32`/`<windows.h>`；`map_files` 用
   `q35_plat_dir_*` + `q35_plat_mmap_ro` 替换 Win32 mmap 调用；
   `unmap_files` 用 `q35_plat_munmap`；OOM guard 用
   `q35_plat_avail_phys` 替换 `GlobalMemoryStatusEx`。

`_strdup` 是 MSVC 专有但不在 `#ifdef _WIN32` 块里，不违反"零 #ifdef"规则，
暂不改（后续可加 `q35_plat_strdup`）。

## Acceptance

- `grep -n '#ifdef _WIN32\|#include <windows.h>' q35_st.c q35_model.c` 无输出。
- `grep -n 'FindFirst\|MapView\|CreateFile\|GlobalMemory\|_fseeki64\|_ftelli64'`
  q35_st.c q35_model.c` 无输出。
- 全量构建 + ctest 12/12 绿。

Status: CLOSED
