#include "q35_plat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <psapi.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

void *q35_plat_read_file(const char *path, size_t *out_len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long end = ftell(f);
    if (end < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    void *buf = malloc((size_t)end + 1u);
    if (!buf) { fclose(f); return NULL; }
    if (end > 0 && fread(buf, 1, (size_t)end, f) != (size_t)end) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    ((char *)buf)[end] = '\0';
    *out_len = (size_t)end;
    return buf;
}

void *q35_plat_aligned_alloc(size_t alignment, size_t size)
{
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void *p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    return p;
#endif
}

/* Large-page allocation for big buffers (KV cache, DeltaNet state).
 * Tries MEM_LARGE_PAGES once (needs SeLockMemoryPrivilege); on success uses
 * large pages for all subsequent calls. Returns NULL when large pages are
 * unavailable — callers fall back to their own allocator (calloc).
 *
 * Never silently substitutes plain VirtualAlloc for the fallback (ticket #31):
 * its demand-zero page faults measured 1.58 -> 2.72 s/tok on the KV cache,
 * and a VirtualAlloc/calloc free-path mix is a segfault waiting to happen. */
void *q35_plat_large_alloc(size_t size)
{
#ifdef _WIN32
    static int lp_state = 0;  /* 0=untested, 1=available, -1=unavailable */
    static SIZE_T lp_min = 0;
    if (lp_state == 0) {
        lp_min = GetLargePageMinimum();
        if (lp_min > 0) {
            void *test = VirtualAlloc(NULL, lp_min,
                                      MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                                      PAGE_READWRITE);
            if (test) { VirtualFree(test, 0, MEM_RELEASE); lp_state = 1; }
            else lp_state = -1;
        } else lp_state = -1;
    }
    if (lp_state == 1 && size >= lp_min) {
        size_t aligned = ((size + lp_min - 1) / lp_min) * lp_min;
        void *p = VirtualAlloc(NULL, aligned,
                               MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                               PAGE_READWRITE);
        if (p) return p;
    }
    return NULL;
#else
    (void)size;
    return NULL;
#endif
}

void q35_plat_large_free(void *p)
{
#ifdef _WIN32
    if (p) VirtualFree(p, 0, MEM_RELEASE);
#else
    free(p);
#endif
}

void q35_plat_aligned_free(void *p)
{
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

void q35_plat_free(void *p) { free(p); }

double q35_plat_now(void)
{
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int init = 0;
    LARGE_INTEGER c;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/* ---- directory enumeration ---- */

struct Q35PlatDir {
#ifdef _WIN32
    HANDLE h;
    WIN32_FIND_DATAA fd;
    int done;
#else
    void *dp;
#endif
};

Q35PlatDir *q35_plat_dir_open(const char *dir)
{
    Q35PlatDir *d = calloc(1, sizeof(*d));
    if (!d) return NULL;
#ifdef _WIN32
    char pat[4096];
    snprintf(pat, sizeof(pat), "%s\\*", dir);
    d->h = FindFirstFileA(pat, &d->fd);
    if (d->h == INVALID_HANDLE_VALUE) { free(d); return NULL; }
#else
    (void)dir; free(d); return NULL;
#endif
    return d;
}

const char *q35_plat_dir_next(Q35PlatDir *d)
{
    if (!d) return NULL;
#ifdef _WIN32
    for (;;) {
        if (d->done) return NULL;
        if (!(d->fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            const char *name = d->fd.cFileName;
            if (!FindNextFileA(d->h, &d->fd)) d->done = 1;
            return name;
        }
        if (!FindNextFileA(d->h, &d->fd)) { d->done = 1; return NULL; }
    }
#else
    return NULL;
#endif
}

void q35_plat_dir_close(Q35PlatDir *d)
{
    if (!d) return;
#ifdef _WIN32
    if (d->h && d->h != INVALID_HANDLE_VALUE) FindClose(d->h);
#endif
    free(d);
}

/* ---- 64-bit file seek/tell ---- */

int q35_plat_fseek64(FILE *f, int64_t offset, int whence)
{
#ifdef _WIN32
    return _fseeki64(f, (__int64)offset, whence);
#else
    return fseeko(f, (long)offset, whence);
#endif
}

int64_t q35_plat_ftell64(FILE *f)
{
#ifdef _WIN32
    return (int64_t)_ftelli64(f);
#else
    return (int64_t)ftello(f);
#endif
}

/* ---- read-only memory mapping ---- */

struct Q35PlatMmap {
#ifdef _WIN32
    HANDLE hf, hm;
    const uint8_t *base;
#else
    void *base;
    size_t size;
#endif
};

Q35PlatMmap *q35_plat_mmap_ro(const char *path, const uint8_t **out_base)
{
#ifdef _WIN32
    Q35PlatMmap *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                        OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    if (m->hf == INVALID_HANDLE_VALUE) { free(m); return NULL; }
    /* Try large pages first (needs SeLockMemoryPrivilege); fall back to
     * 4 KB pages silently. Large pages cut TLB entries ~500x for the 26 GB
     * resident weight set, which is the primary lever for a memory-bound
     * inference engine. */
    m->hm = CreateFileMappingA(m->hf, NULL,
                               PAGE_READONLY | SEC_LARGE_PAGES, 0, 0, NULL);
    if (!m->hm)
        m->hm = CreateFileMappingA(m->hf, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!m->hm) { CloseHandle(m->hf); free(m); return NULL; }
    m->base = (const uint8_t *)MapViewOfFile(m->hm, FILE_MAP_READ, 0, 0, 0);
    if (!m->base) { CloseHandle(m->hm); CloseHandle(m->hf); free(m); return NULL; }
    if (out_base) *out_base = m->base;
    return m;
#else
    (void)path; (void)out_base;
    return NULL;
#endif
}

void q35_plat_munmap(Q35PlatMmap *m)
{
    if (!m) return;
#ifdef _WIN32
    if (m->base) UnmapViewOfFile((void *)m->base);
    if (m->hm) CloseHandle(m->hm);
    if (m->hf && m->hf != INVALID_HANDLE_VALUE) CloseHandle(m->hf);
#endif
    free(m);
}

void q35_plat_mmap_prefetch(Q35PlatMmap *m)
{
    if (!m || !m->base) return;
#ifdef _WIN32
    /* Walk the mapping with VirtualQuery to find total committed size,
     * then prefetch in one shot. */
    const uint8_t *p = m->base;
    MEMORY_BASIC_INFORMATION mbi;
    size_t mapped = 0;
    if (VirtualQuery(p, &mbi, sizeof(mbi))) {
        mapped = mbi.RegionSize;
        while (VirtualQuery(p + mapped, &mbi, sizeof(mbi)) &&
               mbi.State == MEM_COMMIT &&
               mbi.AllocationBase == (void *)p)
            mapped += mbi.RegionSize;
    }
    if (mapped > 0)
        q35_plat_prefetch(p, mapped);
#endif
}

/* ---- available physical memory ---- */

uint64_t q35_plat_avail_phys(void)
{
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms))
        return (uint64_t)ms.ullAvailPhys;
    return 0;
#else
    return 0;
#endif
}

/* ---- prefetch: page in a mapped region ---- */

void q35_plat_prefetch(const void *base, size_t len)
{
    if (!base || !len) return;
#ifdef _WIN32
    /* PrefetchVirtualMemory (Win8+): tells the OS to page in the region
     * with sequential I/O, separating disk faults from compute. */
    typedef BOOL (WINAPI *PFN_PrefetchVirtualMemory)(
        HANDLE, ULONG_PTR, const void *, ULONG);
    static PFN_PrefetchVirtualMemory pfn = NULL;
    static int tried = 0;
    if (!tried) {
        tried = 1;
        HMODULE h = GetModuleHandleA("kernel32.dll");
        if (h) pfn = (PFN_PrefetchVirtualMemory)
            GetProcAddress(h, "PrefetchVirtualMemory");
    }
    if (pfn) {
        /* Win32 entry expects CHAR_COUNT (bytes / 4096) + WIN32_MEMORY_RANGE_ENTRY */
        typedef struct { const void *VirtualAddress; SIZE_T NumberOfBytes; }
            WIN32_MEMORY_RANGE_ENTRY;
        WIN32_MEMORY_RANGE_ENTRY e;
        e.VirtualAddress = base;
        e.NumberOfBytes = len;
        pfn(GetCurrentProcess(), 1, &e, 0);
        return;
    }
    /* fallback: sequential touch to force page-in */
    const volatile uint8_t *p = (const volatile uint8_t *)base;
    size_t step = 4096;
    for (size_t i = 0; i < len; i += step) { (void)p[i]; }
#else
    const volatile uint8_t *p = (const volatile uint8_t *)base;
    size_t step = 4096;
    for (size_t i = 0; i < len; i += step) { (void)p[i]; }
#endif
}

/* ---- peak RSS ---- */

size_t q35_plat_peak_rss(void)
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc))
        return (size_t)pmc.PeakWorkingSetSize;
    return 0;
#else
    return 0;
#endif
}

/* ---- physical core count (SMT-agnostic, for thread auto-tune) ---- */

int q35_plat_num_phys_cores(void)
{
#ifdef _WIN32
    DWORD len = 0;
    int phys = 0;
    GetLogicalProcessorInformation(NULL, &len);
    if (len == 0) return 0;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buf =
        (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc(len);
    if (!buf) return 0;
    if (GetLogicalProcessorInformation(buf, &len)) {
        DWORD n = len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        for (DWORD i = 0; i < n; i++)
            if (buf[i].Relationship == RelationProcessorCore) phys++;
    }
    free(buf);
    return phys;
#elif defined(_SC_NPROCESSORS_ONLN)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 0;
#else
    return 0;
#endif
}

/* ---- strdup / path join ---- */

char *q35_plat_strdup(const char *s)
{
#ifdef _WIN32
    return _strdup(s);
#else
    return strdup(s);
#endif
}

int q35_plat_path_join(char *dst, size_t cap, const char *dir, const char *name)
{
    int n = snprintf(dst, cap, "%s/%s", dir, name);
    return (n >= 0 && (size_t)n < cap) ? 0 : -1;
}

/* ---- TCP server sockets (loopback; Winsock vs POSIX) ---- */

int q35_plat_sock_init(void)
{
#ifdef _WIN32
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0 ? 0 : -1;
#else
    return 0;
#endif
}

int q35_plat_sock_listen(int port)
{
    int fd = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  /* localhost only */
    addr.sin_port = htons((unsigned short)port);
    if (bind(fd, (struct sockaddr *)&addr, sizeof addr) != 0 ||
        listen(fd, 4) != 0) {
        q35_plat_sock_close(fd);
        return -1;
    }
    return fd;
}

int q35_plat_sock_accept(int lfd)
{
#ifdef _WIN32
    SOCKET s = accept(lfd, NULL, NULL);
    if (s == INVALID_SOCKET) return -1;
    /* SOCKET is 64-bit; our fd interface is int. Reject (and close) handles
     * outside int range instead of truncating them into false negatives. */
    if ((uintptr_t)s > (uintptr_t)INT_MAX) {
        closesocket(s);
        return -1;
    }
    return (int)(uintptr_t)s;
#else
    return (int)accept(lfd, NULL, NULL);
#endif
}

int q35_plat_sock_recv(int fd, void *buf, size_t len)
{
    return (int)recv(fd, (char *)buf, (int)len, 0);
}

int q35_plat_sock_send(int fd, const void *buf, size_t len)
{
    return (int)send(fd, (const char *)buf, (int)len, 0);
}

void q35_plat_sock_close(int fd)
{
    if (fd < 0) return;
#ifdef _WIN32
    closesocket(fd);
#else
    close(fd);
#endif
}
