#ifndef Q35_PLAT_H
#define Q35_PLAT_H

/* Minimal platform abstraction layer (PAL). All OS-specific bits live here;
   algorithm files contain zero #ifdef _WIN32. */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Read entire file into a heap buffer (caller frees with q35_plat_free). */
void *q35_plat_read_file(const char *path, size_t *out_len);

/* Aligned allocation (alignment must be power of two >= sizeof(void*)). */
void *q35_plat_aligned_alloc(size_t alignment, size_t size);
void  q35_plat_aligned_free(void *p);

/* Large-page allocation for big buffers (KV cache, state). Returns a
 * large-page mapping when the OS supports it and the privilege is present;
 * returns NULL otherwise so callers fall back to their own allocator. */
void *q35_plat_large_alloc(size_t size);
void  q35_plat_large_free(void *p);

void  q35_plat_free(void *p);

/* Monotonic time in seconds. */
double q35_plat_now(void);

/* ---- directory enumeration ---- */
typedef struct Q35PlatDir Q35PlatDir;
Q35PlatDir *q35_plat_dir_open(const char *dir);
const char *q35_plat_dir_next(Q35PlatDir *d);  /* NULL = done */
void q35_plat_dir_close(Q35PlatDir *d);

/* ---- 64-bit file seek/tell (replaces _fseeki64/_ftelli64) ---- */
int     q35_plat_fseek64(FILE *f, int64_t offset, int whence);
int64_t q35_plat_ftell64(FILE *f);

/* ---- read-only memory mapping ---- */
typedef struct Q35PlatMmap Q35PlatMmap;
Q35PlatMmap *q35_plat_mmap_ro(const char *path, const uint8_t **out_base);
void q35_plat_munmap(Q35PlatMmap *m);

/* Prefetch the entire mapped region into RAM (sequential page-in). */
void q35_plat_mmap_prefetch(Q35PlatMmap *m);

/* ---- available physical memory (bytes), 0 = unknown ---- */
uint64_t q35_plat_avail_phys(void);

/* ---- peak resident set size (bytes), 0 = unknown ---- */
size_t q35_plat_peak_rss(void);

/* ---- number of physical cores (not logical/SMT), <=0 = unknown ---- */
int q35_plat_num_phys_cores(void);

/* ---- portable strdup (MSVC _strdup / POSIX strdup) ---- */
char *q35_plat_strdup(const char *s);

/* ---- path join: dst = dir + "/" + name (forward slash; Windows CRT accepts) ---- */
int q35_plat_path_join(char *dst, size_t cap, const char *dir, const char *name);

/* ---- prefetch: page in a mapped region so the first access doesn't stall.
 * base/len describe the region to prefetch. No-op if unsupported. ---- */
void q35_plat_prefetch(const void *base, size_t len);

/* ---- TCP server sockets (loopback HTTP API; Winsock vs POSIX) ---- */
int  q35_plat_sock_init(void);          /* 0 = ok */
int  q35_plat_sock_listen(int port);    /* listening fd, -1 = error */
int  q35_plat_sock_accept(int lfd);     /* connection fd, -1 = error */
int  q35_plat_sock_recv(int fd, void *buf, size_t len);
int  q35_plat_sock_send(int fd, const void *buf, size_t len);
void q35_plat_sock_close(int fd);

#endif
