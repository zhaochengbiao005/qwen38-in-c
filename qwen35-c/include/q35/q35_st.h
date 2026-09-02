#ifndef Q35_ST_H
#define Q35_ST_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    Q35_DTYPE_F8_E4M3 = 0,
    Q35_DTYPE_BF16,
    Q35_DTYPE_F32,
    Q35_DTYPE_F16,
    Q35_DTYPE_I64,
    Q35_DTYPE_UNKNOWN = -1
} Q35Dtype;

size_t q35_dtype_size(Q35Dtype t);
const char *q35_dtype_name(Q35Dtype t);

typedef struct {
    char *name;
    Q35Dtype dtype;
    uint32_t ndim;
    uint32_t shape[8];
    int file_idx;         /* shard file index */
    uint64_t data_off;    /* absolute file offset of tensor bytes */
    uint64_t data_len;
} Q35Tensor;

typedef struct Q35St Q35St;

typedef enum {
    Q35_ST_OK = 0,
    Q35_ST_ERR_OPEN,
    Q35_ST_ERR_HEADER,     /* bad/truncated header */
    Q35_ST_ERR_DTYPE,
    Q35_ST_ERR_RANGE,      /* data offsets out of file bounds */
    Q35_ST_ERR_IO
} Q35StErr;

/* Predicate: does a filename look like a safetensors shard (not a json sidecar)?
 * Used by both q35_st_open_dir and q35_model's mmap layer so the file
 * ordering stays identical between index and mmap paths. */
int q35_st_is_shard_name(const char *name);

/* open a directory of shard files (*.safetensors / *.safetensors_). */
Q35St *q35_st_open_dir(const char *dir, Q35StErr *err);
/* open a single safetensors file. */
Q35St *q35_st_open_file(const char *path, Q35StErr *err);
void   q35_st_close(Q35St *s);

const Q35Tensor *q35_st_find(const Q35St *s, const char *name);
uint32_t q35_st_tensor_count(const Q35St *s);
const Q35Tensor *q35_st_tensor_at(const Q35St *s, uint32_t i);

/* read tensor bytes into caller buffer (must be >= data_len). 1 ok / 0 io err */
int q35_st_read(const Q35St *s, const Q35Tensor *t, void *dst);

const char *q35_st_err_str(Q35StErr e);

#endif

