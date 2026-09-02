#ifndef Q35_JSON_H
#define Q35_JSON_H

#include <stddef.h>
#include <stdint.h>

/* minimal DOM JSON parser, enough for config.json / safetensors headers.
   DOM lives in one arena; json_free releases everything. */

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } JType;

typedef struct JVal JVal;
typedef struct JPair JPair;
struct JPair { char *key; JVal *val; };

typedef struct {
    JVal **items;
    size_t n;
} JArr;

typedef struct {
    JPair *pairs;
    size_t n;
} JObj;

struct JVal {
    JType t;
    union {
        int b;
        double num;
        char *str;      /* decoded utf-8 */
        JArr arr;
        JObj obj;
    } v;
};

typedef struct Q35Json Q35Json; /* arena handle */

/* parse text (not NUL-terminated needed, uses len). NULL on syntax error. */
Q35Json *q35_json_parse(const char *text, size_t len);
void     q35_json_free(Q35Json *j);

JVal       *q35_json_root(Q35Json *j);
const JVal *q35_obj_get(const JVal *obj, const char *key); /* NULL if absent */
/* typed helpers: return 0/"" if absent or wrong type */
double      q35_jnum(const JVal *obj, const char *key, int *present);
const char *q35_jstr(const JVal *obj, const char *key);
int         q35_jint_present(const JVal *obj, const char *key, long *out);

#endif
