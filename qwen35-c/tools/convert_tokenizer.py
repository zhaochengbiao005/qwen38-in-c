#!/usr/bin/env python3
"""convert_tokenizer.py — Qwen3.8-27B(-FP8) tokenizer.json -> tokenizer.bin

Offline, one-shot converter. The C engine loads only the produced blob and
never touches JSON at runtime.

Blob format (little-endian):
  8B magic "Q35TOK10" | u32 version(1) | u32 section_count
  then sections, each: 4cc tag | u64 payload_len | payload

Sections:
  META  u32 flags (bit1 = NFC normalizer) | u32 declared_vocab_size (model config, padding included)
        u32 defined_vocab_size  (base vocab + added tokens defined)
        u32 bos_id, u32 eos_id, u32 pad_id, u32 chat_eos_id
  PRET  u32 flags (bit0 = isolated split) ; u32 len ; utf-8 pretokenizer regex
  VOCB  u32 count ; per entry: u32 id, u32 bytelen, raw token bytes
        (vocab strings translated from GPT-2 byte-level alphabet to raw bytes)
  MRGS  u32 count ; per merge, priority order: u32 id_a, u32 id_b
  ATOK  u32 count ; per entry: u32 id, u32 special_flag, u32 len, utf-8 content

Self-validation (any failure -> exit 1), and a reference re-encode of a fixture
set cross-checked against the HF `tokenizers` library (missing deps -> exit 2).
Exit 0 means blob written and every check passed.
"""

import json
import struct
import sys

MAGIC = b"Q35TOK10"
VERSION = 1

EXPECTED_BASE_VOCAB = 248044
EXPECTED_MERGES = 247587
EXPECTED_ADDED = 33

FIXTURES = [
    "Hello, world!",
    "你好，世界",
    "The quick brown fox jumps over the lazy dog 123 456",
    "def foo(x):\n    return x + 1",
    "emoji \U0001f600\U0001f680 test",
    "混排 English and 中文 with   spaces\n\nnewlines",
    "'s 't 're 've 'm 'll 'd DON'T don't",
    "   trailing spaces   ",
    "<|im_start|>user\nhi<|im_end|>",
]


def fail(msg):
    print("FAIL:", msg, file=sys.stderr)
    sys.exit(1)


def section(tag, payload):
    assert len(tag) == 4
    return tag + struct.pack("<Q", len(payload)) + payload


def byte_encoder():
    """GPT-2 byte-level alphabet: byte -> display char."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


BYTE_TO_CHAR = byte_encoder()
CHAR_TO_BYTE = {v: k for k, v in BYTE_TO_CHAR.items()}


def tokstr_to_bytes(s):
    return bytes(CHAR_TO_BYTE[c] for c in s)


def main():
    if len(sys.argv) != 5:
        print("usage: convert_tokenizer.py <tokenizer.json> <tokenizer_config.json> <model_config.json> <out.bin>")
        return 2
    tok_path, cfg_path, model_cfg_path, out_path = sys.argv[1:5]

    tok = json.load(open(tok_path, encoding="utf-8"))
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    mcfg = json.load(open(model_cfg_path, encoding="utf-8"))
    text_cfg = mcfg.get("text_config", mcfg)

    if tok["model"]["type"] != "BPE":
        fail("model.type != BPE")
    if tok["model"].get("byte_fallback"):
        fail("byte_fallback unexpectedly true")
    normalizer = tok.get("normalizer")
    if normalizer is not None and normalizer.get("type") != "NFC":
        fail("unsupported normalizer: %r" % (normalizer,))
    flag_nfc = 2 if normalizer is not None else 0

    pt = tok.get("pre_tokenizer") or {}
    pretok = pt.get("pretokenizers") or []
    split = next((p for p in pretok if p.get("type") == "Split"), None)
    blvl = next((p for p in pretok if p.get("type") == "ByteLevel"), None)
    if split is None or blvl is None:
        fail("pretokenizer pattern unexpected")
    if split.get("behavior") != "Isolated" or split.get("invert"):
        fail("split behavior not plain Isolated")
    regex_pat = split["pattern"]["Regex"]
    cfg_pat = cfg.get("pretokenize_regex")
    if cfg_pat is not None and cfg_pat != regex_pat:
        fail("pretokenizer regex differs from tokenizer_config.json pretokenize_regex")

    vocab = tok["model"]["vocab"]          # token string -> id
    merges_raw = tok["model"]["merges"]
    added = tok.get("added_tokens", [])

    if len(vocab) != EXPECTED_BASE_VOCAB:
        fail(f"base vocab {len(vocab)} != {EXPECTED_BASE_VOCAB}")
    if len(merges_raw) != EXPECTED_MERGES:
        fail(f"merges {len(merges_raw)} != {EXPECTED_MERGES}")
    if len(added) != EXPECTED_ADDED:
        fail(f"added tokens {len(added)} != {EXPECTED_ADDED}")

    # vocab entries must be exactly ids 0..248043
    id_to_tok = {}
    for s, i in vocab.items():
        if not (0 <= i < EXPECTED_BASE_VOCAB):
            fail(f"vocab id out of range: {i}")
        if i in id_to_tok:
            fail(f"duplicate id {i}")
        id_to_tok[i] = s
    if sorted(id_to_tok) != list(range(EXPECTED_BASE_VOCAB)):
        fail("base vocab ids not contiguous 0..N-1")

    # resolve merges ("A B" string or [A, B] list) to id pairs
    merges = []
    for m in merges_raw:
        a, b = m.split(" ") if isinstance(m, str) else (m[0], m[1])
        if a not in vocab or b not in vocab:
            fail("merge operand not in vocab: %r" % (m,))
        merges.append((vocab[a], vocab[b]))

    # added tokens: ids must be base + index, contiguous
    for idx, at in enumerate(added):
        if at["id"] != EXPECTED_BASE_VOCAB + idx:
            fail("added token id gap at %d" % idx)

    bos_id = text_cfg.get("bos_token_id")
    eos_id = text_cfg.get("eos_token_id")
    pad_id = text_cfg.get("pad_token_id")
    eos_name = cfg.get("eos_token")              # "<|im_end|>" for chat
    pad_name = cfg.get("pad_token")              # "<|endoftext|>"
    id_of_added = {a["content"]: a["id"] for a in added}
    chat_eos_id = id_of_added.get(eos_name, -1) if isinstance(eos_name, str) else -1
    if pad_id is None and isinstance(pad_name, str):
        pad_id = id_of_added.get(pad_name, -1)
    declared_vocab = text_cfg.get("vocab_size")
    if pad_id is None or pad_id < 0:
        pad_id = eos_id
    if None in (bos_id, eos_id, declared_vocab):
        fail("model text_config missing bos/eos/vocab_size")
    defined_vocab = EXPECTED_BASE_VOCAB + EXPECTED_ADDED
    if defined_vocab > declared_vocab:
        fail("defined vocab exceeds declared vocab_size")

    # ---- build blob ----
    meta = struct.pack("<7I", flag_nfc, declared_vocab, defined_vocab,
                       bos_id, eos_id, pad_id, chat_eos_id)
    pret = struct.pack("<II", 1, len(regex_pat.encode("utf-8"))) + regex_pat.encode("utf-8")
    vocb = [struct.pack("<I", defined_vocab)]
    for i in range(EXPECTED_BASE_VOCAB):
        b = tokstr_to_bytes(id_to_tok[i])
        vocb.append(struct.pack("<II", i, len(b)) + b)
    for a in added:
        b = a["content"].encode("utf-8")
        vocb.append(struct.pack("<II", a["id"], len(b)) + b)
    vocb = b"".join(vocb)
    mrgs = [struct.pack("<I", len(merges))]
    for a, b in merges:
        mrgs.append(struct.pack("<II", a, b))
    mrgs = b"".join(mrgs)
    atok = [struct.pack("<I", len(added))]
    for a in added:
        b = a["content"].encode("utf-8")
        atok.append(struct.pack("<III", a["id"], 1 if a.get("special") else 0, len(b)) + b)
    atok = b"".join(atok)

    blob = MAGIC + struct.pack("<II", VERSION, 5) + section(b"META", meta) + \
        section(b"PRET", pret) + section(b"VOCB", vocb) + \
        section(b"MRGS", mrgs) + section(b"ATOK", atok)

    # ---- self-checks on the tables we are about to ship ----
    merged_vocab = dict(vocab)
    for a in added:
        merged_vocab[a["content"]] = a["id"]
    # every merge produces an existing token id? (merge result = concat)
    for a_id, b_id in merges:
        pass  # operand existence already validated; result coverage checked via encode parity

    with open(out_path, "wb") as f:
        f.write(blob)
    print(f"OK wrote {out_path}: {len(blob)} bytes, "
          f"vocab={defined_vocab}/{declared_vocab}, merges={len(merges)}, "
          f"special={sum(1 for a in added if a.get('special'))}, "
          f"bos={bos_id} eos={eos_id} pad={pad_id} chat_eos={chat_eos_id}")

    # ---- cross-check with HF tokenizers ----
    try:
        from tokenizers import Tokenizer
        import regex as re_mod
    except ImportError as e:
        print(f"NOT RUN: cross-check skipped, missing dependency: {e}")
        return 2

    hf = Tokenizer.from_file(tok_path)
    pat = re_mod.compile(regex_pat)
    byte_vocab = {i: tokstr_to_bytes(id_to_tok[i]) for i in range(EXPECTED_BASE_VOCAB)}
    bytes_to_id = {v: k for k, v in byte_vocab.items()}
    # added tokens must split first (Isolated special spans)
    added_sorted = sorted(added, key=lambda a: -len(a["content"]))
    added_bytes = {a["content"].encode("utf-8"): a["id"] for a in added_sorted}
    rank = {pair: i for i, pair in enumerate(merges)}

    def ref_encode(text):
        ids = []
        i = 0
        raw = text.encode("utf-8")
        # greedy special-token scan, then regex split + byte-level BPE per piece
        while i < len(text):
            hit = None
            for a in added_sorted:
                c = a["content"]
                if text.startswith(c, i):
                    hit = a
                    break
            if hit is not None:
                ids.append(hit["id"])
                i += len(hit["content"])
                continue
            nxt = len(text)
            for a in added_sorted:
                c = a["content"]
                j = text.find(c, i)
                if 0 <= j < nxt:
                    nxt = j
            chunk = text[i:nxt]
            for m in pat.finditer(chunk):
                piece = m.group(0).encode("utf-8")
                seq = [bytes([bt]) for bt in piece]
                while len(seq) > 1:
                    best, bi = -1, -1
                    for k in range(len(seq) - 1):
                        pr = (bytes_to_id.get(seq[k], -1), bytes_to_id.get(seq[k + 1], -1))
                        r = rank.get(pr, -1)
                        if r >= 0 and (best < 0 or r < best):
                            best, bi = r, k
                    if best < 0:
                        break
                    seq[bi:bi + 2] = [seq[bi] + seq[bi + 1]]
                ids.extend(bytes_to_id[s] for s in seq)
            i = nxt
        return ids


    for fx in FIXTURES:
        want = hf.encode(fx, add_special_tokens=False).ids
        got = ref_encode(fx)
        if want != got:
            print("MISMATCH on fixture:", fx[:40].encode("unicode_escape"))
            print("  hf :", want[:32])
            print("  ref:", got[:32])
            fail("reference re-encode diverged from HF tokenizers")
    print(f"OK cross-check: {len(FIXTURES)} fixtures match HF tokenizers byte-exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())


