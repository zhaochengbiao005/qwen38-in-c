import sys, random
from tokenizers import Tokenizer
sys.stdout.reconfigure(encoding="ascii", errors="backslashreplace")
tok = Tokenizer.from_file("tokenizer/src/tokenizer.json")
rng = random.Random(0xBEE5)

# curated edge fixtures (same as before, kept for traceability)
FIXTURES = [
    "Hello, world!", "你好，世界",
    "The quick brown fox jumps over the lazy dog 123 456",
    "def foo(x):\n    return x + 1",
    "emoji \U0001f600\U0001f680 test \U0001f1e8\U0001f1f3",
    "混排 English and 中文 with   spaces\n\nnewlines",
    "'s 't 're 've 'm 'll 'd DON'T don't CAN'T",
    "   trailing spaces   ",
    "<|im_start|>user\nhi<|im_end|>",
    "é cafe é", "나 한",
    "1234567890 3.14159 0x1F", "¡$%&/()=?¿*¨Ç][ºª",
    "line1\r\nline2\rline3\nline4", "🄰🄱🄂 symbols ㌀ 𠜎",
    "https://example.com/path?q=水&x=1#frag",
    "if(a==b){c++;}\t/* comment */", "ΑΒΓ αβγ δέκα Привет мир",
    "0b1101 1e-10 0o777 .5 5.", "んにちは カタカナ ひらがな",
]

# randomized fuzz corpus across unicode planes
POOLS = [
    range(0x20, 0x7f),           # ascii
    range(0xa0, 0x2500),         # latin-1 + extended
    range(0x3000, 0x30ff),       # CJK symbols, kana
    range(0x4e00, 0x4f00),       # CJK
    range(0x1100, 0x1200),       # Hangul jamo
    range(0xac00, 0xad00),       # Hangul syllables
    range(0x1f600, 0x1f650),     # emoji
    range(0x1d400, 0x1d450),     # math alphanumerics
    [0x301, 0x302, 0x308, 0x20d0, 0x20e1, 0xfe0f],  # combining
]
WHITES = [' ', '\t', '\n', '\r', '\v', '\f', ' ', ' ', ' ']
def rc():
    if rng.random() < 0.08: return rng.choice(WHITES)
    p = rng.choice(POOLS)
    return chr(rng.choice(list(p) if isinstance(p, range) else p))

for _ in range(400):
    L = rng.randint(1, 40)
    FIXTURES.append(''.join(rc() for _ in range(L)))
# a few long stress lines
for _ in range(20):
    FIXTURES.append(''.join(rc() for _ in range(400)) + "<|im_end|> tail")

lines = []; total = 0
for fx in FIXTURES:
    ids = tok.encode(fx, add_special_tokens=False).ids
    lines.append(fx.encode("utf-8").hex() + "\t" + ",".join(map(str, ids)))
    total += len(ids)
open("tests/fixtures/encode_fixtures.txt", "w", newline="\n").write("\n".join(lines) + "\n")
print("fixtures:", len(FIXTURES), "total ids:", total)
