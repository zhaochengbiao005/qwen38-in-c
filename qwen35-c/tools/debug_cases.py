from tokenizers import Tokenizer
import sys
sys.stdout.reconfigure(encoding="ascii", errors="backslashreplace")
tok = Tokenizer.from_file("tokenizer/src/tokenizer.json")
cases = ["é cafe é", "¡$%&/()=?¿*¨Ç][ºª", "ΑΒΓ αβγ δέκα Привет мир"]
for c in cases:
    enc = tok.encode(c, add_special_tokens=False)
    print("TEXT", c.encode("unicode_escape"))
    print("ids ", enc.ids)
    print("toks", [t.encode("unicode_escape") for t in enc.tokens])
