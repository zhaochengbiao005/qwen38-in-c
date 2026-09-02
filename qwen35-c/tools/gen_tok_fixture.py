import json, sys
sys.stdout.reconfigure(encoding='ascii', errors='backslashreplace')
tok = json.load(open('tokenizer/src/tokenizer.json', encoding='utf-8'))
vocab = tok['model']['vocab']; added = tok.get('added_tokens', [])
bs = list(range(33,127))+list(range(0xA1,0xAD))+list(range(0xAE,0x100)); cs=bs[:]; n=0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256+n); n+=1
c2b = {chr(c):b for b,c in zip(bs,cs)}
def raw(s): return bytes(c2b[c] for c in s)
targets = ['Hello','Ġworld','ĠĠĠĠ','!','a','Ċ','Ġthe']
cn = '中'.encode('utf-8')
picks = []
for p in targets:
    if p in vocab: picks.append(p)
    else: print('MISS', p.encode('unicode_escape'))
cn_tok = next((s for s in vocab if raw(s) == cn), None)
if cn_tok: picks.append(cn_tok)
else: print('MISS zh token')
out = [(vocab[p], raw(p)) for p in picks]
for a in added[:4]:
    out.append((a['id'], a['content'].encode('utf-8')))
lines = ['/* generated from tokenizer.json - do not edit */', 'typedef struct { unsigned int id; const char *hex; } tok_fixture_t;', 'static const tok_fixture_t TOK_FIXTURES[] = {']
for i,b in out:
    lines.append('    { %du, "%s" },' % (i, b.hex()))
lines.append('};')
lines.append('#define TOK_FIXTURES_N %d' % len(out))
open('tests/fixtures/tok_lookup.h','w',newline='\n').write('\n'.join(lines)+'\n')
print('fixture entries:', len(out))

