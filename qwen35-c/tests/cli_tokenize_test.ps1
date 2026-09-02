# cli test: feeds each fixture through the qwen35 tokenize CLI and compares ids
$tok = "tokenizer/tokenizer.bin"
$exe = "build/qwen35.exe"
$lines = [System.IO.File]::ReadAllText("tests/fixtures/encode_fixtures.txt") -split "`n"
$pass = 0; $fail = 0
foreach ($ln in $lines[0..19]) {   # curated first 20 (incl special tokens + NFC)
    if (-not $ln) { continue }
    $parts = $ln -split "`t"
    if ($parts.Count -ne 2) { continue }
    $b = New-Object byte[] ($parts[0].Length / 2)
    for ($i = 0; $i -lt $b.Length; $i++) { $b[$i] = [Convert]::ToByte($parts[0].Substring($i*2, 2), 16) }
    $text = [System.Text.Encoding]::UTF8.GetString($b)
    $out = & $exe tokenize --tokenizer $tok --prompt $text 2>$null
    if ($LASTEXITCODE -ne 0) { $fail++; continue }
    $want = $parts[1] -replace ",", " "
    if ($out.Trim() -eq $want.Trim()) { $pass++ } else { $fail++; Write-Output "MISMATCH: '$($text.Substring(0,[Math]::Min(20,$text.Length)))'" }
}
Write-Output "cli fixtures: $pass pass, $fail fail"
if ($fail -gt 0) { exit 1 }
