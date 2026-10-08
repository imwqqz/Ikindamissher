# Trains the two side-by-side models with protected paths, so no run can
# clobber the other's checkpoint or tokenizer (the default data/her_model.pt
# and data/input/bpe.json are the clobber attractors).
#
#   uv run python checks.py            # gate
#   .\scripts\run_models.ps1           # memory model, then style model
#   .\scripts\run_models.ps1 -Only memory
#   .\scripts\run_models.ps1 -SkipChecks
#
# Chat afterwards:
#   uv run chatbot.py --ckpt data\her_model_memory.pt
#   uv run chatbot.py --ckpt data\her_model_style.pt
param(
    [ValidateSet("both", "memory", "style")]
    [string]$Only = "both",
    [int]$Epochs = 5,
    [switch]$SkipChecks
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
foreach ($file in @("data\her_model.pt", "data\input\bpe.json")) {
    if (Test-Path $file) {
        Copy-Item $file "$file.$stamp.bak"
        Write-Host "backed up $file -> $file.$stamp.bak"
    }
}

if (-not $SkipChecks) {
    uv run python checks.py
    if ($LASTEXITCODE -ne 0) { throw "checks.py failed; not training" }
}

if ($Only -in @("both", "memory")) {
    Write-Host "== memory model (full corpus, no validation) =="
    uv run her.py --fresh --epochs $Epochs `
        --ckpt           data\her_model_memory.pt `
        --tokenizer-file data\input\bpe_memory.json
    if ($LASTEXITCODE -ne 0) { throw "memory training failed" }
}

if ($Only -in @("both", "style")) {
    Write-Host "== style model (held-out validation, regularized) =="
    # --no-keep-last overrides a config file's keep_last=true, so best-val wins.
    uv run her.py --fresh --epochs $Epochs `
        --val-fraction 0.1 --patience 8 --dropout 0.1 `
        --persona-prefix --persona-repeat 4 --lr 1.0e-4 `
        --no-keep-last `
        --ckpt           data\her_model_style.pt `
        --tokenizer-file data\input\bpe_style.json
    if ($LASTEXITCODE -ne 0) { throw "style training failed" }
}

Write-Host ""
Write-Host "done. compare on novel prompts:"
Write-Host "  uv run chatbot.py --ckpt data\her_model_memory.pt"
Write-Host "  uv run chatbot.py --ckpt data\her_model_style.pt"
