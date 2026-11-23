# Поднимает LLM (Ollama) на хосте + web-контейнер.
# При остановке используй .\stop.ps1 — погасит и контейнер, и LLM.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Model = if ($env:LLM_MODEL) { $env:LLM_MODEL } else { "qwen3:4b-instruct" }

function Test-OllamaUp {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 2
        return $r.StatusCode -eq 200
    } catch {
        return $false
    }
}

Write-Host "Starting Ollama (llama server) on host..."
if (-not (Test-OllamaUp)) {
    $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollamaCmd) {
        $candidates = @(
            "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
            "$env:ProgramFiles\Ollama\ollama.exe"
        )
        foreach ($c in $candidates) {
            if (Test-Path $c) { $ollamaCmd = $c; break }
        }
    } else {
        $ollamaCmd = $ollamaCmd.Source
    }
    if (-not $ollamaCmd) {
        throw "Ollama not found. Install from https://ollama.com"
    }
    Start-Process -FilePath $ollamaCmd -ArgumentList "serve" -WindowStyle Hidden
    $deadline = (Get-Date).AddSeconds(45)
    while (-not (Test-OllamaUp)) {
        if ((Get-Date) -gt $deadline) { throw "Ollama did not start on :11434" }
        Start-Sleep -Seconds 1
    }
}
Write-Host "Ollama is up."

Write-Host "Ensuring model $Model ..."
& ollama pull $Model | Out-Host
# Снимок LightRAG собран при num_ctx=16384. Если модель уже загружена с другим окном, выгружаем её.
try { & ollama stop $Model 2>&1 | Out-Null } catch { }


New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.cache\mo3_ml_hw10_rag" | Out-Null
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.cache\mo3-hw11" | Out-Null

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $env:HW11_REPO_DIR) { $env:HW11_REPO_DIR = $RepoRoot }
$Bundle = if ($env:HW11_INDEX_BUNDLE) { $env:HW11_INDEX_BUNDLE } else { Join-Path $RepoRoot "data\ML-HW11\indices.zip" }
if (Test-Path $Bundle) {
    Write-Host "HW11 indices: $Bundle"
} else {
    Write-Host "WARNING: не найден $Bundle"
    Write-Host "Положите готовый архив в data\ML-HW11\indices.zip или задайте HW11_INDEX_BUNDLE."
    Write-Host "Без него в чате останется только базовая RAG 10-й лабы, LongRAG/LightRAG не запустятся."
}

Write-Host "Starting web container..."
docker compose up -d --build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "OK: http://127.0.0.1:8000"
Write-Host "Stop all (web + llama): .\stop.ps1"
