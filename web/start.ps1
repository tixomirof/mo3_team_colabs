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

New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.cache\mo3_ml_hw10_rag" | Out-Null

Write-Host "Starting web container..."
docker compose up -d --build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "OK: http://127.0.0.1:8000"
Write-Host "Stop all (web + llama): .\stop.ps1"
