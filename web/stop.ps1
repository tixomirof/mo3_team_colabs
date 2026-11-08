# Останавливает web-контейнер и LLM (Ollama / llama-server) на хосте.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$Model = if ($env:LLM_MODEL) { $env:LLM_MODEL } else { "qwen3:4b-instruct" }

Write-Host "Stopping web container..."
docker compose down --remove-orphans

Write-Host "Stopping LLM..."
ollama stop $Model 2>$null
Get-Process -Name "llama-server","ollama","ollama app" -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host "Done."
