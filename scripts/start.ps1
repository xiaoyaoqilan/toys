$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "========================================"
Write-Host "  Finance Multi-Source RAG Platform"
Write-Host "========================================" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "[setup] creating virtualenv ..." -ForegroundColor Yellow
    python -m venv .venv
}

$PyExe = ".venv\Scripts\python.exe"

Write-Host "[setup] installing dependencies ..." -ForegroundColor Yellow
& $PyExe -m pip install --upgrade pip | Out-Host
& $PyExe -m pip install -r requirements.txt | Out-Host

$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "."

Write-Host ""
Write-Host "[run] starting FastAPI server on http://127.0.0.1:8000 ..." -ForegroundColor Green
& $PyExe -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
