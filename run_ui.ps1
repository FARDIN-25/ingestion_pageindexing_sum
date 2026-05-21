# Run PageIndex UI with full terminal logging
Set-Location $PSScriptRoot
.\.venv\Scripts\Activate.ps1

$env:PYTHONUNBUFFERED = "1"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " PageIndex UI  ->  http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host " Logs: terminal + logs\app.log" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# --reload hides worker logs on Windows; use without reload for reliable logging
.\.venv\Scripts\uvicorn.exe app:app `
  --host 127.0.0.1 `
  --port 8000 `
  --log-level info `
  --log-config logging_config.yaml `
  --access-log
