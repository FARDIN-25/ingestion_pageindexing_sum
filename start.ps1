# PageIndex — start server WITH terminal logging (recommended)
Set-Location $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " PageIndex — logs in THIS window" -ForegroundColor Cyan
Write-Host " Do NOT use: uvicorn --reload" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

function Test-PortFree([int]$Port) {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse("127.0.0.1"), $Port)
    try {
        $listener.Start()
        $listener.Stop()
        return $true
    } catch {
        return $false
    }
}

function Stop-ListenersOnPort([int]$Port) {
    $pids = @(
        Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    )
    foreach ($pid in $pids) {
        if ($pid -and $pid -gt 0) {
            Write-Host "Stopping PID $pid on port $Port..." -ForegroundColor Yellow
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            taskkill /F /PID $pid 2>$null | Out-Null
        }
    }
    Start-Sleep -Seconds 2
}

# Try to free 8000 (all owning PIDs, not just the first)
Stop-ListenersOnPort 8000

$port = 8000
if (-not (Test-PortFree 8000)) {
    Write-Host "Port 8000 still blocked (often a ghost PID after uvicorn --reload)." -ForegroundColor Yellow
    Write-Host "Using port 8001 instead." -ForegroundColor Yellow
    $port = 8001
    $env:PAGEINDEX_PORT = "8001"
} else {
    $env:PAGEINDEX_PORT = "8000"
}

Write-Host ""
Write-Host " Open: http://127.0.0.1:$port" -ForegroundColor Green
Write-Host " Health: http://127.0.0.1:$port/api/health" -ForegroundColor Green
Write-Host ""

.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe run_pageindex_server.py
