# Sign Bridge - reliable one-click start (kills stale servers, uses .venv, opens app)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  Write-Host "Missing .venv — creating it..."
  $pyLauncher = $null
  foreach ($c in @("py -3.13", "py -3.12", "py -3.11", "python")) {
    try {
      $ver = Invoke-Expression "$c -c `"import sys; print(sys.version)`"" 2>$null
      if ($ver) { $pyLauncher = $c; break }
    } catch {}
  }
  if (-not $pyLauncher) {
    Write-Host "No Python found. Install Python 3.11+ then retry."
    exit 1
  }
  if ($pyLauncher -like "py *") {
    Invoke-Expression "$pyLauncher -m venv `"$root\.venv`""
  } else {
    & $pyLauncher -m venv "$root\.venv"
  }
  if (-not (Test-Path $py)) {
    Write-Host "Failed to create .venv"
    exit 1
  }
  Write-Host "Installing web dependencies..."
  & $py -m pip install -q --upgrade pip
  & $py -m pip install -q "fastapi>=0.110" "uvicorn[standard]>=0.27" python-multipart numpy Pillow python-dotenv requests tqdm
}

function Test-ApiHealthy {
  try {
    $r = Invoke-RestMethod "http://127.0.0.1:8000/api/health" -TimeoutSec 2
    return ($r.ok -eq $true)
  } catch {
    return $false
  }
}

function Get-Port8000Pid {
  $conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($conn) { return $conn.OwningProcess }
  return $null
}

function Stop-Port8000 {
  $procId = Get-Port8000Pid
  if ($procId) {
    Write-Host "Stopping stale process on port 8000 (PID $procId)..."
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    Start-Sleep 1
  }
}

if ((Get-Port8000Pid) -and -not (Test-ApiHealthy)) {
  Stop-Port8000
}

if (-not (Test-ApiHealthy)) {
  if (Get-Port8000Pid) { Stop-Port8000 }
  Write-Host "Starting Sign Bridge API..."
  Start-Process -FilePath $py `
    -ArgumentList @("-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000") `
    -WorkingDirectory $root `
    -WindowStyle Hidden

  $ready = $false
  for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 500
    if (Test-ApiHealthy) { $ready = $true; break }
  }
  if (-not $ready) {
    Write-Host "API did not start. Try: .\.venv\Scripts\python.exe run_webapp.py"
    exit 1
  }
}

Write-Host "API ready at http://127.0.0.1:8000"

$exe = Join-Path $root "src-tauri\target\debug\sign-bridge.exe"
if (Test-Path $exe) {
  $running = Get-Process -Name "sign-bridge" -ErrorAction SilentlyContinue
  if ($running) {
    $shell = New-Object -ComObject WScript.Shell
    $shell.AppActivate("Sign Bridge") | Out-Null
    Write-Host "Sign Bridge window focused."
  } else {
    Start-Process -FilePath $exe -WorkingDirectory $root
    Write-Host "Sign Bridge desktop app launched."
  }
} else {
  Start-Process "http://127.0.0.1:8000"
  Write-Host "Opened in browser (desktop exe not built)."
}
