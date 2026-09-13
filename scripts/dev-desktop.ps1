# Fast desktop launcher — opens the window immediately; API starts in parallel.
# Cursor LIVE edits: set SIGN_BRIDGE_LIVE=1 or use scripts/dev-desktop-live.ps1

$ErrorActionPreference = "SilentlyContinue"
$root = (Resolve-Path "$PSScriptRoot\..").Path
$exe = Join-Path $root "src-tauri\target\debug\sign-bridge.exe"
$log = Join-Path $env:TEMP "sign-bridge-live.log"
$live = $env:SIGN_BRIDGE_LIVE -eq "1"

Set-Location $root
"==== OPEN $(Get-Date -Format 'HH:mm:ss') live=$live ====" | Out-File $log -Encoding utf8

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
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    Start-Sleep 1
  }
}

function Start-ApiServer {
  $py = Join-Path $root ".venv\Scripts\python.exe"
  if (-not (Test-Path $py)) { $py = "python" }

  if (Get-Port8000Pid) {
    if (Test-ApiHealthy) {
      "API already healthy on :8000" | Out-File $log -Append -Encoding utf8
      return
    }
    Stop-Port8000
  }

  $args = @(
    "-m", "uvicorn", "backend.main:app",
    "--host", "127.0.0.1", "--port", "8000"
  )
  if ($live) {
    $args += @("--reload", "--reload-dir", "web", "--reload-dir", "src", "--reload-dir", "backend", "--reload-dir", "data")
  }

  Start-Process -FilePath $py `
    -ArgumentList $args `
    -WorkingDirectory $root `
    -WindowStyle Hidden
  "Started API live=$live" | Out-File $log -Append -Encoding utf8
}

# 1) Kick API in background if needed — do NOT block the window
if (-not (Test-ApiHealthy)) {
  Start-ApiServer
}

# 2) Open desktop window immediately
if (-not (Test-Path $exe)) {
  Write-Host "Missing exe. Build once with: npm run tauri:build"
  "Missing exe" | Out-File $log -Append -Encoding utf8
  exit 1
}

$running = Get-Process -Name "sign-bridge" -ErrorAction SilentlyContinue
if ($running) {
  $shell = New-Object -ComObject WScript.Shell
  $shell.AppActivate("Sign Bridge") | Out-Null
  "Focused existing window" | Out-File $log -Append -Encoding utf8
} else {
  Start-Process -FilePath $exe -WorkingDirectory $root
  "Launched exe" | Out-File $log -Append -Encoding utf8
}

exit 0
