# Install Desktop + Start Menu shortcuts for Sign Bridge LIVE

$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path
$vbs = Join-Path $root "scripts\launch-desktop-hidden.vbs"
$icon = Join-Path $root "src-tauri\icons\icon.ico"

$desktop = [Environment]::GetFolderPath("Desktop")
# Prefer OneDrive Desktop if that is where other project shortcuts live
$oneDriveDesktop = Join-Path $env:USERPROFILE "OneDrive\Desktop"
if (Test-Path $oneDriveDesktop) { $desktop = $oneDriveDesktop }

$startMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"

$targets = @(
  (Join-Path $desktop "Sign Bridge.lnk"),
  (Join-Path $startMenu "Sign Bridge.lnk")
)

$wsh = New-Object -ComObject WScript.Shell
foreach ($path in $targets) {
  $sc = $wsh.CreateShortcut($path)
  $sc.TargetPath = "wscript.exe"
  $sc.Arguments = "//B `"$vbs`""
  $sc.WorkingDirectory = $root
  $sc.WindowStyle = 7
  $sc.Description = "Sign Bridge LIVE - Cursor edits auto-sync to the app"

  if (Test-Path $icon) { $sc.IconLocation = "$icon,0" }
  $sc.Save()
  Write-Host "Installed: $path"
}

Write-Host ""
Write-Host "Double-click 'Sign Bridge' on your Desktop to open the app."
