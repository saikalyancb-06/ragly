<#  Start the Ragly backend on http://127.0.0.1:8765  (API docs: http://127.0.0.1:8765/docs)
    -Mode normal|snapdragon   choose the answer engine at start (otherwise the last used one)
#>
param([ValidateSet("", "normal", "snapdragon")] [string]$Mode = "")
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$vpy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) { throw "Run scripts\setup_windows.ps1 first." }
if ($Mode) {
    & $vpy -c "from ragly_backend.config import save_user_settings as s; s({'engine_mode': '$Mode'})"
}
# The vision model (CLIP) is what makes Images and Photo Search work. Fetch it once if missing.
if (-not (Test-Path (Join-Path $Root "models\clip-vit-base-patch32\visual.onnx"))) {
    Write-Host "Vision model missing - downloading it once (about 150 MB)..." -ForegroundColor Yellow
    & $vpy scripts\download_models.py --clip
}
Write-Host "Ragly -> http://127.0.0.1:8765  (API docs at /docs, Ctrl+C to stop)" -ForegroundColor Green
Start-Process "http://127.0.0.1:8765"
& $vpy -m ragly_backend
