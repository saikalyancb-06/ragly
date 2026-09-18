<#  One command that checks this install and fixes what it can.

      powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1

    It installs missing Python packages, the voice engines, the vision model, and - on a
    Snapdragon machine - the Qualcomm ONNX Runtime so embeddings can use the Hexagon NPU.
    Nothing here claims hardware it cannot see: the report at the end prints what is really
    available on this machine.
#>
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$vpy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) { throw "Run scripts\setup_windows.ps1 first (it creates the virtual environment)." }

function Step($text) { Write-Host "`n$text" -ForegroundColor Cyan }

Step "1/5  Build"
& $vpy -c "import sys; sys.path.insert(0, r'$Root'); from ragly_backend import __version__, BUILD; print(f'   build {__version__} ({BUILD})')"

Step "2/5  Python packages"
# --only-binary keeps pip from trying to compile anything from source, which is what turns a
# five minute setup into an hour on a machine without build tools.
& $vpy -m pip install --quiet --only-binary=:all: -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "   some packages failed - rerun with a working internet connection" -ForegroundColor Red }
else { Write-Host "   core packages ready" -ForegroundColor Green }

& $vpy -m pip install --quiet --only-binary=:all: -r requirements-ocr.txt
if ($LASTEXITCODE -ne 0) { Write-Host "   OCR/voice packages failed - text still works, dictation may not" -ForegroundColor Yellow }
else { Write-Host "   OCR and voice engines ready" -ForegroundColor Green }

# Windows blocks speech recognition until the account has accepted the speech privacy policy.
# The packages install fine without it, so this is checked separately or dictation fails at
# the first click with "The speech privacy policy was not accepted".
$privKey = "HKCU:\SOFTWARE\Microsoft\Speech_OneCore\Settings\OnlineSpeechPrivacy"
$accepted = 0
try { $accepted = (Get-ItemProperty -Path $privKey -Name "HasAccepted" -ErrorAction Stop).HasAccepted } catch { $accepted = 0 }
if ($accepted -ne 1) {
    Write-Host "   Windows speech consent not granted - granting it for this user..." -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot "enable_speech.ps1")
} else {
    Write-Host "   Windows speech consent granted" -ForegroundColor Green
}

Step "3/5  Qualcomm NPU runtime"
$machine = & $vpy -c "import platform; print(platform.machine())"
$qnn = & $vpy -c "import onnxruntime as o; print('yes' if any(p.startswith('QNN') for p in o.get_available_providers()) else 'no')"
if ($qnn -eq "yes") {
    Write-Host "   Hexagon NPU provider already available" -ForegroundColor Green
} elseif ($machine -match "ARM64|aarch64") {
    Write-Host "   Snapdragon machine without the QNN provider - installing onnxruntime-qnn..." -ForegroundColor Yellow
    & $vpy -m pip install --quiet --only-binary=:all: onnxruntime-qnn
    $qnn = & $vpy -c "import onnxruntime as o; print('yes' if any(p.startswith('QNN') for p in o.get_available_providers()) else 'no')"
    if ($qnn -eq "yes") { Write-Host "   Hexagon NPU provider installed" -ForegroundColor Green }
    else { Write-Host "   Could not install it. Embeddings stay on the CPU; everything still works." -ForegroundColor Yellow }
} else {
    Write-Host "   Not a Snapdragon machine - nothing to install" -ForegroundColor DarkGray
}

Step "4/5  Models"
$clip = Join-Path $Root "models\clip-vit-base-patch32\visual.onnx"
if (-not (Test-Path $clip)) {
    Write-Host "   Vision model missing - downloading (about 150 MB)..." -ForegroundColor Yellow
    & $vpy scripts\download_models.py --clip
}
$embed = Join-Path $Root "models\bge-small-en-v1.5\model.onnx"
$llm = Join-Path $Root "models\qwen2.5-3b-instruct-q4_k_m.gguf"
if (-not (Test-Path $embed) -or -not (Test-Path $llm)) {
    Write-Host "   Answer or embedding model missing - downloading (about 2.3 GB)..." -ForegroundColor Yellow
    & $vpy scripts\download_models.py --llm 3b
}
Write-Host "   models checked" -ForegroundColor Green

Step "5/5  What this machine can run"
& $vpy scripts\snapdragon_check.py

Write-Host "`nIf anything above says MISSING, rerun this script with a working internet connection."
Write-Host "Then start the app:  powershell -ExecutionPolicy Bypass -File scripts\run.ps1`n" -ForegroundColor Green
