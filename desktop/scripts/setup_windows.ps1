<#
  Ragly backend setup for Windows (x64 laptops and Snapdragon ARM64 PCs).
  Run from the "desktop" folder in PowerShell:
      powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
  Options:
      -Llm 3b|1.5b|0.5b     answer model size for llama.cpp (default 3b)
      -SkipLlama            don't download llama.cpp
      -SkipModels           don't download models
      -SkipGenieX           don't install GenieX even on Snapdragon
#>
param(
    [ValidateSet("3b", "1.5b", "0.5b")] [string]$Llm = "3b",
    [switch]$SkipLlama,
    [switch]$SkipModels,
    [switch]$SkipGenieX
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # makes Invoke-WebRequest much faster
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# ---------- 1. Detect hardware ----------
Step "Detecting hardware"
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$isArm = $cpu.Architecture -eq 12
$isSnapdragon = $isArm -and ($cpu.Name -match "Snapdragon|Qualcomm")
Write-Host "CPU: $($cpu.Name)  | ARM64: $isArm | Snapdragon: $isSnapdragon"

# ---------- 2. Python venv ----------
Step "Setting up Python virtual environment"
$py = $null
foreach ($cand in @("py -3.12", "py -3.11", "py -3.10", "python")) {
    try {
        $v = Invoke-Expression "$cand -c `"import sys;print(sys.version_info[:2] >= (3,10))`"" 2>$null
        if ($v -eq "True") { $py = $cand; break }
    } catch { }
}
if (-not $py) {
    throw "Python 3.10+ not found. Install it from https://www.python.org/downloads/windows/ (on Snapdragon pick the ARM64 installer) and tick 'Add to PATH'."
}
Write-Host "Using: $py"
if (-not (Test-Path ".venv")) { Invoke-Expression "$py -m venv .venv" }
$vpy = Join-Path $Root ".venv\Scripts\python.exe"
& $vpy -m pip install --quiet --upgrade pip
# --only-binary: never compile from source on a machine that has no build tools
& $vpy -m pip install --only-binary=:all: -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed. On ARM64, if a package has no ARM64 wheel, install x64 Python instead and rerun." }

Step "Installing OCR (built-in Windows OCR)"
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"   # an OCR problem must never stop the setup
try {
    & $vpy -m pip install --only-binary=:all: -r requirements-ocr.txt
    $ocr = & $vpy -c "from ragly_backend.ingest import OCR; s = OCR.status(); print(s['backend'] or ('NONE: ' + str(s['error'])))" 2>$null
    if ($ocr -match "^NONE") {
        Write-Warning "OCR not available: $ocr"
        Write-Warning "Scanned pages will be skipped. If the English OCR pack is missing, run in an admin PowerShell:"
        Write-Warning '  Add-WindowsCapability -Online -Name "Language.OCR~~~en-US~0.0.1.0"'
    } else {
        Write-Host "OCR backend: $ocr" -ForegroundColor Green
    }
} catch {
    Write-Warning "OCR install failed: $_  (text PDFs still work)"
}
$ErrorActionPreference = $prevEap

# ---------- 3. llama.cpp ----------
if (-not $SkipLlama) {
    Step "Downloading llama.cpp (llama-server)"
    $dest = Join-Path $Root "bin\llama"
    if (Test-Path (Join-Path $dest "llama-server.exe")) {
        Write-Host "llama-server.exe already present"
    } else {
        # "latest" can be a source-only release, so scan recent releases for the Windows CPU build
        $pattern = if ($isArm) { "^llama-b\d+-bin-win-cpu-arm64\.zip$" } else { "^llama-b\d+-bin-win-cpu-x64\.zip$" }
        $asset = $null; $tag = $null
        try {
            $rels = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30" -Headers @{ "User-Agent" = "ragly-setup" }
            foreach ($r in $rels) {
                $a = $r.assets | Where-Object { $_.name -match $pattern } | Select-Object -First 1
                if ($a) { $asset = $a; $tag = $r.tag_name; break }
            }
        } catch {
            Write-Warning "GitHub API unavailable ($_). Using the pinned build."
        }
        if (-not $asset) {
            $tag = "b10964"   # known-good build (Sep 2026)
            $arch = if ($isArm) { "arm64" } else { "x64" }
            $name = "llama-$tag-bin-win-cpu-$arch.zip"
            $asset = [pscustomobject]@{ name = $name; browser_download_url = "https://github.com/ggml-org/llama.cpp/releases/download/$tag/$name" }
        }
        $rel = [pscustomobject]@{ tag_name = $tag }
        Write-Host "Release $($rel.tag_name): $($asset.name)"
        $zip = Join-Path $env:TEMP $asset.name
        Invoke-WebRequest $asset.browser_download_url -OutFile $zip -UseBasicParsing
        $tmp = Join-Path $env:TEMP "ragly-llama"
        if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
        Expand-Archive $zip -DestinationPath $tmp -Force
        $exe = Get-ChildItem $tmp -Recurse -Filter "llama-server.exe" | Select-Object -First 1
        if (-not $exe) { throw "llama-server.exe not found inside $($asset.name)" }
        New-Item -ItemType Directory -Force $dest | Out-Null
        Copy-Item (Join-Path $exe.DirectoryName "*") $dest -Recurse -Force
        Remove-Item $zip, $tmp -Recurse -Force
        Write-Host "Installed to bin\llama"
    }
}

# ---------- 3b. Qualcomm NPU runtime (Snapdragon only) ----------
if ($isSnapdragon) {
    Step "Qualcomm ONNX Runtime (Hexagon NPU)"
    & $vpy -m pip install --quiet --only-binary=:all: onnxruntime-qnn
    $qnn = & $vpy -c "import onnxruntime as o; print('yes' if any(p.startswith('QNN') for p in o.get_available_providers()) else 'no')"
    if ($qnn -eq "yes") { Write-Host "Hexagon NPU provider available" -ForegroundColor Green }
    else { Write-Host "No QNN provider: embeddings and image search run on the CPU." -ForegroundColor Yellow }
}

# ---------- 4. Models ----------
if (-not $SkipModels) {
    Step "Downloading models (about 2.5 GB: answer model, embeddings, vision model)"
    & $vpy scripts\download_models.py --llm $Llm --clip
    if ($LASTEXITCODE -ne 0) { throw "model download failed" }
}

# ---------- 5. GenieX (Snapdragon only) ----------
if ($isSnapdragon -and -not $SkipGenieX) {
    Step "GenieX (Snapdragon NPU engine)"
    $gx = Get-Command geniex -ErrorAction SilentlyContinue
    if (-not $gx) {
        $installer = Join-Path $Root "bin\geniex-cli.exe"
        New-Item -ItemType Directory -Force (Split-Path $installer) | Out-Null
        Invoke-WebRequest "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/geniex-cli.exe" -OutFile $installer -UseBasicParsing
        Write-Host "Launching the GenieX installer. If SmartScreen warns: More info -> Run anyway." -ForegroundColor Yellow
        Start-Process $installer -Wait
        $gx = Get-Command geniex -ErrorAction SilentlyContinue
    }
    if ($gx) {
        $model = (Get-Content engines.json -Raw | ConvertFrom-Json).snapdragon.model
        Write-Host "Pulling $model (pick Q4_0 / defaults if prompted)"
        geniex pull $model
        if ($LASTEXITCODE -ne 0) { Write-Warning "geniex pull failed - run 'geniex pull --help' and pull the model manually." }
    } else {
        Write-Warning "geniex not on PATH yet. Open a new terminal after the installer finishes, then run: geniex pull ai-hub-models/Qwen3-4B-Instruct-2507"
    }
} elseif (-not $isSnapdragon) {
    Write-Host "`nNot a Snapdragon PC: GenieX skipped (Normal / llama.cpp mode only)."
}

Step "Running tests"
& $vpy -m pytest -q tests

Write-Host "`nSetup complete. Start the backend with:  powershell -ExecutionPolicy Bypass -File scripts\run.ps1" -ForegroundColor Green
