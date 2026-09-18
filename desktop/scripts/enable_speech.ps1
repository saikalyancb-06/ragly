<#
  enable_speech.ps1 - let Windows use speech on this account.

  Windows blocks every speech-recognition call until the account has accepted the speech
  privacy policy once; the failure looks like
      OSError [WinError -2147199735] The speech privacy policy was not accepted
  This sets the same per-user flag the Settings toggle sets. No admin rights, current user
  only, nothing is downloaded and no audio leaves the machine because of this script.

  Undo:  Settings > Privacy & security > Speech > turn Speech recognition off.
#>

$ErrorActionPreference = "Stop"
$key = "HKCU:\SOFTWARE\Microsoft\Speech_OneCore\Settings\OnlineSpeechPrivacy"

Write-Host "Ragly - enabling Windows speech for $env:USERNAME" -ForegroundColor Cyan

if (-not (Test-Path $key)) { New-Item -Path $key -Force | Out-Null }
New-ItemProperty -Path $key -Name "HasAccepted" -Value 1 -PropertyType DWord -Force | Out-Null

$now = (Get-ItemProperty -Path $key -Name "HasAccepted").HasAccepted
if ($now -eq 1) {
    Write-Host "  Speech consent granted." -ForegroundColor Green
} else {
    Write-Host "  Could not set the flag. Turn it on by hand: Settings > Privacy & security > Speech." -ForegroundColor Yellow
    exit 1
}

# The English speech pack is what recognises words on-device. Report whether it is present.
$speech = Get-WindowsCapability -Online -Name "Language.Speech~~~en-US*" -ErrorAction SilentlyContinue
if ($speech -and $speech.State -ne "Installed") {
    Write-Host "  English speech pack is not installed (dictation accuracy will be poor or unavailable)." -ForegroundColor Yellow
    Write-Host '  In an admin PowerShell: Add-WindowsCapability -Online -Name "Language.Speech~~~en-US~0.0.1.0"' -ForegroundColor Yellow
} elseif ($speech) {
    Write-Host "  English speech pack installed." -ForegroundColor Green
}

# Microphone consent is a separate switch; without it recognition ends as USER_CANCELED.
$micKey = "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"
$mic = (Get-ItemProperty -Path $micKey -Name "Value" -ErrorAction SilentlyContinue).Value
if ($mic -eq "Allow") {
    Write-Host "  Microphone allowed for desktop apps." -ForegroundColor Green
} else {
    Write-Host "  Microphone is NOT allowed for desktop apps - dictation will cancel itself." -ForegroundColor Yellow
    Write-Host "  Settings > Privacy & security > Microphone: turn on 'Microphone access' and" -ForegroundColor Yellow
    Write-Host "  'Let desktop apps access your microphone'." -ForegroundColor Yellow
    Start-Process "ms-settings:privacy-microphone"
}

Write-Host ""
Write-Host "Reload the Ragly page - the mic button should go live. No restart needed." -ForegroundColor Cyan
