# Builds the release zip: the exe, the ONNX Runtime provider DLLs, the MSVC runtime and the licence files.
# The models are NOT in the zip: the app downloads them on its first start (see README, "First start").
#
#   powershell -ExecutionPolicy Bypass -File scripts\package.ps1 [-TargetDir target] [-SkipBuild]
param(
    [string]$TargetDir = "target",
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$version = (Select-String -Path Cargo.toml -Pattern '^version = "(.+)"' | Select-Object -First 1).Matches[0].Groups[1].Value
$name = "PixelSpriteStudio-v$version-win-x64"

if (-not $SkipBuild) {
    $env:CARGO_TARGET_DIR = $TargetDir
    cargo build --release --bin animator_ui
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
}

$rel = Join-Path $TargetDir "release"
$out = Join-Path $root "dist\$name"
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force $out | Out-Null

Copy-Item (Join-Path $rel "animator_ui.exe") (Join-Path $out "PixelSpriteStudio.exe")
# ONNX Runtime itself is linked into the exe; these are its provider libraries
foreach ($f in "onnxruntime_providers_shared.dll", "onnxruntime_providers_cuda.dll", "DirectML.dll") {
    Copy-Item (Join-Path $rel $f) $out
}
# app-local MSVC runtime, so the app starts on a machine without the Visual C++ redistributable
foreach ($f in "msvcp140.dll", "msvcp140_1.dll", "msvcp140_atomic_wait.dll", "vcruntime140.dll", "vcruntime140_1.dll") {
    Copy-Item (Join-Path $env:SystemRoot "System32\$f") $out
}
foreach ($f in "LICENSE", "THIRD_PARTY.md") { Copy-Item $f $out }
Copy-Item "scripts\README_RELEASE.txt" (Join-Path $out "README.txt")

$zip = Join-Path $root "dist\$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $out -DestinationPath $zip -CompressionLevel Optimal
"{0}  {1:N1} MB" -f $zip, ((Get-Item $zip).Length / 1MB)
(Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
