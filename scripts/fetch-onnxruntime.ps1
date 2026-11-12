# Windows: download ONNX Runtime with DirectML (works on any DX12 GPU: NVIDIA, AMD, Intel).
#   powershell -ExecutionPolicy Bypass -File scripts\fetch-onnxruntime.ps1
#   ... -Flavor cuda   for the NVIDIA CUDA 12 build instead
param([string]$Flavor = "directml", [string]$Version = "1.31.1")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $Root "third_party\onnxruntime"
$Tmp = New-Item -ItemType Directory -Path (Join-Path $env:TEMP ("ort-" + [guid]::NewGuid()))

if ($Flavor -eq "directml") {
    # DirectML builds ship as a NuGet package (a zip).
    $Url = "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/$Version"
    Invoke-WebRequest $Url -OutFile "$Tmp\ort.zip"
    Expand-Archive "$Tmp\ort.zip" -DestinationPath "$Tmp\pkg"
    New-Item -ItemType Directory -Force -Path "$Dest\lib" | Out-Null
    Copy-Item "$Tmp\pkg\runtimes\win-x64\native\*" "$Dest\lib\" -Force
} else {
    $Pkg = "onnxruntime-win-x64-gpu_cuda12-$Version"
    Invoke-WebRequest "https://github.com/microsoft/onnxruntime/releases/download/v$Version/$Pkg.zip" -OutFile "$Tmp\ort.zip"
    Expand-Archive "$Tmp\ort.zip" -DestinationPath $Tmp
    if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
    Copy-Item -Recurse "$Tmp\$Pkg" $Dest
}
Remove-Item -Recurse -Force $Tmp
Write-Host "installed ONNX Runtime $Version ($Flavor) -> $Dest"
