# Windows Verilator timing support needs a recent C++ compiler. The regression
# can use local tool bundles or explicit existing installations via these paths.
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$verilatorBuildRoot = if ($env:NN_ACCEL_VERILATOR_BUILD_ROOT) {
    [System.IO.Path]::GetFullPath($env:NN_ACCEL_VERILATOR_BUILD_ROOT)
} else {
    Join-Path $env:USERPROFILE ".codex/scratch/latency-slash-verilator"
}
if ($verilatorBuildRoot -match '\s') {
    throw "GNU make needs a build path without spaces. Set NN_ACCEL_VERILATOR_BUILD_ROOT accordingly."
}
$ossCadRoot = if ($env:NN_ACCEL_OSS_CAD_SUITE) {
    $env:NN_ACCEL_OSS_CAD_SUITE
} else {
    Join-Path $projectRoot "build/tools/oss-cad-suite"
}
$devkitRoot = if ($env:NN_ACCEL_W64DEVKIT) {
    $env:NN_ACCEL_W64DEVKIT
} else {
    Join-Path $projectRoot "build/tools/w64devkit"
}
if (-not (Test-Path -LiteralPath (Join-Path $ossCadRoot "bin/verilator_bin.exe"))) {
    throw "Verilator was not found. Set NN_ACCEL_OSS_CAD_SUITE to an OSS CAD Suite installation."
}
if (-not (Test-Path -LiteralPath (Join-Path $devkitRoot "bin/g++.exe")) -or
    -not (Test-Path -LiteralPath (Join-Path $devkitRoot "bin/make.exe"))) {
    throw "The C++ compiler was not found. Set NN_ACCEL_W64DEVKIT to a recent w64devkit installation."
}

# GNU make recipes cannot reliably handle spaces in the Verilator runtime path.
# Model artifacts use a separate space-free path; RTL and logs stay in the project.
$verilatorFileSystem = New-Object -ComObject Scripting.FileSystemObject
$ossCadRootShort = $verilatorFileSystem.GetFolder($ossCadRoot).ShortPath
$devkitRootShort = $verilatorFileSystem.GetFolder($devkitRoot).ShortPath
$projectRootShort = $verilatorFileSystem.GetFolder($projectRoot).ShortPath
$verilator = Join-Path $ossCadRootShort "bin/verilator_bin.exe"
$env:VERILATOR_ROOT = ((Join-Path $ossCadRootShort "share/verilator") -replace '\\', '/')
$env:PATH = "$(Join-Path $devkitRootShort 'bin');$(Join-Path $ossCadRootShort 'bin');$(Join-Path $ossCadRootShort 'lib');$env:PATH"
