# Shared by the regression scripts: locate Questa, bridge its license
# variable, and list the accelerator core RTL in dependency order.

$projectRoot = Split-Path -Parent $PSScriptRoot

# Prefer the recent Questa installed with Quartus.  PATH on machines upgraded
# from older Quartus releases often still points at ModelSim 20.x.
$candidateBins = @()
if ($env:NN_ACCEL_QUESTA_BIN) {
    $candidateBins += $env:NN_ACCEL_QUESTA_BIN
}
$candidateBins += "C:\altera_lite\25.1std\questa_fse\win64"
$candidateBins += @(Get-Command vsim -All -ErrorAction SilentlyContinue |
    Where-Object { $_.Source -match "questa" } |
    ForEach-Object { Split-Path -Parent $_.Source })

$questaBin = $candidateBins |
    Where-Object { Test-Path -LiteralPath (Join-Path $_ "vsim.exe") } |
    Select-Object -First 1

if (-not $questaBin) {
    throw "Questa was not found. Set NN_ACCEL_QUESTA_BIN to the directory containing vlib.exe, vlog.exe, and vsim.exe."
}

$vlib = Join-Path $questaBin "vlib.exe"
$vlog = Join-Path $questaBin "vlog.exe"
$vsim = Join-Path $questaBin "vsim.exe"

# Bridge Intel's SALT_LICENSE_FILE to Questa 2025.1+ SALT_LICENSE_SERVER
# for this process only; leave the machine-wide environment unchanged.
if (-not $env:SALT_LICENSE_SERVER) {
    $userSaltLicense = [Environment]::GetEnvironmentVariable(
        "SALT_LICENSE_SERVER", "User")
    if ($userSaltLicense) {
        $env:SALT_LICENSE_SERVER = $userSaltLicense
    } elseif ($env:SALT_LICENSE_FILE) {
        $env:SALT_LICENSE_SERVER = $env:SALT_LICENSE_FILE
    }
}

$coreSources = @(
    "memory/signedFifo.sv",
    "weightStationaryVariant/weightedVectorReduction.sv",
    "weightStationaryVariant/weightStationaryProcessingElement.sv",
    "weightStationaryVariant/weightStationarySystolicArray.sv",
    "weightStationaryVariant/weightStationaryMatrixMultiplier.sv",
    "weightStationaryVariant/nnAccelerator.sv"
) | ForEach-Object { Join-Path $projectRoot $_ }
