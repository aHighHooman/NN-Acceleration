$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $projectRoot "build/modelsim"

$candidateBins = @()
if ($env:NN_ACCEL_QUESTA_BIN) {
    $candidateBins += $env:NN_ACCEL_QUESTA_BIN
}
$candidateBins += "C:\altera_lite\25.1std\questa_fse\win64"
$candidateBins += @(Get-Command vsim -All -ErrorAction SilentlyContinue |
    ForEach-Object { Split-Path -Parent $_.Source })

$questaBin = $candidateBins |
    Where-Object { Test-Path -LiteralPath (Join-Path $_ "vsim.exe") } |
    Select-Object -First 1

if (-not $questaBin) {
    throw "Questa/ModelSim was not found. Set NN_ACCEL_QUESTA_BIN to the directory containing vlib.exe, vlog.exe, and vsim.exe."
}

$vlib = Join-Path $questaBin "vlib.exe"
$vlog = Join-Path $questaBin "vlog.exe"
$vsim = Join-Path $questaBin "vsim.exe"

if (-not $env:SALT_LICENSE_SERVER) {
    $userSaltLicense = [Environment]::GetEnvironmentVariable(
        "SALT_LICENSE_SERVER", "User")
    if ($userSaltLicense) {
        $env:SALT_LICENSE_SERVER = $userSaltLicense
    } elseif ($env:SALT_LICENSE_FILE) {
        $env:SALT_LICENSE_SERVER = $env:SALT_LICENSE_FILE
    }
}

if (Test-Path -LiteralPath $buildDir) {
    Remove-Item -LiteralPath $buildDir -Recurse -Force
}
New-Item -ItemType Directory -Path $buildDir | Out-Null

$sources = @(
    (Join-Path $projectRoot "memory/signedFifo.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/reluActivation.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/activationLayer.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/weightedVectorReduction.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/multiplierBlockWeightStationary.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/systolicArrayWeightStationary.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/matrixMultiplierWeightStationary.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/nnAccelerator.sv"),
    (Join-Path $projectRoot "SPI_Module.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/matrixMultiplierWeightStationarySPI.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/matrixMultiplierWeightStationary_tb.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/matrixMultiplierWeightStationarySPI_tb.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/nnAccelerator_tb.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/matrixWeightUpdateWave_tb.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/weightedVectorReduction_tb.sv")
)

Push-Location $buildDir
try {
    & $vlib work
    if ($LASTEXITCODE -ne 0) { throw "vlib failed." }

    & $vlog -sv @sources
    if ($LASTEXITCODE -ne 0) { throw "vlog failed." }

    & $vsim -c work.matrixMultiplierWeightStationary_tb `
        -l core-regression.log -do "run -all; quit -code [coverage attribute -name TESTSTATUS] -f"
    if ($LASTEXITCODE -ne 0) { throw "Core regression failed." }

    & $vsim -c work.matrixMultiplierWeightStationarySPI_tb `
        -l spi-regression.log -do "run -all; quit -code [coverage attribute -name TESTSTATUS] -f"
    if ($LASTEXITCODE -ne 0) { throw "SPI regression failed." }

    & $vsim -c work.weightedVectorReduction_tb `
        -l reduction-regression.log -do "run -all; quit -code [coverage attribute -name TESTSTATUS] -f"
    if ($LASTEXITCODE -ne 0) { throw "Weighted reduction regression failed." }

    & $vsim -c work.nnAccelerator_tb `
        -l accelerator-regression.log -do "run -all; quit -code [coverage attribute -name TESTSTATUS] -f"
    if ($LASTEXITCODE -ne 0) { throw "Accelerator target/comparator regression failed." }

    & $vsim -c work.nnAcceleratorPhase5J_3x3_tb `
        -l accelerator-phase5j-regression.log -do "run -all; quit -code [coverage attribute -name TESTSTATUS] -f"
    if ($LASTEXITCODE -ne 0) { throw "Phase 5J 3x3 learning-boundary regression failed." }

    & $vsim -c work.matrixWeightUpdateWave_tb `
        -l matrix-update-regression.log -do "run -all; quit -code [coverage attribute -name TESTSTATUS] -f"
    if ($LASTEXITCODE -ne 0) { throw "Matrix update-wave regression failed." }
}
finally {
    Pop-Location
}

Write-Output "PASS: core, SPI, reduction, Phase 5J accelerator, and matrix update-wave regressions completed."
