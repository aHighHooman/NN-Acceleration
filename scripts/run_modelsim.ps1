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

function Invoke-RtlTest {
    param([string]$Top, [string]$Log, [string]$Failure)
    & $vsim -c "work.$Top" -l $Log `
        -do "run -all; quit -code [coverage attribute -name TESTSTATUS] -f"
    if ($LASTEXITCODE -ne 0) { throw $Failure }
    $logText = Get-Content -Raw $Log
    if ($logText -match "\*\* (Error|Fatal):") {
        throw "$Failure See $(Join-Path $buildDir $Log)"
    }
}

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
    (Join-Path $projectRoot "weightStationaryVariant/relu.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/outputActivation.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/weightedVectorReduction.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/weightStationaryProcessingElement.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/weightStationarySystolicArray.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/weightStationaryMatrixMultiplier.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/nnAccelerator.sv"),
    (Join-Path $projectRoot "SPI_Module.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/weightStationaryMatrixMultiplierTop.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/weightStationaryMatrixMultiplier_tb.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/weightStationaryMatrixMultiplierTop_tb.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/matrixWeightUpdateWave_tb.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/weightedVectorReduction_tb.sv")
)

Push-Location $projectRoot
try {
    & python -m unittest discover -s reference -p "test_*.py"
    if ($LASTEXITCODE -ne 0) { throw "Python golden-reference tests failed." }
}
finally {
    Pop-Location
}

Push-Location $buildDir
try {
    & $vlib work
    if ($LASTEXITCODE -ne 0) { throw "vlib failed." }

    & $vlog -sv @sources
    if ($LASTEXITCODE -ne 0) { throw "vlog failed." }

    # N=1 is intentionally outside the supported matrix boundary.  This also
    # protects the one-entry activation skid from being mistaken for the old
    # global FIFO-depth>=2 restriction.
    & $vsim -c work.weightStationaryMatrixMultiplier -GN=1 `
        -l invalid-parameter.log -do "run 1ns; quit -f"
    if (-not (Select-String -Path invalid-parameter.log -SimpleMatch `
            -Pattern "WIDTH>=1 and N>=2" -Quiet)) {
        throw "Invalid parameter check failed for an unexpected reason."
    }

    Invoke-RtlTest "weightedVectorReduction_tb" "reduction-regression.log" `
        "Weighted reduction regression failed."
    Invoke-RtlTest "weightStationaryMatrixMultiplier_tb" "core-regression.log" `
        "Core regression failed."
    Invoke-RtlTest "matrixWeightUpdateWave_tb" "matrix-update-regression.log" `
        "Matrix update-wave regression failed."

    & pwsh -File (Join-Path $PSScriptRoot "run_rtl_reference_compare.ps1")
    if ($LASTEXITCODE -ne 0) { throw "Golden RTL comparison failed." }

    & pwsh -File (Join-Path $PSScriptRoot "run_uvm.ps1")
    if ($LASTEXITCODE -ne 0) { throw "UVM protocol regression failed." }

    Invoke-RtlTest "weightStationaryMatrixMultiplierTop_tb" "spi-regression.log" `
        "SPI regression failed."
}
finally {
    Pop-Location
}

Write-Output "PASS: parameter bounds, Python references, local RTL units, golden RTL comparison, UVM protocol, and SPI regressions completed."
