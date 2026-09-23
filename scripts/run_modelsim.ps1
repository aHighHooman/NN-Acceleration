$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "questa_env.ps1")
$buildDir = Join-Path $projectRoot "build/modelsim"

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

if (Test-Path -LiteralPath $buildDir) {
    Remove-Item -LiteralPath $buildDir -Recurse -Force
}
New-Item -ItemType Directory -Path $buildDir | Out-Null

$sources = $coreSources + (@(
    "SPI_Module.sv",
    "weightStationaryVariant/weightStationaryMatrixMultiplierTop.sv",
    "weightStationaryVariant/weightStationaryMatrixMultiplier_tb.sv",
    "weightStationaryVariant/weightStationaryMatrixMultiplierTop_tb.sv",
    "weightStationaryVariant/matrixWeightUpdateWave_tb.sv",
    "weightStationaryVariant/weightedVectorReduction_tb.sv"
) | ForEach-Object { Join-Path $projectRoot $_ })

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

    # Reject N=1 matrices with the parameter message, not an elaboration error.
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
