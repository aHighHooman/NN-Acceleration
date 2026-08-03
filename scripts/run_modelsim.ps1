$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $projectRoot "build/modelsim"

$requiredCommands = @("vlib", "vlog", "vsim")
foreach ($command in $requiredCommands) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required ModelSim command '$command' was not found on PATH."
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
    (Join-Path $projectRoot "weightStationaryVariant/multiplierBlockWeightStationary.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/systolicArrayWeightStationary.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/matrixMultiplierWeightStationary.sv"),
    (Join-Path $projectRoot "SPI_Module.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/matrixMultiplierWeightStationarySPI.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/matrixMultiplierWeightStationary_tb.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/matrixMultiplierWeightStationarySPI_tb.sv")
)

Push-Location $buildDir
try {
    & vlib work
    if ($LASTEXITCODE -ne 0) { throw "vlib failed." }

    & vlog -sv @sources
    if ($LASTEXITCODE -ne 0) { throw "vlog failed." }

    & vsim -c work.matrixMultiplierWeightStationary_tb `
        -l core-regression.log -do "run -all; quit -f"
    if ($LASTEXITCODE -ne 0) { throw "Core regression failed." }

    & vsim -c work.matrixMultiplierWeightStationarySPI_tb `
        -l spi-regression.log -do "run -all; quit -f"
    if ($LASTEXITCODE -ne 0) { throw "SPI regression failed." }

    & vsim -c work.matrixMultiplierWeightStationary -GINPUT_FIFO_DEPTH=1 `
        -l invalid-parameter.log -do "run -all; quit -f"
    if (-not (Select-String -Path invalid-parameter.log -SimpleMatch `
            -Pattern "WIDTH>=1, N>=2, FIFO depths>=2" -Quiet)) {
        throw "Invalid parameter check failed for an unexpected reason."
    }
}
finally {
    Pop-Location
}

Write-Output "PASS: core, asynchronous-clock SPI, and parameter-bound regressions completed."
