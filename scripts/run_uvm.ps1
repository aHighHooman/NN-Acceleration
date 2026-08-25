param(
    [ValidateSet(
        "nn_uvm_smoke_test",
        "nn_uvm_relu_test",
        "nn_uvm_starter_regression_test"
    )]
    [string]$TestName = "nn_uvm_starter_regression_test"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $projectRoot "build/uvm"

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
    throw "Questa was not found. Set NN_ACCEL_QUESTA_BIN to the directory containing vlog.exe and vsim.exe."
}

$vlib = Join-Path $questaBin "vlib.exe"
$vlog = Join-Path $questaBin "vlog.exe"
$vsim = Join-Path $questaBin "vsim.exe"
$questaRoot = Split-Path -Parent $questaBin
$uvmSource = Join-Path $questaRoot "verilog_src/uvm-1.1d/src"

# Questa 2025.1+ reads SALT_LICENSE_SERVER.  Some Intel installers still leave
# a valid node-locked path in SALT_LICENSE_FILE, so bridge it for this process
# without changing the user's machine-wide environment.
if (-not $env:SALT_LICENSE_SERVER -and $env:SALT_LICENSE_FILE) {
    $env:SALT_LICENSE_SERVER = $env:SALT_LICENSE_FILE
}

if (-not (Test-Path -LiteralPath (Join-Path $uvmSource "uvm_macros.svh"))) {
    throw "Bundled UVM macro source was not found below $questaRoot."
}

if (Test-Path -LiteralPath $buildDir) {
    Remove-Item -LiteralPath $buildDir -Recurse -Force
}
New-Item -ItemType Directory -Path $buildDir | Out-Null

$rtlSources = @(
    (Join-Path $projectRoot "memory/signedFifo.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/reluActivation.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/activationLayer.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/multiplierBlockWeightStationary.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/systolicArrayWeightStationary.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/matrixMultiplierWeightStationary.sv"),
    (Join-Path $projectRoot "SPI_Module.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/matrixMultiplierWeightStationarySPI.sv")
)

$uvmSources = @(
    (Join-Path $projectRoot "uvm/nn_uvm_if.sv"),
    (Join-Path $projectRoot "uvm/nn_uvm_pkg.sv"),
    (Join-Path $projectRoot "uvm/nn_uvm_tb_top.sv")
)

Push-Location $buildDir
try {
    & $vlib work
    if ($LASTEXITCODE -ne 0) { throw "vlib failed." }

    # Questa ships a compiled mtiUvm library.  Using it lets the directed UVM
    # starter run with the FPGA Starter license.  Constrained randomization and
    # covergroups still require the separate svverification feature.
    & $vlog -sv -L mtiUvm -timescale 1ns/1ps "+incdir+$uvmSource" @rtlSources @uvmSources
    if ($LASTEXITCODE -ne 0) { throw "RTL/UVM testbench compilation failed." }

    $logPath = Join-Path $buildDir "$TestName.log"
    & $vsim -c -L mtiUvm work.nn_uvm_tb_top "+UVM_TESTNAME=$TestName" `
        -l $logPath -do "run -all; quit -f"
    if ($LASTEXITCODE -ne 0) { throw "$TestName simulation failed." }

    $logText = Get-Content -Raw $logPath
    if ($logText -match "UVM_(ERROR|FATAL)\s*:\s*[1-9]") {
        throw "$TestName completed with UVM errors. See $logPath"
    }
}
finally {
    Pop-Location
}

Write-Output "PASS: $TestName completed with Questa at $questaBin"
