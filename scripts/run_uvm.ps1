param(
    [ValidateSet(
        "nn_uvm_smoke_test",
        "nn_uvm_regression_test",
        "nn_uvm_training_test"
    )]
    [string]$TestName = "nn_uvm_regression_test",
    [string]$Seed = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $projectRoot "build/uvm"
$uvmDir = Join-Path $projectRoot "uvm"

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
if (-not $env:SALT_LICENSE_SERVER) {
    $userSaltLicense = [Environment]::GetEnvironmentVariable(
        "SALT_LICENSE_SERVER", "User")
    if ($userSaltLicense) {
        $env:SALT_LICENSE_SERVER = $userSaltLicense
    } elseif ($env:SALT_LICENSE_FILE) {
        $env:SALT_LICENSE_SERVER = $env:SALT_LICENSE_FILE
    }
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
    (Join-Path $projectRoot "weightStationaryVariant/weightedVectorReduction.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/multiplierBlockWeightStationary.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/systolicArrayWeightStationary.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/matrixMultiplierWeightStationary.sv"),
    (Join-Path $projectRoot "weightStationaryVariant/nnAccelerator.sv")
)

$uvmSources = @(
    (Join-Path $projectRoot "uvm/nn_core_if.sv"),
    (Join-Path $projectRoot "uvm/nn_training_if.sv"),
    (Join-Path $projectRoot "uvm/nn_uvm_pkg.sv"),
    (Join-Path $projectRoot "uvm/nn_training_uvm_pkg.sv"),
    (Join-Path $projectRoot "uvm/nn_uvm_tb_top.sv"),
    (Join-Path $projectRoot "uvm/nn_training_uvm_tb_top.sv")
)

Push-Location $buildDir
try {
    & $vlib work
    if ($LASTEXITCODE -ne 0) { throw "vlib failed." }

    # Questa ships a compiled mtiUvm library.  The environment intentionally
    # uses seeded $urandom stimulus and counter-based coverage so it remains
    # runnable when svverification-licensed constrained randomization and
    # covergroups are unavailable.
    & $vlog -sv -L mtiUvm -timescale 1ns/1ps "+incdir+$uvmSource" "+incdir+$uvmDir" @rtlSources @uvmSources
    if ($LASTEXITCODE -ne 0) { throw "RTL/UVM testbench compilation failed." }

    $logPath = Join-Path $buildDir "$TestName.log"
    $topLevel = if ($TestName -eq "nn_uvm_training_test") {
        "work.nn_training_uvm_tb_top"
    } else {
        "work.nn_uvm_tb_top"
    }
    $simArgs = @(
        "-c",
        "-L", "mtiUvm",
        $topLevel,
        "+UVM_TESTNAME=$TestName"
    )
    if ($Seed -ne "") {
        $simArgs += "+NN_SEED=$Seed"
    }
    $simArgs += @("-l", $logPath, "-do", "run -all; quit -f")
    & $vsim @simArgs
    if ($LASTEXITCODE -ne 0) { throw "$TestName simulation failed." }

    $logText = Get-Content -Raw $logPath
    if ($logText -match "UVM_(ERROR|FATAL)\s*:\s*[1-9]") {
        throw "$TestName completed with UVM errors. See $logPath"
    }
    if ($logText -match "\*\* Error:") {
        throw "$TestName completed with simulator/assertion errors. See $logPath"
    }
}
finally {
    Pop-Location
}

Write-Output "PASS: $TestName completed with Questa at $questaBin"
