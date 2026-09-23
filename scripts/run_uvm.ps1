param(
    [ValidateSet(
        "nn_uvm_smoke_test",
        "nn_uvm_regression_test"
    )]
    [string]$TestName = "nn_uvm_regression_test",
    [string]$Seed = ""
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "questa_env.ps1")
$buildDir = Join-Path $projectRoot "build/uvm"
$uvmDir = Join-Path $projectRoot "uvm"
$questaRoot = Split-Path -Parent $questaBin
$uvmSource = Join-Path $questaRoot "verilog_src/uvm-1.1d/src"

if (-not (Test-Path -LiteralPath (Join-Path $uvmSource "uvm_macros.svh"))) {
    throw "Bundled UVM macro source was not found below $questaRoot."
}

if (Test-Path -LiteralPath $buildDir) {
    Remove-Item -LiteralPath $buildDir -Recurse -Force
}
New-Item -ItemType Directory -Path $buildDir | Out-Null

$uvmSources = @(
    (Join-Path $projectRoot "uvm/nn_core_if.sv"),
    (Join-Path $projectRoot "uvm/nn_uvm_pkg.sv"),
    (Join-Path $projectRoot "uvm/nn_uvm_tb_top.sv")
)

Push-Location $buildDir
try {
    & $vlib work
    if ($LASTEXITCODE -ne 0) { throw "vlib failed." }

    # Use Questa's compiled mtiUvm with seeded $urandom and local assertions;
    # constrained randomization and covergroups require svverification licenses.
    & $vlog -sv -L mtiUvm -timescale 1ns/1ps "+incdir+$uvmSource" "+incdir+$uvmDir" @coreSources @uvmSources
    if ($LASTEXITCODE -ne 0) { throw "RTL/UVM testbench compilation failed." }

    $logPath = Join-Path $buildDir "$TestName.log"
    $simArgs = @(
        "-c",
        "-L", "mtiUvm",
        "work.nn_uvm_tb_top",
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
