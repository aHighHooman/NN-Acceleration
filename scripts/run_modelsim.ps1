$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "questa_env.ps1")
$buildDir = Join-Path $projectRoot "build/modelsim"
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

function Invoke-RtlTest {
    param([string]$Top, [string]$Log)
    & $vsim -c "work.$Top" -l $Log -do "run -all; quit -f" *> "$Log.console"
    $text = Get-Content -LiteralPath $Log -Raw
    if ($LASTEXITCODE -or $text -match "\*\* (Error|Fatal):" -or $text -notmatch "PASS:") {
        Get-Content -LiteralPath $Log -Tail 30
        throw "RTL regression failed: $Top"
    }
    Get-Content -LiteralPath $Log | Where-Object { $_ -match "PASS:" }
}

Push-Location $projectRoot
try {
    & python -m unittest discover -s reference -p "test_*.py"
    if ($LASTEXITCODE) { throw "Python references failed" }
} finally { Pop-Location }
$sources = $coreSources + (@(
    "weightStationaryVariant/weightStationaryMatrixMultiplier_tb.sv",
    "weightStationaryVariant/matrixWeightUpdateWave_tb.sv",
    "weightStationaryVariant/weightedVectorReduction_tb.sv"
) | ForEach-Object { Join-Path $projectRoot $_ })
Push-Location $buildDir
try {
    if (-not (Test-Path -LiteralPath work)) { & $vlib work; if ($LASTEXITCODE) { throw "vlib failed" } }
    & $vlog -sv @sources *> compile.log
    if ($LASTEXITCODE) { Get-Content -LiteralPath compile.log -Tail 30; throw "RTL compilation failed" }
    foreach ($badN in @(1,2,4,8)) {
        $log = "invalid-N$badN.log"
        & $vsim -c work.weightStationaryMatrixMultiplier "-GN=$badN" -l $log -do "run 1ns; quit -f" *> "$log.console"
        if (-not (Select-String -LiteralPath $log -SimpleMatch "Fountain requires" -Quiet)) {
            throw "Invalid N=$badN did not produce its intended parameter diagnostic"
        }
    }
    Write-Output "PASS: unsupported single/even dimensions reject with a parameter diagnostic."
    $diagnostics = @(
        "Fountain MAC valid mismatch", "Fountain MAC tag mismatch", "Fountain join tag mismatch",
        "Array configuration requires an idle valid row", "Array input before weight commit"
    )
    for ($fault=0; $fault -lt $diagnostics.Count; $fault++) {
        $log = "protocol-fault-$fault.log"
        & $vsim -c work.matrixProtocolFault_tb "-GFAULT=$fault" -l $log -do "run -all; quit -f" *> "$log.console"
        if (-not (Select-String -LiteralPath $log -SimpleMatch $diagnostics[$fault] -Quiet)) {
            Get-Content -LiteralPath $log -Tail 20
            throw "Protocol fault $fault did not produce its intended assertion"
        }
    }
    Write-Output "PASS: missing/mistagged MAC operands, mistagged joins, busy writes and premature input are detected."
    Invoke-RtlTest "weightedVectorReduction_tb" "reduction-regression.log"
    Invoke-RtlTest "weightStationaryMatrixMultiplier_tb" "core-regression.log"
    Invoke-RtlTest "matrixWeightUpdateWave_tb" "matrix-update-regression.log"
} finally { Pop-Location }
& pwsh -File (Join-Path $PSScriptRoot "run_rtl_reference_compare.ps1") -ArtifactDir (Join-Path $projectRoot "build/rtl_reference_compare/three")
if ($LASTEXITCODE) { throw "Three-layer reference comparison failed" }
& pwsh -File (Join-Path $PSScriptRoot "run_rtl_reference_compare.ps1") -ArtifactDir (Join-Path $projectRoot "build/rtl_reference_compare/five") -N 5
if ($LASTEXITCODE) { throw "N=5 reference comparison failed" }
& pwsh -File (Join-Path $PSScriptRoot "run_rtl_reference_compare.ps1") -ArtifactDir (Join-Path $projectRoot "build/rtl_reference_compare/routed") -FractionBits 2 -RouteStages 1 -JoinLatency 2
if ($LASTEXITCODE) { throw "Fixed-point/routed reference comparison failed" }
& pwsh -File (Join-Path $PSScriptRoot "run_rtl_reference_compare.ps1") -ArtifactDir (Join-Path $projectRoot "build/rtl_reference_compare/single") -Layers 1 -FractionBits 4
if ($LASTEXITCODE) { throw "Single-layer reference comparison failed" }
& pwsh -File (Join-Path $PSScriptRoot "run_uvm.ps1")
if ($LASTEXITCODE) { throw "Public-port UVM regression failed" }
Write-Output "PASS: Fountain arithmetic, enabled-step timing, loading, global stalls, all-layer learning, RTL/reference states and UVM."
