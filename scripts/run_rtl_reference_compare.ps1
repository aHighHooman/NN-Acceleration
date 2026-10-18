param(
    [string]$ArtifactDir = "",
    [ValidateSet(3, 5, 9)][int]$N = 3,
    [ValidateRange(1, 10)][int]$Layers = 3,
    [ValidateRange(0, 8)][int]$FractionBits = 0,
    [ValidateRange(0, 8)][int]$RouteStages = 0,
    [ValidateRange(1, 8)][int]$JoinLatency = 1
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "questa_env.ps1")
if ($ArtifactDir -eq "") { $ArtifactDir = Join-Path $projectRoot "build/rtl_reference_compare" }
$ArtifactDir = [System.IO.Path]::GetFullPath($ArtifactDir)
New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null
$stimulus = Join-Path $ArtifactDir "stimulus.txt"
$trace = Join-Path $ArtifactDir "rtl_trace.txt"
$modelArgs = @("--n", $N, "--layers", $Layers, "--fraction-bits", $FractionBits, "--g", $RouteStages, "--j", $JoinLatency)
Push-Location $projectRoot
try {
    & python -m reference.rtl_reference_compare generate $stimulus @modelArgs
    if ($LASTEXITCODE) { throw "Fountain stimulus generation failed" }
} finally { Pop-Location }
Push-Location $ArtifactDir
try {
    if (-not (Test-Path -LiteralPath "work")) { & $vlib work; if ($LASTEXITCODE) { throw "vlib failed" } }
    $sources = $coreSources + (Join-Path $projectRoot "weightStationaryVariant/nnAcceleratorStateTrace_tb.sv")
    & $vlog -sv -timescale 1ns/1ps @sources *> compile.log
    if ($LASTEXITCODE) { Get-Content -LiteralPath compile.log -Tail 30; throw "State-trace compilation failed" }
    & $vsim -c work.nnAcceleratorStateTrace_tb "-GN=$N" "-GLAYERS=$Layers" "-GFRACTION_BITS=$FractionBits" `
        "-GG=$RouteStages" "-GJ=$JoinLatency" "+STIMULUS=$stimulus" "+TRACE=$trace" `
        -l simulation.log -do "run -all; quit -f" *> console.log
    $logText = Get-Content -LiteralPath simulation.log -Raw
    if ($LASTEXITCODE -or $logText -match "\*\* (Error|Fatal):" -or $logText -notmatch "PASS:") {
        Get-Content -LiteralPath simulation.log -Tail 30
        throw "State-trace simulation failed"
    }
} finally { Pop-Location }
Push-Location $projectRoot
try {
    & python -m reference.rtl_reference_compare compare $stimulus $trace @modelArgs
    if ($LASTEXITCODE) { throw "Fountain RTL/reference comparison failed" }
} finally { Pop-Location }
