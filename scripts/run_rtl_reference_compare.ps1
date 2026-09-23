param([string]$ArtifactDir = "")

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "questa_env.ps1")
if ($ArtifactDir -eq "") { $ArtifactDir = Join-Path $projectRoot "build/rtl_reference_compare" }
$ArtifactDir = [System.IO.Path]::GetFullPath($ArtifactDir)
New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null

$stimulus = Join-Path $ArtifactDir "stimulus.txt"
$trace = Join-Path $ArtifactDir "rtl_trace.txt"
$work = Join-Path $ArtifactDir "work"
if (Test-Path -LiteralPath $work) { Remove-Item -LiteralPath $work -Recurse -Force }
Push-Location $projectRoot
try { & python -m reference.rtl_reference_compare generate $stimulus; if ($LASTEXITCODE) { throw "stimulus generation failed" } }
finally { Pop-Location }

Push-Location $ArtifactDir
try {
    & $vlib work
    if ($LASTEXITCODE) { throw "vlib failed" }
    $sources = $coreSources + (Join-Path $projectRoot "weightStationaryVariant/nnAcceleratorStateTrace_tb.sv")
    & $vlog -sv -timescale 1ns/1ps @sources
    if ($LASTEXITCODE) { throw "RTL state-trace testbench compilation failed" }
    & $vsim -c work.nnAcceleratorStateTrace_tb "+STIMULUS=$stimulus" "+TRACE=$trace" `
        -l (Join-Path $ArtifactDir "simulation.log") -do "run -all; quit -f"
    if ($LASTEXITCODE) { throw "RTL state-trace simulation failed" }
}
finally { Pop-Location }

Push-Location $projectRoot
try { & python -m reference.rtl_reference_compare compare $stimulus $trace; if ($LASTEXITCODE) { throw "RTL trace comparison failed" } }
finally { Pop-Location }
