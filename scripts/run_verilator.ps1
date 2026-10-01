param(
    [ValidateRange(1, 16)][int]$BuildJobs = 2,
    [switch]$ConventionalBaseline,
    [string]$BaselineRef = "09a409725e348b324f5cf87b61b907c2215d1e33"
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "verilator_env.ps1")
$artifactDir = Join-Path $projectRoot "build/verilator"
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
$coreSources = @(
    "memory/signedFifo.sv",
    "weightStationaryVariant/weightStationaryProcessingElement.sv",
    "weightStationaryVariant/weightStationarySystolicArray.sv",
    "weightStationaryVariant/weightStationaryMatrixMultiplier.sv",
    "weightStationaryVariant/weightedVectorReduction.sv",
    "weightStationaryVariant/nnAccelerator.sv"
)

function Build-VerilatorTest {
    param([string]$Top, [string]$Directory, [string[]]$ExtraArguments = @(),
          [string[]]$SourceFiles = @())
    $modelDir = Join-Path $verilatorBuildRoot $Directory
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
    $buildLog = Join-Path $artifactDir "$Top-build.log"
    $sources = if ($SourceFiles.Count) { $SourceFiles } else { $coreSources }
    if ($Top.EndsWith("_tb")) {
        $sources += "weightStationaryVariant/$Top.sv"
    }
    & $verilator --binary --timing --assert -Wno-fatal -j $BuildJobs `
        --top-module $Top --Mdir ($modelDir -replace '\\', '/') `
        @ExtraArguments @sources *> $buildLog
    if ($LASTEXITCODE -ne 0) {
        Get-Content -LiteralPath $buildLog -Tail 25
        throw "Verilator build failed for $Top. See $buildLog"
    }
    return Join-Path $modelDir "V$Top.exe"
}

function Invoke-VerilatorTest {
    param([string]$Top, [string]$Directory)
    $binary = Build-VerilatorTest $Top $Directory
    $runLog = Join-Path $artifactDir "$Top-run.log"
    & $binary *> $runLog
    if ($LASTEXITCODE -ne 0) {
        Get-Content -LiteralPath $runLog -Tail 25
        throw "Verilator simulation failed for $Top. See $runLog"
    }
    Get-Content -LiteralPath $runLog | Where-Object { $_ -match "PASS:" }
}

Push-Location $projectRoot
try {
    & python -m unittest discover -s reference -p "test_*.py"
    if ($LASTEXITCODE -ne 0) { throw "Python reference tests failed." }
    Invoke-VerilatorTest "weightedVectorReduction_tb" "reduction"
    Invoke-VerilatorTest "weightStationaryMatrixMultiplier_tb" "core"
    Invoke-VerilatorTest "matrixWeightUpdateWave_tb" "wave"

    $invalidBinary = Build-VerilatorTest "weightStationaryMatrixMultiplier" "invalid" @("-GN=1")
    $invalidLog = Join-Path $artifactDir "invalid-parameter.log"
    & $invalidBinary *> $invalidLog
    if ($LASTEXITCODE -eq 0 -or -not (Select-String -LiteralPath $invalidLog `
            -SimpleMatch -Pattern "WIDTH>=1 and N>=2" -Quiet)) {
        throw "N=1 parameter rejection did not produce its intended diagnostic."
    }
    Write-Output "PASS: unsupported N=1 rejects with the parameter diagnostic."

    $traceBinary = Build-VerilatorTest "nnAcceleratorStateTrace_tb" "trace"
    $stimulus = Join-Path $artifactDir "stimulus.txt"
    $trace = Join-Path $artifactDir "rtl_trace.txt"
    & python -m reference.rtl_reference_compare generate $stimulus
    if ($LASTEXITCODE -ne 0) { throw "State-trace stimulus generation failed." }
    & $traceBinary "+STIMULUS=$stimulus" "+TRACE=$trace" *> (Join-Path $artifactDir "trace-run.log")
    if ($LASTEXITCODE -ne 0) { throw "RTL state-trace simulation failed." }
    & python -m reference.rtl_reference_compare compare $stimulus $trace
    if ($LASTEXITCODE -ne 0) { throw "RTL/reference comparison failed." }

    if ($ConventionalBaseline) {
        $baselineDir = Join-Path $artifactDir "baseline-rtl"
        New-Item -ItemType Directory -Force -Path $baselineDir | Out-Null
        $baselineSources = @()
        foreach ($name in @("weightStationaryProcessingElement",
                            "weightStationarySystolicArray",
                            "weightStationaryMatrixMultiplier")) {
            $source = & git show "${BaselineRef}:weightStationaryVariant/$name.sv"
            if ($LASTEXITCODE -ne 0) { throw "Cannot read conventional baseline $BaselineRef." }
            $path = Join-Path $baselineDir "$name.sv"
            Set-Content -LiteralPath $path -Value $source -Encoding utf8
            $baselineSources += $path
        }
        $baselineBinary = Build-VerilatorTest "weightStationaryMatrixMultiplier_tb" `
            "conventional" @("+define+CONVENTIONAL_BASELINE") $baselineSources
        $baselineLog = Join-Path $artifactDir "conventional-run.log"
        & $baselineBinary *> $baselineLog
        if ($LASTEXITCODE -ne 0) { throw "Conventional-array comparison failed. See $baselineLog" }
        Get-Content -LiteralPath $baselineLog | Where-Object { $_ -match "PASS:" }
        Write-Output "PASS: conventional and inward cores used identical signed/random vectors and the same dot-product scoreboard."
    }
}
finally {
    Pop-Location
}
Write-Output "PASS: Verilator references, arithmetic, inward array, loading, learning, stalls, and RTL traces. UVM runs separately with Questa."
