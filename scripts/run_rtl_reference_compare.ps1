param([string]$ArtifactDir = "")

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if ($ArtifactDir -eq "") { $ArtifactDir = Join-Path $projectRoot "build/rtl_reference_compare" }
$ArtifactDir = [System.IO.Path]::GetFullPath($ArtifactDir)
New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null

$candidateBins = @()
if ($env:NN_ACCEL_QUESTA_BIN) { $candidateBins += $env:NN_ACCEL_QUESTA_BIN }
$candidateBins += "C:\altera_lite\25.1std\questa_fse\win64"
$candidateBins += @(Get-Command vsim -All -ErrorAction SilentlyContinue | ForEach-Object { Split-Path -Parent $_.Source })
$questaBin = $candidateBins | Where-Object { Test-Path -LiteralPath (Join-Path $_ "vsim.exe") } | Select-Object -First 1
if (-not $questaBin) { throw "Questa/ModelSim was not found. Set NN_ACCEL_QUESTA_BIN." }
if (-not $env:SALT_LICENSE_SERVER) {
    $userSaltLicense = [Environment]::GetEnvironmentVariable("SALT_LICENSE_SERVER", "User")
    if ($userSaltLicense) { $env:SALT_LICENSE_SERVER = $userSaltLicense }
    elseif ($env:SALT_LICENSE_FILE) { $env:SALT_LICENSE_SERVER = $env:SALT_LICENSE_FILE }
}

$stimulus = Join-Path $ArtifactDir "stimulus.txt"
$trace = Join-Path $ArtifactDir "rtl_trace.txt"
$work = Join-Path $ArtifactDir "work"
if (Test-Path -LiteralPath $work) { Remove-Item -LiteralPath $work -Recurse -Force }
Push-Location $projectRoot
try { & python -m reference.rtl_reference_compare generate $stimulus; if ($LASTEXITCODE) { throw "stimulus generation failed" } }
finally { Pop-Location }

Push-Location $ArtifactDir
try {
    & (Join-Path $questaBin "vlib.exe") work
    if ($LASTEXITCODE) { throw "vlib failed" }
    $sources = @(
        (Join-Path $projectRoot "memory/signedFifo.sv"),
        (Join-Path $projectRoot "weightStationaryVariant/relu.sv"),
        (Join-Path $projectRoot "weightStationaryVariant/outputActivation.sv"),
        (Join-Path $projectRoot "weightStationaryVariant/weightedVectorReduction.sv"),
        (Join-Path $projectRoot "weightStationaryVariant/weightStationaryProcessingElement.sv"),
        (Join-Path $projectRoot "weightStationaryVariant/weightStationarySystolicArray.sv"),
        (Join-Path $projectRoot "weightStationaryVariant/weightStationaryMatrixMultiplier.sv"),
        (Join-Path $projectRoot "weightStationaryVariant/nnAccelerator.sv"),
        (Join-Path $projectRoot "weightStationaryVariant/nnAcceleratorStateTrace_tb.sv")
    )
    & (Join-Path $questaBin "vlog.exe") -sv -timescale 1ns/1ps @sources
    if ($LASTEXITCODE) { throw "RTL state-trace testbench compilation failed" }
    & (Join-Path $questaBin "vsim.exe") -c work.nnAcceleratorStateTrace_tb "+STIMULUS=$stimulus" "+TRACE=$trace" `
        -l (Join-Path $ArtifactDir "simulation.log") -do "run -all; quit -f"
    if ($LASTEXITCODE) { throw "RTL state-trace simulation failed" }
}
finally { Pop-Location }

Push-Location $projectRoot
try { & python -m reference.rtl_reference_compare compare $stimulus $trace; if ($LASTEXITCODE) { throw "RTL trace comparison failed" } }
finally { Pop-Location }
