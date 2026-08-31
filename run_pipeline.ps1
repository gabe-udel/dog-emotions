<#
End-to-end: DogFLW -> conversion table -> COCO -> head surgery -> fine-tune -> eval -> video

Windows port of run_pipeline.sh. Same seven steps, same arguments.

  .\run_pipeline.ps1
  .\run_pipeline.ps1 -P1Epochs 1 -P2Epochs 3
  .\run_pipeline.ps1 -Threads 6

Assumes src\extract_dogflw.py and src\build_dogboxes.py have already run
(build_dogboxes.py takes ~42 min - do not re-run it casually).
#>
[CmdletBinding()]
param(
    [int]$P1Epochs = 1,
    [int]$P2Epochs = 4,
    [int]$BatchSize = 8,
    # This box is 6 physical / 12 logical cores; OMP does best on physical cores.
    [int]$Threads = 6,
    # 0 = load in-process. Windows spawns (no fork) so workers are usually a net loss here.
    [int]$Workers = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$PY = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $PY)) {
    throw "venv python not found at $PY - create it with the setup steps in CLAUDE.md."
}
$env:OMP_NUM_THREADS = $Threads

function Step($msg) {
    Write-Host ''
    Write-Host "=============== $msg ===============" -ForegroundColor Cyan
}

# $ErrorActionPreference does not trap nonzero exits from native exes, so check explicitly.
function Invoke-Py {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PyArgs)
    & $PY @PyArgs
    if ($LASTEXITCODE -ne 0) { throw "FAILED (exit $LASTEXITCODE): python $($PyArgs -join ' ')" }
}

function Newest($pattern) {
    $f = Get-ChildItem $pattern -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $f) { throw "no file matching $pattern" }
    return $f.FullName
}

Step '1/7  SuperAnimal <-> DogFLW correspondence'
Invoke-Py src\analyze_correspondence.py

Step '2/7  COCO dataset (39 SuperAnimal + added DogFLW face keypoints)'
Invoke-Py src\make_coco.py

Step '3/7  extend the heatmap head'
Invoke-Py src\extend_head.py

Step '4/7  fine-tune, phase 1: frozen backbone (head only)'
Invoke-Py src\train_dogface.py --run-name phase1 --epochs $P1Epochs --batch-size $BatchSize `
    --unfreeze none --lr-head 1e-3 --save-epochs 1 --workers $Workers `
    --snapshot model_weights\superanimal_quadruped_hrnet_w32_dogface.pt

$P1 = Newest 'dlc_project\phase1\snapshot-*.pt'
Write-Host "phase 1 snapshot: $P1"

Step '5/7  fine-tune, phase 2: full network, low backbone LR'
Invoke-Py src\train_dogface.py --run-name phase2 --epochs $P2Epochs --batch-size $BatchSize `
    --unfreeze stage4 --lr-backbone 1e-5 --lr-head 2e-4 --save-epochs 1 `
    --workers $Workers --snapshot $P1

$FINAL = Newest 'dlc_project\phase2\snapshot-*.pt'
New-Item -ItemType Directory -Force -Path model_weights | Out-Null
Copy-Item $FINAL 'model_weights\superanimal_quadruped_dogface_final.pt' -Force
Copy-Item 'dlc_project\phase2\pytorch_config.yaml' 'model_weights\pytorch_config.yaml' -Force
Write-Host 'final snapshot: model_weights\superanimal_quadruped_dogface_final.pt'

Step '6/7  evaluate on the DogFLW test split'
New-Item -ItemType Directory -Force -Path outputs | Out-Null
& $PY src\evaluate.py --config model_weights\pytorch_config.yaml `
    --snapshot model_weights\superanimal_quadruped_dogface_final.pt `
    --out outputs\evaluation.json | Tee-Object -FilePath outputs\evaluation.txt
if ($LASTEXITCODE -ne 0) { throw "FAILED (exit $LASTEXITCODE): evaluate.py" }

Step '7/7  run on the dog-walking video'
Invoke-Py src\run_video.py --video videos\mixkit_1476.mp4 `
    --out outputs\dog_walk_dogface_comparison.mp4 `
    --config model_weights\pytorch_config.yaml `
    --snapshot model_weights\superanimal_quadruped_dogface_final.pt `
    --compare --width 960 --smooth 3 --start-frame 36 --max-frames 180 `
    --also-solo outputs\dog_walk_dogface.mp4

Invoke-Py src\make_figures.py --config model_weights\pytorch_config.yaml `
    --snapshot model_weights\superanimal_quadruped_dogface_final.pt

Write-Host ''
Write-Host 'pipeline complete' -ForegroundColor Green
