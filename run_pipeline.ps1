# Build and train the two-stage face cascade, end to end.
#
# Assumes src\extract_dogflw.py and src\build_dogboxes.py have already run - their
# outputs (data\dogflw\, data\sa_dogboxes.npz) are on disk and build_dogboxes takes
# about 42 minutes, so do not re-run it casually.
#
#   .\run_pipeline.ps1                      # full run
#   .\run_pipeline.ps1 -SkipTrain           # rebuild data + tests, stop before training
#   .\run_pipeline.ps1 -Epochs 6 -Threads 8
#
# PowerShell 5.1 has no `&&`, so every step checks $LASTEXITCODE explicitly.

param(
    [int]$Epochs = 4,
    [int]$BatchSize = 8,
    [int]$Threads = 6,
    [int]$Workers = 0,
    [double]$Pad = 1.8,
    [int]$PosDistThresh = 8,
    [string]$RunName = "face1",
    [switch]$SkipTrain
)

$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
$env:OMP_NUM_THREADS = "$Threads"
$env:PYTHONUNBUFFERED = "1"

function Step($n, $what) {
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor Cyan
    Write-Host "  $n. $what" -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor Cyan
}

function Check($what) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $what (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

if (-not (Test-Path $py)) {
    Write-Host "No venv at $py - run setup.bat first." -ForegroundColor Red
    exit 1
}
foreach ($need in @("data\dogflw\annotations.json", "data\sa_dogboxes.npz")) {
    if (-not (Test-Path $need)) {
        Write-Host "Missing $need - run src\extract_dogflw.py / src\build_dogboxes.py first." -ForegroundColor Red
        exit 1
    }
}

Step 1 "Tests - coordinate transforms, face box, decoding, splits"
& $py -m pytest tests\ -q
Check "tests"

Step 2 "Train / val / test split (val carved out of train; test untouched)"
& $py src\splits.py
Check "splits"

Step 3 "COCO dataset from DERIVED face boxes"
& $py src\build_face_coco.py --pad $Pad
Check "build_face_coco"

if ($SkipTrain) {
    Write-Host ""
    Write-Host "-SkipTrain set: stopping before training." -ForegroundColor Yellow
    exit 0
}

Step 4 "Train the 46-keypoint face model"
& $py src\train_face.py --run-name $RunName --epochs $Epochs --batch-size $BatchSize `
    --workers $Workers --pos-dist-thresh $PosDistThresh --eval-every 1 --max-snapshots 20
Check "train_face"

$snap = Get-ChildItem "face_project\$RunName\snapshot-*.pt" |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $snap) {
    Write-Host "No snapshot written by training." -ForegroundColor Red
    exit 1
}
Write-Host "using snapshot $($snap.FullName)"

Step 5 "Evaluate on VALIDATION (tune here; test is scored once, separately)"
& $py src\evaluate_face.py --split val `
    --config "face_project\$RunName\pytorch_config.yaml" --snapshot $snap.FullName
Check "evaluate_face"

Write-Host ""
Write-Host "Done. To score the held-out test split ONCE, when tuning is finished:" -ForegroundColor Green
Write-Host "  $py src\evaluate_face.py --split test --snapshot `"$($snap.FullName)`"" -ForegroundColor Green
Write-Host ""
Write-Host "To render a video:" -ForegroundColor Green
Write-Host "  $py src\run_video.py --video `"Happy lab.mov`" --snapshot `"$($snap.FullName)`"" -ForegroundColor Green
