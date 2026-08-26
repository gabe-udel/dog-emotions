#!/usr/bin/env bash
# End-to-end: DogFLW -> conversion table -> COCO -> head surgery -> fine-tune -> eval -> video
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

step() { echo; echo "=============== $* ==============="; }

step "1/7  SuperAnimal <-> DogFLW correspondence"
$PY src/analyze_correspondence.py

step "2/7  COCO dataset (39 SuperAnimal + added DogFLW face keypoints)"
$PY src/make_coco.py

step "3/7  extend the heatmap head"
$PY src/extend_head.py

step "4/7  fine-tune, phase 1: frozen backbone (head only)"
$PY src/train_dogface.py --run-name phase1 --epochs "${P1_EPOCHS:-1}" --batch-size 8 \
    --unfreeze none --lr-head 1e-3 --save-epochs 1 \
    --snapshot model_weights/superanimal_quadruped_hrnet_w32_dogface.pt

P1=$(ls -t dlc_project/phase1/snapshot-*.pt | head -1)
echo "phase 1 snapshot: $P1"

step "5/7  fine-tune, phase 2: full network, low backbone LR"
$PY src/train_dogface.py --run-name phase2 --epochs "${P2_EPOCHS:-4}" --batch-size 8 \
    --unfreeze stage4 --lr-backbone 1e-5 --lr-head 2e-4 --save-epochs 1 --snapshot "$P1"

FINAL=$(ls -t dlc_project/phase2/snapshot-*.pt | head -1)
cp "$FINAL" model_weights/superanimal_quadruped_dogface_final.pt
cp dlc_project/phase2/pytorch_config.yaml model_weights/pytorch_config.yaml
echo "final snapshot: model_weights/superanimal_quadruped_dogface_final.pt"

step "6/7  evaluate on the DogFLW test split"
$PY src/evaluate.py --config model_weights/pytorch_config.yaml \
    --snapshot model_weights/superanimal_quadruped_dogface_final.pt \
    --out outputs/evaluation.json | tee outputs/evaluation.txt

step "7/7  run on the dog-walking video"
mkdir -p outputs
$PY src/run_video.py --video videos/mixkit_1476.mp4 \
    --out outputs/dog_walk_dogface_comparison.mp4 \
    --config model_weights/pytorch_config.yaml \
    --snapshot model_weights/superanimal_quadruped_dogface_final.pt \
    --compare --width 960 --smooth 3 --start-frame 36 --max-frames 180 \
    --also-solo outputs/dog_walk_dogface.mp4

$PY src/make_figures.py --config model_weights/pytorch_config.yaml \
    --snapshot model_weights/superanimal_quadruped_dogface_final.pt

echo; echo "pipeline complete"
