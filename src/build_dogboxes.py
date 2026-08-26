"""Pass 1 over DogFLW: SuperAnimal detector -> whole-dog box, then SuperAnimal pose -> 39 keypoints.

Whole-dog boxes (not the DogFLW face box) are used deliberately: the fine-tuned model is
top-down, so its training crops must match what it will see at inference, where a detector
supplies a whole-animal box.  The 39 predicted keypoints become memory-replay pseudo-labels.
"""
from __future__ import annotations
import json, sys, time
import numpy as np
sys.path.insert(0, "src")
import superanimal as sa

sa.DETECTOR = "fasterrcnn_mobilenet_v3_large_fpn"
OUT = "data/sa_dogboxes.npz"
MIN_SCORE = 0.5


def iou_and_containment(box, face):
    """box, face are xyxy. Returns (iou, fraction of `face` inside `box`)."""
    ix1, iy1 = max(box[0], face[0]), max(box[1], face[1])
    ix2, iy2 = min(box[2], face[2]), min(box[3], face[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ab = (box[2] - box[0]) * (box[3] - box[1])
    af = (face[2] - face[0]) * (face[3] - face[1])
    union = ab + af - inter
    return (inter / union if union > 0 else 0.0), (inter / af if af > 0 else 0.0)


def main():
    recs = json.load(open("data/dogflw/annotations.json"))
    _, (pose_runner, det_runner) = sa.build_runners("cpu", max_individuals=3, with_detector=True)

    boxes = np.zeros((len(recs), 4), np.float32)     # xyxy, the crop box used
    srcs = np.zeros(len(recs), np.int8)              # 2=detector+contains face, 1=detector, 0=fallback
    scores = np.zeros(len(recs), np.float32)
    poses = np.zeros((len(recs), 39, 3), np.float32)

    t, B = time.time(), 16
    for i in range(0, len(recs), B):
        chunk = recs[i:i + B]
        paths = [f"data/dogflw/{r['file']}" for r in chunk]
        dets = det_runner.inference(images=paths)

        pose_items = []
        for k, (r, d) in enumerate(zip(chunk, dets)):
            j = i + k
            face = np.array(r["bbox_xyxy"], float)
            bb = np.asarray(d["bboxes"], float).reshape(-1, 4)   # xywh
            sc = np.asarray(d.get("bbox_scores", []), float).reshape(-1)
            cand = []
            for b, s in zip(bb, sc):
                if s < MIN_SCORE:
                    continue
                xyxy = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])
                _, contain = iou_and_containment(xyxy, face)
                cand.append((contain >= 0.8, s, xyxy))
            if cand:
                good = [c for c in cand if c[0]]
                pick = max(good or cand, key=lambda c: c[1])
                boxes[j], scores[j], srcs[j] = pick[2], pick[1], (2 if pick[0] else 1)
            else:
                # no usable detection: fall back to the face box grown by 60%
                w, h = face[2] - face[0], face[3] - face[1]
                boxes[j] = [face[0] - .3 * w, face[1] - .3 * h, face[2] + .3 * w, face[3] + .3 * h]
                srcs[j] = 0
            x1, y1, x2, y2 = boxes[j]
            pose_items.append((paths[k], np.array([[x1, y1, x2 - x1, y2 - y1]])))

        for k, p in enumerate(sa.pose_on_boxes(pose_runner, pose_items)):
            poses[i + k] = p[0]

        done = min(i + B, len(recs)); el = time.time() - t
        if (i // B) % 10 == 0:
            print(f"  {done}/{len(recs)}  {el:.0f}s  eta {el/done*(len(recs)-done)/60:.1f}min", flush=True)

    np.savez_compressed(OUT, boxes=boxes, srcs=srcs, scores=scores, poses=poses,
                        ids=np.array([r["id"] for r in recs]),
                        splits=np.array([r["split"] for r in recs]))
    print("wrote", OUT)
    print("box source: detector+contains-face", int((srcs == 2).sum()),
          " detector-only", int((srcs == 1).sum()), " fallback", int((srcs == 0).sum()))


if __name__ == "__main__":
    main()
