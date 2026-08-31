"""Skeletons, palettes and frame rendering for the labelled videos."""
from __future__ import annotations
import json
import cv2
import numpy as np

# ---- SuperAnimal-Quadruped skeleton (by bodypart name) -----------------------------
SA_EDGES = [
    ("nose", "upper_jaw"), ("upper_jaw", "lower_jaw"),
    ("nose", "mouth_end_right"), ("nose", "mouth_end_left"),
    ("nose", "right_eye"), ("nose", "left_eye"),
    ("right_eye", "right_earbase"), ("right_earbase", "right_earend"),
    ("left_eye", "left_earbase"), ("left_earbase", "left_earend"),
    ("right_eye", "left_eye"),
    ("nose", "throat_base"), ("throat_base", "throat_end"), ("throat_end", "neck_base"),
    ("right_earbase", "neck_end"), ("left_earbase", "neck_end"), ("neck_end", "neck_base"),
    ("neck_base", "back_base"), ("back_base", "back_middle"), ("back_middle", "back_end"),
    ("back_end", "tail_base"), ("tail_base", "tail_end"),
    ("back_base", "front_left_thai"), ("front_left_thai", "front_left_knee"),
    ("front_left_knee", "front_left_paw"),
    ("back_base", "front_right_thai"), ("front_right_thai", "front_right_knee"),
    ("front_right_knee", "front_right_paw"),
    ("back_end", "back_left_thai"), ("back_left_thai", "back_left_knee"),
    ("back_left_knee", "back_left_paw"),
    ("back_end", "back_right_thai"), ("back_right_thai", "back_right_knee"),
    ("back_right_knee", "back_right_paw"),
    ("belly_bottom", "body_middle_right"), ("belly_bottom", "body_middle_left"),
    ("body_middle_right", "back_middle"), ("body_middle_left", "back_middle"),
]

# ---- contours over the DogFLW face landmarks, given by DogFLW index ----------------
FACE_CONTOURS = [
    [0, 2, 4, 6, 8, 10, 12, 0],            # right ear
    [1, 3, 5, 7, 9, 11, 13, 1],            # left ear
    [16, 20, 18, 22, 16],                  # right eye
    [17, 21, 19, 23, 17],                  # left eye
    [25, 26, 33, 32, 34, 27, 25],          # nose / nostrils
    [30, 28, 24, 29, 31],                  # muzzle top
    [39, 36, 35, 37, 40],                  # upper lip
    [39, 43, 45, 44, 40],                  # lower lip
    [41, 42],                              # chin
    [38, 35], [38, 45],                    # mouth centre
]

REGION_COLOR = {           # BGR
    "ear":    (60, 170, 255),
    "head":   (60, 220, 255),
    "eye":    (255, 90, 220),
    "nose":   (90, 240, 120),
    "muzzle": (60, 230, 230),
    "mouth":  (90, 110, 255),
}
SA_COLOR = (255, 200, 60)          # SuperAnimal body keypoints (BGR: light blue)
SA_EDGE_COLOR = (215, 165, 45)


class Renderer:
    def __init__(self, keypoint_map_path="data/keypoint_map.json"):
        km = json.load(open(keypoint_map_path))
        self.km = km
        self.bodyparts = km["bodyparts"]
        self.sa_bodyparts = km["superanimal_bodyparts"]
        self.idx = {b: i for i, b in enumerate(self.bodyparts)}
        self.d2m = {int(k): v for k, v in km["dogflw_to_model_idx"].items()}
        self.regions = km["regions"]
        from keypoint_scheme import DOGFLW_NAMES
        self.dogflw_names = DOGFLW_NAMES
        self.n_sa = len(self.sa_bodyparts)
        self.sa_edges = [(self.idx[a], self.idx[b]) for a, b in SA_EDGES
                         if a in self.idx and b in self.idx]
        # A trimmed map (analyze_correspondence.py --trim) has no channel for the dropped
        # contour points, so skip them rather than KeyError. A contour reduced to a single
        # surviving point draws nothing.
        self.face_contours = [seg for seg in
                              ([self.d2m[i] for i in c if i in self.d2m] for c in FACE_CONTOURS)
                              if len(seg) > 1]
        self.face_indices = sorted(set(self.d2m.values()))

    def colour_of(self, model_idx: int):
        for di, mi in self.d2m.items():
            if mi == model_idx and mi >= self.n_sa:
                return REGION_COLOR[self.regions[self.dogflw_names[di]]]
        return SA_COLOR

    def draw(self, img, kpts, pcut=0.25, r=3, thick=1, face_only=False, scale=1.0,
             lines=True):
        """kpts: (K,3) x,y,score in image coordinates. Draws in place.

        lines=False draws bare points with no skeleton edges or face contours. The
        contours connect landmarks in an assumed order, so where a landmark is
        misplaced the line exaggerates it into a visible spike - points alone show
        the model's actual output.
        """
        vis = kpts[:, 2] >= pcut
        s = max(1, int(round(scale)))
        if lines and not face_only:
            for a, b in self.sa_edges:
                if vis[a] and vis[b]:
                    cv2.line(img, tuple(kpts[a, :2].astype(int)), tuple(kpts[b, :2].astype(int)),
                             SA_EDGE_COLOR, thick * s, cv2.LINE_AA)
        if lines:
            for c in self.face_contours:
                for a, b in zip(c[:-1], c[1:]):
                    if vis[a] and vis[b]:
                        col = self.colour_of(a) if self.colour_of(a) == self.colour_of(b) else (200, 200, 200)
                        cv2.line(img, tuple(kpts[a, :2].astype(int)), tuple(kpts[b, :2].astype(int)),
                                 col, max(1, thick * s), cv2.LINE_AA)
        for i in range(len(kpts)):
            if not vis[i]:
                continue
            if face_only and i not in self.face_indices:
                continue
            p = tuple(kpts[i, :2].astype(int))
            cv2.circle(img, p, r * s + 1, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(img, p, r * s, self.colour_of(i), -1, cv2.LINE_AA)
        return img


def banner(img, text, sub=None, org=(18, 16), scale=0.62):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
    pad = 8
    hh = h + 2 * pad + (int(h * 0.95) + 4 if sub else 0)
    ov = img.copy()
    cv2.rectangle(ov, (org[0] - pad, org[1] - pad), (org[0] + w + 220, org[1] + hh), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.45, img, 0.55, 0, img)
    cv2.putText(img, text, (org[0], org[1] + h + 2), cv2.FONT_HERSHEY_DUPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (org[0], org[1] + 2 * h + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    scale * 0.78, (190, 220, 255), 1, cv2.LINE_AA)
    return img


def legend(img, entries, org, scale=0.5, line_h=20):
    x, y = org
    ov = img.copy()
    cv2.rectangle(ov, (x - 10, y - 16), (x + 210, y + line_h * len(entries)), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.45, img, 0.55, 0, img)
    for k, (label, col) in enumerate(entries):
        yy = y + k * line_h
        cv2.circle(img, (x + 4, yy - 4), 4, col, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x + 16, yy), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (235, 235, 235), 1, cv2.LINE_AA)
    return img
