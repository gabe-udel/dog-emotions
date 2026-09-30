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
    """Draws a cascade result: 39 body keypoints and 46 face landmarks, kept separate.

    The unified model needed a `dogflw_to_model_idx` map to find the face channels
    inside a 76-output array. The cascade returns two arrays from two models, each in
    its own natural ordering, so the indirection - and the keypoint_map.json it read -
    is gone. Body index is SuperAnimal's 0..38; face index is DogFLW's 0..45.
    """

    def __init__(self, sa_bodyparts: list[str]):
        from keypoint_scheme import ACTIVE, CH_OF, DOGFLW_NAMES, REGION_OF
        self.sa_bodyparts = list(sa_bodyparts)
        self.idx = {b: i for i, b in enumerate(self.sa_bodyparts)}
        self.dogflw_names = DOGFLW_NAMES
        self.region_of = REGION_OF
        self.active = list(ACTIVE)
        self.sa_edges = [(self.idx[a], self.idx[b]) for a, b in SA_EDGES
                         if a in self.idx and b in self.idx]
        # FACE_CONTOURS is written in DogFLW indices; the face array is in CHANNEL
        # order. Translate, dropping points the model no longer predicts - a contour
        # reduced to fewer than two surviving points draws nothing.
        self.face_contours = [seg for seg in
                              ([CH_OF[i] for i in c if i in CH_OF]
                               for c in FACE_CONTOURS)
                              if len(seg) > 1]

    def face_colour(self, channel: int):
        """Colour for a model channel, resolved through its DogFLW index."""
        return REGION_COLOR[self.region_of[self.active[channel]]]

    def draw(self, img, body=None, face=None, pcut=0.25, r=3, thick=1, scale=1.0,
             lines=True, face_only=False):
        """Draw in place. `body` is (39,3), `face` is (46,3); either may be None.

        lines=False draws bare points with no skeleton edges or face contours. The
        contours connect landmarks in an assumed order, so where a landmark is
        misplaced the line exaggerates it into a visible spike - points alone show the
        model's actual output.
        """
        s = max(1, int(round(scale)))
        if body is not None and not face_only:
            vis = body[:, 2] >= pcut
            if lines:
                for a, b in self.sa_edges:
                    if vis[a] and vis[b]:
                        cv2.line(img, tuple(body[a, :2].astype(int)),
                                 tuple(body[b, :2].astype(int)),
                                 SA_EDGE_COLOR, thick * s, cv2.LINE_AA)
            for i in range(len(body)):
                if vis[i]:
                    self._dot(img, body[i, :2], SA_COLOR, r, s)

        if face is not None:
            vis = face[:, 2] >= pcut
            if lines:
                for c in self.face_contours:
                    for a, b in zip(c[:-1], c[1:]):
                        if a < len(face) and b < len(face) and vis[a] and vis[b]:
                            ca, cb = self.face_colour(a), self.face_colour(b)
                            cv2.line(img, tuple(face[a, :2].astype(int)),
                                     tuple(face[b, :2].astype(int)),
                                     ca if ca == cb else (200, 200, 200),
                                     max(1, thick * s), cv2.LINE_AA)
            for i in range(len(face)):
                if vis[i]:
                    self._dot(img, face[i, :2], self.face_colour(i), r, s)
        return img

    @staticmethod
    def _dot(img, xy, colour, r, s):
        p = tuple(np.asarray(xy).astype(int))
        cv2.circle(img, p, r * s + 1, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(img, p, r * s, colour, -1, cv2.LINE_AA)


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


# Feature-name prefix -> colour, so a row in the readout matches the landmarks it was
# computed from. tail and motion have no facial region, so they get their own.
FEATURE_COLOR = {
    "ear":    REGION_COLOR["ear"],
    "head":   REGION_COLOR["head"],
    "mouth":  REGION_COLOR["mouth"],
    "eye":    REGION_COLOR["eye"],
    "nose":   REGION_COLOR["nose"],
    "tail":   (140, 210, 120),
    "motion": (200, 200, 200),
    "view":   (170, 170, 170),
}
MISSING = (110, 110, 120)


def feature_color(name: str):
    return FEATURE_COLOR.get(name.split("_", 1)[0], (210, 210, 210))


def feature_panel(img, values, ranges, angular=frozenset(), org=None,
                  width=330, line_h=27, scale=0.44, title="live features"):
    """Per-frame feature readout, one row per feature, with a clip-relative bar.

    A NaN is drawn as a dimmed row reading '--', never omitted and never shown as 0.
    features.py refuses to write 0 for a missing measurement because 0 is a real value
    there - 0 tuck means the tail is against the hips, i.e. maximum fear. A panel that
    quietly dropped the row, or drew an empty bar, would hand that rule straight back.

    The bar is scaled to the range this feature actually covered in THIS clip, so it
    shows movement, not absolute magnitude. Two clips' bars are not comparable.

    Args:
        values: {feature name: value for this frame}
        ranges: {feature name: (lo, hi)} over the whole clip
        angular: names measured in degrees, labelled with a degree sign
    """
    h, w = img.shape[:2]
    x, y = org if org else (w - width - 16, 60)
    rows = list(values.items())
    box_h = line_h * len(rows) + 34

    ov = img.copy()
    cv2.rectangle(ov, (x - 12, y - 26), (x + width, y + box_h - 26), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    cv2.putText(img, title, (x - 4, y - 8), cv2.FONT_HERSHEY_DUPLEX, 0.46,
                (235, 235, 235), 1, cv2.LINE_AA)

    for k, (name, val) in enumerate(rows):
        yy = y + k * line_h + 12
        ok = val is not None and np.isfinite(val)
        col = feature_color(name) if ok else MISSING
        unit = "deg" if name in angular else ""
        txt = f"{val:.2f}{unit}" if ok else "--"

        cv2.putText(img, name, (x - 4, yy), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    col if ok else MISSING, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(img, txt, (x + width - tw - 14, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (245, 245, 245) if ok else MISSING, 1, cv2.LINE_AA)

        bx0, bx1, by = x - 4, x + width - 14, yy + 7
        cv2.line(img, (bx0, by), (bx1, by), (58, 58, 64), 3, cv2.LINE_AA)
        lo, hi = ranges.get(name, (None, None))
        if ok and lo is not None and hi is not None and hi > lo:
            f = min(max((val - lo) / (hi - lo), 0.0), 1.0)
            cv2.line(img, (bx0, by), (int(bx0 + f * (bx1 - bx0)), by), col, 3,
                     cv2.LINE_AA)
    return img
