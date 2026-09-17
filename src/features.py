"""Per-frame emotion features from the two-stage cascade's keypoints.

Every feature is a pure function of one frame's (body pose, face landmarks, face box).
Temporal features - wag, rhythms, rates, motion energy - are NOT here; they operate on
series of these outputs and live in the aggregation layer, after the noise-floor prune
decides which of these survive.

Two rules this module exists to enforce:

**Look up keypoints by NAME, never by index.** The SuperAnimal channel order at runtime
is not the order the test fixtures list: `back_base` is 19, `back_end` 20, `back_middle`
21, and the rear legs interleave (`back_left_paw` 30, `back_left_thai` 31). Hard-coding
an index silently computes a different feature.

**A feature is NaN unless every keypoint it needs clears the confidence cut.** Never 0.
Zero is a real value here - 0 tuck means the tail is touching the hips, i.e. maximum
fear - so a missing measurement written as 0 is a confident wrong answer.

Where an earlier design decision was a judgement call between two formulations, BOTH are
computed, suffixed to say which is which. The point of this module is to feed a
noise-floor experiment that picks between them on measured signal-to-noise rather than
on argument.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------------------
# Keypoint access
# --------------------------------------------------------------------------------------

NAN = float("nan")

# Features measured in degrees. Defined here, beside the formulas that produce them, so
# that anything summarising or differentiating them knows to use circular statistics.
# A plain mean or standard deviation on a series resting near +/-180 reports a wild
# spread for a perfectly steady value: head_tilt measured 145 degrees of "spread" on
# real video where the circular figure is 1.23.
ANGULAR_FEATURES = frozenset({
    "ear_angle_L_eyeframe", "ear_angle_R_eyeframe",
    "ear_angle_L_basevec", "ear_angle_R_basevec",
    "tail_height", "head_tilt", "head_pitch",
})


class Points:
    """Name -> (x, y) for one frame, with the confidence cut already applied.

    Missing / low-confidence lookups return None, and every formula below propagates
    that to NaN rather than substituting a value.
    """

    def __init__(self, arr: np.ndarray | None, names: list[str], pcut: float) -> None:
        self.ok: dict[str, np.ndarray] = {}
        self.conf: dict[str, float] = {}
        if arr is None:
            return
        for i, n in enumerate(names):
            if i >= len(arr):
                break
            x, y, c = float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2])
            self.conf[n] = c
            if c >= pcut and math.isfinite(x) and math.isfinite(y):
                self.ok[n] = np.array([x, y], dtype=float)

    def __call__(self, *names: str) -> tuple[np.ndarray, ...] | None:
        """All named points, or None if any is unavailable."""
        got = []
        for n in names:
            p = self.ok.get(n)
            if p is None:
                return None
            got.append(p)
        return tuple(got)

    def mid(self, *names: str) -> np.ndarray | None:
        got = self(*names)
        return None if got is None else np.mean(np.stack(got), axis=0)

    def score(self, *names: str) -> float:
        vals = [self.conf.get(n, 0.0) for n in names]
        return min(vals) if vals else 0.0


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------

R90 = np.array([[0.0, -1.0], [1.0, 0.0]])


def unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else v / n


def signed_angle(a: np.ndarray, w: np.ndarray) -> float:
    """Signed angle FROM w TO a, degrees, in (-180, 180]."""
    cross = float(w[0] * a[1] - w[1] * a[0])
    dot = float(np.dot(w, a))
    return math.degrees(math.atan2(cross, dot))


def fit_line(pts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """Total-least-squares line through >=2 points: (origin, unit direction).

    Ordinary least squares would fail on a vertical arrangement; the principal
    eigenvector of the covariance has no preferred axis.
    """
    if len(pts) < 2:
        return None
    P = np.stack(pts)
    o = P.mean(axis=0)
    d = P - o
    if len(pts) == 2:
        return (o, unit(d[1] - d[0]) if unit(d[1] - d[0]) is not None else None)
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    return o, vt[0]


# --------------------------------------------------------------------------------------
# Reference frames
# --------------------------------------------------------------------------------------

@dataclass
class HeadFrame:
    E_R: np.ndarray
    E_L: np.ndarray
    mid: np.ndarray
    u: np.ndarray
    v: np.ndarray
    iod: float


def head_frame(f: Points) -> HeadFrame | None:
    """Eye-anchored frame. Built from the most accurate landmarks on the face
    (eye corners, 0.0096-0.0131) so that everything referenced to it inherits their
    precision rather than the ear landmarks' 0.039."""
    R = f.mid("eye_right_medial", "eye_right_lateral")
    L = f.mid("eye_left_medial", "eye_left_lateral")
    if R is None or L is None:
        return None
    u = unit(L - R)
    if u is None:
        return None
    iod = float(np.linalg.norm(L - R))
    v = R90 @ u
    mid = 0.5 * (R + L)
    # Orient "up the skull" AWAY from the muzzle, using the nose.
    #
    # This used to key off head_top, and that was a silent 180-degree flip generator:
    # head_top's mean confidence is 0.39, so it drops below the cut often, and on those
    # frames the sign correction simply did not run - leaving v at R90's arbitrary sign.
    # Every ear angle then jumped 180 degrees between consecutive frames. Confidence
    # cannot warn you either: head_top has the weakest confidence-error correlation on
    # the whole face (rho -0.10 to -0.13). nose_top is confidence 0.71, error 0.0099.
    # Explicit None checks, not `a or b`: these are numpy arrays, and `or` evaluates
    # their truth value, which raises for anything with more than one element.
    ref = f.ok.get("nose_top")
    if ref is None:
        ref = f.ok.get("nose_bottom")
    if ref is None:
        # Without a dependable orientation, v's sign is a coin flip and every feature
        # built on it is too. Refuse the frame rather than emit one that may be mirrored.
        return None
    if float(np.dot(v, mid - ref)) < 0:
        v = -v
    return HeadFrame(R, L, mid, u, v, iod)


@dataclass
class BodyFrame:
    b: np.ndarray
    n: np.ndarray
    L: float
    ground_o: np.ndarray | None
    ground_d: np.ndarray | None
    ground_n: np.ndarray | None


PAWS = ("front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw")


def body_frame(s: Points) -> BodyFrame | None:
    got = s("back_base", "back_end")
    if got is None:
        return None
    base, end = got
    L = float(np.linalg.norm(end - base))
    b = unit(end - base)
    if b is None or L < 1e-6:
        return None

    ground_o = ground_d = ground_n = None
    paws = [s.ok[p] for p in PAWS if p in s.ok]
    fit = fit_line(paws)
    if fit is not None and fit[1] is not None:
        ground_o, ground_d = fit
        # An SVD returns +/-v with no preferred sign, so the raw ground direction flips
        # arbitrarily between frames. Anything measured against it (head_tilt) would
        # jump 180 degrees for no physical reason, and differentiating that jump
        # manufactures an enormous rate. Pin the sign to the body axis.
        if float(np.dot(ground_d, b)) < 0:
            ground_d = -ground_d
        gn = R90 @ ground_d
        # point the normal at the dog, so "height" is positive above the ground
        if float(np.dot(gn, base - ground_o)) < 0:
            gn = -gn
        ground_n = gn
    return BodyFrame(b, R90 @ b, L, ground_o, ground_d, ground_n)


def height(bf: BodyFrame, q: np.ndarray | None) -> float:
    if q is None or bf.ground_o is None or bf.ground_n is None:
        return NAN
    return float(np.dot(q - bf.ground_o, bf.ground_n))


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------

def frame_features(body: np.ndarray | None, face: np.ndarray | None,
                   face_side: float | None, sa_names: list[str],
                   face_names: list[str], pcut: float = 0.35) -> dict[str, float]:
    """All per-frame features for one frame. Missing inputs give NaN, never 0."""
    s = Points(body, sa_names, pcut)
    f = Points(face, face_names, pcut)
    hf = head_frame(f)
    bf = body_frame(s)
    S = float(face_side) if face_side and face_side > 0 else NAN

    out: dict[str, float] = {}

    # ---------------------------------------------------------------- ears
    # Two formulations on purpose. 'eyeframe' measures the tip against the eye-anchored
    # skull frame and touches only one ear landmark, so it carries less noise; but it
    # conflates ear rotation with ear length and head shape, which 'basevec' partly
    # cancels. Without a per-dog baseline the cheaper-noise version may well be the
    # worse feature across dogs. Measure, don't guess.
    for side, tip, inner in (("L", "ear_left_tip", "ear_left_inner_base"),
                             ("R", "ear_right_tip", "ear_right_inner_base")):
        sign = 1.0 if side == "L" else -1.0
        a_eye = a_base = NAN
        if hf is not None and tip in f.ok:
            q = f.ok[tip] - hf.mid
            a_eye = sign * signed_angle(q, hf.v)
        got = f(tip, inner)
        if hf is not None and got is not None:
            d = got[0] - got[1]
            a_base = sign * signed_angle(d, hf.v)
        out[f"ear_angle_{side}_eyeframe"] = a_eye
        out[f"ear_angle_{side}_basevec"] = a_base
        # ear length: forward rotation foreshortens rather than rotating in 2-D, so the
        # angle alone cannot distinguish a forward ear from a shorter one
        out[f"ear_len_{side}"] = (float(np.linalg.norm(got[0] - got[1])) / S
                                  if got is not None and math.isfinite(S) else NAN)

    tips = f("ear_left_tip", "ear_right_tip")
    bases = f("ear_left_inner_base", "ear_right_inner_base")
    tip_sep = float(np.linalg.norm(tips[0] - tips[1])) if tips else NAN
    # Normalised by inter-ocular distance, not by the ear bases. Measured against ground
    # truth on 574 val images the base-separation variant lost on every axis: error
    # 0.706 vs 0.239, SNR 1.02 vs 4.55, coverage 287 vs 383 images. The eye corners are
    # simply a better ruler than a short span between two of the noisiest landmarks.
    out["ear_closeness_iod"] = (tip_sep / hf.iod
                                if tips and hf is not None and hf.iod > 1e-9 else NAN)

    if tips and hf is not None and math.isfinite(S):
        t_mid = 0.5 * (tips[0] + tips[1])
        out["ear_height"] = float(np.dot(t_mid - hf.mid, hf.v)) / S
    else:
        out["ear_height"] = NAN

    # ---------------------------------------------------------------- tail
    tail = s("tail_base", "tail_end")
    spine = s("back_middle", "back_end")
    out["tail_height"] = (signed_angle(tail[1] - tail[0], spine[1] - spine[0])
                          if tail and spine else NAN)

    hips = s.mid("back_left_thai", "back_right_thai")
    out["tail_tuck"] = (float(np.linalg.norm(tail[1] - hips)) / bf.L
                        if tail and hips is not None and bf is not None else NAN)

    out["tail_lateral"] = (float(np.dot(tail[1] - tail[0], bf.n)) / bf.L
                           if tail and bf is not None else NAN)
    out["tail_len"] = (float(np.linalg.norm(tail[1] - tail[0])) / bf.L
                       if tail and bf is not None else NAN)
    out["tail_conf"] = s.score("tail_base", "tail_end")

    # ---------------------------------------------------------------- body
    front = s.mid("front_left_thai", "front_right_thai")
    rear = s.mid("back_left_thai", "back_right_thai")
    if bf is not None:
        hf_front, hf_rear = height(bf, front), height(bf, rear)
        # subtraction, not a ratio: a ratio divides by hip height, which reaches zero
        # the instant a dog sits
        out["posture"] = ((hf_front - hf_rear) / bf.L
                          if math.isfinite(hf_front) and math.isfinite(hf_rear) else NAN)
        out["head_height"] = (height(bf, s.ok.get("nose")) / bf.L
                              if "nose" in s.ok else NAN)
    else:
        out["posture"] = out["head_height"] = NAN

    # ---------------------------------------------------------------- head
    # Referenced to the ground line, not the video frame: handheld footage is rarely
    # level, and a tilted camera would otherwise be recorded as a tilted dog.
    # Referenced to the FORWARD direction. ground_d is pinned to the body axis, which
    # runs back_base -> back_end, i.e. toward the TAIL; against it a level dog read
    # +/-180 and alternated sign frame to frame across the wrap. Negating puts neutral
    # at 0, where a derivative is continuous.
    tilt = NAN
    if hf is not None and bf is not None:
        ref = -bf.ground_d if bf.ground_d is not None else -bf.b
        tilt = signed_angle(hf.u, ref)
    out["head_tilt"] = tilt
    out["head_tilt_used_ground"] = (1.0 if bf is not None and bf.ground_d is not None
                                    else 0.0)

    # Referenced to the FORWARD body direction (-b runs back_end -> back_base, i.e.
    # toward the head), not to b itself. Against b a neutral dog reads about 158 deg,
    # parked next to the +/-180 wrap where any derivative goes discontinuous; against
    # forward it reads near 0. Negated so positive = nose up, since image y runs down.
    neck = s("neck_end",)
    out["head_pitch"] = (-signed_angle(s.ok["nose"] - neck[0], -bf.b)
                         if neck and "nose" in s.ok and bf is not None else NAN)

    # ---------------------------------------------------------------- face
    # Mouth opening as a JAW measure: philtrum (error 0.0091) to chin_upper (0.0171),
    # rather than lip_upper to lip_lower (0.023-0.025 either end). Measured on 574 val
    # images the jaw form wins on every axis - error 0.0201 vs 0.0269, SNR 3.76 vs 2.12,
    # coverage 539 vs 420 images - so the lip form is gone.
    ph_ch = f("philtrum", "chin_upper")
    out["mouth_open_jaw"] = (float(np.linalg.norm(ph_ch[0] - ph_ch[1])) / S
                             if ph_ch and math.isfinite(S) else NAN)

    # REMOVED, and the removals are the most consequential result of the validation.
    # Each was measured against ground truth on 574 val images as
    # SNR = (spread across dogs / its own error) - 1:
    #
    #   eyelid_open         0.03   error 0.0114 vs spread 0.0116
    #   eye_width           0.05   error 0.0122 vs spread 0.0125, and bias -0.0096
    #   lip_raise_upperlip  0.00   error 0.0127 vs spread 0.0127
    #   lip_raise_philtrum  0.69   error 0.0104 vs spread 0.0134
    #
    # In each case the measurement error is as large as everything the feature has to
    # say. This is not a resolution problem that a better crop fixes: eyelid_open rides
    # the two most accurate landmarks on the whole face (0.0096-0.0107) and still fails,
    # because the lid gap itself is only a few pixels. Two capabilities die with them -
    # there is NO usable eye-aperture signal, and no usable snarl signal, since a snarl
    # holds the mouth nearly shut and so is invisible to mouth_open_jaw. Reinstating
    # either needs better landmarks, not better arithmetic.

    # ---------------------------------------------------------------- trust
    out["view_yaw"] = hf.iod / S if hf is not None and math.isfinite(S) else NAN
    nt = f("nose_top",)
    if nt and hf is not None:
        dl = float(np.linalg.norm(nt[0] - hf.E_L))
        dr = float(np.linalg.norm(nt[0] - hf.E_R))
        out["view_dir"] = (dl - dr) / (dl + dr) if (dl + dr) > 1e-9 else NAN
    else:
        out["view_dir"] = NAN
    out["face_px"] = S
    out["body_L"] = bf.L if bf is not None else NAN
    out["n_paws"] = float(sum(1 for p in PAWS if p in s.ok))

    return out


def feature_names(sa_names: list[str], face_names: list[str]) -> list[str]:
    """Column order, from one empty call - the single source of truth."""
    return list(frame_features(None, None, None, sa_names, face_names).keys())
