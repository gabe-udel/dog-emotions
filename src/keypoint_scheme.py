"""Naming and grouping of the 46 DogFLW facial landmarks.

DogFLW ships coordinates only - no landmark manual - so the groups below are derived
from the dataset itself: the mean face shape (each face normalised into its own
annotation bounding box, averaged over all 4335 images) is almost perfectly bilaterally
symmetric, giving 20 mirror pairs and 6 midline points.  Side labels ("right"/"left")
follow the SuperAnimal convention (the *animal's* right, which appears on image-left for
a dog facing the camera) and are verified against the SuperAnimal model's own
right_eye / left_eye predictions in analyze_correspondence.py.
"""

# DogFLW index -> descriptive name.  The index is kept in the name for the ear-contour
# points, where the mean shape gives an unambiguous ordering around the ear but no
# anatomical label.
DOGFLW_NAMES = {
    # ---- ears: 7 contour points each, ordered inner-base -> tip -> outer-base ----
    0: "ear_right_inner_base",  1: "ear_left_inner_base",
    2: "ear_right_inner_mid",   3: "ear_left_inner_mid",
    4: "ear_right_inner_low",   5: "ear_left_inner_low",
    6: "ear_right_tip",         7: "ear_left_tip",
    8: "ear_right_outer_low",   9: "ear_left_outer_low",
    10: "ear_right_outer_mid",  11: "ear_left_outer_mid",
    12: "ear_right_outer_base", 13: "ear_left_outer_base",
    # ---- top of head, between the ears ----
    14: "head_top_right", 15: "head_top_left",
    # ---- eyes: medial / lateral canthus + upper / lower lid ----
    16: "eye_right_medial", 17: "eye_left_medial",
    18: "eye_right_lateral", 19: "eye_left_lateral",
    20: "eye_right_upper_lid", 21: "eye_left_upper_lid",
    22: "eye_right_lower_lid", 23: "eye_left_lower_lid",
    # ---- muzzle / nose bridge ----
    24: "nose_bridge",
    28: "muzzle_right_upper", 29: "muzzle_left_upper",
    30: "cheek_right", 31: "cheek_left",
    # ---- nose ----
    25: "nose_top", 26: "nostril_right", 27: "nostril_left",
    32: "nose_bottom", 33: "nose_right_lower", 34: "nose_left_lower",
    # ---- lips / mouth / chin ----
    36: "lip_upper_right", 37: "lip_upper_left",
    35: "philtrum",
    38: "mouth_center",
    39: "mouth_corner_right", 40: "mouth_corner_left",
    43: "lip_lower_right", 44: "lip_lower_left",
    45: "lip_lower_center",
    41: "chin_upper", 42: "chin_bottom",
}

REGION_OF = {}
for _i, _n in DOGFLW_NAMES.items():
    REGION_OF[_i] = ("ear" if _n.startswith("ear") else
                     "head" if _n.startswith("head") else
                     "eye" if _n.startswith("eye") else
                     "nose" if _n.startswith(("nose", "nostril")) else
                     "muzzle" if _n.startswith(("muzzle", "cheek")) else "mouth")

assert len(DOGFLW_NAMES) == 46 and len(set(DOGFLW_NAMES.values())) == 46


# ---------------------------------------------------------------------------------
# Which landmarks a heatmap detector can actually find.  Measured on the 479-image
# DogFLW test split, not assumed - per-region NME / PCK@5%:
#
#     eye     0.0278  94.9%      ear      0.0890  27.6%
#     nose    0.0301  87.0%      head top 0.1250   0.6%   <- chance
#     mouth   0.0344  80.6%
#     muzzle  0.0524  46.7%
#
# Training moved nose/muzzle/mouth by 30-47% over an untrained warm start and the
# ear/head group by 6-8%, i.e. barely at all.  Two distinct causes:
#
#   * the skull top has no local texture, but its position IS implied by the rest of
#     the head - so a shape model derives it at 0.074 vs the CNN's 0.130 (see
#     shape_refine.py).  Detected badly, derived well.
#   * the ear contour points have strong local texture but are NOT globally
#     determined (an ear can be perked or flopped independently of the face), and
#     appear to be annotated as evenly spaced points along a smooth edge.  Neither
#     the CNN, a shape model, nor conditioning on breed or ear type recovers them.
# The split is per LANDMARK, not per region - grouping by region hides that the ear
# points differ enormously from each other.  Measured NME / PCK@5% on the test split:
#
#     ear_*_inner_base   0.054 0.059   49.3% 41.8%   <- as good as the muzzle
#     ear_*_tip          0.079 0.080   53.2% 54.9%   <- better PCK than the muzzle
#     ear_*_inner_mid    0.083 0.088   22.8% 20.3%   <- mediocre, still informative
#     ear_*_outer_base   0.090 0.096   25.5% 22.5%
#     ear_*_inner_low    0.091 0.093   21.7% 21.1%
#     ear_*_outer_low    0.097 0.099   19.2% 20.3%
#     ear_*_outer_mid    0.114 0.122    7.9%  6.5%   <- dead
#     head_top_*         0.122 0.128    0.6%  0.6%   <- dead as detections...
#
# ...but head_top is recovered by shape_refine.py at 0.074, so it stays in RELIABLE.
# Only the two outer-mid points survive nothing: they are the midpoint of a smooth ear
# edge, which the annotation does not pin down and no model tested can recover.
# Hide sparingly - a weak landmark the viewer can see and judge beats a missing one.
# ...and then both of those were fixed too.  ear_correct.py subtracts the systematic
# per-ear-type bias and takes the whole ear region from 0.0882 to 0.0631 NME, with
# ear_*_outer_mid going 0.122 -> 0.080 and 0.113 -> 0.066.  Nothing is now measurably
# unlearnable, so nothing is hidden: UNRELIABLE stays as a mechanism for future
# analysis to populate, but it is empty on the evidence available.
UNRELIABLE: list[int] = []
SHAPE_DERIVED = [14, 15]                 # head top - detected badly, derived well
RELIABLE = [i for i in range(46) if i not in UNRELIABLE]                  # 46

assert len(RELIABLE) + len(UNRELIABLE) == 46


# ---------------------------------------------------------------------------------
# Optional trim: drop dense *contour* sampling, keep every functional landmark.
# Enable with `analyze_correspondence.py --trim`.  76 outputs -> 63.
#
# Ears: 14 points (7 per ear) is a contour trace of the pinna.  Orientation - forward
# vs pinned back, which is the part that carries meaning - is a base->tip vector, so
# one base and the tip per ear is enough.
TRIM_DROP_EAR = [2, 3, 4, 5, 8, 9, 10, 11, 12, 13]

# Eyelids: the upper-lid points are added channels and are the redundant ones.
# 22/23 (lower lids) are deliberately NOT dropped - see the note below.
TRIM_DROP_LID = [20, 21]

# Chin: chin_upper duplicates the lower-lip contour.  42 (chin_bottom) is NOT dropped -
# see below.
TRIM_DROP_JAW = [41]

# Not dropped, on purpose: 22 eye_right_lower_lid, 23 eye_left_lower_lid and
# 42 chin_bottom are *merged* landmarks - they are the DogFLW ground truth attached to
# SuperAnimal's own right_eye / left_eye / lower_jaw channels.  Dropping them would not
# remove a single output channel (those channels belong to the pretrained 39), it would
# only downgrade their supervision from ground truth to pseudo-label.  They are free to
# keep, and they are what makes eyelid aperture and jaw drop measurable at all: the
# canthi alone give eye *width*, which barely moves during a squint or a blink.
TRIM_DROP = set(TRIM_DROP_EAR + TRIM_DROP_LID + TRIM_DROP_JAW)
TRIM_KEEP = [i for i in range(46) if i not in TRIM_DROP]

assert len(TRIM_DROP) == 13 and len(TRIM_KEEP) == 33
