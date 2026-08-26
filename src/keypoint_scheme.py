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
