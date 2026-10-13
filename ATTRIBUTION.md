# Attribution and licence terms

This project is a **non-commercial research prototype**. It may be shared and
demonstrated for free. It may not be sold, monetised, put behind ads, or used to
promote a business.

The constraint comes from the data and weights it is built on, not from a preference.

## Face landmarks — DogFLW

**DogFLW: Dog Facial Landmarks in the Wild**, Martvel et al.
Licensed **CC BY-NC 4.0** (<https://creativecommons.org/licenses/by-nc/4.0/>).

CC BY-NC permits sharing and adaptation for non-commercial purposes, with
attribution. The restriction is **non-commercial**, not non-public — so a free demo is
permitted, and the credit is a condition of it rather than a courtesy.

DogFLW is itself built on **Stanford Dogs** (Khosla et al.), which carries its own
non-commercial research terms.

The licence propagates to anything trained on the data, which includes the
`face_project/face38` and `face_project/face1` weights in this repository.

## Body pose — DeepLabCut SuperAnimal-Quadruped

**SuperAnimal-Quadruped**, Ye et al., Mathis Lab, distributed through the
DeepLabCut Model Zoo.

The weights carry a **Modified MIT** licence granting use for academic and
non-commercial purposes, requiring that the copyright notice travel with copies, and
prohibiting use that deliberately harms an animal. Commercial licensing is handled by
EPFL-TTO.

These weights are run **unmodified** here — no fine-tuning, no head surgery.

## Video clips

| clip | licence |
|---|---|
| Mixkit 1476 | Mixkit Free License |
| `dog_walk_sf.ogv` | CC BY-SA 3.0 |

**StanfordExtra v12** carries its own terms and is deliberately **not** committed to
this repository.

## Where this notice appears at runtime

It is not kept only in this file, because a front end that reads the API would
otherwise never see it:

- `pupchat/config.py::LICENCE_NOTICE` — printed by the CLI on every run
- `pupchat/config.py::ATTRIBUTION` — served by `GET /api/health`
- the web page renders both in the sidebar, from the API response rather than from
  hard-coded HTML, so the page and the server cannot drift apart

`tests/test_pupchat_api.py` and `tests/test_pupchat_guard.py` pin all of the above.

## Not medical or behavioural advice

Pupchat describes measurements and relays a language model's reading of them. It is
educational. It does not diagnose, and it is not a substitute for a veterinarian or a
certified behaviourist.
