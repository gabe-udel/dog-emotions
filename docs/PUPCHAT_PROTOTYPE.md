# Pupchat prototype

`video + question -> keypoints -> features -> evidence -> AI provider -> answer`

A layer on top of the measurement pipeline. It changes no model, training or evaluation
code; it reads what that pipeline produces, describes it in plain English, and hands the
description to a model along with the video.

> **Prototype only.** DogFLW is CC BY-NC 4.0 and the SuperAnimal weights follow the
> DeepLabCut Model Zoo terms, so this must not be deployed publicly or used
> commercially. It is educational guidance, not a vet and not a certified behaviourist.

---

## 1. Set a key

```powershell
copy .env.example .env
notepad .env
```

```ini
PUPCHAT_PROVIDER=openai      # or: gemini
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

`.env` is gitignored; `.env.example` is the tracked template. No key ever appears in
code, in a request payload, or in the usage log.

Install the SDKs if they are not already present:

```powershell
.venv\Scripts\python.exe -m pip install openai google-genai python-dotenv gradio
```

---

## 2. The command line

```powershell
.venv\Scripts\python.exe -m pupchat ask `
    --video "videos\whie pom sample 15.mp4" `
    --question "My pom keeps doing this when guests arrive. What is he telling me?"
```

It prints the answer and then stays open for follow-up questions. Follow-ups do **not**
re-run the keypoint model and do **not** re-upload the video.

| flag | effect |
|---|---|
| `--provider openai\|gemini` | overrides `PUPCHAT_PROVIDER` for one run |
| `--model <name>` | overrides the configured model |
| `--show-evidence` | print the evidence text sent to the model |
| `--show-request` | print the full request before sending, media shortened |
| `--dry-run` | build and print the request, send nothing, **no key needed** |
| `--no-follow-up` | answer once and exit |

Measurements only, no key required:

```powershell
.venv\Scripts\python.exe -m pupchat evidence --video "videos\sample 13.mp4" --json
```

---

## 3. The web page

```powershell
.venv\Scripts\python.exe -m pupchat.web
```

Opens a local Gradio page: upload a video, ask, keep chatting. There is a
**Show the measurements** button so the evidence can be read directly.

**The first question on a new video takes minutes.** The keypoint pass runs on CPU at
roughly 1.2 seconds per frame. Every later question on the same video is fast, because
measurements are cached by the video's SHA-256.

---

## 4. The two providers

The only real difference is what media they can take. The system prompt and the evidence
text are byte-identical either way — a test asserts that, because otherwise comparing
the two would compare two prompts rather than two models.

| | OpenAI | Gemini |
|---|---|---|
| media | sampled frames, base64 | the video file itself |
| why | the API has no video input ([docs](https://developers.openai.com/api/docs/guides/images-vision): only `input_text` and `input_image`) | native video ([docs](https://ai.google.dev/gemini-api/docs/video-understanding)) |
| sampling | **2 fps under 15 s**, 1 fps above, capped at 32 frames | server-side, Google's choice |
| large files | n/a | Files API above 100 MB, inline below |
| default model | `gpt-6.1-sol` | `gemini-3.8-flash` |

Adding a third provider means one subclass of `pupchat.providers.base.Provider` and one
line in the registry. Nothing else in Pupchat branches on provider name.

---

## 5. The evidence summary

Two tiers, decided by measured reliability rather than by a hard-coded list:

- **Timestamped moments** only for features whose within-clip SNR is at least 1 —
  currently `head_pitch`, `tail_height`, `head_tilt`, `head_height`, `tail_tuck`,
  `view_yaw`. An excursion counts when the series leaves its clip baseline by more than
  2x the feature's **own measured error**.
- **Clip-level descriptions**, with no timing at all, for features that separate dogs
  but not moments.

Where `outputs/feature_error.json` and `outputs/feature_snr.json` disagree, ground truth
wins — the same rule `aggregate.load_noise_floors` already uses. That promotes
`view_yaw` (1.61 by ground truth, 0.94 by the noise model) and demotes `ear_height`
(0.33 against 1.42), and it reproduces the six above without naming any of them.

Also enforced:

- **Coverage** is stated for every feature. Below 40% (configurable) it is reported as
  unavailable instead.
- **Unavailable features are always listed, with a reason** — and are never dropped to
  save words. An earlier version trimmed bullets off the end when it ran over budget and
  silently removed 8 of 10 unavailable entries on a real clip. Long lists are now
  grouped by reason instead, and tests assert every entry survives a tight budget.
- **Tail wag rate is listed as unavailable whenever the oscillation gate refuses it.**
  That is not the same as "the dog did not wag", and the system prompt says so.
- **No emotion words.** `config.BANNED_WORDS` covers affect terms and also
  interpretation wearing body-language clothes — `relaxed`, `alert`, `submissive`. A
  test checks real output against the list.

---

## 6. Caching and cost

| what | where |
|---|---|
| keypoints, features, evidence | `outputs/pupchat/<hash>/` |
| sampled frames (OpenAI) | `outputs/pupchat/<hash>/vision_frames.json` |
| token usage and estimated cost | `outputs/pupchat/usage.jsonl` |

Keyed by SHA-256 of the file, so the same clip under two names is processed once. Bump
`pipeline.CACHE_VERSION` when anything upstream changes meaning.

The usage log is one JSON object per line: provider, model, tokens, cached tokens,
reasoning tokens, estimated USD, seconds, finish reason, and whether it was a follow-up.
Prices live in `config.PRICING`; an unpriced model logs `null` rather than `0`, so a
missing price cannot quietly understate a bill.

### Reasoning tokens are billed, and the two providers report them differently

Found on the first live call, and it is a real money bug if you get it wrong:

* **Gemini** reports `candidates_token_count` (the visible answer) and
  `thoughts_token_count` separately, and **charges for both**. Measured: 214 visible
  against 682 thinking. Billing only the visible count understates the output charge
  roughly fourfold.
* **OpenAI** already includes reasoning inside `output_tokens`, so adding it again
  would overstate.

`normalise_usage` makes `output_tokens` mean "the billed total" on both, with
`reasoning_tokens` carried alongside for transparency. A test pins each direction.

That same budget is also why `max_output_tokens` defaults to **3000**. At 900 the first
real answer was truncated mid-word, because thinking consumed 682 of it before the
answer began. Truncation is now surfaced as a warning rather than returned as if it
were a whole answer.

---

## 7. Tests

```powershell
.venv\Scripts\python.exe -m pytest tests\test_pupchat_evidence.py tests\test_pupchat_client.py -q
```

63 tests. Every API call is mocked: **no test needs a key, opens a socket, or spends
money.** They cover the tier rule, the no-false-timestamp guarantee, low-coverage
reporting, the wag gate, the banned-word list, the trimming regression, both providers'
request shapes, media handling, usage normalisation and cost.

---

## 8. Layout

```
pupchat/
  config.py              every tunable; provider settings; banned words; prices
  reliability.py         merges the two measurement files, ground truth wins
  evidence.py            build_evidence / evidence_to_text / detect_events
  pipeline.py            video -> keypoints -> features -> evidence, cached
  client.py              provider-agnostic: media prep, send, accounting
  providers/
    base.py              Provider ABC; MEDIA_FRAMES vs MEDIA_VIDEO
    openai_provider.py   frames
    gemini_provider.py   native video
  prompts/system.md      the system prompt, editable
  cli.py / __main__.py   python -m pupchat
  web.py                 gradio page
  _src.py                bridge to src/, heavy imports kept lazy
```


---

## 9. Keypoint smoothing, and an honest result

A One Euro filter runs over the keypoint tracks before features are computed. Its cutoff
rises with speed, so it smooths hard when a keypoint is nearly still and backs off when
it moves - which is the property that matters here, because the fast movements are the
signal.

**Measured on `videos/whie pom sample 15.mp4` at 12.5 fps.** Jitter is the median
per-frame residual displacement with whole-frame translation removed, so camera pan and
the dog walking do not count.

| beta | jitter vs raw | 1 Hz kept | 3 Hz kept | 5 Hz kept |
|---|---|---|---|---|
| 0.7 (paper default) | **-17%** | 92% | 58% | 43% |
| 2.0 | -10% | 97% | 81% | 71% |
| **5.0 (shipped)** | **-5%** | 99% | **92%** | **87%** |
| 20.0 | -1% | 100% | 98% | 96% |

Shipped numbers on that clip: face **4.137 -> 3.921 px**, body **7.417 -> 7.101 px**.

**The honest reading is that the gain is small, and it has to be.** One Euro adapts on
*speed*, and per-frame jitter is itself fast - the filter cannot tell a 4 px tracking
wobble from a 4 px real movement, because at this frame rate they look identical. So the
setting that preserves a wag barely touches the jitter, and the setting that removes the
jitter eats the wag.

Note the first row: at the paper's default beta, One Euro keeps **less** of a 3 Hz wag
(58%) than the rolling median already in the pipeline (68.6%). Shipping that default
would have been a regression dressed as an improvement. beta is 5.0 for that reason.

Also worth knowing: the keypoint pass strides 25 fps down to 12.5, which puts 5 Hz at
2.5 samples per cycle. Some of the loss in the table is the sampling rate, not the
filter - raising `KeypointConfig.target_fps` would help more than retuning beta.

---

## 10. Flagged frames, stillness, weight shift

Three additions that ride alongside the tiered evidence in `evidence.json` rather than
inside it, because none has a measured `error_sd` and none may be presented on the same
footing as a validated feature.

**`flagged_frames`** - frames where a feature jumped and snapped back. The test that
separates a glitch from a movement is what the series does either side: if the before
and after baselines match and only the middle deviates, it is a spike; if they differ,
the dog moved. Threshold is 4x the feature's own measured error. Flagged, never
repaired - a spike is evidence the measurement is unreliable at that moment, and
smoothing it away would destroy exactly that signal. Complementary to the RANSAC
upstream, which only sees face keypoints.

**`stillness`** - runs where movement stays inside each feature's own noise floor for
2+ seconds. Reported both for all features at once and per feature, because whole-dog
stillness needs every feature still simultaneously and a single jittery one vetoes it;
without the per-feature runs an empty list reads as "never stopped moving" when it means
"one measurement never settled".

> It is deliberately **not** called freezing. Freezing is a behavioural label with a
> specific meaning; what is measured is "no movement above measurement error", and a dog
> standing calmly produces the same reading as a dog frozen in fear. Naming it freezing
> in the evidence would be exactly the interpretation this layer exists to keep out.

**`weight_shift`** - where the hips sit between the front and rear paw contacts, 0 over
the rear and 1 over the front, normalised by the paw span. **Unavailable when the paws
are not visible**, with the visibility of each paw reported either way. It carries
`source: "no reliability measurement yet"` even when available, because it has never
been differenced against ground truth.

---

## 11. Answer format, and how it is checked

`prompts/system.md` makes Pupchat a warm educational guide rather than a liability
notice. The five-section format for a new video is **What I see / What it likely means /
Do / Don't / Try this next**; follow-ups answer conversationally and do not repeat it.
Raw numbers never appear - "his ears are pinned back", not "ears angled 110 degrees".
A professional referral is one sentence at the end, only when warranted, never leading.

Rules the rewrite did not mention are carried over from the previous prompt, because
each guards against a specific way of overclaiming: the ground-truth vs estimated
reliability split, "an unmeasured thing means no information, not an absence", never
inventing a timestamp for a clip-level measurement, and never diagnosing.

`pupchat/answer_check.py` turns those rules into pure functions, so they are testable
without the API:

```python
check(answer, new_video=True)   # five sections present and in order
check(answer, new_video=False)  # follow-up must NOT repeat the format
raw_degrees(answer)             # "110 degrees", "-45.5 deg"
leaked_jargon(answer)           # face-widths, SNR, keypoints, coverage
referral_report(answer)         # at most one, and in the last quarter
aversive_advice(answer)         # prong/shock/choke, negation-aware
```

The same functions run in the tests against fixtures and against live answers, so what
is asserted offline and what is measured online cannot drift apart. **They check shape,
not quality**: a warm, confident, wrong answer passes every one of them.

One subtlety worth keeping: `aversive_advice` is negation-aware. A plain keyword match
flags "Don't scold or punish him" as aversive advice, which is backwards - the prompt's
whole Don't section is built from sentences like that.

---

## 12. Seeing how a feature was calculated

`pupchat trace` joins the raw keypoint confidences to the feature values for the same
frame, so a NaN stops being a dead end. It reads the cache and recomputes nothing.

```powershell
.venv\Scripts\python.exe -m pupchat trace --video "<file>"               # mid frame
.venv\Scripts\python.exe -m pupchat trace --video "<file>" --time 3.2    # at 3.2s
.venv\Scripts\python.exe -m pupchat trace --video "<file>" --failed-only
```

It prints two tables. First, keypoint reliability across the clip, weakest first - this
is what a coverage number actually means, because **a feature is NaN unless every
keypoint it needs clears the cut, so the weakest keypoint caps every feature touching
it**. Then one frame in detail: which keypoints fell below the cut, which features
could not be computed, and the values of those that could.

Everything else the cache exposes, for scripting:

| file | contents |
|---|---|
| `frames.json` | every feature, every frame |
| `clip.json` | the aggregated clip row, with `__coverage` / `__level` / `__rate` |
| `evidence.json` | tiers, events, unavailable reasons |
| `keypoints.npz` | raw body/face arrays with names - what `trace` reads |

And to watch the numbers move against the video, the research pipeline already draws
them on the frames:

```powershell
.venv\Scripts\python.exe src
un_video.py --video "<file>" `
    --snapshot face_projectace38\snapshot-best-008.pt --ransac --feature-overlay
```

---

## 13. Driving it from your own front end

Google Stitch exports a **front end only** - HTML/CSS, Tailwind, Vue, Angular, Flutter
or SwiftUI. It generates no server, so a Stitch design has nothing to call until the
pipeline is exposed over HTTP. That is what `pupchat/api.py` is for.

```powershell
.venv\Scripts\python.exe -m pupchat.api        # http://127.0.0.1:8000, docs at /docs
```

| endpoint | purpose |
|---|---|
| `POST /api/analyse` | upload a clip, measure it, get the evidence. Minutes the first time, instant after - cached by the file's own hash |
| `POST /api/ask` | question about an analysed clip; pass back `history` for follow-ups that re-measure nothing |
| `GET /api/evidence/{hash}` | the evidence text and the full JSON |
| `GET /api/track/{hash}` | per-frame feature values with timestamps, for scrubbing alongside the video |
| `GET /api/trace/{hash}?frame=N` | keypoint confidences joined to feature values: why something is missing |
| `GET /api/overlay/{hash}` | the video with keypoints and a live feature panel drawn on it |
| `POST /api/flag` / `GET /api/flags` | flagged answers |

Two things worth knowing when wiring a front end to it:

**CORS is wide open.** A Stitch export opened from the filesystem or served on another
port is a different origin, and a silent CORS failure looks like a broken backend. Safe
only because this is a local prototype that must not be deployed.

**NaN is converted to `null` at the boundary.** Python writes bare `NaN` happily;
Starlette and every strict JSON parser refuse it. A missing measurement is genuinely
null, and `null` is something a front end can test for.

### Seeing the maths on the video

`GET /api/overlay/{hash}` renders the keypoints plus a live feature readout onto the
cached frames - the same `draw.Renderer` and `draw.feature_panel` the research pipeline
uses, so the two cannot disagree. It redraws from the cache rather than re-running the
cascade, so it costs seconds rather than minutes.

Pair it with `GET /api/track/{hash}` to drive a scrubber: that returns every feature's
value per frame with timestamps, so a UI can show the numbers moving against the video
without re-rendering anything.

---

## 14. Proving accuracy on YOUR footage

Every accuracy figure in this project was measured on DogFLW and StanfordExtra stills.
That is the right baseline and **it does not transfer**: `mouth_open_jaw` runs at 94%
coverage on DogFLW stills and 10% on one of these videos. A number measured on someone
else's photographs is not a claim about your camera, breed, lighting or framing.

`pupchat/groundtruth.py` applies the project's own method to your clips:

1. **Sample frames evenly** - `sample_indices()`. Not chosen by eye: picking the frames
   where the dog looks clear measures the easy cases, which is the bias that makes an
   accuracy number flattering. Deterministic, so a second session covers the same frames.
2. **Label them by hand**, with the model's prediction never drawn. Showing the guess
   for the point being labelled anchors the click to it, and the run then measures
   agreement rather than accuracy - the same rule `src/label_withers.py` follows.
3. **Difference predicted against labelled** - `score_keypoints()`. Normalised by the
   per-frame box diagonal, so the numbers sit beside `evaluate_face` and
   `check_body_accuracy` rather than needing translation.
4. **Recompute the features from both sets of keypoints** - `score_features()`.

**Step 4 is the one people skip, and it is the one that matters.** A 3 px keypoint error
tells you nothing about `ear_angle` until you measure it, and the relationship is not
proportional: a feature built as a difference between two nearby landmarks cancels most
of a shared error, while a ratio with a small denominator amplifies it. A test pins
exactly that - the same shared 5 px shift produces 0.0 error in a difference feature and
2.5 in a ratio.

`compare_to_dogflw()` then puts your error beside the published one per feature. A ratio
well above 1 means the DogFLW figure does not describe your footage, and anything built
on it - the SNR tiers, the timestamp eligibility, the noise floors - needs rechecking
against the local number.

---

## 15. Verified live

**Gemini, 2026-10-11**, `videos/whie pom sample 15.mp4` (7.5 s), `gemini-3.8-flash`:

| | |
|---|---|
| first question | 1,874 in / 896 out (490 thinking), 13.1 s, $0.0048 |
| follow-up | 2,409 in, 15.3 s, $0.0053 - no re-measure, no re-upload |
| measurements | cached from the first run, reused instantly |

The answer recommended a certified behaviourist and a vet check in its second paragraph
without being asked, used the ear measurements explicitly ("angled at over -100 degrees"),
and said plainly "we can't see his body or tail in this clip" - the unavailable list
doing its job.

**OpenAI is not yet verified live** - no key was available. Its request assembly, frame
sampling, redaction and usage normalisation are covered by tests and by `--dry-run`, but
no live round trip has been made.
