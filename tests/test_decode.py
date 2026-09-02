"""Heatmap decoding.

The regression these guard against is the one that cost this project the most: argmax
decoding without an offset map collapses co-located keypoints onto identical
coordinates. `test_subpixel_separates_peaks_in_the_same_cell` fails if that ever
returns.
"""
from __future__ import annotations

import numpy as np
import pytest

from decode import decode_heatmaps, distinct_positions


def _gauss(h: int, w: int, cx: float, cy: float, sigma: float = 1.5) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))


def _stack(peaks: list[tuple[float, float]], h: int = 64, w: int = 64) -> np.ndarray:
    """(1, H, W, J) with one Gaussian per keypoint."""
    return np.stack([_gauss(h, w, x, y) for x, y in peaks], axis=-1)[None]


def test_recovers_integer_peak_positions() -> None:
    peaks = [(10.0, 20.0), (33.0, 5.0), (60.0, 61.0)]
    out = decode_heatmaps(_stack(peaks), stride=4.0, n_keypoints=3)
    assert out.shape == (1, 3, 3)
    for j, (x, y) in enumerate(peaks):
        assert out[0, j, 0] == pytest.approx(x * 4.0 + 2.0, abs=1e-6)
        assert out[0, j, 1] == pytest.approx(y * 4.0 + 2.0, abs=1e-6)


def test_recovers_subpixel_peak_positions() -> None:
    """A peak between cells must decode between cells."""
    peaks = [(10.4, 20.3), (33.5, 5.5)]
    out = decode_heatmaps(_stack(peaks), stride=4.0, n_keypoints=2)
    for j, (x, y) in enumerate(peaks):
        assert out[0, j, 0] == pytest.approx(x * 4.0 + 2.0, abs=0.6)
        assert out[0, j, 1] == pytest.approx(y * 4.0 + 2.0, abs=0.6)


def test_subpixel_beats_argmax_on_offgrid_peaks() -> None:
    peaks = [(10.45, 20.35), (33.4, 5.45), (7.3, 51.4)]
    hm = _stack(peaks)
    truth = np.array([[x * 4.0 + 2.0, y * 4.0 + 2.0] for x, y in peaks])
    sub = decode_heatmaps(hm, 4.0, subpixel=True, n_keypoints=3)[0, :, :2]
    arg = decode_heatmaps(hm, 4.0, subpixel=False, n_keypoints=3)[0, :, :2]
    assert np.abs(sub - truth).mean() < np.abs(arg - truth).mean()


def test_subpixel_separates_peaks_in_the_same_cell() -> None:
    """THE regression test. Two landmarks peaking in one cell must not decode to the
    same point - that is what rendered 46 face channels as 16 dots."""
    hm = _stack([(20.2, 30.2), (20.4, 30.45)])
    arg = decode_heatmaps(hm, 4.0, subpixel=False, n_keypoints=2)
    sub = decode_heatmaps(hm, 4.0, subpixel=True, n_keypoints=2)
    assert distinct_positions(arg) == 1, "argmax should collapse them (premise of the test)"
    assert distinct_positions(sub) == 2


def test_offsets_are_bounded_to_half_a_cell() -> None:
    """A vertex further than half a cell means the fit is not describing this peak."""
    rng = np.random.default_rng(3)
    hm = rng.random((2, 16, 16, 5))
    sub = decode_heatmaps(hm, 4.0, subpixel=True, n_keypoints=5)
    arg = decode_heatmaps(hm, 4.0, subpixel=False, n_keypoints=5)
    assert np.abs(sub[:, :, :2] - arg[:, :, :2]).max() <= 0.5 * 4.0 + 1e-9


def test_border_peak_keeps_cell_centre() -> None:
    """No neighbour on one side, so no parabola - must not read across the wrap."""
    hm = np.zeros((1, 8, 8, 1))
    hm[0, 0, 0, 0] = 1.0
    out = decode_heatmaps(hm, 4.0, n_keypoints=1)
    assert out[0, 0, 0] == pytest.approx(2.0)
    assert out[0, 0, 1] == pytest.approx(2.0)


def test_flat_heatmap_does_not_produce_nan() -> None:
    hm = np.full((1, 8, 8, 3), 0.5)
    out = decode_heatmaps(hm, 4.0, n_keypoints=3)
    assert np.isfinite(out).all()


def test_score_is_the_peak_value() -> None:
    hm = _stack([(10.0, 10.0)])
    hm *= 0.75
    out = decode_heatmaps(hm, 4.0, n_keypoints=1)
    assert out[0, 0, 2] == pytest.approx(0.75, abs=1e-9)


def test_accepts_bjhw_layout() -> None:
    bhwj = _stack([(10.0, 20.0), (30.0, 40.0)])
    bjhw = np.transpose(bhwj, (0, 3, 1, 2))
    a = decode_heatmaps(bhwj, 4.0, n_keypoints=2)
    b = decode_heatmaps(bjhw, 4.0, n_keypoints=2)
    assert np.allclose(a, b)


def test_ambiguous_layout_raises_rather_than_guessing() -> None:
    with pytest.raises(ValueError, match="no axis of size"):
        decode_heatmaps(np.zeros((1, 8, 8, 4)), 4.0, n_keypoints=7)


def test_rejects_non_4d() -> None:
    with pytest.raises(ValueError, match="must be 4-D"):
        decode_heatmaps(np.zeros((8, 8, 4)), 4.0)
