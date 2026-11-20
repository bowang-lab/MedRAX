"""Tests for the intensity scale used before ``xrv.datasets.normalize``.

The module is loaded by path: importing ``medrax.tools`` loads every tool and its models.

Run from the repository root:  pytest tests/test_intensity.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torchxrayvision as xrv

_spec = importlib.util.spec_from_file_location(
    "intensity", Path(__file__).parents[1] / "medrax" / "tools" / "intensity.py"
)
intensity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(intensity)
xrv_maxval = intensity.xrv_maxval

IMG8 = np.array([[0, 64], [128, 255]], dtype=np.uint8)


def test_8bit_keeps_scale_255():
    assert xrv_maxval(IMG8) == 255.0


def test_uint16_within_8bit_range_keeps_scale_255():
    # Worked before the change, so it must keep working.
    assert xrv_maxval(IMG8.astype(np.uint16)) == 255.0


@pytest.mark.parametrize(
    ("image", "maxval"),
    [
        (IMG8.astype(np.uint16) * 257, 65535),  # full-range 16-bit
        (np.round(IMG8 * (4095 / 255)).astype(np.uint16), 4095),  # 12-bit stored in uint16
    ],
)
def test_wider_images_normalize_like_8bit_with_input_maxval(image, maxval):
    expected = xrv.datasets.normalize(IMG8, 255)
    actual = xrv.datasets.normalize(image, xrv_maxval(image, maxval))
    np.testing.assert_allclose(actual, expected, atol=2048 / maxval)


def test_wider_images_without_input_maxval_explain_what_to_pass():
    with pytest.raises(ValueError, match="input_maxval"):
        xrv_maxval(np.array([[0, 4095]], dtype=np.uint16))


def test_pixels_above_input_maxval_are_rejected():
    with pytest.raises(ValueError, match="outside"):
        xrv_maxval(np.array([[0, 65535]], dtype=np.uint16), 4095)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_input_maxval_is_rejected(bad):
    with pytest.raises(ValueError, match="positive"):
        xrv_maxval(IMG8, bad)
