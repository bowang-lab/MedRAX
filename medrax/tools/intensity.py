"""Intensity scale for images passed to ``xrv.datasets.normalize``."""

from typing import Optional

import numpy as np


def xrv_maxval(image: np.ndarray, input_maxval: Optional[float] = None) -> float:
    """Return the ``maxval`` to pass to ``xrv.datasets.normalize`` for ``image``.

    Without ``input_maxval`` the scale stays 255, as before. Images with pixels
    above 255 (such as 16-bit PNGs) used to fail inside ``normalize``; they now
    need an explicit ``input_maxval``, because the dtype does not say how many
    bits the pixels use: 12-bit values are often stored in ``uint16``. The scale
    is never taken from the image's own maximum, which would change its contrast.

    Raises:
        ValueError: If pixels exceed the scale, or ``input_maxval`` is invalid.
    """
    image = np.asarray(image)

    if input_maxval is not None:
        if not np.isfinite(input_maxval) or input_maxval <= 0:
            raise ValueError(f"input_maxval must be a positive number, got {input_maxval}.")
        if image.size and (image.min() < 0 or image.max() > input_maxval):
            raise ValueError(
                f"Pixel values [{image.min()}, {image.max()}] are outside "
                f"[0, input_maxval={input_maxval}]."
            )
        return float(input_maxval)

    if image.size and image.max() > 255:
        raise ValueError(
            f"{image.dtype} image has pixel values up to {image.max()}, above the 8-bit "
            "range. Pass input_maxval when creating the tool, e.g. 4095 for 12-bit "
            "pixels stored in uint16 or 65535 for full-range 16-bit images."
        )

    return 255.0
