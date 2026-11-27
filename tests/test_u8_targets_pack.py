# Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

"""The uint8 target packer: uint8 passes through, float is [0, 1] sRGB, and any other integer type is refused."""
from __future__ import annotations

import numpy as np
import pytest

from trainer.backend.render.targets import srgb01_to_u8


def test_uint8_passes_through_and_float_is_quantized() -> None:
    u8 = np.arange(16, dtype=np.uint8).reshape(2, 2, 4)
    assert np.shares_memory(srgb01_to_u8(u8), u8)
    np.testing.assert_array_equal(srgb01_to_u8(u8.astype(np.float32) / 255.0), u8)


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.int64, np.bool_])
def test_other_integer_targets_are_refused(dtype) -> None:
    """A raw uint16 decode read as [0, 1] floats clips almost every pixel to 255 without an error."""
    with pytest.raises(TypeError, match="uint8"):
        srgb01_to_u8(np.full((2, 2, 4), 1000, np.uint16).astype(dtype))
