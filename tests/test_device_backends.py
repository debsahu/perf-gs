# Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

"""Device backend names and shader-define parsing (no GPU)."""
from __future__ import annotations

from typing import get_args

import pytest

from trainer.backend.compute import device


def test_backend_type_names_every_backend_create_device_accepts() -> None:
    assert set(get_args(device.Backend)) == set(device._BACKEND_MAP)


@pytest.mark.parametrize(("defines", "expected"), [
    (None, False), ({}, False), ({"X": "0"}, False), ({"X": " 0 "}, False), ({"X": "1"}, True),
    ({"X": "2"}, True), ({"X": "0x1"}, True), ({"Y": "1"}, False)])
def test_define_enabled_follows_slang_if(defines, expected) -> None:
    assert device.define_enabled(defines, "X") is expected


def test_define_enabled_refuses_a_non_integer() -> None:
    with pytest.raises(ValueError, match="not an integer"):
        device.define_enabled({"X": "yes"}, "X")
