"""Built-in test scenes."""
from __future__ import annotations

from typing import Tuple

import mitsuba as mi
import numpy as np

CORNELL_MAX_DEPTH = 5


def modified_cornell_box(resolution: Tuple[int, int] = (256, 256)) -> dict:
    """Mitsuba's Cornell box with the ceiling light flipped about X and lowered by 0.2."""
    cbox = mi.cornell_box()
    cbox["integrator"]["max_depth"] = CORNELL_MAX_DEPTH
    cbox["sensor"]["film"]["width"], cbox["sensor"]["film"]["height"] = resolution

    transform = np.array(cbox["light"]["to_world"].matrix)
    position = transform[:3, 3].copy()
    flip_x = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    transform = flip_x @ transform
    transform[:3, 3] = position - np.array([0, 0.2, 0])
    cbox["light"]["to_world"] = mi.ScalarTransform4f(transform)
    return cbox
