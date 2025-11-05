from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List
import numpy as np
from .utils import Band

@dataclass(frozen=True)
class Tile:
    id: int
    if1_center_hz: float
    bw_hz: float
    rf_center_hz: float

def make_tiles(if1_min, if1_max, rf_min, rf_max, bw_list, if1_step, rf_step) -> List[Tile]:
    tiles: List[Tile] = []
    tid = 0
    for bw in bw_list:
        cmin = if1_min + bw/2
        cmax = if1_max - bw/2
        if cmin > cmax:
            continue
        if1_cs = np.arange(cmin, cmax + 0.5*if1_step, if1_step)
        rf_cs = np.arange(rf_min, rf_max + 0.5*rf_step, rf_step)
        for ic in if1_cs:
            for rc in rf_cs:
                tiles.append(Tile(id=tid, if1_center_hz=float(ic), bw_hz=float(bw), rf_center_hz=float(rc)))
                tid += 1
    return tiles
