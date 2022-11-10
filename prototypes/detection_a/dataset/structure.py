#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field


@dataclass
class BoundBox:
    x1: int  # left top
    y1: int  # left top
    x2: int  # right bottom
    y2: int  # right bottom
    w: int = field(init=False)
    h: int = field(init=False)

    def __post_init__(self):
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1


@dataclass
class Mask:
    origin_x: int
    origin_y: int
    b64: str

@dataclass
class ToothLabel:
    mask: Mask
    type: str
    img_id: str
    # id: str = field(default_factory=lambda: next(teeth_counter))
    # id: str = field(default_factory=lambda: next(st.session_state.teeth_counter))


@dataclass
class CariesLabel:
    type: str
    bbox: BoundBox
    img_id: str
    # id: str = field(default_factory=lambda: next(caries_counter))
    # id: str = field(default_factory=lambda: next(st.session_state.caries_counter))


@dataclass
class GlobalLabel:
    bacteria_sm: float
    bacteria_sa: float
    direction: str
    anomaly: bool
    img_id: str
