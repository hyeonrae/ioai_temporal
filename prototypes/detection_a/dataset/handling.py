#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import glob
import numpy as np
import pandas as pd

def get_subentries(root='.', only_dir: bool=False):
    with os.scandir(root) as entries:
        for entry in entries:
            if only_dir and not entry.is_dir():         continue
            if not only_dir and not entry.is_file():    continue

            yield entry.name

def subdirs(root='.'):
    for entry_name in get_subentries(root=root, only_dir=True):
        yield entry_name

def subfiles(root='.'):
    for entry_name in get_subentries(root=root, only_dir=False):
        yield entry_name
