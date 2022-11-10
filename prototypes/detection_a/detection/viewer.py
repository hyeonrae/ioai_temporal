#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import cv2

ROOT = 'results'

def main():
    view = st.columns(2)
    img = cv2.imread(f"{ROOT}/pred.png")
    view[0].image(img)
    img = cv2.imread(f"{ROOT}/gt.png")
    view[1].image(img)
    

if __name__ == "__main__":
    main()
