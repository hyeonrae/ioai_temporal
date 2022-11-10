#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import cv2
import json
import zlib
import base64
import time
import shutil
from pathlib import Path

import streamlit as st
from handling import subdirs, subfiles
from structure import BoundBox, Mask, ToothLabel, CariesLabel, GlobalLabel

"""
### Dataset Wragling
```
- name: qrayset
- description: camera-qray pair images for caries detection
```
"""

SRCPATH = "../data"
OUTPATH = "../data_anony"

IMG_FILE_FORMAT = "jpg"
ANN_FILE_FORMAT = "json"


def get_patient_list(round_id):
    root = os.path.join(SRCPATH, round_id)

    modal_root = {}
    modal_root["camera"] = os.path.join(root, "camera")
    modal_root["qray"] = os.path.join(root, "qray")

    pbc = patients_by_camera = sorted(list(subdirs(modal_root["camera"])))
    pbq = patients_by_qray = sorted(list(subdirs(modal_root["qray"])))

    len_of_pbc = len(pbc)
    len_of_pbq = len(pbq)

    assert pbc == pbq, f"pbc: {len_of_pbc}, pbq: {len_of_pbq}, Unmatched patient lists"

    pathes = {}
    for p_id in pbc:
        pathes["img"] = os.path.join(modal_root["camera"], p_id, "img")
        pathes["ann"] = os.path.join(modal_root["camera"], p_id, "ann")
        pathes["qray"] = os.path.join(modal_root["qray"], p_id, "img")

        yield (p_id, pathes)


def get_unit(pathes):
    img_list = sorted(list(subfiles(pathes["img"])))
    qray_list = sorted(list(subfiles(pathes["qray"])))
    ann_list = sorted(list(subfiles(pathes["ann"])))

    assert len(img_list) == len(qray_list)

    for i, q, a in zip(img_list, qray_list, ann_list):
        yield (
            os.path.join(pathes["img"], i),
            os.path.join(pathes["qray"], q),
            os.path.join(pathes["ann"], a),
        )


import hashlib

def get_hash_value(in_str, in_digest_bytes_size=64):
    assert 1 <= in_digest_bytes_size and in_digest_bytes_size <= 64
    blake  = hashlib.blake2b(in_str.encode('utf-8'), digest_size=in_digest_bytes_size)
    return blake.hexdigest()

def gen(root):
    for r_id in sorted(list(subdirs(root))):
        for p_id, pathes in get_patient_list(r_id):
            cnt_total = 0

            for i, q, a in get_unit(pathes):
                u_id = f"{r_id}{p_id}{cnt_total}"
                u_id = get_hash_value(u_id, 10)
                print(r_id, p_id, cnt_total, u_id)
                yield {
                    "id": cnt_total,
                    "uid": u_id,
                    "r_id": r_id,
                    "p_id": p_id,
                    "c_path": i,
                    "q_path": q,
                    "a_path": a,
                }
                cnt_total += 1

@st.cache(show_spinner=False)
def load_filelist():
    print(f"load filelist from {SRCPATH}")
    return list(gen(SRCPATH))


# data -> bitmap
def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    # n = np.fromstring(z, np.uint8)
    n = np.frombuffer(z, np.uint8)
    # mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(np.uint8)
    return mask


def draw_box(img, bbox: BoundBox, color=(255,255,255), thickness=5):
    return cv2.rectangle(img, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, thickness)

def draw_carries_box(img, bbox: BoundBox, cavity: bool = False):
    if cavity:
        color = (150, 255, 0)
    else:
        color = (0, 180, 150)
    thickness = 3

    return draw_box(img, bbox, color, thickness)


def draw_mask(img, b64, origin):
    imh, imw, _ = img.shape
    data_b64 = b64

    local_mask = base64_2_mask(data_b64)
    local_shape = local_mask.shape

    img_mask = np.zeros([imh, imw], dtype=np.uint8)
    img_zero = np.zeros([imh, imw], dtype=np.uint8)

    img_mask[
        origin[1] : origin[1] + local_shape[0], origin[0] : origin[0] + local_shape[1]
    ] += local_mask

    color_mask = cv2.merge((img_mask, img_zero, img_mask))
    # color_mask = cv2.merge((img_mask, img_mask, img_mask))
    # color_mask = cv2.blur(color_mask, (9, 9))

    return cv2.addWeighted(img, 1.0, color_mask, 0.2, 0)
    # return cv2.addWeighted(img, 1.0, color_mask, 1.0, 0)


def object_from_ann(path: str = None, uid: str = None):
    if not path:
        return
    if not uid:
        return

    with open(path) as json_file:
        ann = json.load(json_file)

    _global = {
        'bacteria_sm': str(.0),
        'bacteria_sa': str(.0),
        'direction': '설측',  # 설측, 교합측, 순측
        'anomaly': False,  # 우식여부
        'img_id': uid,
    }

    _teeth = []
    _caries = []

    teeth_type = None

    for tag in ann["tags"]:
        if tag['name'] == "치아종류":
            teeth_type = str(tag['value'])
        elif tag['name'] == "촬영방향":
            _global['direction'] = str(tag['value'])
        elif tag['name'] == "세균_SM":
            _global['bacteria_sm'] = float(tag['value'])
        elif tag['name'] == "세균_SA":
            _global['bacteria_sa'] = float(tag['value'])
        else:
            pass


    for obj in ann["objects"]:
        # print(f"////objs. - {obj['classTitle']}")
        class_title = obj["classTitle"]

        if class_title == "Caries":
            _global['anomaly'] = True
            caries_type = None
            caries_bbox = None

            for tag in obj["tags"]:  # rectangle type 에 대한 tags
                if tag["name"] == "우식유형":  # 초기, 와동,
                    caries_type = tag["value"]
            caries_bbox = [
                obj["points"]["exterior"][0][0],
                obj["points"]["exterior"][0][1],
                obj["points"]["exterior"][1][0],
                obj["points"]["exterior"][1][1],
            ]

            _caries.append(
                CariesLabel(
                    caries_type,
                    BoundBox(*caries_bbox),
                    uid
                )
            )

        elif class_title == "Tooth":

            _data_b64 = obj["bitmap"]["data"]
            _origin = obj["bitmap"]["origin"]

            #tooth_labels['masks'].append(_data_b64)
            #tooth_labels['origin'].append(_origin)

            _teeth.append(
                ToothLabel(
                    Mask(*_origin, _data_b64),
                    teeth_type,
                    uid
                )
            )
        else:
            pass

    __global = GlobalLabel(
        _global["bacteria_sm"],
        _global["bacteria_sa"],
        _global["direction"],
        _global["anomaly"],
        _global["img_id"]
    )

    return __global, _caries, _teeth


def view_imgs(c_path, q_path, caries_info, teeth_info):
    view_size = (300, 300)

    img = cv2.imread(c_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for c in caries_info:
        cavity = False
        if c.type == "와동":
            cavity = True

        img = draw_carries_box(img, c.bbox, cavity)
    for t in teeth_info:
        img = draw_mask(img, t.mask.b64, (t.mask.origin_x, t.mask.origin_y))

    img = cv2.resize(img, dsize=view_size, interpolation=cv2.INTER_LINEAR)

    qray = cv2.imread(q_path, cv2.IMREAD_COLOR)
    qray = cv2.cvtColor(qray, cv2.COLOR_BGR2RGB)
    qray = cv2.resize(qray, dsize=view_size, interpolation=cv2.INTER_LINEAR)

    st.image(
        np.hstack((img, qray)),
        caption=f"fig. left is camera, right is qray",
    )


def view(cur):
    c_path, q_path, a_path = cur["c_path"], cur["q_path"], cur["a_path"]
    _global, _caries, _teeth = object_from_ann(a_path, cur["uid"])
    st.write(f"**[{cur['id']}] {cur['r_id']} > {cur['p_id']} > `{cur['uid']}`**")

    path_expander = st.expander("pathes", True)
    path_expander.caption(c_path)
    path_expander.caption(q_path)
    path_expander.caption(a_path)

    view_imgs(c_path, q_path, _caries, _teeth)

    g_expander = st.expander("global", True)
    g_expander.write(f"- global > anomaly > `{_global.anomaly}`")
    # direction: 설측 / 교합측 / 순측
    g_expander.write(f"- global > direction > `{_global.direction}`")
    g_expander.write(f"- global > bacteria_sm > `{_global.bacteria_sm}`")
    g_expander.write(f"- global > bacteria_sa > `{_global.bacteria_sa}`")

    for c in _caries:
        c_expander = st.expander("caries", True)
        c_expander.write(f"- caries> bbox > [`{c.bbox.x1}`,`{c.bbox.y1}`,`{c.bbox.w}`,`{c.bbox.h}`]")
        # cavity / noncavity
        c_expander.write(f"- caries> type > `{c.type}`")

    for t in _teeth:
        t_expander = st.expander("teeth", True)
        t_expander.write(f"- teeth > origin > [`{t.mask.origin_x}`,`{t.mask.origin_y}`]")
        t_expander.write(f"- teeth > maks > `{t.mask.b64[:10]}...`")
        # 전치(anterior) / 소구치(posterior-premolar) / 대구치(posterior-molar)
        t_expander.write(f"- teeth> type > `{t.type}`")

    return _global, _caries, _teeth


def save_format_anony(cur):
    u_id = cur['uid']
    srcpath_camera = cur['c_path'] 
    srcpath_qray = cur['q_path']
    srcpath_ann = cur['a_path']

    # predefined:
    #   OUTPATH = "out"
    outdir_camera = os.path.join(OUTPATH, 'camera')
    outdir_qray = os.path.join(OUTPATH, 'qray')
    outdir_ann = os.path.join(OUTPATH, 'ann')

    # predefined:
    #   IMG_FILE_FORMAT = "jpg"
    #   ANN_FILE_FORMAT = "json"
    outpath_camera = os.path.join(outdir_camera, f'{u_id}.{IMG_FILE_FORMAT}')
    outpath_qray = os.path.join(outdir_qray, f'{u_id}.{IMG_FILE_FORMAT}')
    outpath_ann = os.path.join(outdir_ann, f'{u_id}.{ANN_FILE_FORMAT}')

    Path(outdir_camera).mkdir(exist_ok=True)
    Path(outdir_qray).mkdir(exist_ok=True)
    Path(outdir_ann).mkdir(exist_ok=True)

    print(f"  from {srcpath_camera}")
    print(f"  to   {outpath_camera}")
    try:
        shutil.copyfile(srcpath_camera, outpath_camera)
        shutil.copyfile(srcpath_qray, outpath_qray)
        shutil.copyfile(srcpath_ann, outpath_ann)
    except:
        print("Error occurred while copying file.")

    outpath_ann_g = os.path.join(outdir_ann, f'global.{ANN_FILE_FORMAT}')
    outpath_ann_c = os.path.join(outdir_ann, f'carries.{ANN_FILE_FORMAT}')
    outpath_ann_t = os.path.join(outdir_ann, f'teeth.{ANN_FILE_FORMAT}')
    

def save_format_anony_all(filelist):
    file_max = len(filelist)

    progress_bar = st.progress(0)
    for i, cur in enumerate(filelist):
        print(f"[{i}] {cur['uid']} saved")
        save_format_anony(cur)
        progress_bar.progress( (i + 1) / file_max )


def main():
    filelist = load_filelist()

    if "curidx" not in st.session_state:
        st.session_state.curidx = 0

    # layout
    w1 = st.empty()
    st.write("---")  # control
    w2 = st.empty()
    w3 = st.empty()

    file_min, file_max = 0, len(filelist) - 1

    def downidx(idx, _min, _max):
        return idx - 1 if idx > file_min else _max

    def upidx(idx, _min, _max):
        return idx + 1 if idx < file_max else _min

#    with w1:
#        with st.container():
#            st.write(1)
#            st.write(2)
#            st.write(3)
#            time.sleep(0.5)
#
#        for i in range(5):
#            st.write(i)
#            time.sleep(0.5)

    with w1.container():
        cview = st.columns([1, 1, 5, 1, 2])
        btn_down = cview[0].button("  <  ")
        btn_up = cview[1].button("  >  ")
        btn_save = cview[3].button("save")
        btn_save_all = cview[4].button("save all")

        if btn_down:
            st.session_state.curidx = downidx(
                st.session_state.curidx, file_min, file_max
            )
        if btn_up:
            st.session_state.curidx = upidx(st.session_state.curidx, file_min, file_max)

        st.write(st.session_state.curidx)

    cur = filelist[st.session_state.curidx]

    if btn_save_all:
        #pbar = w2.progress(0)
        #with w3:
        #    #for i in range(file_max):
        #    for i in range(10):
        #        _cur = filelist[i]
        #        with w3.container():
        #            view(_cur)  # TODO bug
        #        pbar.progress( (i + 1) / file_max )
        #        time.sleep(0.01)
        with w2:
            save_format_anony_all(filelist)
    else:
        with w3.container():
            view(cur)


if __name__ == "__main__":
    main()
