#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd

from handling import subfiles
from structure import BoundBox, Mask, ToothLabel, CariesLabel, GlobalLabel

SRCPATH = "../data_anony"

IMG_FILE_FORMAT = "jpg"
ANN_FILE_FORMAT = "json"


class QsetAnnSupervisely:
    def __init__(self, path: str = None, _id: str = None):
        self.path = path
        self.id = _id
        self.file = None
        self.ann = None

        self.size = None

        self.n_caries = 0
        self.b_caries_anomal = False

        self.n_tooth = 0

        self._global = None
        self._caries = []
        self._tooth = []

    def __enter__(self):
        self.file = open(self.path)
        self.ann = json.load(self.file)
        self.process()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    def get_global(self):
        return self._global

    def get_caries(self):
        return self._caries

    def get_tooth(self):
        return self._tooth

    def process(self):
        _global_anomaly = False
        _global_direction = None
        _global_bacteria_sm = None
        _global_bacteria_sa = None
        _tooth_type = None

        self.size = (
            self.ann["size"]["width"],
            self.ann["size"]["height"],
        )

        for g_tag in self.ann["tags"]:
            g_tag_name = g_tag["name"]
            g_tag_value = str(g_tag["value"])

            if g_tag_name == "치아종류":
                _tooth_type = g_tag_value
            elif g_tag_name == "촬영방향":
                _global_direction = g_tag_value
            elif g_tag_name == "세균_SM":
                _global_bacteria_sm = g_tag_value
            elif g_tag_name == "세균_SA":
                _global_bacteria_sa = g_tag_value
            else:
                pass

        for obj in self.ann["objects"]:
            obj_class_title = obj["classTitle"]
            obj_tags = obj["tags"]  # rectangle type 에 대한 tags

            if obj_class_title == "Caries":
                self.n_caries += 1
                # if not self.b_caries_anomal:
                if not _global_anomaly:
                    _global_anomaly = True
                    self.b_caries_anomal = True

                _global_anomaly = True
                _caries_type = None
                _caries_bbox = None

                """
                obj_points_exterior = 
                    [[top_left x, top_left y], [bottom_right x, bottom_right y]]
                """
                obj_points_exterior = obj["points"]["exterior"]

                for tag in obj_tags:
                    tag_name = tag["name"]
                    tag_value = tag["value"]

                    if tag_name == "우식유형":
                        # 초기(noncavity), 와동(cavity)
                        _caries_type = tag_value

                """
                caries_bbox = [
                    obj_points_exterior[0][1],
                    obj_points_exterior[0][0],
                    obj_points_exterior[1][1],
                    obj_points_exterior[1][0],
                ]
                """
                caries_bbox = sum(obj_points_exterior, [])
                # caries_bbox = [
                #     obj_points_exterior[0][1],
                #     obj_points_exterior[0][0],
                #     obj_points_exterior[1][1],
                #     obj_points_exterior[1][0],
                # ]

                self._caries.append(
                    CariesLabel(_caries_type, BoundBox(*caries_bbox), self.id)
                )

            elif obj_class_title == "Tooth":
                self.n_tooth += 1

                obj_bitmap_data = obj["bitmap"]["data"]
                obj_bitmap_origin = obj["bitmap"]["origin"]

                # _data_b64 = obj["bitmap"]["data"]
                # _origin = obj["bitmap"]["origin"]

                _tooth_segmentation_mask = obj_bitmap_data
                _tooth_segmentation_origin = obj_bitmap_origin

                self._tooth.append(
                    ToothLabel(
                        Mask(*_tooth_segmentation_origin, _tooth_segmentation_mask),
                        _tooth_type,  # TODO need change
                        self.id,
                    )
                )

            else:
                pass

        self._global = GlobalLabel(
            _global_bacteria_sm,
            _global_bacteria_sa,
            _global_direction,
            _global_anomaly,
            self.id,
        )


def main():
    # load old anns
    # process old format to new format
    # save new anns
    cameras = list(subfiles(f"{SRCPATH}/camera"))
    qrays = list(subfiles(f"{SRCPATH}/qray"))
    anns = list(subfiles(f"{SRCPATH}/ann"))

    n_camera = len(cameras)
    n_qray = len(qrays)
    n_ann = len(anns)

    n_caries = 0
    n_tooth = 0

    n_caries_nomal = 0
    n_caries_anomal = 0

    n_cavity = 0
    n_non_cavity = 0
    n_caries_type_missing = 0

    n_tooth_anterior = 0
    n_tooth_premolar = 0
    n_tooth_molar = 0
    n_tooth_type_missing = 0

    # aggregation / analysis
    for fname in anns:
        _id = os.path.splitext(fname)[0]
        ann_path = os.path.join(SRCPATH, "ann", fname)
        with QsetAnnSupervisely(ann_path, _id) as ann:
            n_caries += ann.n_caries
            n_tooth += ann.n_tooth

            if ann.get_global().anomaly:
                n_caries_anomal += 1
            else:
                n_caries_nomal += 1

            for c in ann.get_caries():
                if c.type == "와동":
                    n_cavity += 1
                elif c.type == "초기":
                    n_non_cavity += 1
                else:
                    n_caries_type_missing += 1

            for t in ann.get_tooth():
                # 소구치 / 대구치/ 전치
                if t.type == "소구치":
                    n_tooth_premolar += 1
                elif t.type == "대구치":
                    n_tooth_molar += 1
                elif t.type == "전치":
                    n_tooth_anterior += 1
                else:
                    n_tooth_type_missing += 1

    # print out
    print("== dataset info ==")
    print("# file numbers")
    print(f"  - images: {n_camera}")
    print(f"  - qrays: {n_qray}")
    print(f"  - anns: {n_ann}")

    print()
    print("# privacy")
    print(f"  - patient: [anonymization]")

    print()
    print("# target numbers")
    print(f"  - caries: {n_caries}")
    print(f"  - tooth: {n_tooth}")

    print()
    print("# ratio")
    print(f"  - nomal : anomal = {n_caries_nomal} : {n_caries_anomal}")
    print(
        f"  - cavatiy : noncavity : (missing) = {n_cavity} : {n_non_cavity} : ({n_caries_type_missing})"
    )
    print(
        f"  - anterior : premolar : molar : (missing) = {n_tooth_anterior} : {n_tooth_premolar} : {n_tooth_molar} : ({n_tooth_type_missing})"
    )

    print("---")

    # new save file 1
    # csv type

    # 1) classification
    #   columns=['image_id', 'width', 'height', 'anomaly']
    # 2) localization
    #   columns=['image_id', 'width', 'height', 'x', 'y', 'w', 'h']
    # 3) segmentation
    #   columns=['image_id', 'width', 'height', 'mask']

    column_type_1 = ["image_id", "width", "height", "anomaly", "direction"]
    column_type_2 = ["image_id", "width", "height", "x", "y", "w", "h", "type"]
    column_type_3 = ["image_id", "width", "height", "origin_x", "origin_y", "mask", "type"]

    data_1 = []
    data_2 = []
    data_3 = []
    for fname in anns:
        _id = os.path.splitext(fname)[0]
        ann_path = os.path.join(SRCPATH, "ann", fname)
        with QsetAnnSupervisely(ann_path, _id) as ann:
            #print(ann._global.img_id)
            data_1.append(
                {
                    "image_id": ann._global.img_id,
                    "width": ann.size[0],
                    "height": ann.size[1],
                    "anomaly": ann._global.anomaly,
                    "direction": ann._global.direction,
                }
            )

            for i in ann.get_caries():
                data_2.append(
                    {
                        "image_id": ann._global.img_id,
                        "width": ann.size[0],
                        "height": ann.size[1],
                        "x": i.bbox.x1,
                        "y": i.bbox.y1,
                        "w": i.bbox.w,
                        "h": i.bbox.h,
                        "type": i.type,
                    }
                )

            for i in ann.get_tooth():
                data_3.append(
                    {
                        "image_id": ann._global.img_id,
                        "width": ann.size[0],
                        "height": ann.size[1],
                        "origin_x": i.mask.origin_x,
                        "origin_y": i.mask.origin_y,
                        "mask": i.mask.b64,
                        "type": i.type,
                    }
                )
                pass

    df_1 = pd.DataFrame(
        data_1,
        columns=column_type_1
    )
    print(df_1)

    df_2 = pd.DataFrame(
        data_2,
        columns=column_type_2
    )
    print(df_2)

    df_3 = pd.DataFrame(
        data_3,
        columns=column_type_3
    )
    print(df_3)

    df_1.to_csv(f"{SRCPATH}/global.csv", mode='w')
    df_2.to_csv(f"{SRCPATH}/caries.csv", mode='w')
    df_3.to_csv(f"{SRCPATH}/tooth.csv", mode='w')




if __name__ == "__main__":
    main()
