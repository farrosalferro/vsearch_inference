from abc import ABC
from .utils import *
from .recursion import *
from .sliding_window import *


class OVDProcessor(ABC):
    def __init__(self, args):
        self.args = args

    def load_model(self):
        pass

    def run_detection(self, image, target_objects):
        detection_info = {
            "target_objects": target_objects,
            "bbox": [],
            "is_bbox": []
        }
        if self.args.detection_args["detection_method"] == "recursion":
            for target_object in target_objects:
                bbox, is_bbox = run_recursion(self.get_predictions, image, target_object, self.args.detection_args["min_region_size"], self.args.detection_args["max_depth"], self.args.detection_args["score_threshold"])
                detection_info["bbox"].append(bbox)
                detection_info["is_bbox"].append(is_bbox)
        elif self.args.detection_args["detection_method"] == "sliding_window":
            for target_object in target_objects:
                bbox, is_bbox = run_sliding_window(self.get_predictions, image, target_object, self.args.detection_args["tile_size"], self.args.detection_args["overlap_ratio"], self.args.detection_args["score_threshold"])
                detection_info["bbox"].append(bbox)
                detection_info["is_bbox"].append(is_bbox)
        else:
            raise ValueError(f"Invalid detection method: {self.args.detection_args['detection_method']}")

        return detection_info

    def get_predictions(self, image, target_object):
        pass