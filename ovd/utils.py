import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from collections import namedtuple
import torch

def create_padded_crop_bbox(original_bbox, target_size=336, image_width=None, image_height=None):
    """
    Create a fixed-size bounding box for cropping, centered on the original bbox.
    Adjust position if near edges while ensuring original bbox is included.
    
    Args:
        original_bbox: [x1, y1, x2, y2] - original bounding box coordinates
        target_size: int - target square size (default 336)
        image_width: int - image width
        image_height: int - image height
    
    Returns:
        [x1, y1, x2, y2] - padded crop box coordinates
    """
    # if original bbox is larger than target_size, no need to pad
    x1, y1, x2, y2 = original_bbox

    if x2 - x1 > target_size or y2 - y1 > target_size:
        return original_bbox
    
    # Calculate center of original bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Calculate half size
    half_size = target_size // 2
    
    # Initial crop box centered on original bbox center
    crop_x1 = center_x - half_size
    crop_y1 = center_y - half_size
    crop_x2 = center_x + half_size
    crop_y2 = center_y + half_size
    
    # Adjust if going outside image bounds
    # Shift horizontally if needed
    if crop_x1 < 0:
        shift = -crop_x1
        crop_x1 += shift
        crop_x2 += shift
    elif crop_x2 > image_width:
        shift = crop_x2 - image_width
        crop_x1 -= shift
        crop_x2 -= shift
    
    # Shift vertically if needed    
    if crop_y1 < 0:
        shift = -crop_y1
        crop_y1 += shift
        crop_y2 += shift
    elif crop_y2 > image_height:
        shift = crop_y2 - image_height
        crop_y1 -= shift
        crop_y2 -= shift
    
    # Final clamp to image bounds (in case image is smaller than target_size)
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(image_width, crop_x2)
    crop_y2 = min(image_height, crop_y2)
    
    return [int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)]

def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    box1, box2: [x1, y1, x2, y2] format
    """
    # Calculate intersection area
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def convert_xywh_to_xyxy(bbox, img_width, img_height):
    """
    Convert bounding box from (x, y, w, h) to (x1, y1, x2, y2) format.
    """
    x, y, w, h = bbox
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1, y1, x2, y2]

def get_bbox_center(bbox):
    """
    Get the center point of a bounding box.
    bbox: [x1, y1, x2, y2] format
    Returns: [center_x, center_y]
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return [center_x, center_y]


def calculate_euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    point1, point2: [x, y] format
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def safe_to_list(item):
    """
    Safely convert tensor or array to list, handling cases where item is already a list.
    """
    if hasattr(item, 'tolist'):
        return item.tolist()
    elif isinstance(item, (list, tuple)):
        return list(item)
    else:
        return item


def safe_to_float(item):
    """
    Safely convert tensor or scalar to float, handling cases where item is already a float.
    """
    if hasattr(item, 'item'):
        return item.item()
    elif isinstance(item, (int, float)):
        return float(item)
    else:
        return item

def center_to_corners_format(boxes):
    """
    Convert bounding boxes from center format (cx, cy, w, h) to corners format (x1, y1, x2, y2).
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def scale_boxes(boxes, target_sizes):
    """
    Scale batch of bounding boxes to the target sizes.
    """
    if isinstance(target_sizes, (list, tuple)):
        image_height = torch.tensor([i[0] for i in target_sizes])
        image_width = torch.tensor([i[1] for i in target_sizes])
    elif isinstance(target_sizes, torch.Tensor):
        image_height, image_width = target_sizes.unbind(1)
    else:
        raise ValueError("`target_sizes` must be a list, tuple or torch.Tensor")

    # for owlv2 image is padded to max size unlike owlvit, that's why we have to scale boxes to max size
    max_size = torch.max(image_height, image_width)

    scale_factor = torch.stack([max_size, max_size, max_size, max_size], dim=1)
    scale_factor = scale_factor.unsqueeze(1).to(boxes.device)
    boxes = boxes * scale_factor
    return boxes