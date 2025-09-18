from PIL import Image, ImageDraw, ImageFont
import os
import math
from collections import namedtuple
from .utils import safe_to_list, safe_to_float, create_padded_crop_bbox

# Named tuple for tile information
TileInfo = namedtuple('TileInfo', ['tile_image', 'x_offset', 'y_offset', 'tile_id'])

def generate_sliding_window_tiles(image, tile_size=512, overlap_ratio=0.25):
    """
    Generate overlapping tiles from an image using sliding window approach.
    
    Args:
        image: PIL Image object
        tile_size: Size of each tile (square)
        overlap_ratio: Overlap ratio between adjacent tiles (0.0 to 1.0)
    
    Returns:
        List of TileInfo namedtuples containing tile images and their positions
    """
    image_width, image_height = image.size
    step_size = int(tile_size * (1 - overlap_ratio))
    
    tiles = []
    tile_id = 0
    
    # Generate tiles with sliding window
    for y in range(0, image_height, step_size):
        for x in range(0, image_width, step_size):
            # Calculate tile boundaries
            x1 = x
            y1 = y
            x2 = min(x + tile_size, image_width)
            y2 = min(y + tile_size, image_height)
            
            # Skip tiles that are too small (less than 50% of target size)
            tile_width = x2 - x1
            tile_height = y2 - y1
            min_acceptable_size = tile_size * 0.5
            
            if tile_width < min_acceptable_size or tile_height < min_acceptable_size:
                continue
            
            # Crop the tile
            tile_image = image.crop((x1, y1, x2, y2))
            
            # If tile is smaller than target size, pad it to maintain aspect ratio
            if tile_width < tile_size or tile_height < tile_size:
                # Create a new image with target size and paste the tile
                padded_tile = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
                padded_tile.paste(tile_image, (0, 0))
                tile_image = padded_tile
            
            tiles.append(TileInfo(
                tile_image=tile_image,
                x_offset=x1,
                y_offset=y1,
                tile_id=f"tile_{tile_id}"
            ))
            tile_id += 1
    
    return tiles

def map_tile_coords_to_original(box_coords, tile_info, original_tile_size):
    """
    Map bounding box coordinates from tile space to original image space.
    
    Args:
        box_coords: [x1, y1, x2, y2] in tile coordinates
        tile_info: TileInfo object containing tile position
        original_tile_size: Original size of the tile before any padding
    
    Returns:
        [x1, y1, x2, y2] in original image coordinates
    """
    x1, y1, x2, y2 = box_coords
    
    # Map coordinates back to original image
    original_x1 = tile_info.x_offset + x1
    original_y1 = tile_info.y_offset + y1
    original_x2 = tile_info.x_offset + x2
    original_y2 = tile_info.y_offset + y2
    
    return [original_x1, original_y1, original_x2, original_y2]

def calculate_detection_quality_score(box_coords, tile_info, tile_size):
    """
    Calculate a quality score for a detection based on how centered it is in the tile.
    Detections near the center of tiles are generally more reliable.
    
    Args:
        box_coords: [x1, y1, x2, y2] in tile coordinates
        tile_info: TileInfo object
        tile_size: Size of the tile
    
    Returns:
        Quality score (0.0 to 1.0, higher is better)
    """
    x1, y1, x2, y2 = box_coords
    
    # Calculate detection center
    det_center_x = (x1 + x2) / 2
    det_center_y = (y1 + y2) / 2
    
    # Calculate tile center
    tile_center_x = tile_size / 2
    tile_center_y = tile_size / 2
    
    # Calculate distance from tile center (normalized by tile size)
    distance_x = abs(det_center_x - tile_center_x) / (tile_size / 2)
    distance_y = abs(det_center_y - tile_center_y) / (tile_size / 2)
    
    # Quality score decreases with distance from center
    # Use max distance to ensure score is between 0 and 1
    max_distance = max(distance_x, distance_y)
    quality_score = max(0.0, 1.0 - max_distance)
    
    return quality_score

def merge_overlapping_detections(detections, iou_threshold=0.5):
    """
    Merge overlapping detections using Non-Maximum Suppression approach.
    
    Args:
        detections: List of (box_coords, score, label, quality_score) tuples
        iou_threshold: IoU threshold for considering detections as overlapping
    
    Returns:
        List of merged detections
    """
    if not detections:
        return []
    
    # Sort detections by combined score (detection score * quality score)
    detections.sort(key=lambda x: x[1] * x[3], reverse=True)
    
    merged = []
    used = set()
    
    for i, (box1, score1, label1, quality1) in enumerate(detections):
        if i in used:
            continue
            
        # Find all overlapping detections
        overlapping = [(box1, score1, label1, quality1)]
        used.add(i)
        
        for j, (box2, score2, label2, quality2) in enumerate(detections[i+1:], i+1):
            if j in used:
                continue
                
            # Calculate IoU
            iou = calculate_iou(box1, box2)
            if iou > iou_threshold and label1 == label2:
                overlapping.append((box2, score2, label2, quality2))
                used.add(j)
        
        # If multiple overlapping detections, merge them
        if len(overlapping) > 1:
            merged_detection = merge_detection_group(overlapping)
        else:
            merged_detection = overlapping[0]
            
        merged.append(merged_detection)
    
    return merged

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2] format
    
    Returns:
        IoU value (0.0 to 1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

def merge_detection_group(detections):
    """
    Merge a group of overlapping detections into a single detection.
    
    Args:
        detections: List of (box_coords, score, label, quality_score) tuples
    
    Returns:
        Single merged detection tuple
    """
    # Use weighted average based on detection score * quality score
    total_weight = sum(score * quality for _, score, _, quality in detections)
    
    if total_weight == 0:
        # Fallback to simple average
        boxes = [box for box, _, _, _ in detections]
        avg_box = [
            sum(box[i] for box in boxes) / len(boxes)
            for i in range(4)
        ]
        max_score = max(score for _, score, _, _ in detections)
        best_label = detections[0][2]  # Use first label
        avg_quality = sum(quality for _, _, _, quality in detections) / len(detections)
        return (avg_box, max_score, best_label, avg_quality)
    
    # Weighted average of bounding boxes
    weighted_box = [0, 0, 0, 0]
    for box, score, _, quality in detections:
        weight = (score * quality) / total_weight
        for i in range(4):
            weighted_box[i] += box[i] * weight
    
    # Take maximum score and quality
    max_score = max(score for _, score, _, _ in detections)
    max_quality = max(quality for _, _, _, quality in detections)
    best_label = detections[0][2]  # Use label from highest scoring detection
    
    return (weighted_box, max_score, best_label, max_quality)

def run_sliding_window(get_predictions, image, target_object, tile_size=512, overlap_ratio=0.25, score_threshold=0.3):
    """
    Run sliding window object detection on a large image.
    
    Args:
        ovd_processor: OVD processor object with get_predictions method
        image: PIL Image object
        target_object: Target object description string
        tile_size: Size of each sliding window tile
        overlap_ratio: Overlap ratio between adjacent tiles
        score_threshold: Minimum confidence score for detections
    
    Returns:
        Tuple (bounding_box, found):
            - bounding_box: [x1, y1, x2, y2] coordinates of best detection or whole image
            - found: Boolean indicating if target object was found above threshold
    """
    print(f"Running sliding window detection with tile_size={tile_size}, overlap={overlap_ratio}")
    
    # Generate sliding window tiles
    tiles = generate_sliding_window_tiles(image, tile_size, overlap_ratio)
    print(f"Generated {len(tiles)} tiles for processing")
    
    all_detections = []
    
    # Process each tile
    for i, tile_info in enumerate(tiles):
        try:
            print(f"Processing tile {i+1}/{len(tiles)}: {tile_info.tile_id}")
            
            # Get predictions for this tile
            boxes, scores, labels = get_predictions(tile_info.tile_image, target_object)
            
            # Process each detection in this tile
            for box, score, label in zip(boxes, scores, labels):
                score_val = safe_to_float(score)
                box_coords = safe_to_list(box)
                
                # Calculate quality score based on position in tile
                quality_score = calculate_detection_quality_score(box_coords, tile_info, tile_size)
                
                # Map coordinates back to original image
                original_coords = map_tile_coords_to_original(
                    box_coords, tile_info, tile_size
                )
                
                # Store detection with all relevant information
                detection = (original_coords, score_val, label, quality_score)
                all_detections.append(detection)
                
                print(f"  Found detection: score={score_val:.3f}, quality={quality_score:.3f}")
        
        except Exception as e:
            print(f"Error processing tile {tile_info.tile_id}: {str(e)}")
            continue
    
    if not all_detections:
        print("No detections found in any tile")
        return [0, 0, image.width, image.height], False
    
    print(f"Found {len(all_detections)} total detections before merging")
    
    # Merge overlapping detections
    merged_detections = merge_overlapping_detections(all_detections, iou_threshold=0.3)
    print(f"After merging: {len(merged_detections)} detections")
    
    # Find best detection above threshold
    best_detection = None
    best_combined_score = 0
    
    for box_coords, score, label, quality in merged_detections:
        combined_score = score * quality
        
        print(f"Detection: score={score:.3f}, quality={quality:.3f}, combined={combined_score:.3f}")
        
        if score >= score_threshold and combined_score > best_combined_score:
            best_detection = (box_coords, score, label, quality)
            best_combined_score = combined_score
    
    if best_detection:
        print(f"Found target object with score {best_detection[1]:.3f}")
        # Create padded crop bbox around the detection
        padded_bbox = create_padded_crop_bbox(
            best_detection[0], 
            target_size=min(512, min(image.width, image.height)),
            image_width=image.width, 
            image_height=image.height
        )
        return padded_bbox, True
    else:
        print(f"No detections above threshold {score_threshold}")
        return [0, 0, image.width, image.height], False

def save_sliding_window_visualization(image, tiles, detections, output_path):
    """
    Save a visualization of the sliding window tiles and detections.
    
    Args:
        image: Original PIL Image
        tiles: List of TileInfo objects
        detections: List of detection tuples
        output_path: Path to save visualization
    """
    # Create a copy of the original image for drawing
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    # Draw tile boundaries
    for tile_info in tiles:
        x1 = tile_info.x_offset
        y1 = tile_info.y_offset
        x2 = x1 + tile_info.tile_image.width
        y2 = y1 + tile_info.tile_image.height
        
        # Draw tile boundary in blue
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
    
    # Draw detections
    font = ImageFont.load_default()
    for i, (box_coords, score, label, quality) in enumerate(detections):
        x1, y1, x2, y2 = [int(coord) for coord in box_coords]
        
        # Draw detection box in green
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        
        # Add label
        label_text = f"{label} ({score:.3f})"
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x1, y1-20), label_text, font)
        else:
            w, h = draw.textsize(label_text, font)
            bbox = (x1, y1-20, w + x1, y1-20 + h)
        
        draw.rectangle(bbox, fill="green")
        draw.text((x1, y1-20), label_text, fill="white", font=font)
    
    vis_image.save(output_path)
    print(f"Saved sliding window visualization to {output_path}")
