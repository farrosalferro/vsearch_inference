from PIL import Image, ImageDraw, ImageFont
import os
from collections import namedtuple
from .utils import *
import heapq

RegionTask = namedtuple('RegionTask', ['priority_score', 'region_coords', 'image_crop', 'depth', 'region_id'])

def divide_image_into_regions(image_width, image_height):
    """
    Divide image into 4 equal regions: top-left, top-right, bottom-left, bottom-right.
    Returns region coordinates as [x1, y1, x2, y2] for each region.
    """
    mid_x = image_width // 2
    mid_y = image_height // 2
    
    regions = {
        'top_left': [0, 0, mid_x, mid_y],
        'top_right': [mid_x, 0, image_width, mid_y],
        'bottom_left': [0, mid_y, mid_x, image_height],
        'bottom_right': [mid_x, mid_y, image_width, image_height]
    }
    
    return regions

def calculate_overlap_area(box, region):
    """
    Calculate the area of overlap between a bounding box and a region.
    Both box and region are in [x1, y1, x2, y2] format.
    """
    # Calculate intersection area
    x1_inter = max(box[0], region[0])
    y1_inter = max(box[1], region[1])
    x2_inter = min(box[2], region[2])
    y2_inter = min(box[3], region[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    return (x2_inter - x1_inter) * (y2_inter - y1_inter)

def categorize_boxes_to_regions(boxes, scores, image_width, image_height):
    """
    Categorize bounding boxes into 4 regions based on their location.
    If overlap between regions, assign to region with most overlap area.
    
    Returns: dict with region names as keys and list of (box_idx, score) as values.
    """
    regions = divide_image_into_regions(image_width, image_height)
    region_assignments = {region_name: [] for region_name in regions.keys()}
    
    for i, (box, score) in enumerate(zip(boxes, scores)):
        # Convert box to list format if it's a tensor
        if hasattr(box, 'tolist'):
            box_coords = box.tolist()
        else:
            box_coords = box
            
        max_overlap_area = 0
        assigned_region = None
        
        for region_name, region_coords in regions.items():
            overlap_area = calculate_overlap_area(box_coords, region_coords)
            if overlap_area > max_overlap_area:
                max_overlap_area = overlap_area
                assigned_region = region_name
        
        if assigned_region and max_overlap_area > 0:
            # Convert score to float if it's a tensor
            score_val = score.item() if hasattr(score, 'item') else float(score)
            region_assignments[assigned_region].append((i, score_val))
    
    return region_assignments, regions

def calculate_region_scores(region_assignments):
    """
    Calculate total score for each region by summing scores of assigned bounding boxes.
    """
    region_scores = {}
    for region_name, box_data in region_assignments.items():
        total_score = sum(score for _, score in box_data)
        region_scores[region_name] = total_score
    
    return region_scores

def crop_image_region(image_pil, region_coords):
    """
    Crop image based on region coordinates.
    region_coords: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = region_coords
    return image_pil.crop((x1, y1, x2, y2))

def save_crop_with_detection(image_crop, boxes, scores, labels, output_dir, base_name, depth, region_id, query_text):
    """
    Save cropped image with detected bounding boxes and return saved path.
    """
    # Create output filename with depth indicator
    filename = f"{base_name}_depth{depth}_{region_id}_detections.jpg"
    output_path = os.path.join(output_dir, filename)
    
    # Draw bounding boxes on the crop
    draw = ImageDraw.Draw(image_crop)
    font = ImageFont.load_default()
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        # Convert box to coordinates
        if hasattr(box, 'tolist'):
            x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]
        else:
            x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Convert score to float
        score_val = score.item() if hasattr(score, 'item') else float(score)
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        
        # Add label with confidence score
        label_text = f"{label} ({score_val:.3f})"
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x1, y1-15), label_text, font)
        else:
            w, h = draw.textsize(label_text, font)
            bbox = (x1, y1-15, w + x1, y1-15 + h)
        draw.rectangle(bbox, fill="green")
        draw.text((x1, y1-15), label_text, fill="white")
    
    # Add region info at the top
    info_text = f"Depth {depth} | Region: {region_id} | Query: {query_text}"
    if hasattr(font, "getbbox"):
        bbox = draw.textbbox((5, 5), info_text, font)
    else:
        w, h = draw.textsize(info_text, font)
        bbox = (5, 5, w + 5, 5 + h)
    draw.rectangle(bbox, fill="blue")
    draw.text((5, 5), info_text, fill="white")
    
    image_crop.save(output_path)
    return output_path

def save_crop_with_padding(image_pil, bbox, output_dir, base_name, target_object, target_size):
    """
    Save a crop with padding centered on the bounding box.
    """
    # Create padded crop bbox
    padded_bbox = create_padded_crop_bbox(
        bbox, target_size=target_size, 
        image_width=image_pil.width, image_height=image_pil.height
    )
    crop_x1, crop_y1, crop_x2, crop_y2 = padded_bbox
    
    # Crop the padded region
    cropped_image = image_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    
    # Resize to exactly target_size if needed (in case of edge constraints)
    if cropped_image.size != (target_size, target_size):
        cropped_image = cropped_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Save the crop
    filename = f"{base_name}_{target_object}_detection.jpg"
    output_path = os.path.join(output_dir, filename)
    cropped_image.save(output_path)
    
    return padded_bbox, output_path

def save_highest_scoring_region(image_crop, output_dir, base_name, target_object, region_id):
    """
    Save the highest scoring region without resizing.
    """
    filename = f"{base_name}_{target_object}_highest_score_region.jpg"
    output_path = os.path.join(output_dir, filename)
    image_crop.save(output_path)
    return output_path

def run_recursion(get_predictions, image, target_object, min_region_size, max_depth, score_threshold):
    region_counter = 0
    region_queue = []
    highest_scoring_region = None
    highest_score = -1

    initial_task = RegionTask(
        priority_score=1.0,
        region_coords=[0, 0, image.width, image.height],
        image_crop=image,
        depth=0,
        region_id="full_image")

    heapq.heappush(region_queue, (-initial_task.priority_score, region_counter, initial_task))
    region_counter += 1

    while region_queue:
        neg_priority, _, current_task = heapq.heappop(region_queue)
        current_priority = -neg_priority

        print(f"Processing region: {current_task.region_id} (depth={current_task.depth}, priority={current_priority:.3f})")

        if current_priority > highest_score:
            highest_score = current_priority
            highest_scoring_region = current_task

        region_width = current_task.region_coords[2] - current_task.region_coords[0]
        region_height = current_task.region_coords[3] - current_task.region_coords[1]

        if region_width < min_region_size or region_height < min_region_size or current_task.depth >= max_depth:
            print(f"  Stopping recursion: size=({region_width}x{region_height}), depth={current_task.depth}")
            continue

        try:
            top_boxes, top_scores, top_labels = get_predictions(current_task.image_crop, target_object)
            
            # print(f"  Found {len(top_boxes)} predictions (inference time: {timing_info['total_time']:.3f}s)")
            
            # Check if any prediction has score above threshold
            for box, score, label in zip(top_boxes, top_scores, top_labels):
                score_val = safe_to_float(score)
                if score_val >= score_threshold:
                    # print(f"  Found detection above threshold: {score_val:.3f}")
                    
                    # Convert box coordinates to original image space
                    box_coords = safe_to_list(box)
                    x1, y1, x2, y2 = box_coords
                    original_x1 = current_task.region_coords[0] + x1
                    original_y1 = current_task.region_coords[1] + y1
                    original_x2 = current_task.region_coords[0] + x2
                    original_y2 = current_task.region_coords[1] + y2
                    
                    original_bbox = [original_x1, original_y1, original_x2, original_y2]

                    output_bbox = create_padded_crop_bbox(
                        original_bbox, target_size=min_region_size, 
                        image_width=image.width, image_height=image.height
                    )

                    return output_bbox, True
            
            # No high-score detections, proceed with region division
            # print(f"  No detections above threshold, dividing into sub-regions")
            
            # Take top 10 predictions for region analysis
            analysis_boxes = top_boxes[:10] if len(top_boxes) >= 10 else top_boxes
            analysis_scores = top_scores[:10] if len(top_scores) >= 10 else top_scores
            
            if len(analysis_boxes) == 0:
                # print(f"  No predictions for region analysis")
                continue
                
            # Categorize boxes into regions
            region_assignments, regions = categorize_boxes_to_regions(
                analysis_boxes, analysis_scores, 
                current_task.image_crop.width, current_task.image_crop.height
            )
            
            # Calculate region scores
            region_scores = calculate_region_scores(region_assignments)
            
            # Add sub-regions to queue
            for region_name, region_coords in regions.items():
                region_score = region_scores[region_name]
                
                if region_score > 0:  # Only add regions with assigned boxes
                    # Adjust region coordinates relative to original image
                    adjusted_coords = [
                        current_task.region_coords[0] + region_coords[0],
                        current_task.region_coords[1] + region_coords[1], 
                        current_task.region_coords[0] + region_coords[2],
                        current_task.region_coords[1] + region_coords[3]
                    ]
                    
                    # Crop the sub-region
                    region_crop = crop_image_region(current_task.image_crop, region_coords)
                    
                    # Create new task
                    sub_region_id = f"{current_task.region_id}_{region_name}"
                    new_task = RegionTask(
                        priority_score=region_score,
                        region_coords=adjusted_coords,
                        image_crop=region_crop,
                        depth=current_task.depth + 1,
                        region_id=sub_region_id
                    )
                    
                    heapq.heappush(region_queue, (-region_score, region_counter, new_task))
                    region_counter += 1
                    
                    print(f"    Added sub-region {sub_region_id} with score {region_score:.3f}")
                
        except Exception as e:
            print(f"  Error processing region {current_task.region_id}: {str(e)}")
            continue

    # No detection found, save highest scoring region
    if highest_scoring_region:
        # print(f"No detection found. Saving highest scoring region: {highest_scoring_region.region_id} (score={highest_score:.3f})")
        output_bbox = highest_scoring_region.region_coords
        
        return output_bbox, False