from PIL import Image, ImageDraw, ImageFont

def both_prompting(ovd_results, question, image):
    bbox = ovd_results.get("bbox", [])
    target_objects = ovd_results.get("target_objects", [])
    
    draw = ImageDraw.Draw(image)
    if len(bbox) == 1:
        colors = ["red"]
    elif len(bbox) == 2:
        colors = ["red", "blue"]
    else:
        # Generate different colors for more than 2 bboxes
        color_palette = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta", "brown", "pink"]
        colors = [color_palette[i % len(color_palette)] for i in range(len(bbox))]

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
    
    implicit_bbox = "Given the following item and its bounding boxes:\n"
    for box, target_object, color in zip(bbox, target_objects, colors):
        implicit_bbox += f"{target_object}: {color} box\n"

        if isinstance(box, list) and len(box) == 4:
            x, y, w, h = box
            
            # Ensure coordinates are valid
            if all(isinstance(coord, (int, float)) for coord in [x, y, w, h]):
                # Draw rectangle
                draw.rectangle([x, y, x+w, y+h], outline=color, width=3)

    implicit_bbox += "Please answer the following question:\n"
    question = implicit_bbox + question

    return question, image