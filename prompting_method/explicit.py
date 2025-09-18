from PIL import Image, ImageDraw, ImageFont

def explicit_prompting(ovd_results, question, image):
    bbox = ovd_results.get("bbox", [])
    target_objects = ovd_results.get("target_object", [])

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

    for box in bbox:
        # Validate box format
        if isinstance(box, list) and len(box) == 4:
            x, y, w, h = box
            
            # Ensure coordinates are valid
            if all(isinstance(coord, (int, float)) for coord in [x, y, w, h]):
                # Draw rectangle
                draw.rectangle([x, y, x+w, y+h], outline="red", width=3)

    return question, image