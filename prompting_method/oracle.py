from PIL import Image

def oracle_prompting(ovd_results, question, image):
    bbox = ovd_results.get("bbox", [])
    target_objects = ovd_results.get("target_object", [])

    if len(bbox) == 1:
        crop = image.crop((bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]))
    else:
        # in this case, merge the bounding box into one
        x1, y1, x2, y2 = bbox[0]
        for box in bbox:
            x1 = min(x1, box[0])
            y1 = min(y1, box[1])
            x2 = max(x2, box[2])
            y2 = max(y2, box[3])
        crop = image.crop((x1, y1, x2, y2))

    return question, crop