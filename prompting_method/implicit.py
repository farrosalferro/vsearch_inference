from PIL import Image

def implicit_prompting(ovd_results, question, image):
    bbox = ovd_results.get("bbox", [])
    target_objects = ovd_results.get("target_object", [])

    implicit_bbox = "Given the following item and its bounding boxes:\n"
    for box, target_object in zip(bbox, target_objects):
        implicit_bbox += f"{target_object}: {box}\n"
    implicit_bbox += "Please answer the following question:\n"

    question = implicit_bbox + question

    return question, image