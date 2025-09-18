

def generate_relative_position_prompt(question, owlv2_result, image_size):
    target_objects_prompt = "The image size is " + str(image_size[0]) + "x" + str(image_size[1]) + ".\n"
    target_objects_prompt += "You are given the following information to answer your question:\n"
    for key, values in owlv2_result.items():
        for feature, value in values.items():
            if feature == "saved_crop":
                continue
            if feature == "bbox":
                # info = f"The object {key} is located at the position [{round(value['x1'], 2)}, {round(value['y1'], 2)}, {round(value['x2'], 2)}, {round(value['y2'])}]."
                info = f"The object {key} is located at the position {value}."
            elif feature == "region":
                info = f"The object {key} is located at the region {value}."
        target_objects_prompt += info + "\n"
    target_objects_prompt += "Please answer the following question:\n"
    target_objects_prompt += question
    return target_objects_prompt