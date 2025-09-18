from abc import ABC
from prompting_method import explicit_prompting, implicit_prompting, both_prompting, oracle_prompting


class MLLMProcessor(ABC):

    def __init__(self, args):
        self.args = args

    def get_prompting_method(self, prompting_method):
        if prompting_method == "explicit":
            return explicit_prompting
        elif prompting_method == "implicit":
            return implicit_prompting
        elif prompting_method == "both":
            return both_prompting
        elif prompting_method == "oracle":
            return oracle_prompting
        else:
            raise ValueError(f"Invalid prompting method: {prompting_method}")

    def extract_information(self, sample, image, output_template, ovd_results):
        category = sample["category"]
        question = sample["text"]
        # image_base_name = os.path.splitext(os.path.basename(image_file))[0]

        prompting_method = self.get_prompting_method(self.args.prompting_method[category])
        question, image = prompting_method(ovd_results, question, image)

        output_template["modified_prompt"] = question

        return image, question, output_template

    def generate_answer(self, **kwargs):
        pass

    def load_model(self, **kwargs):
        pass