import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from .template import TEMPLATE
from .base import MLLMProcessor

class LlavaHfProcessor(MLLMProcessor):
    def __init__(self, args):
        super().__init__(args)
        self.model, self.processor = self.load_model()

    def load_model(self):
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.args.mllm_model_path,
            # torch_dtype=torch.float16
            torch_dtype="auto", 
            low_cpu_mem_usage=True,
            device_map=self.args.device_map
        )
        processor = AutoProcessor.from_pretrained(self.args.mllm_model_path)
        return model, processor

    def _preprocess_for_qa(self, sample, image, output_template, ovd_results):
        image, question, output_template = self.extract_information(sample, image, output_template, ovd_results)

        prompt = self.args.extra_prompt + question
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant"
            },
            {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image",
                    "image": image,
                },
                ],
            },
        ]

        input_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=image, text=input_prompt, return_tensors="pt").to(self.model.device)
        return inputs, output_template

    def _preprocess_for_noun_extraction(self, sample):
        question = sample["text"].split("\n")[0]
        messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TEMPLATE.format(question=question)},
                    ],
                }
            ]
        input_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=input_prompt, return_tensors="pt").to(self.model.device)
        return inputs, None

    def generate_answer(self, sample, image, output_template, phase="qa", ovd_results=None):
        if phase == "qa":
            inputs, output_template = self._preprocess_for_qa(sample, image, output_template, ovd_results)
        elif phase == "noun_extraction":
            inputs, output_template = self._preprocess_for_noun_extraction(sample)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=1024, 
                do_sample=False
            )

        output_text = self._postprocess_answer(inputs.input_ids, output_ids)
        if output_template is not None:
            output_template["pred_response"] = output_text

        return output_text, output_template

    def _postprocess_answer(self, input_ids, output_ids):
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        return output_text