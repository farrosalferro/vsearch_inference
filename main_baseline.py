import os
import json
import argparse
from utils import load_mllm, SampleTemplate, fill_sample_information, time_function
from PIL import Image
from config_parser import ConfigParser, config_to_args
from tqdm import tqdm

def main(args):
    # load data - keep original structure
    records = {
        "config": vars(args),  # Convert Args object to dictionary
        "record": [],
        "time": {
            "target_objects_extraction": 0,
            "ovd_detection": 0,
            "qa_generation": 0
        }
    }

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    mllm_processor = load_mllm(args)

    for sample in tqdm(questions):
        # Create a typed template - starts as empty dict but has type hints
        sample_template: SampleTemplate = {}
        image_file = os.path.join(args.image_folder, sample["image"])
        image = Image.open(image_file)
        fill_sample_information(sample_template, sample)
        
        (_, sample_template), qa_generation_time = time_function(
            mllm_processor.generate_answer, sample, image, sample_template, phase="baseline", ovd_results=None
        )
        sample_template["mllm_time"] = qa_generation_time
        records["time"]["qa_generation"] += qa_generation_time

        # Add the completed sample to records
        records["record"].append(sample_template)

        # Save the entire records structure after each sample
        with open(answers_file, "w") as f:
            json.dump(records, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Inference Pipeline with YAML Configuration")
    parser.add_argument("--config", "-c", type=str, required=True, 
                       help="Path to YAML configuration file")
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = ConfigParser.load_config(args.config)
    
    # Convert config to args object for backward compatibility
    config_args = config_to_args(config)
    
    # Run main function
    main(config_args)