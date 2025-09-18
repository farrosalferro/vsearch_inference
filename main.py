import os
import json
import argparse
from utils import load_mllm, load_ovd, SampleTemplate, fill_sample_information, time_function
from PIL import Image
from config_parser import ConfigParser, config_to_args

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
    ovd_processor = load_ovd(args)

    for sample in questions[150:]:
        # Create a typed template - starts as empty dict but has type hints
        sample_template: SampleTemplate = {}
        image_file = os.path.join(args.image_folder, sample["image"])
        image = Image.open(image_file)
        fill_sample_information(sample_template, sample)

        (extracted_nouns, _), noun_extraction_time = time_function(
            mllm_processor.generate_answer, sample, None, None, phase="noun_extraction"
        )
        extracted_nouns = [noun.strip() for noun in extracted_nouns.split(",")]
        sample_template["target_objects"] = extracted_nouns
        sample_template["target_objects_extraction_time"] = noun_extraction_time
        records["time"]["target_objects_extraction"] += noun_extraction_time

        ovd_results, ovd_detection_time = time_function(
            ovd_processor.run_detection, image, extracted_nouns
        )
        sample_template.update(ovd_results)
        sample_template["ovd_time"] = ovd_detection_time
        records["time"]["ovd_detection"] += ovd_detection_time
        
        (_, sample_template), qa_generation_time = time_function(
            mllm_processor.generate_answer, sample, image, sample_template, phase="qa", ovd_results=ovd_results
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