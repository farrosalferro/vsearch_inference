from mllm import QwenProcessor, LlavaHfProcessor
from ovd import LLMDetProcessor, OWLv2Processor, MMGroundingDINOProcessor
from typing import TypedDict, List, Optional, Dict, Callable, Any, Tuple
import time

def time_function(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Times the execution of a function and returns both the result and execution time.
    
    Args:
        func: The function to time
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Tuple of (function_result, execution_time_in_seconds)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

class SampleTemplate(TypedDict, total=False):
    # sample information
    sample_id: str
    original_prompt: str
    gt_response: str
    category: str
    image_path: str

    # target objects extraction
    target_objects: Optional[List[str]]
    target_objects_extraction_time: Optional[float]

    # ovd results
    bbox: Optional[List]
    is_bbox: Optional[List[bool]]
    ovd_time: Optional[float]

    # mllm results
    modified_prompt: str
    pred_response: str
    mllm_time: Optional[float]

def load_mllm(args):
    if "llava" in args.mllm:
        mllm_processor = LlavaHfProcessor(args)
    elif "qwen" in args.mllm:
        mllm_processor = QwenProcessor(args)
    else:
        raise ValueError(f"Invalid MLLM: {args.mllm}")
    
    return mllm_processor

def load_ovd(args):
    if "llmdet" in args.ovd:
        ovd_processor = LLMDetProcessor(args)
    elif "owlv2" in args.ovd:
        ovd_processor = OWLv2Processor(args)
    elif "mmgroundingdino" in args.ovd:
        ovd_processor = MMGroundingDINOProcessor(args)
    else:
        raise ValueError(f"Invalid OVD: {args.ovd}")
    
    return ovd_processor

def fill_sample_information(sample_template: SampleTemplate, sample: Dict):
    sample_template["sample_id"] = sample["question_id"]
    sample_template["original_prompt"] = sample["text"]
    sample_template["gt_response"] = sample["label"]
    sample_template["category"] = sample["category"]
    sample_template["image_path"] = sample["image"]

