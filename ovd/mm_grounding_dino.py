from .base import OVDProcessor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers.image_utils import load_image
import torch
import warnings
from .utils import *

class MMGroundingDINOProcessor(OVDProcessor):
    def __init__(self, args):
        super().__init__(args)
        self.model, self.processor = self.load_model()

    def load_model(self):
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.args.ovd_model_path).to(self.args.device_map)
        processor = AutoProcessor.from_pretrained(self.args.ovd_model_path)
        return model, processor

    def get_predictions(self, image, target_object):
        image = load_image(image)
        queries = [q.strip() for q in target_object.split('.') if q.strip()]
            
        # Format queries as expected by LLMDet - wrap each query in a list
        text_labels = []
        for query in queries:
            # Add "a " prefix if not already present for better detection
            if not query.lower().startswith(('a ', 'an ', 'the ')):
                query = f"a {query}"
            text_labels.append(query)
        text_labels = [text_labels]
        
        # If no queries after processing, use the original text
        if not text_labels or not text_labels[0]:
            text_labels = [[f"a {target_object}"]]
        
        # Process the image and text
        inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(self.args.device_map)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use a lower threshold for initial processing to get more candidates
        low_threshold = min(0.1, self.args.detection_args["score_threshold"])
        
        # Suppress the FutureWarning about 'labels' key
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.grounding_dino.processing_grounding_dino")
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                threshold=low_threshold,
                target_sizes=[(image.height, image.width)]
            )
        
        # Extract results for the first (and only) image
        result = results[0]
        boxes = result["boxes"]
        scores = result["scores"]
        
        # Handle labels - use text_labels if available, otherwise fall back to labels
        if "text_labels" in result:
            labels = result["text_labels"]
        else:
            labels = result["labels"]
        
        # Sort by scores to get top predictions
        if len(boxes) > 0:
            # Convert to lists for sorting
            boxes_list = [safe_to_list(box) for box in boxes]
            scores_list = [safe_to_float(score) for score in scores]
            labels_list = list(labels)
            
            # Sort by scores (descending)
            sorted_indices = sorted(range(len(scores_list)), key=lambda i: scores_list[i], reverse=True)
            
            # Get top predictions (up to 20 for region analysis)
            top_k = min(20, len(sorted_indices))
            
            top_boxes = [boxes_list[i] for i in sorted_indices[:top_k]]
            top_scores = [scores_list[i] for i in sorted_indices[:top_k]]
            top_labels = [labels_list[i] for i in sorted_indices[:top_k]]
            
            # Convert back to tensors for consistency
            top_boxes = torch.tensor(top_boxes) if top_boxes else torch.empty(0, 4)
            top_scores = torch.tensor(top_scores) if top_scores else torch.empty(0)
        else:
            top_boxes = torch.empty(0, 4)
            top_scores = torch.empty(0)
            top_labels = []

        return top_boxes, top_scores, top_labels