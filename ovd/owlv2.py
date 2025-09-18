from .base import OVDProcessor
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers.image_utils import load_image
import torch
from .utils import *

class OWLv2Processor(OVDProcessor):
    def __init__(self, args):
        super().__init__(args)
        self.model, self.processor = self.load_model()

    def load_model(self):
        model = Owlv2ForObjectDetection.from_pretrained(self.args.ovd_model_path).to(self.args.device_map)
        processor = Owlv2Processor.from_pretrained(self.args.ovd_model_path)
        return model, processor

    def get_predictions(self, image, target_object):
        image = image.convert("RGB")
        queries = [q.strip() for q in target_object.split('.') if q.strip()]
        
        # Format queries as expected by OWLv2 - each query should be a separate item
        text_queries = []
        for query in queries:
            text_queries.append(f"a photo of {query}")
        
        # Process the image and text
        inputs = self.processor(text=[text_queries], images=image, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get logits and boxes from outputs
        batch_logits, batch_boxes = outputs.logits, outputs.pred_boxes
        batch_size = len(batch_logits)
        
        # Get scores and labels (similar to the original post_process_object_detection)
        batch_class_logits = torch.max(batch_logits, dim=-1)
        batch_scores = torch.sigmoid(batch_class_logits.values)
        batch_labels = batch_class_logits.indices
        
        # Convert boxes from center format to corners format
        batch_boxes = center_to_corners_format(batch_boxes)
        
        # Scale boxes to image size
        target_sizes = torch.tensor([(image.height, image.width)])
        batch_boxes = scale_boxes(batch_boxes, target_sizes)
        
        # Get results for the first (and only) image
        scores = batch_scores[0]
        labels = batch_labels[0]
        boxes = batch_boxes[0]
        
        # Sort by scores to get top and bottom predictions
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_scores = scores[sorted_indices]
        sorted_labels = labels[sorted_indices]
        sorted_boxes = boxes[sorted_indices]
        
        # Get top 10 and bottom 10
        num_predictions = len(sorted_scores)
        top_k = min(10, num_predictions)
        
        top_boxes = sorted_boxes[:top_k]
        top_scores = sorted_scores[:top_k]
        top_labels = sorted_labels[:top_k]
        
        # Convert labels to text labels
        top_text_labels = [text_queries[label.item()] if label.item() < len(text_queries) else f"class_{label.item()}" for label in top_labels]    
        
        return top_boxes, top_scores, top_text_labels
        
        