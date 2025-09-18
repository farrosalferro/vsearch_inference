import yaml
import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration class that holds all parameters for the full inference pipeline."""
    
    # Data paths
    question_file: str = ""
    answers_file: str = ""
    image_folder: str = ""
    
    # Model selection
    mllm: str = "llava"  # llava, qwen
    ovd: str = "owlv2"   # llmdet, owlv2
    device_map: str = "cuda"
    
    # MLLM parameters
    mllm_model_path: str = ""
    mllm_model_base: str = None
    extra_prompt: str = ""
    temperature: float = 0.2
    top_p: float = None
    num_beams: int = 1
    
    # OVD parameters
    ovd_model_path: str = ""
    detection_args: Dict[str, Any] = None
    
    # Prompting method by category
    prompting_method: Dict[str, str] = None
    
    def __post_init__(self):
        """Set default prompting method if not provided."""
        if self.prompting_method is None:
            self.prompting_method = {
                "direct_attributes": "explicit",
                "relative_position": "implicit"
            }


class ConfigParser:
    """Parser for YAML configuration files."""
    
    @staticmethod
    def load_config(config_path: str) -> Config:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Config object with loaded parameters
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML file is malformed
            ValueError: If required parameters are missing
        """
        config_path = os.path.expanduser(config_path)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as file:
                yaml_data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")
        
        if not yaml_data:
            yaml_data = {}
        
        # Create Config object with loaded data
        config = Config()
        
        # Update config attributes with YAML data
        for key, value in yaml_data.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Unknown configuration parameter '{key}' found in {config_path}")
        
        # Validate required parameters
        ConfigParser._validate_config(config)
        
        return config
    
    @staticmethod
    def _validate_config(config: Config) -> None:
        """
        Validate that required configuration parameters are set.
        
        Args:
            config: Config object to validate
            
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        required_params = {
            'question_file': "Path to questions JSON file",
            'answers_file': "Path to output answers file", 
            'image_folder': "Path to folder containing images",
            'mllm_model_path': "Path to the MLLM model",
            'ovd_model_path': "Path to the OVD model"
        }
        
        for param, description in required_params.items():
            value = getattr(config, param)
            if not value or value == "":
                raise ValueError(f"Required parameter '{param}' is missing. {description}")
        
        # Validate model choices
        valid_mllm = ["llava", "qwen"]
        if config.mllm.split("_")[0] not in valid_mllm:
            raise ValueError(f"Invalid MLLM '{config.mllm}'. Must be one of: {valid_mllm}")
        
        valid_ovd = ["llmdet", "owlv2", "mmgroundingdino"]  
        if config.ovd.split("_")[0] not in valid_ovd:
            raise ValueError(f"Invalid OVD '{config.ovd}'. Must be one of: {valid_ovd}")
        
        # Validate prompting methods
        valid_prompting_methods = ["explicit", "implicit", "both", "oracle"]
        for category, method in config.prompting_method.items():
            if method not in valid_prompting_methods:
                raise ValueError(f"Invalid prompting method '{method}' for category '{category}'. Must be one of: {valid_prompting_methods}")
    
    @staticmethod
    def save_config(config: Config, config_path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config: Config object to save
            config_path: Path where to save the YAML file
        """
        config_path = os.path.expanduser(config_path)
        
        # Convert config to dictionary
        config_dict = {}
        for key, value in config.__dict__.items():
            if not key.startswith('_'):
                config_dict[key] = value
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Save to YAML file
        with open(config_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)
        
        print(f"Configuration saved to: {config_path}")


# Utility function for backward compatibility with argparse
def config_to_args(config: Config):
    """
    Convert Config object to an argparse.Namespace-like object.
    This allows existing code that expects args.* to work with the new config system.
    
    Args:
        config: Config object
        
    Returns:
        Namespace-like object with config attributes
    """
    class Args:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    return Args(config.__dict__)
