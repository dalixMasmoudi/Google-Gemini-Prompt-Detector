#!/usr/bin/env python3
"""
Configuration Module for GuardiaVision Gemini

This module handles configuration management, validation, and default settings
for the GuardiaVision detection system.

Classes:
    Config: Main configuration class with validation and defaults
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class DetectionConfig:
    """Configuration for detection parameters."""
    model_name: str = "gemini-2.5-flash-preview-05-20"
    temperature: float = 0.2
    thinking_budget: int = 0
    min_confidence: float = 0.7
    max_objects: int = 20
    remove_duplicates: bool = True
    iou_threshold: float = 0.5


@dataclass
class ImageConfig:
    """Configuration for image processing parameters."""
    max_size: int = 1024
    min_size: int = 512
    enhance_quality: bool = True
    contrast: float = 1.2
    brightness: float = 1.0
    sharpness: float = 1.1
    color: float = 1.0
    jpeg_quality: int = 95


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    show_confidence: bool = True
    show_labels: bool = True
    default_font_size: int = 12
    default_line_width: int = 3
    colors: List[str] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = [
                '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
                '#00FFFF', '#FFA500', '#800080', '#FFC0CB', '#A52A2A'
            ]


class Config:
    """
    Main configuration manager for GuardiaVision Gemini.
    
    This class provides centralized configuration management with validation,
    default values, and environment variable support. It handles all aspects
    of system configuration including detection, image processing, and visualization.
    
    Features:
        - Environment variable integration
        - Configuration validation
        - JSON config file support
        - Default value management
        - Type checking and conversion
        
    Example:
        >>> config = Config()
        >>> config.load_from_file("config.json")
        >>> detector_config = config.get_detection_config()
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file (str, optional): Path to JSON configuration file
                                       If provided, loads configuration from file
        """
        # Initialize with default configurations
        self.detection = DetectionConfig()
        self.image = ImageConfig()
        self.visualization = VisualizationConfig()
        
        # API key management
        self.api_key = self._get_api_key()
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # Override with environment variables
        self._load_from_environment()
    
    def _get_api_key(self) -> str:
        """
        Get API key from environment variables.
        
        Returns:
            str: Google API key
            
        Raises:
            ValueError: If API key is not found or empty
        """
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            # Check for common config files
            config_files = ['.env', 'config.json', 'secrets.json']
            for config_file in config_files:
                if os.path.exists(config_file):
                    try:
                        if config_file.endswith('.json'):
                            with open(config_file, 'r') as f:
                                data = json.load(f)
                                api_key = data.get('api_key') or data.get('google_api_key')
                                if api_key:
                                    break
                    except Exception:
                        continue
        
        if not api_key:
            raise ValueError(
                "Google API key not found. Please set GOOGLE_API_KEY environment variable "
                "or provide it in a config file."
            )
        
        return api_key
    
    def _load_from_environment(self):
        """Load configuration overrides from environment variables."""
        # Detection configuration
        if os.getenv('GEMINI_MODEL_NAME'):
            self.detection.model_name = os.getenv('GEMINI_MODEL_NAME')
        
        if os.getenv('GEMINI_TEMPERATURE'):
            try:
                self.detection.temperature = float(os.getenv('GEMINI_TEMPERATURE'))
            except ValueError:
                pass
        
        if os.getenv('MIN_CONFIDENCE'):
            try:
                self.detection.min_confidence = float(os.getenv('MIN_CONFIDENCE'))
            except ValueError:
                pass
        
        if os.getenv('MAX_OBJECTS'):
            try:
                self.detection.max_objects = int(os.getenv('MAX_OBJECTS'))
            except ValueError:
                pass
        
        # Image configuration
        if os.getenv('MAX_IMAGE_SIZE'):
            try:
                self.image.max_size = int(os.getenv('MAX_IMAGE_SIZE'))
            except ValueError:
                pass
        
        if os.getenv('ENHANCE_IMAGES'):
            self.image.enhance_quality = os.getenv('ENHANCE_IMAGES').lower() in ('true', '1', 'yes')
    
    def load_from_file(self, config_file: str):
        """
        Load configuration from JSON file.
        
        Args:
            config_file (str): Path to JSON configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid JSON
            
        Example:
            >>> config.load_from_file("my_config.json")
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            # Update detection config
            if 'detection' in data:
                for key, value in data['detection'].items():
                    if hasattr(self.detection, key):
                        setattr(self.detection, key, value)
            
            # Update image config
            if 'image' in data:
                for key, value in data['image'].items():
                    if hasattr(self.image, key):
                        setattr(self.image, key, value)
            
            # Update visualization config
            if 'visualization' in data:
                for key, value in data['visualization'].items():
                    if hasattr(self.visualization, key):
                        setattr(self.visualization, key, value)
            
            # Update API key if present
            if 'api_key' in data:
                self.api_key = data['api_key']
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_file}: {str(e)}")
    
    def save_to_file(self, config_file: str):
        """
        Save current configuration to JSON file.
        
        Args:
            config_file (str): Output file path
            
        Example:
            >>> config.save_to_file("saved_config.json")
        """
        config_data = {
            'detection': asdict(self.detection),
            'image': asdict(self.image),
            'visualization': asdict(self.visualization)
        }
        
        # Don't save API key to file for security
        # config_data['api_key'] = self.api_key
        
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def validate(self) -> List[str]:
        """
        Validate current configuration.
        
        Returns:
            List[str]: List of validation errors (empty if valid)
            
        Example:
            >>> errors = config.validate()
            >>> if errors:
            ...     print("Configuration errors:", errors)
        """
        errors = []
        
        # Validate detection config
        if not (0.0 <= self.detection.temperature <= 2.0):
            errors.append("Detection temperature must be between 0.0 and 2.0")
        
        if not (0.0 <= self.detection.min_confidence <= 1.0):
            errors.append("Minimum confidence must be between 0.0 and 1.0")
        
        if not (1 <= self.detection.max_objects <= 100):
            errors.append("Max objects must be between 1 and 100")
        
        if not (0.0 <= self.detection.iou_threshold <= 1.0):
            errors.append("IoU threshold must be between 0.0 and 1.0")
        
        # Validate image config
        if not (128 <= self.image.max_size <= 4096):
            errors.append("Max image size must be between 128 and 4096")
        
        if not (64 <= self.image.min_size <= 2048):
            errors.append("Min image size must be between 64 and 2048")
        
        if self.image.min_size >= self.image.max_size:
            errors.append("Min image size must be less than max image size")
        
        if not (0.1 <= self.image.contrast <= 3.0):
            errors.append("Image contrast must be between 0.1 and 3.0")
        
        if not (0.1 <= self.image.brightness <= 3.0):
            errors.append("Image brightness must be between 0.1 and 3.0")
        
        if not (1 <= self.image.jpeg_quality <= 100):
            errors.append("JPEG quality must be between 1 and 100")
        
        # Validate visualization config
        if not (6 <= self.visualization.default_font_size <= 72):
            errors.append("Default font size must be between 6 and 72")
        
        if not (1 <= self.visualization.default_line_width <= 20):
            errors.append("Default line width must be between 1 and 20")
        
        return errors
    
    def get_detection_config(self) -> DetectionConfig:
        """Get detection configuration."""
        return self.detection
    
    def get_image_config(self) -> ImageConfig:
        """Get image processing configuration."""
        return self.image
    
    def get_visualization_config(self) -> VisualizationConfig:
        """Get visualization configuration."""
        return self.visualization
    
    def get_api_key(self) -> str:
        """Get API key."""
        return self.api_key
