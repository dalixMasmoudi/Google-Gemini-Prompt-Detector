"""
GuardiaVision Gemini - Advanced Object Detection and Segmentation

A modular, production-ready computer vision system using Google's Gemini 2.5 
for accurate object detection and segmentation based on natural language prompts.

Author: GuardiaVision Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GuardiaVision Team"

from .detector import GeminiDetector
from .image_processor import ImageProcessor
from .visualizer import Visualizer
from .config import Config

__all__ = [
    "GeminiDetector",
    "ImageProcessor", 
    "Visualizer",
    "Config"
]
