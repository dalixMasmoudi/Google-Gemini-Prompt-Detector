#!/usr/bin/env python3
"""
Image Processing Module for GuardiaVision Gemini

This module handles all image preprocessing, enhancement, and utility functions
required for optimal object detection performance. It provides methods for
image loading, resizing, enhancement, and format conversion.

Classes:
    ImageProcessor: Main class for image processing operations
"""

import os
from typing import Tuple, Optional, Union
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


class ImageProcessor:
    """
    Advanced image processor for computer vision tasks.
    
    This class provides comprehensive image preprocessing capabilities optimized
    for object detection tasks. It handles various image formats, sizes, and
    quality enhancement operations.
    
    Features:
        - Smart resizing with aspect ratio preservation
        - Quality enhancement (contrast, sharpness, brightness)
        - Format conversion and validation
        - Batch processing support
        
    Example:
        >>> processor = ImageProcessor()
        >>> image, original_size = processor.load_and_preprocess("photo.jpg")
        >>> enhanced = processor.enhance_image(image, contrast=1.2)
    """
    
    def __init__(self, 
                 default_max_size: int = 1024,
                 default_min_size: int = 512,
                 default_quality: int = 95):
        """
        Initialize the image processor.
        
        Args:
            default_max_size (int): Default maximum dimension for resizing (pixels)
                                  Higher values preserve more detail but use more memory
                                  Recommended: 1024 for balance, 2048 for high detail
            default_min_size (int): Default minimum dimension for upscaling (pixels)
                                  Ensures images aren't too small for detection
                                  Recommended: 512 for most cases
            default_quality (int): Default JPEG quality for saving (1-100)
                                 Higher values = better quality, larger files
                                 Recommended: 95 for production, 85 for storage
        """
        self.default_max_size = max(1024, default_max_size)  # Minimum 256px
        self.default_min_size = max(1024, default_min_size)  # Minimum 128px
        self.default_quality = max(1, min(default_quality, 100))  # Clamp 1-100
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    def load_image(self, image_path: str) -> Tuple[Image.Image, Tuple[int, int]]:
        """
        Load an image from file path with validation.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Tuple[Image.Image, Tuple[int, int]]: (loaded_image, original_size)
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If file format is not supported
            IOError: If image cannot be loaded
            
        Example:
            >>> image, size = processor.load_image("photo.jpg")
            >>> print(f"Loaded {size[0]}x{size[1]} image")
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check file extension
        _, ext = os.path.splitext(image_path.lower())
        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {ext}. Supported: {self.supported_formats}")
        
        try:
            image = Image.open(image_path)
            original_size = image.size
            
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                if image.mode == 'RGBA':
                    # Create white background for transparent images
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                    image = background
                else:
                    image = image.convert('RGB')
            
            return image, original_size
            
        except Exception as e:
            raise IOError(f"Failed to load image {image_path}: {str(e)}")
    
    def smart_resize(self, 
                    image: Image.Image,
                    max_size: Optional[int] = None,
                    min_size: Optional[int] = None,
                    maintain_aspect: bool = True) -> Image.Image:
        """
        Intelligently resize image for optimal detection performance.
        
        This method applies smart resizing logic that:
        - Preserves aspect ratio by default
        - Downscales large images to reduce processing time
        - Upscales small images to improve detection accuracy
        - Uses high-quality resampling algorithms
        
        Args:
            image (Image.Image): Input PIL Image
            max_size (int, optional): Maximum dimension in pixels
                                    None uses default_max_size
                                    Higher values = more detail, slower processing
            min_size (int, optional): Minimum dimension in pixels
                                    None uses default_min_size
                                    Higher values = better small object detection
            maintain_aspect (bool): Whether to preserve aspect ratio
                                  True = no distortion, may have different final size
                                  False = exact size, may distort image
                                  
        Returns:
            Image.Image: Resized image
            
        Example:
            >>> resized = processor.smart_resize(image, max_size=2048, min_size=1024)
            >>> # Image will be between 1024-2048 pixels in largest dimension
        """
        if max_size is None:
            max_size = self.default_max_size
        if min_size is None:
            min_size = self.default_min_size
            
        current_width, current_height = image.size
        max_current = max(current_width, current_height)
        min_current = min(current_width, current_height)
        
        # Determine if resizing is needed
        if max_current > max_size:
            # Downscale large image
            if maintain_aspect:
                ratio = max_size / max_current
                new_size = (int(current_width * ratio), int(current_height * ratio))
            else:
                new_size = (max_size, max_size)
                
            return image.resize(new_size, Image.Resampling.LANCZOS)
            
        elif max_current < min_size:
            # Upscale small image
            if maintain_aspect:
                ratio = min_size / max_current
                new_size = (int(current_width * ratio), int(current_height * ratio))
            else:
                new_size = (min_size, min_size)
                
            return image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Image size is already optimal
        return image

    def enhance_image(self,
                     image: Image.Image,
                     contrast: float = 1.0,
                     brightness: float = 1.0,
                     sharpness: float = 1.0,
                     color: float = 1.0) -> Image.Image:
        """
        Enhance image quality for better detection performance.

        Args:
            image (Image.Image): Input PIL Image
            contrast (float): Contrast enhancement factor (1.0 = no change)
            brightness (float): Brightness adjustment factor (1.0 = no change)
            sharpness (float): Sharpness enhancement factor (1.0 = no change)
            color (float): Color saturation factor (1.0 = no change)

        Returns:
            Image.Image: Enhanced image
        """
        from PIL import ImageEnhance

        enhanced = image.copy()

        # Apply enhancements in optimal order
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(brightness)

        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(contrast)

        if color != 1.0:
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(color)

        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(sharpness)

        return enhanced

    def preprocess_for_detection(self,
                                image_path: str,
                                enhance_quality: bool = True,
                                preserve_original_size: bool = False,
                                force_resize_to: Optional[Tuple[int, int]] = (1024, 1024),
                                max_size: Optional[int] = None,
                                min_size: Optional[int] = None,
                                enhancement_params: Optional[dict] = None) -> Tuple[Image.Image, Tuple[int, int]]:
        """
        Complete preprocessing pipeline for object detection.

        This method combines loading, resizing, and enhancement into a single
        convenient function optimized for detection tasks.

        Args:
            image_path (str): Path to the image file
            enhance_quality (bool): Whether to apply quality enhancements
            preserve_original_size (bool): Whether to keep original image dimensions
            force_resize_to (Tuple[int, int], optional): Force resize to specific dimensions
            max_size (int, optional): Maximum image dimension
            min_size (int, optional): Minimum image dimension
            enhancement_params (dict, optional): Custom enhancement parameters

        Returns:
            Tuple[Image.Image, Tuple[int, int]]: (processed_image, original_size)
        """
        # Load image
        image, original_size = self.load_image(image_path)

        # Apply resizing logic
        if preserve_original_size:
            print(f"📏 Preserving original image size: {original_size}")
        elif force_resize_to:
            # Force resize to specific dimensions
            target_width, target_height = force_resize_to
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            print(f"📏 Image force-resized from {original_size} to {image.size}")
        else:
            # Apply smart resizing
            image = self.smart_resize(image, max_size=max_size, min_size=min_size)
            print(f"📏 Image smart-resized from {original_size} to {image.size}")

        # Apply enhancements if requested
        if enhance_quality:
            if enhancement_params is None:
                # Default enhancement parameters optimized for detection
                enhancement_params = {
                    "contrast": 1.2,
                    "brightness": 1.0,
                    "sharpness": 1.1,
                    "color": 1.0
                }

            image = self.enhance_image(image, **enhancement_params)
            print(f"✨ Image enhanced with parameters: {enhancement_params}")

        return image, original_size

    def load_original_image(self, image_path: str) -> Tuple[Image.Image, Tuple[int, int]]:
        """
        Load image at its original resolution without any processing.

        This method is used in the new workflow to maintain the original image
        for final visualization while processing is done on a resized version.

        Args:
            image_path (str): Path to the image file

        Returns:
            Tuple[Image.Image, Tuple[int, int]]: (original_image, original_size)

        Example:
            >>> original_img, size = processor.load_original_image("photo.jpg")
            >>> # Image is loaded at its native resolution
        """
        return self.load_image(image_path)

    def save_image(self,
                  image: Image.Image,
                  output_path: str,
                  quality: Optional[int] = None,
                  optimize: bool = True) -> None:
        """
        Save processed image to file.

        Args:
            image (Image.Image): PIL Image to save
            output_path (str): Output file path
            quality (int, optional): JPEG quality (1-100), uses default if None
            optimize (bool): Whether to optimize file size
                           True = smaller files, slightly slower saving
                           False = faster saving, larger files

        Raises:
            IOError: If unable to save image

        Example:
            >>> processor.save_image(processed_image, "output.jpg", quality=90)
        """
        if quality is None:
            quality = self.default_quality

        try:
            # Determine format from extension
            _, ext = os.path.splitext(output_path.lower())

            if ext in {'.jpg', '.jpeg'}:
                image.save(output_path, 'JPEG', quality=quality, optimize=optimize)
            elif ext == '.png':
                image.save(output_path, 'PNG', optimize=optimize)
            else:
                # Default to JPEG for other formats
                image.save(output_path, 'JPEG', quality=quality, optimize=optimize)

        except Exception as e:
            raise IOError(f"Failed to save image to {output_path}: {str(e)}")

    def get_image_info(self, image: Image.Image) -> dict:
        """
        Get comprehensive information about an image.

        Args:
            image (Image.Image): PIL Image to analyze

        Returns:
            dict: Image information including size, mode, format, etc.

        Example:
            >>> info = processor.get_image_info(image)
            >>> print(f"Size: {info['size']}, Mode: {info['mode']}")
        """
        return {
            "size": image.size,
            "width": image.size[0],
            "height": image.size[1],
            "mode": image.mode,
            "format": getattr(image, 'format', 'Unknown'),
            "has_transparency": image.mode in ('RGBA', 'LA') or 'transparency' in image.info,
            "aspect_ratio": image.size[0] / image.size[1] if image.size[1] > 0 else 0
        }
