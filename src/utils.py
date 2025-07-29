#!/usr/bin/env python3
"""
Utilities Module for GuardiaVision Gemini

This module contains utility functions and helper classes used throughout
the GuardiaVision detection system.

Functions:
    - File and path utilities
    - Validation helpers
    - Performance monitoring
    - Logging setup
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import hashlib
from functools import wraps


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str, optional): Path to log file. If None, logs to console only.
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        >>> logger = setup_logging("DEBUG", "detection.log")
        >>> logger.info("Detection started")
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    logger = logging.getLogger('guardiavision')
    logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_image_path(image_path: str) -> bool:
    """
    Validate that an image path exists and has a supported format.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        bool: True if valid, False otherwise
        
    Example:
        >>> if validate_image_path("photo.jpg"):
        ...     print("Valid image")
    """
    if not os.path.exists(image_path):
        return False
    
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    _, ext = os.path.splitext(image_path.lower())
    
    return ext in supported_extensions


def ensure_output_directory(output_path: str) -> str:
    """
    Ensure output directory exists and return the full path.
    
    Args:
        output_path (str): Output file path
        
    Returns:
        str: Absolute path with directory created
        
    Example:
        >>> path = ensure_output_directory("results/detection.jpg")
        >>> # Creates 'results' directory if it doesn't exist
    """
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return output_path


def generate_output_filename(input_path: str, 
                           suffix: str = "detected",
                           output_dir: Optional[str] = None) -> str:
    """
    Generate an output filename based on input path.
    
    Args:
        input_path (str): Input file path
        suffix (str): Suffix to add to filename
        output_dir (str, optional): Output directory. If None, uses input directory.
        
    Returns:
        str: Generated output path
        
    Example:
        >>> output = generate_output_filename("photo.jpg", "detected")
        >>> # Returns "photo_detected.jpg"
    """
    input_path = Path(input_path)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_path.parent
    
    # Generate new filename
    stem = input_path.stem
    extension = input_path.suffix
    
    output_filename = f"{stem}_{suffix}{extension}"
    return str(output_dir / output_filename)


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate MD5 hash of a file for caching/comparison purposes.
    
    Args:
        file_path (str): Path to file
        
    Returns:
        str: MD5 hash string
        
    Example:
        >>> hash_val = calculate_file_hash("image.jpg")
        >>> print(f"File hash: {hash_val}")
    """
    hash_md5 = hashlib.md5()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()


def timing_decorator(func):
    """
    Decorator to measure function execution time.
    
    Example:
        >>> @timing_decorator
        ... def detect_objects():
        ...     # detection code
        ...     pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"⏱️  {func.__name__} completed in {execution_time:.2f} seconds")
        
        return result
    return wrapper


class PerformanceMonitor:
    """
    Performance monitoring utility for tracking detection metrics.
    
    This class helps track various performance metrics during detection
    operations including timing, memory usage, and success rates.
    
    Example:
        >>> monitor = PerformanceMonitor()
        >>> monitor.start_operation("detection")
        >>> # ... detection code ...
        >>> monitor.end_operation("detection")
        >>> print(monitor.get_summary())
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.operations = {}
        self.start_times = {}
        self.total_detections = 0
        self.successful_detections = 0
    
    def start_operation(self, operation_name: str):
        """
        Start timing an operation.
        
        Args:
            operation_name (str): Name of the operation to track
        """
        self.start_times[operation_name] = time.time()
    
    def end_operation(self, operation_name: str, success: bool = True):
        """
        End timing an operation and record results.
        
        Args:
            operation_name (str): Name of the operation
            success (bool): Whether the operation was successful
        """
        if operation_name not in self.start_times:
            return
        
        duration = time.time() - self.start_times[operation_name]
        
        if operation_name not in self.operations:
            self.operations[operation_name] = {
                'count': 0,
                'total_time': 0.0,
                'successes': 0,
                'failures': 0,
                'min_time': float('inf'),
                'max_time': 0.0
            }
        
        op_stats = self.operations[operation_name]
        op_stats['count'] += 1
        op_stats['total_time'] += duration
        op_stats['min_time'] = min(op_stats['min_time'], duration)
        op_stats['max_time'] = max(op_stats['max_time'], duration)
        
        if success:
            op_stats['successes'] += 1
        else:
            op_stats['failures'] += 1
        
        del self.start_times[operation_name]
    
    def record_detection_result(self, num_detections: int):
        """
        Record detection results.
        
        Args:
            num_detections (int): Number of objects detected
        """
        self.total_detections += num_detections
        if num_detections > 0:
            self.successful_detections += 1
    
    def get_summary(self) -> str:
        """
        Get performance summary report.
        
        Returns:
            str: Formatted performance report
        """
        if not self.operations:
            return "No operations recorded."
        
        summary_lines = []
        summary_lines.append("PERFORMANCE SUMMARY")
        summary_lines.append("=" * 40)
        
        for op_name, stats in self.operations.items():
            avg_time = stats['total_time'] / stats['count']
            success_rate = (stats['successes'] / stats['count']) * 100
            
            summary_lines.append(f"\n{op_name.upper()}:")
            summary_lines.append(f"  Operations: {stats['count']}")
            summary_lines.append(f"  Success Rate: {success_rate:.1f}%")
            summary_lines.append(f"  Average Time: {avg_time:.2f}s")
            summary_lines.append(f"  Min Time: {stats['min_time']:.2f}s")
            summary_lines.append(f"  Max Time: {stats['max_time']:.2f}s")
            summary_lines.append(f"  Total Time: {stats['total_time']:.2f}s")
        
        if self.total_detections > 0:
            summary_lines.append(f"\nDETECTION STATS:")
            summary_lines.append(f"  Total Objects Detected: {self.total_detections}")
            summary_lines.append(f"  Successful Detection Runs: {self.successful_detections}")
        
        return "\n".join(summary_lines)
    
    def reset(self):
        """Reset all performance metrics."""
        self.operations.clear()
        self.start_times.clear()
        self.total_detections = 0
        self.successful_detections = 0


def save_detection_results(detections: List[Dict], 
                         output_path: str,
                         metadata: Optional[Dict] = None):
    """
    Save detection results to JSON file.
    
    Args:
        detections (List[Dict]): Detection results
        output_path (str): Output JSON file path
        metadata (Dict, optional): Additional metadata to include
        
    Example:
        >>> save_detection_results(detections, "results.json", 
        ...                        {"image": "photo.jpg", "timestamp": "2024-01-01"})
    """
    output_data = {
        "detections": detections,
        "count": len(detections),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    if metadata:
        output_data["metadata"] = metadata
    
    ensure_output_directory(output_path)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def load_detection_results(input_path: str) -> Tuple[List[Dict], Optional[Dict]]:
    """
    Load detection results from JSON file.

    Args:
        input_path (str): Input JSON file path

    Returns:
        Tuple[List[Dict], Optional[Dict]]: (detections, metadata)

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Results file not found: {input_path}")

    try:
        with open(input_path, 'r') as f:
            data = json.load(f)

        detections = data.get("detections", [])
        metadata = data.get("metadata")

        return detections, metadata

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in results file: {str(e)}")


def transform_coordinates_to_original(detections: List[Dict],
                                    original_size: Tuple[int, int],
                                    processed_size: Tuple[int, int] = (1024, 1024)) -> List[Dict]:
    """
    Transform detection coordinates from processed image size back to original image size.

    This function handles the coordinate transformation needed when detection is performed
    on a resized image but results need to be displayed on the original image.

    Args:
        detections (List[Dict]): List of detection dictionaries with normalized coordinates (0-1000)
        original_size (Tuple[int, int]): Original image dimensions (width, height)
        processed_size (Tuple[int, int]): Processed image dimensions (width, height)

    Returns:
        List[Dict]: Detections with coordinates transformed to original image size

    Note:
        Input coordinates are expected to be normalized (0-1000 range).
        Output coordinates will be in pixel coordinates for the original image.

    Example:
        >>> detections = [{"label": "car", "box_2d": [100, 200, 300, 400]}]
        >>> original_size = (2048, 1536)  # Original image size
        >>> processed_size = (1024, 1024)  # Size used for detection
        >>> transformed = transform_coordinates_to_original(detections, original_size, processed_size)
        >>> # Coordinates will be scaled to fit the 2048x1536 original image
    """
    if not detections:
        return detections

    original_width, original_height = original_size
    processed_width, processed_height = processed_size

    # Calculate scaling factors
    width_scale = original_width / processed_width
    height_scale = original_height / processed_height

    print(f"🔄 Transforming coordinates from {processed_size} to {original_size}")
    print(f"   Scale factors: width={width_scale:.3f}, height={height_scale:.3f}")

    transformed_detections = []

    for detection in detections:
        if 'box_2d' not in detection:
            transformed_detections.append(detection)
            continue

        # Get normalized coordinates (0-1000 range)
        y0, x0, y1, x1 = detection['box_2d']

        # Convert to pixel coordinates in processed image (0-1024 range)
        x0_processed = (x0 / 1000.0) * processed_width
        x1_processed = (x1 / 1000.0) * processed_width
        y0_processed = (y0 / 1000.0) * processed_height
        y1_processed = (y1 / 1000.0) * processed_height

        # Transform to original image coordinates
        x0_original = x0_processed * width_scale
        x1_original = x1_processed * width_scale
        y0_original = y0_processed * height_scale
        y1_original = y1_processed * height_scale

        # Ensure coordinates are within bounds
        x0_original = max(0, min(x0_original, original_width))
        x1_original = max(0, min(x1_original, original_width))
        y0_original = max(0, min(y0_original, original_height))
        y1_original = max(0, min(y1_original, original_height))

        # Create transformed detection
        transformed_detection = detection.copy()
        transformed_detection['box_2d'] = [
            int(y0_original), int(x0_original),
            int(y1_original), int(x1_original)
        ]

        # Transform segmentation mask if present
        if 'mask' in detection and detection['mask']:
            try:
                transformed_mask = []
                for point in detection['mask']:
                    if isinstance(point, list) and len(point) == 2:
                        # Convert from normalized coordinates to processed image pixels
                        mask_x_processed = (point[0] / 1000.0) * processed_width
                        mask_y_processed = (point[1] / 1000.0) * processed_height

                        # Transform to original image coordinates
                        mask_x_original = mask_x_processed * width_scale
                        mask_y_original = mask_y_processed * height_scale

                        # Ensure coordinates are within bounds
                        mask_x_original = max(0, min(mask_x_original, original_width))
                        mask_y_original = max(0, min(mask_y_original, original_height))

                        transformed_mask.append([int(mask_x_original), int(mask_y_original)])

                if len(transformed_mask) >= 3:
                    transformed_detection['mask'] = transformed_mask
                    print(f"✅ Transformed segmentation mask with {len(transformed_mask)} points")
                else:
                    print(f"Warning: Insufficient mask points after transformation")

            except Exception as e:
                print(f"Warning: Error transforming mask: {e}")

        # Add transformation metadata
        transformed_detection['coordinate_transform'] = {
            'original_coords': [y0, x0, y1, x1],
            'transformed_coords': [int(y0_original), int(x0_original), int(y1_original), int(x1_original)],
            'scale_factors': {'width': width_scale, 'height': height_scale}
        }

        transformed_detections.append(transformed_detection)

    print(f"✅ Transformed {len(transformed_detections)} detection coordinates")
    return transformed_detections
