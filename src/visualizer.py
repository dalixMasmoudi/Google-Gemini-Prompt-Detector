#!/usr/bin/env python3
"""
Visualization Module for GuardiaVision Gemini

This module handles all visualization tasks including drawing bounding boxes,
creating detection overlays, generating reports, and saving results.

Classes:
    Visualizer: Main class for visualization operations
"""

from typing import List, Dict, Tuple, Optional, Union
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os


class Visualizer:
    """
    Advanced visualization system for object detection results.
    
    This class provides comprehensive visualization capabilities for detection
    results including bounding box drawing, confidence display, label formatting,
    and report generation.
    
    Features:
        - Customizable bounding box styles
        - Automatic color assignment
        - Font size optimization
        - Confidence score display
        - Detection reports and statistics
        
    Example:
        >>> visualizer = Visualizer()
        >>> result_image = visualizer.draw_detections(image, detections)
        >>> visualizer.save_result(result_image, "output.jpg")
    """
    
    def __init__(self,
                 default_colors: Optional[List[str]] = None,
                 default_font_size: int = 12,
                 default_line_width: int = 3):
        """
        Initialize the visualizer with default styling options.
        
        Args:
            default_colors (List[str], optional): List of hex color codes for bounding boxes
                                                Default provides 10 distinct colors
            default_font_size (int): Base font size for labels
                                   Actual size is auto-adjusted based on image size
            default_line_width (int): Base line width for bounding boxes
                                    Actual width is auto-adjusted based on image size
        """
        self.default_colors = default_colors or [
            '#FF0000',  # Red
            '#00FF00',  # Green  
            '#0000FF',  # Blue
            '#FFFF00',  # Yellow
            '#FF00FF',  # Magenta
            '#00FFFF',  # Cyan
            '#FFA500',  # Orange
            '#800080',  # Purple
            '#FFC0CB',  # Pink
            '#A52A2A'   # Brown
        ]
        
        self.default_font_size = max(8, default_font_size)
        self.default_line_width = max(1, default_line_width)
        
        # Font loading with fallbacks
        self.font_paths = [
            "arial.ttf",
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:/Windows/Fonts/arial.ttf"  # Windows
        ]
    
    def _load_font(self, size: int) -> ImageFont.ImageFont:
        """
        Load the best available font for the system.
        
        Args:
            size (int): Font size in pixels
            
        Returns:
            ImageFont.ImageFont: Loaded font object
        """
        for font_path in self.font_paths:
            try:
                return ImageFont.truetype(font_path, size)
            except (OSError, IOError):
                continue
        
        # Fallback to default font
        return ImageFont.load_default()
    
    def _calculate_optimal_sizes(self, image: Image.Image) -> Tuple[int, int]:
        """
        Calculate optimal font size and line width based on image dimensions.
        
        Args:
            image (Image.Image): Input image
            
        Returns:
            Tuple[int, int]: (font_size, line_width)
        """
        min_dimension = min(image.size)
        
        # Scale font size based on image size
        font_size = max(self.default_font_size, min_dimension // 50)
        font_size = min(font_size, 48)  # Cap at reasonable maximum
        
        # Scale line width based on image size
        line_width = max(self.default_line_width, min_dimension // 200)
        line_width = min(line_width, 10)  # Cap at reasonable maximum
        
        return font_size, line_width
    
    def draw_detections(self,
                       image: Image.Image,
                       detections: List[Dict],
                       show_confidence: bool = True,
                       show_labels: bool = True,
                       colors: Optional[List[str]] = None,
                       font_size: Optional[int] = None,
                       line_width: Optional[int] = None,
                       coordinates_are_pixels: bool = False) -> Image.Image:
        """
        Draw bounding boxes and labels on image for all detections.

        This method creates a visual representation of detection results by
        drawing colored bounding boxes with optional labels and confidence scores.

        Args:
            image (Image.Image): Input PIL Image
            detections (List[Dict]): List of detection dictionaries with keys:
                                   - 'box_2d': [y0, x0, y1, x1] coordinates
                                   - 'label': Object description string
                                   - 'confidence': Detection confidence (0.0-1.0)
            show_confidence (bool): Whether to display confidence scores in labels
            show_labels (bool): Whether to display text labels
            colors (List[str], optional): Custom color list (hex codes)
            font_size (int, optional): Custom font size
            line_width (int, optional): Custom line width
            coordinates_are_pixels (bool): Whether coordinates are already in pixel format
                                         True = coordinates are pixel values
                                         False = coordinates are normalized (0-1000)

        Returns:
            Image.Image: New image with detection overlays

        Example:
            >>> # With normalized coordinates (0-1000)
            >>> result = visualizer.draw_detections(image, detections, coordinates_are_pixels=False)

            >>> # With pixel coordinates (after transformation)
            >>> result = visualizer.draw_detections(image, detections, coordinates_are_pixels=True)
        """
        if not detections:
            print("⚠️  No detections to draw")
            return image.copy()
        
        # Use provided colors or defaults
        colors = colors or self.default_colors
        
        # Calculate optimal sizes
        if font_size is None or line_width is None:
            calc_font_size, calc_line_width = self._calculate_optimal_sizes(image)
            font_size = font_size or calc_font_size
            line_width = line_width or calc_line_width
        
        # Create a copy and drawing context
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)
        
        # Load font
        font = self._load_font(font_size)
        
        print(f"🎨 Drawing {len(detections)} detections")
        
        for i, detection in enumerate(detections):
            # Get bounding box coordinates
            y0, x0, y1, x1 = detection['box_2d']

            # Convert coordinates based on format
            if coordinates_are_pixels:
                # Coordinates are already in pixel format
                x0_px, y0_px, x1_px, y1_px = int(x0), int(y0), int(x1), int(y1)
            else:
                # Convert from normalized coordinates (0-1000) to pixel coordinates
                img_width, img_height = image.size
                x0_px = int((x0 / 1000.0) * img_width)
                x1_px = int((x1 / 1000.0) * img_width)
                y0_px = int((y0 / 1000.0) * img_height)
                y1_px = int((y1 / 1000.0) * img_height)
            
            # Select color (cycle through available colors)
            color = colors[i % len(colors)]
            
            # Draw segmentation mask if present, otherwise draw bounding box
            if 'mask' in detection and detection['mask']:
                try:
                    # Convert mask coordinates to pixels
                    mask_points = []
                    for point in detection['mask']:
                        if coordinates_are_pixels:
                            mask_x, mask_y = int(point[0]), int(point[1])
                        else:
                            mask_x = int((point[0] / 1000.0) * image.size[0])
                            mask_y = int((point[1] / 1000.0) * image.size[1])
                        mask_points.append((mask_x, mask_y))

                    # Draw segmentation mask as polygon
                    if len(mask_points) >= 3:
                        draw.polygon(mask_points, outline=color, width=line_width)
                        print(f"🎨 Drew segmentation mask with {len(mask_points)} points")
                    else:
                        # Fallback to bounding box if mask is invalid
                        draw.rectangle([x0_px, y0_px, x1_px, y1_px], outline=color, width=line_width)
                except Exception as e:
                    print(f"Warning: Error drawing mask for '{detection.get('label', 'Unknown')}': {e}")
                    # Fallback to bounding box
                    draw.rectangle([x0_px, y0_px, x1_px, y1_px], outline=color, width=line_width)
            else:
                # Draw bounding box rectangle
                draw.rectangle(
                    [x0_px, y0_px, x1_px, y1_px],
                    outline=color,
                    width=line_width
                )
            
            # Draw label if requested
            if show_labels:
                # Prepare label text
                label = detection['label']
                if show_confidence and 'confidence' in detection:
                    confidence = detection['confidence']
                    label += f" ({confidence:.2f})"
                
                # Calculate label position (above the box)
                label_y = max(0, y0_px - font_size - 5)
                
                # Draw label background for better readability
                bbox = draw.textbbox((x0_px, label_y), label, font=font)
                draw.rectangle(
                    [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], 
                    fill=color
                )
                
                # Draw label text
                draw.text((x0_px, label_y), label, fill='white', font=font)
        
        return result_image

    def generate_detection_report(self,
                                detections: List[Dict],
                                target_objects: str,
                                image_info: Optional[Dict] = None) -> str:
        """
        Generate a comprehensive text report of detection results.

        This method creates a detailed summary of detection results including
        statistics, object counts, confidence scores, and other relevant metrics.

        Args:
            detections (List[Dict]): List of detection dictionaries
            target_objects (str): Original target objects string
            image_info (Dict, optional): Image information dictionary
                                       (from ImageProcessor.get_image_info)

        Returns:
            str: Formatted detection report

        Example:
            >>> report = visualizer.generate_detection_report(
            ...     detections, "cars, trucks", image_info
            ... )
            >>> print(report)
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("GUARDIAVISION DETECTION REPORT")
        report_lines.append("=" * 60)

        # Basic information
        report_lines.append(f"Target Objects: {target_objects}")
        report_lines.append(f"Total Detections: {len(detections)}")

        if image_info:
            report_lines.append(f"Image Size: {image_info['width']}x{image_info['height']}")
            report_lines.append(f"Image Format: {image_info.get('format', 'Unknown')}")

        report_lines.append("")

        if detections:
            # Individual detections
            report_lines.append("DETECTED OBJECTS:")
            report_lines.append("-" * 40)

            for i, det in enumerate(detections, 1):
                confidence = det.get('confidence', 'N/A')
                if isinstance(confidence, float):
                    confidence = f"{confidence:.3f}"

                # Calculate box size
                y0, x0, y1, x1 = det['box_2d']
                box_width = x1 - x0
                box_height = y1 - y0

                report_lines.append(
                    f"{i:2d}. {det['label']:<25} "
                    f"Confidence: {confidence:<6} "
                    f"Size: {box_width}x{box_height}"
                )

            report_lines.append("")

            # Object type statistics
            object_counts = {}
            confidence_stats = {}

            for det in detections:
                # Extract object type (first word of label)
                obj_type = det['label'].split()[0].lower()
                object_counts[obj_type] = object_counts.get(obj_type, 0) + 1

                # Collect confidence scores
                if 'confidence' in det and isinstance(det['confidence'], (int, float)):
                    if obj_type not in confidence_stats:
                        confidence_stats[obj_type] = []
                    confidence_stats[obj_type].append(det['confidence'])

            # Object counts
            report_lines.append("OBJECT COUNTS:")
            report_lines.append("-" * 20)
            for obj_type, count in sorted(object_counts.items()):
                report_lines.append(f"  {obj_type.capitalize()}: {count}")

            report_lines.append("")

            # Confidence statistics
            if confidence_stats:
                report_lines.append("CONFIDENCE STATISTICS:")
                report_lines.append("-" * 30)

                for obj_type, confidences in confidence_stats.items():
                    avg_conf = sum(confidences) / len(confidences)
                    min_conf = min(confidences)
                    max_conf = max(confidences)

                    report_lines.append(
                        f"  {obj_type.capitalize()}: "
                        f"Avg={avg_conf:.3f}, Min={min_conf:.3f}, Max={max_conf:.3f}"
                    )
        else:
            report_lines.append("No objects detected matching the specified criteria.")

        report_lines.append("")
        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    def save_result(self,
                   image: Image.Image,
                   output_path: str,
                   quality: int = 95) -> None:
        """
        Save visualization result to file.

        Args:
            image (Image.Image): Result image with visualizations
            output_path (str): Output file path
            quality (int): JPEG quality (1-100)

        Raises:
            IOError: If unable to save image
        """
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create directory if there is one
                os.makedirs(output_dir, exist_ok=True)

            # Save with appropriate format
            _, ext = os.path.splitext(output_path.lower())
            if ext in {'.jpg', '.jpeg'}:
                image.save(output_path, 'JPEG', quality=quality, optimize=True)
            elif ext == '.png':
                image.save(output_path, 'PNG', optimize=True)
            else:
                # Default to JPEG
                image.save(output_path, 'JPEG', quality=quality, optimize=True)

            print(f"✅ Result saved to: {output_path}")

        except Exception as e:
            raise IOError(f"Failed to save result to {output_path}: {str(e)}")

    def create_matplotlib_visualization(self,
                                      image: Image.Image,
                                      detections: List[Dict],
                                      title: str = "Detection Results",
                                      figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create a matplotlib figure with detection visualization.

        This method provides an alternative visualization using matplotlib,
        which can be useful for jupyter notebooks or scientific reports.

        Args:
            image (Image.Image): Input image
            detections (List[Dict]): Detection results
            title (str): Figure title
            figsize (Tuple[int, int]): Figure size (width, height) in inches

        Returns:
            plt.Figure: Matplotlib figure object

        Example:
            >>> fig = visualizer.create_matplotlib_visualization(image, detections)
            >>> fig.savefig("detection_plot.png", dpi=300, bbox_inches='tight')
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Display image
        ax.imshow(np.array(image))
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')

        # Draw bounding boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.default_colors)))

        for i, detection in enumerate(detections):
            y0, x0, y1, x1 = detection['box_2d']

            # Convert to pixel coordinates
            img_width, img_height = image.size
            x0_px = (x0 / 1000.0) * img_width
            x1_px = (x1 / 1000.0) * img_width
            y0_px = (y0 / 1000.0) * img_height
            y1_px = (y1 / 1000.0) * img_height

            # Create rectangle patch
            width = x1_px - x0_px
            height = y1_px - y0_px

            rect = patches.Rectangle(
                (x0_px, y0_px), width, height,
                linewidth=2, edgecolor=colors[i % len(colors)],
                facecolor='none'
            )
            ax.add_patch(rect)

            # Add label
            label = detection['label']
            if 'confidence' in detection:
                label += f" ({detection['confidence']:.2f})"

            ax.text(
                x0_px, y0_px - 5, label,
                fontsize=10, color=colors[i % len(colors)],
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
            )

        plt.tight_layout()
        return fig
