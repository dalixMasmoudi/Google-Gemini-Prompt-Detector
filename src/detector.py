#!/usr/bin/env python3
"""
Core Detection Module for GuardiaVision Gemini

This module contains the main GeminiDetector class responsible for object detection
and segmentation using Google's Gemini 2.5 model. It provides high-level interfaces
for detecting objects based on natural language prompts.

Classes:
    GeminiDetector: Main detector class for object detection and segmentation
"""

import json
import re
from typing import List, Dict, Tuple, Optional, Union
from google import genai
from google.genai import types
from PIL import Image


class GeminiDetector:
    """
    Advanced object detector using Google's Gemini 2.5 model.
    
    This class provides sophisticated object detection and segmentation capabilities
    using natural language prompts. It can detect multiple object types with high
    accuracy and return bounding boxes or segmentation masks.
    
    Attributes:
        client: Google GenAI client instance
        model_name: Name of the Gemini model to use
        safety_settings: Safety configuration for the model
        
    Example:
        >>> detector = GeminiDetector(api_key="your_api_key")
        >>> results = detector.detect_objects("image.jpg", "cars, trucks")
        >>> print(f"Found {len(results)} objects")
    """
    
    def __init__(self, 
                 api_key: str,
                 model_name: str = "gemini-2.5-flash-preview-05-20",
                 temperature: float = 0.5,
                 thinking_budget: int = 0):
        """
        Initialize the Gemini detector.
        
        Args:
            api_key (str): Google API key for Gemini access
            model_name (str): Gemini model name to use. Default is gemini-2.5-flash-preview-05-20
            temperature (float): Model temperature for response consistency. 
                               Lower values (0.0-0.3) give more consistent results.
                               Higher values (0.7-1.0) give more creative responses.
            thinking_budget (int): Computational budget for model thinking. 
                                 0 = no thinking, higher values allow more reasoning time.
                                 
        Raises:
            ValueError: If api_key is empty or invalid
            ConnectionError: If unable to connect to Gemini API
        """
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be empty")
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        
        # Enhanced system instructions for better accuracy
        self.bounding_box_system_instructions = """
        Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
        If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
        Format: [{"label": "object_name", "box_2d": [y0, x0, y1, x1], "confidence": 0.95}]
        """

        # Segmentation instructions for detailed object masks
        self.segmentation_system_instructions = """
        Return both bounding boxes AND segmentation masks as a JSON array. Never return code fencing. Limit to 25 objects.
        If an object is present multiple times, name them according to their unique characteristic.
        Format: [{"label": "object_name", "box_2d": [y0, x0, y1, x1], "confidence": 0.95, "mask": [[x1,y1], [x2,y2], ..., [x1,y1]]}]
        """

        # Safety settings for production use
        self.safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ]
    

    def parse_detection_response(self, response_text: str) -> List[Dict]:
        """
        Parse Gemini model response into structured detection results with validation.

        This method handles JSON parsing, confidence score extraction, and bounding box validation.

        Args:
            response_text (str): Raw response from Gemini API

        Returns:
            List[Dict]: Validated detection results with proper confidence scores
        """
        try:
            # Step 1: Check for empty response
            if not response_text or response_text.strip() == "":
                print("Warning: Empty response from Gemini model")
                return []

            # Step 2: Clean the response text
            cleaned = response_text.strip()

            # Remove markdown code blocks if present
            if '```json' in cleaned:
                import re
                match = re.search(r'```json\s*(.*?)\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1).strip()
            elif cleaned.startswith('```') and cleaned.endswith('```'):
                cleaned = cleaned.strip('`').strip()

            # Step 3: Parse JSON
            data_list = json.loads(cleaned)

            if not isinstance(data_list, list):
                print(f"Warning: Expected list, got {type(data_list)}")
                return []

            # Step 4: Validate and enhance each detection
            validated_detections = []

            for i, detection in enumerate(data_list):
                if not isinstance(detection, dict):
                    print(f"Warning: Detection {i} is not a dictionary, skipping")
                    continue

                # Validate required fields
                if 'box_2d' not in detection or 'label' not in detection:
                    print(f"Warning: Detection {i} missing required fields, skipping")
                    continue

                # Validate and fix confidence score
                confidence = detection.get('confidence')
                if confidence is None:
                    # Assign default confidence if missing
                    confidence = 0.8
                    print(f"Warning: Detection '{detection.get('label', 'Unknown')}' missing confidence, assigned default 0.8")
                elif not isinstance(confidence, (int, float)):
                    try:
                        confidence = float(confidence)
                    except (ValueError, TypeError):
                        confidence = 0.8
                        print(f"Warning: Invalid confidence for '{detection.get('label', 'Unknown')}', assigned default 0.8")

                # Ensure confidence is in valid range
                confidence = max(0.0, min(1.0, float(confidence)))

                # Validate bounding box coordinates
                box_2d = detection.get('box_2d', [])
                if not isinstance(box_2d, list) or len(box_2d) != 4:
                    print(f"Warning: Invalid box_2d format for '{detection.get('label', 'Unknown')}', skipping")
                    continue

                try:
                    y0, x0, y1, x1 = [float(coord) for coord in box_2d]

                    # Validate coordinate bounds (0-1000 normalized)
                    if not all(0 <= coord <= 1000 for coord in [y0, x0, y1, x1]):
                        print(f"Warning: Coordinates out of bounds for '{detection.get('label', 'Unknown')}', skipping")
                        continue

                    # Validate box dimensions (x1 > x0, y1 > y0)
                    if x1 <= x0 or y1 <= y0:
                        print(f"Warning: Invalid box dimensions for '{detection.get('label', 'Unknown')}', skipping")
                        continue

                    # Check minimum box size (at least 30x20 in normalized coordinates)
                    width = x1 - x0
                    height = y1 - y0
                    if width < 30 or height < 20:
                        print(f"Warning: Box too small for '{detection.get('label', 'Unknown')}' ({width:.1f}x{height:.1f}), skipping")
                        continue

                    # Create validated detection
                    validated_detection = {
                        'label': str(detection.get('label', 'Unknown')),
                        'box_2d': [int(y0), int(x0), int(y1), int(x1)],
                        'confidence': round(confidence, 3)
                    }

                    # Add segmentation mask if present
                    if 'mask' in detection:
                        mask = detection.get('mask')
                        if isinstance(mask, list) and len(mask) > 2:
                            # Validate mask coordinates
                            try:
                                validated_mask = []
                                for point in mask:
                                    if isinstance(point, list) and len(point) == 2:
                                        x, y = float(point[0]), float(point[1])
                                        if 0 <= x <= 1000 and 0 <= y <= 1000:
                                            validated_mask.append([int(x), int(y)])

                                if len(validated_mask) >= 3:  # Minimum 3 points for a polygon
                                    validated_detection['mask'] = validated_mask
                                    print(f"✅ Validated segmentation mask with {len(validated_mask)} points")
                                else:
                                    print(f"Warning: Insufficient mask points for '{detection.get('label', 'Unknown')}'")
                            except (ValueError, TypeError) as e:
                                print(f"Warning: Invalid mask format for '{detection.get('label', 'Unknown')}': {e}")
                        else:
                            print(f"Warning: Invalid mask data for '{detection.get('label', 'Unknown')}'")


                    validated_detections.append(validated_detection)

                except (ValueError, TypeError) as e:
                    print(f"Warning: Error processing coordinates for '{detection.get('label', 'Unknown')}': {e}")
                    continue

            print(f"✅ Validated {len(validated_detections)}/{len(data_list)} detections")
            return validated_detections

        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON response: {e}")
            print(f"Response text: {response_text[:200]}...")
            return []
        except Exception as e:
            print(f"Error: Unexpected error parsing response: {e}")
            return []

    def validate_detection(self, detection: Dict) -> bool:
        """
        Validate a single detection result.

        Performs comprehensive validation of detection data including
        coordinate bounds, data types, and logical consistency.

        Args:
            detection (Dict): Detection dictionary to validate

        Returns:
            bool: True if detection is valid, False otherwise

        Validation checks:
            - Required fields present (label, box_2d)
            - Bounding box has 4 coordinates
            - Coordinates are numeric and within valid range (0-1000)
            - Box coordinates are logically consistent (x0 < x1, y0 < y1)
            - Confidence score is valid (0.0-1.0) if present
        """
        required_fields = ['label', 'box_2d']

        # Check required fields
        for field in required_fields:
            if field not in detection:
                return False

        # Validate bounding box
        box = detection['box_2d']
        if not isinstance(box, list) or len(box) != 4:
            return False

        # Check coordinate validity (normalized 0-1000)
        try:
            y0, x0, y1, x1 = [float(coord) for coord in box]
        except (ValueError, TypeError):
            return False

        # Check coordinate bounds and logical consistency
        if not (0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000):
            return False

        # Check confidence if present
        if 'confidence' in detection:
            conf = detection['confidence']
            if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
                return False

        return True

    def filter_detections_by_confidence(self,
                                      detections: List[Dict],
                                      min_confidence: float = 0.7) -> List[Dict]:
        """
        Filter detections by minimum confidence threshold.

        Args:
            detections (List[Dict]): List of detection dictionaries
            min_confidence (float): Minimum confidence threshold (0.0-1.0)
                                  Higher values = more strict filtering
                                  Lower values = more permissive filtering

        Returns:
            List[Dict]: Filtered detections meeting confidence threshold

        Note:
            Detections without confidence scores are assigned default value of 0.8
        """
        min_confidence = max(0.0, min(min_confidence, 1.0))  # Clamp to valid range
        return [det for det in detections if det.get('confidence', 0.8) >= min_confidence]

    def filter_detections_by_target_objects(self,
                                           detections: List[Dict],
                                           target_objects: str) -> List[Dict]:
        """
        Filter detections to only include objects that match the target objects.

        This method ensures that only objects specifically requested by the user
        are returned, preventing unwanted detections of other objects in the image.

        Args:
            detections (List[Dict]): List of detection dictionaries
            target_objects (str): Comma-separated list of target objects

        Returns:
            List[Dict]: Filtered detections matching only target objects

        Example:
            >>> target_objects = "license plate, license plates"
            >>> # Will only return detections with labels containing "license" and "plate"
        """
        if not target_objects or not detections:
            return detections

        # Parse target objects and create search terms
        targets = [obj.strip().lower() for obj in target_objects.split(',')]

        # Create expanded search terms for better matching
        search_terms = set()
        for target in targets:
            search_terms.add(target)
            # Add singular/plural variations
            if target.endswith('s'):
                search_terms.add(target[:-1])  # Remove 's'
            else:
                search_terms.add(target + 's')  # Add 's'

            # Add individual words for compound terms
            words = target.split()
            search_terms.update(words)

        filtered_detections = []

        for detection in detections:
            label = detection.get('label', '').lower()

            # Check if any search term matches the detection label
            is_match = False
            for term in search_terms:
                if term in label:
                    is_match = True
                    break

            if is_match:
                filtered_detections.append(detection)
            else:
                print(f"🚫 Filtered out non-matching detection: '{detection.get('label', 'Unknown')}'")

        print(f"🎯 Filtered detections: {len(filtered_detections)}/{len(detections)} match target objects")
        return filtered_detections

    def filter_detections_by_size(self,
                                 detections: List[Dict],
                                 min_width_pixels: int = 30,
                                 min_height_pixels: int = 20,
                                 image_size: Tuple[int, int] = (1024, 1024)) -> List[Dict]:
        """
        Filter out detections with bounding boxes that are too small.

        This method removes false positive detections that have unreasonably small
        bounding boxes, which are likely to be noise or irrelevant objects.

        Args:
            detections (List[Dict]): List of detection dictionaries
            min_width_pixels (int): Minimum width in pixels
            min_height_pixels (int): Minimum height in pixels
            image_size (Tuple[int, int]): Image dimensions (width, height)

        Returns:
            List[Dict]: Filtered detections with reasonable sizes

        Example:
            >>> # Filter out boxes smaller than 30x20 pixels
            >>> filtered = detector.filter_detections_by_size(detections, 30, 20)
        """
        if not detections:
            return detections

        img_width, img_height = image_size
        filtered_detections = []

        for detection in detections:
            if 'box_2d' not in detection:
                filtered_detections.append(detection)
                continue

            # Get normalized coordinates (0-1000)
            y0, x0, y1, x1 = detection['box_2d']

            # Convert to pixel dimensions
            width_pixels = ((x1 - x0) / 1000.0) * img_width
            height_pixels = ((y1 - y0) / 1000.0) * img_height

            # Check if box meets minimum size requirements
            if width_pixels >= min_width_pixels and height_pixels >= min_height_pixels:
                filtered_detections.append(detection)
            else:
                label = detection.get('label', 'Unknown')
                print(f"🚫 Filtered out small detection: '{label}' ({width_pixels:.1f}x{height_pixels:.1f} pixels)")

        print(f"📏 Size filtering: {len(filtered_detections)}/{len(detections)} detections meet size requirements")
        return filtered_detections

    def remove_duplicate_detections(self,
                                  detections: List[Dict],
                                  iou_threshold: float = 0.5) -> List[Dict]:
        """
        Remove duplicate detections using Intersection over Union (IoU).

        This method eliminates overlapping detections of the same object by
        calculating IoU between bounding boxes and keeping only the highest
        confidence detection for each group of overlapping boxes.

        Args:
            detections (List[Dict]): List of detection dictionaries
            iou_threshold (float): IoU threshold for considering detections as duplicates
                                 0.0 = no filtering (all detections kept)
                                 0.5 = moderate filtering (50% overlap threshold)
                                 0.8 = strict filtering (80% overlap threshold)
                                 1.0 = only exact matches removed

        Returns:
            List[Dict]: Filtered detections with duplicates removed

        Algorithm:
            1. Sort detections by confidence (highest first)
            2. For each detection, check IoU with all kept detections
            3. If IoU > threshold with any kept detection, mark as duplicate
            4. Keep only non-duplicate detections
        """
        def calculate_iou(box1: List[float], box2: List[float]) -> float:
            """Calculate Intersection over Union of two bounding boxes."""
            y0_1, x0_1, y1_1, x1_1 = box1
            y0_2, x0_2, y1_2, x1_2 = box2

            # Calculate intersection
            x0_i = max(x0_1, x0_2)
            y0_i = max(y0_1, y0_2)
            x1_i = min(x1_1, x1_2)
            y1_i = min(y1_1, y1_2)

            if x1_i <= x0_i or y1_i <= y0_i:
                return 0.0

            intersection = (x1_i - x0_i) * (y1_i - y0_i)

            # Calculate union
            area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
            area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0.0

        if not detections:
            return []

        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x.get('confidence', 0.8), reverse=True)

        filtered = []
        for det in detections:
            is_duplicate = False
            for kept_det in filtered:
                if calculate_iou(det['box_2d'], kept_det['box_2d']) > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(det)

        return filtered

    def detect_objects(self,
                      image: Image.Image,
                      target_objects: str,
                      mode: str = "bounding_box",
                      min_confidence: float = 0.7,
                      max_objects: int = 25,
                      remove_duplicates: bool = True,
                      iou_threshold: float = 0.5) -> Tuple[List[Dict], str]:
        """
        Perform object detection on an image.

        This is the main detection method that orchestrates the entire detection
        pipeline from prompt creation to result filtering.

        Args:
            image (Image.Image): PIL Image object to analyze
            target_objects (str): Comma-separated list of objects to detect
            mode (str): Detection mode - "bounding_box" or "segmentation"
            min_confidence (float): Minimum confidence threshold for filtering
                                  0.0-1.0, higher values = stricter filtering
            max_objects (int): Maximum number of objects to detect (1-50)
            remove_duplicates (bool): Whether to remove duplicate detections
            iou_threshold (float): IoU threshold for duplicate removal (0.0-1.0)

        Returns:
            Tuple[List[Dict], str]: (filtered_detections, raw_response)
                - filtered_detections: List of validated detection dictionaries
                - raw_response: Raw text response from Gemini model

        Raises:
            ValueError: If image is None or target_objects is empty
            ConnectionError: If unable to connect to Gemini API

        Example:
            >>> image = Image.open("photo.jpg")
            >>> detections, raw = detector.detect_objects(image, "cars, trucks")
            >>> print(f"Found {len(detections)} vehicles")
        """
        if image is None:
            raise ValueError("Image cannot be None")
        if not target_objects or target_objects.strip() == "":
            raise ValueError("Target objects cannot be empty")

        # Create detection prompt
        prompt = target_objects

        # Configure generation
        config = types.GenerateContentConfig(
            temperature=.5,
            safety_settings=self.safety_settings,
            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget)
        )

        if mode == "bounding_box":
            config.system_instruction = self.bounding_box_system_instructions
        elif mode == "segmentation":
            config.system_instruction = self.segmentation_system_instructions
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'bounding_box' or 'segmentation'")

        # Generate content
        print(f"🔍 Detecting: {target_objects}")
        print("⏳ Processing with Gemini...")

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image],
                config=config
            )

            # Check if response has text attribute and content
            if not hasattr(response, 'text'):
                raise ConnectionError("Invalid response format from Gemini API")

            raw_response = response.text

            # Check for empty response
            if not raw_response or raw_response.strip() == "":
                print("⚠️  Warning: Gemini API returned empty response")
                print("   This might be due to:")
                print("   - API rate limiting")
                print("   - Content policy restrictions")
                print("   - Temporary service issues")
                print("   - Invalid API key")
                raw_response = "[]"  # Return empty array to avoid parsing errors

        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower():
                raise ConnectionError(f"API quota exceeded: {error_msg}")
            elif "permission" in error_msg.lower() or "auth" in error_msg.lower():
                raise ConnectionError(f"Authentication error - check your API key: {error_msg}")
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                raise ConnectionError(f"Network connection error: {error_msg}")
            else:
                raise ConnectionError(f"Failed to get response from Gemini: {error_msg}")

        # Parse and validate detections
        detections = self.parse_detection_response(raw_response)

        # Filter invalid detections
        valid_detections = [det for det in detections if self.validate_detection(det)]

        # Filter by target objects (most important filter)
        valid_detections = self.filter_detections_by_target_objects(
            valid_detections, target_objects
        )

        # Filter by minimum size (remove small false positives)
        valid_detections = self.filter_detections_by_size(
            valid_detections,
            min_width_pixels=30,
            min_height_pixels=20,
            image_size=image.size
        )

        # Apply confidence filtering
        if min_confidence > 0:
            valid_detections = self.filter_detections_by_confidence(
                valid_detections, min_confidence
            )

        # Remove duplicates if requested
        if remove_duplicates and len(valid_detections) > 1:
            valid_detections = self.remove_duplicate_detections(
                valid_detections, iou_threshold
            )

        print(f"✅ Found {len(valid_detections)} valid detections")

        return valid_detections, raw_response
