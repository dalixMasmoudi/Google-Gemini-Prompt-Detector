# Gemino - Gemini + Vision + AI.
LLM based detection model. Detect whatever you want in an image with a simple prompt.
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-green.svg)]()

A modular, production-ready computer vision system using Google's Gemini 2.5 model for accurate object detection and segmentation based on natural language prompts.

## 🚀 Features

- **Natural Language Detection**: Detect objects using simple text descriptions
- **High Accuracy**: Powered by Google's Gemini 2.5 model for precise results
- **Standardized Processing**: All images are automatically resized to 1024x1024 for consistent results
- **Dual Modes**: Support for both bounding box detection and segmentation
- **Modular Architecture**: Clean, maintainable code structure
- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **Flexible Configuration**: JSON config files and environment variables
- **Performance Monitoring**: Built-in timing and success rate tracking
- **Rich Visualization**: Customizable bounding boxes with confidence scores
- **VS Code Integration**: Pre-configured debugging setups for development

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Architecture](#architecture)
- [Performance](#performance)
- [Contributing](#contributing)

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- Google API key for Gemini access
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
https://github.com/dalixMasmoudi/Google-Gemini-Prompt-Detector.git
cd Google-Gemini-Prompt-Detector
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up API Key

Get your Google API key from [Google AI Studio](https://aistudio.google.com/app/apikey) and set it up:

**Option 1: Environment Variable**
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

**Option 2: Create .env file**
```bash
cp .env.example .env
# Edit .env and add your API key
```

## 🚀 Quick Start

### Basic Detection

```bash
python main.py --image photo.jpg --objects "cars, trucks"
```

### License Plate Detection

```bash
python main.py --image car.png --objects "license plate" --min-confidence 0.8
```

### Segmentation Mode

```bash
python main.py --image photo.jpg --objects "people" --mode segmentation
```

### Save Raw Results

```bash
python main.py --image photo.jpg --objects "vehicles" --save-raw --performance
```

### Standardized 1024x1024 Processing

```bash
# All images are automatically resized to 1024x1024 for consistent results
python main.py --image photo.jpg --objects "license plate"

# Output images with bounding boxes are also 1024x1024
python main.py --image large_image.jpg --objects "cars, trucks"
```

## ⚙️ Configuration

### Configuration File

Create a `config.json` file to customize behavior:

```json
{
  "detection": {
    "model_name": "gemini-2.5-flash-preview-05-20",
    "temperature": 0.0,
    "min_confidence": 0.7,
    "max_objects": 20,
    "remove_duplicates": true,
    "iou_threshold": 0.5
  },
  "image": {
    "max_size": 1024,
    "min_size": 512,
    "enhance_quality": true,
    "contrast": 1.2,
    "brightness": 1.0,
    "sharpness": 1.1
  },
  "visualization": {
    "show_confidence": true,
    "show_labels": true,
    "default_font_size": 12,
    "colors": ["#FF0000", "#00FF00", "#0000FF"]
  }
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | Required |
| `GEMINI_MODEL_NAME` | Model to use | gemini-2.5-flash-preview-05-20 |
| `MIN_CONFIDENCE` | Minimum confidence threshold | 0.7 |
| `MAX_OBJECTS` | Maximum objects to detect | 20 |
| `MAX_IMAGE_SIZE` | Maximum image dimension | 1024 |
| `ENHANCE_IMAGES` | Enable image enhancement | true |

## 📚 Usage Examples

### 1. Vehicle Detection

```bash
# Detect various vehicle types
python main.py --image traffic.jpg --objects "cars, trucks, buses, motorcycles"

# High confidence vehicle detection
python main.py --image highway.jpg --objects "vehicles" --min-confidence 0.9
```

### 2. Security Applications

```bash
# Person detection for security
python main.py --image security_cam.jpg --objects "people, persons" --save-raw

# License plate recognition
python main.py --image parking.jpg --objects "license plates" --mode segmentation
```

### 3. Retail and Inventory

```bash
# Product detection
python main.py --image shelf.jpg --objects "bottles, cans, boxes" --max-objects 50

# Custom output location
python main.py --image inventory.jpg --objects "products" --output results/inventory_detected.jpg
```

### 4. Medical and Scientific

```bash
# Medical imaging (example)
python main.py --image xray.jpg --objects "anomalies, fractures" --min-confidence 0.95

# Laboratory equipment
python main.py --image lab.jpg --objects "microscopes, beakers, test tubes"
```

## 🏗 Architecture

The system follows a modular architecture with clear separation of concerns:

```
src/
├── detector.py      # Core detection logic using Gemini
├── image_processor.py # Image loading, preprocessing, enhancement
├── visualizer.py    # Result visualization and reporting
├── config.py        # Configuration management
└── utils.py         # Utilities and helper functions

main.py              # Command-line interface
config.json          # Default configuration
requirements.txt     # Python dependencies
```

### Key Components

1. **GeminiDetector**: Handles communication with Google's Gemini API
2. **ImageProcessor**: Manages image loading, resizing, and enhancement
3. **Visualizer**: Creates visual outputs and generates reports
4. **Config**: Manages configuration from files and environment variables
5. **Utils**: Provides logging, performance monitoring, and file operations

## 📊 Performance

### Typical Performance Metrics

- **Detection Time**: 2-5 seconds per image (depending on size and complexity)
- **Accuracy**: 85-95% for common objects with proper prompts
- **Memory Usage**: ~200-500MB during processing
- **Supported Formats**: JPG, PNG, BMP, TIFF, WebP

### Optimization Tips

1. **Image Size**: Keep images under 1024px for faster processing
2. **Specific Prompts**: Use specific object descriptions for better accuracy
3. **Confidence Threshold**: Adjust based on your accuracy requirements
4. **Batch Processing**: Process multiple images in sequence for efficiency

## 🔧 API Reference

### Command Line Interface

```bash
python main.py [OPTIONS]

Required Arguments:
  --image, -i          Path to input image file
  --objects, -o        Comma-separated list of objects to detect

Optional Arguments:
  --mode, -m           Detection mode: bounding_box or segmentation
  --output             Output image path (auto-generated if not specified)
  --config, -c         Path to JSON configuration file
  --min-confidence     Minimum confidence threshold (0.0-1.0)
  --max-objects        Maximum number of objects to detect
  --save-raw           Save raw detection results to JSON file
  --no-enhance         Disable image quality enhancement
  --verbose, -v        Enable verbose logging
  --performance        Show performance summary
```

### Programmatic Usage

```python
from src.detector import GeminiDetector
from src.image_processor import ImageProcessor
from src.visualizer import Visualizer
from src.config import Config

# Initialize components
config = Config()
detector = GeminiDetector(api_key=config.get_api_key())
processor = ImageProcessor()
visualizer = Visualizer()

# Process image
image, original_size = processor.preprocess_for_detection("photo.jpg")
detections, raw_response = detector.detect_objects(image, "cars, trucks")
result_image = visualizer.draw_detections(image, detections)

# Save results
visualizer.save_result(result_image, "output.jpg")
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/guardiavision/Google-Gemini-Prompt-Detector.git
cd Google-Gemini-Prompt-Detector
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run tests
python test_modular.py

# Install in development mode
pip install -e .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full API Documentation](https://guardiavision.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/guardiavision/guardiavision-gemini/issues)
- **Discussions**: [GitHub Discussions](https://github.com/guardiavision/guardiavision-gemini/discussions)

## 🙏 Acknowledgments

- Google for the powerful Gemini 2.5 model
- The open-source community for inspiration and tools
- Contributors and users who help improve this project

---

**Made with ❤️ by dali**
