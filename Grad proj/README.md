# Face Cropping and Enhancement Script

This script detects faces in an image, crops them, and enhances their quality and resolution.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Simply run the script:

```bash
python crop_and_enhance_faces.py
```

The script will:
1. Detect all faces in `frame_038.jpg`
2. Crop each face with padding
3. Enhance quality using:
   - Sharpening filters
   - Contrast enhancement
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Denoising
   - High-quality upscaling (LANCZOS resampling)
4. Save enhanced faces to the `cropped faces` directory

## Configuration

You can modify these parameters in the script:
- `min_face_size`: Minimum face size to process (default: 50 pixels)
- `target_resolution`: Target resolution for upscaling (default: 512 pixels)
- `output_dir`: Directory to save cropped faces (default: "cropped faces")

## Output

Enhanced face images will be saved as:
- `face_001_conf_0.XX.jpg`
- `face_002_conf_0.XX.jpg`
- etc.

Each filename includes the confidence score from face detection.

