import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
from pathlib import Path
import urllib.request

def download_dnn_models():
    """
    Download OpenCV DNN face detection model files if they don't exist
    """
    prototxt_path = "deploy.prototxt"
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"
    
    if not os.path.exists(prototxt_path):
        print("Downloading prototxt file...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            prototxt_path
        )
    
    if not os.path.exists(model_path):
        print("Downloading model file (this may take a moment)...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            model_path
        )
    
    return prototxt_path, model_path

def detect_faces_opencv(image_path):
    """
    Detect faces using OpenCV's DNN face detector (more accurate than Haar cascades)
    """
    # Try to download model files if they don't exist
    try:
        prototxt_path, model_path = download_dnn_models()
    except Exception as e:
        print(f"Could not download DNN models: {e}")
        print("Using Haar Cascade detector as fallback...")
        return detect_faces_haar(image_path)
    
    # Check if model files exist, if not, use Haar cascade as fallback
    if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
        print("DNN model files not found. Using Haar Cascade detector...")
        return detect_faces_haar(image_path)
    
    # Load the DNN model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    (h, w) = image.shape[:2]
    
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
    
    # Pass the blob through the network and obtain detections
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    confidence_threshold = 0.5
    
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            # Compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure the bounding box is within image bounds
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            # Add padding for better crop
            padding = 20
            startX = max(0, startX - padding)
            startY = max(0, startY - padding)
            endX = min(w, endX + padding)
            endY = min(h, endY + padding)
            
            faces.append((startX, startY, endX, endY, confidence))
    
    return image, faces

def detect_faces_haar(image_path):
    """
    Detect faces using Haar Cascade (fallback method) with improved parameters
    """
    # Try multiple cascade files for better detection
    cascade_files = [
        'haarcascade_frontalface_default.xml',
        'haarcascade_frontalface_alt.xml',
        'haarcascade_frontalface_alt2.xml'
    ]
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    all_faces = []
    
    # Try each cascade file
    for cascade_file in cascade_files:
        cascade_path = cv2.data.haarcascades + cascade_file
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            continue
        
        # Use more sensitive parameters for better detection
        faces_detected = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # Smaller steps for better detection
            minNeighbors=3,     # Lower threshold
            minSize=(20, 20),   # Smaller minimum size
            flags=cv2.CASCADE_SCALE_IMAGE,
            maxSize=(500, 500)  # Maximum face size
        )
        
        for (x, y, w, h) in faces_detected:
            all_faces.append((x, y, w, h))
    
    # Remove duplicate faces (faces that overlap significantly)
    unique_faces = []
    for (x, y, w, h) in all_faces:
        # Check if this face overlaps significantly with any existing face
        is_duplicate = False
        for (ex, ey, ew, eh) in unique_faces:
            # Calculate overlap
            overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
            overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
            overlap_area = overlap_x * overlap_y
            face_area = w * h
            if overlap_area > face_area * 0.5:  # 50% overlap threshold
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_faces.append((x, y, w, h))
    
    # Convert to final format with padding
    faces = []
    for (x, y, w, h) in unique_faces:
        # Add padding
        padding = 30
        startX = max(0, x - padding)
        startY = max(0, y - padding)
        endX = min(image.shape[1], x + w + padding)
        endY = min(image.shape[0], y + h + padding)
        faces.append((startX, startY, endX, endY, 1.0))
    
    return image, faces

def enhance_image_quality(pil_image, target_size=None):
    """
    Enhance image quality using multiple techniques
    """
    # Convert to RGB if needed
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Apply sharpening
    pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.1)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.2)
    
    # If target size is specified, upscale using high-quality resampling
    if target_size:
        # Use LANCZOS resampling for best quality upscaling
        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
    
    return pil_image

def enhance_with_opencv(cv_image):
    """
    Additional enhancement using OpenCV techniques
    """
    # Convert to LAB color space for better enhancement
    lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Apply slight denoising
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    return enhanced

def crop_and_enhance_faces(image_path, output_dir="cropped faces", min_face_size=50, target_resolution=512):
    """
    Main function to crop faces and enhance their quality
    """
    # Get the directory of the script to ensure output is in the same folder
    script_dir = os.path.dirname(os.path.abspath(__file__)) if __file__ else os.getcwd()
    output_path = os.path.join(script_dir, output_dir)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Output folder: {output_path}")
    print(f"Loading image: {image_path}")
    
    # Detect faces - try DNN first, then Haar cascade
    image = None
    faces = []
    
    try:
        print("Attempting face detection with DNN model...")
        image, faces = detect_faces_opencv(image_path)
        if len(faces) == 0:
            print("DNN detection found no faces. Trying Haar Cascade...")
            image, faces = detect_faces_haar(image_path)
    except Exception as e:
        print(f"Error in DNN face detection: {e}")
        print("Trying Haar Cascade as fallback...")
        try:
            image, faces = detect_faces_haar(image_path)
        except Exception as e2:
            print(f"Error in Haar Cascade detection: {e2}")
            return
    
    if len(faces) == 0:
        print("No faces detected in the image.")
        return
    
    print(f"Found {len(faces)} face(s)")
    
    # Process each face
    for i, (startX, startY, endX, endY, confidence) in enumerate(faces):
        # Calculate face dimensions
        face_width = endX - startX
        face_height = endY - startY
        
        # Skip very small faces
        if face_width < min_face_size or face_height < min_face_size:
            print(f"Skipping face {i+1} (too small: {face_width}x{face_height})")
            continue
        
        print(f"Processing face {i+1} (confidence: {confidence:.2f}, size: {face_width}x{face_height})")
        
        # Crop the face
        face_crop = image[startY:endY, startX:endX]
        
        # Convert to PIL for enhancement
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        
        # Calculate target size maintaining aspect ratio
        aspect_ratio = face_width / face_height
        if aspect_ratio > 1:
            target_width = target_resolution
            target_height = int(target_resolution / aspect_ratio)
        else:
            target_height = target_resolution
            target_width = int(target_resolution * aspect_ratio)
        
        # Enhance quality and upscale
        enhanced_face = enhance_image_quality(face_pil, target_size=(target_width, target_height))
        
        # Additional OpenCV enhancement
        enhanced_cv = cv2.cvtColor(np.array(enhanced_face), cv2.COLOR_RGB2BGR)
        enhanced_cv = enhance_with_opencv(enhanced_cv)
        
        # Convert back to PIL for final save
        final_face = Image.fromarray(cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2RGB))
        
        # Save the enhanced face
        face_output_path = os.path.join(output_path, f"face_{i+1:03d}_conf_{confidence:.2f}.jpg")
        final_face.save(face_output_path, "JPEG", quality=95, optimize=True)
        print(f"Saved: {face_output_path}")
    
    print(f"\nAll faces processed and saved to '{output_path}' directory")

if __name__ == "__main__":
    # Input image path
    image_path = "frame_066.jpg"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        exit(1)
    
    # Process faces
    crop_and_enhance_faces(
        image_path=image_path,
        output_dir="cropped faces",
        min_face_size=50,  # Minimum face size in pixels
        target_resolution=512  # Target resolution (will maintain aspect ratio)
    )

