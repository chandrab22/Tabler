import cv2
import numpy as np
from PIL import Image, ImageOps

def preprocess_image(image_path, output_path, target_size=(1280, 1280)):
    """
    Applies EXIF auto-orient when present, then resizes the image to fit within
    target_size while maintaining aspect ratio and pads the remaining area with
    white pixels (fit white edges in).
    """
    try:
        with Image.open(image_path) as pil_img:
            # Match Roboflow's "Auto-Orient: Applied" behavior for images with EXIF rotation.
            pil_img = ImageOps.exif_transpose(pil_img)
            img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    target_w, target_h = target_size
    print(f"  [Preprocess] Raw: {w}x{h} -> Target: {target_w}x{target_h}")

    # Calculate scale factor to fit within target_size
    scale = min(target_w / w, target_h / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a white canvas
    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255

    # Compute centering offsets
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Paste resized image into center of canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

    # Save preprocessed image
    cv2.imwrite(output_path, canvas)
    return output_path
