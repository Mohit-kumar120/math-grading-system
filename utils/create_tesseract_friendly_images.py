# utils/create_tesseract_friendly_images.py
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import random
import numpy as np

def create_tesseract_friendly_image(text, filename):
    """Create test images that Tesseract can actually read"""
    # Larger image for better OCR
    width, height = 600, 300
    
    # More realistic paper background
    bg_color = (random.randint(235, 245), random.randint(235, 245), random.randint(230, 240))
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Use a standard, clear font (not handwriting)
    font = get_clear_font(48)  # Larger font size
    
    # Get text size
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        text_width = len(text) * 30
        text_height = 60
    
    # Center the text
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw text in solid black (good for OCR)
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    
    # Add realistic noise and texture
    img = add_ocr_friendly_noise(img)
    
    # Save as high-quality PNG (better for OCR than JPEG)
    os.makedirs('test_images_ocr', exist_ok=True)
    img.save(f'test_images_ocr/{filename}', 'PNG')
    print(f"✅ Created Tesseract-friendly image: test_images_ocr/{filename}")

def get_clear_font(size):
    """Get a clear, standard font that OCR can read easily"""
    clear_fonts = [
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\times.ttf", 
        "C:\\Windows\\Fonts\\cour.ttf",
        "arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    
    for font_path in clear_fonts:
        try:
            font = ImageFont.truetype(font_path, size)
            print(f"📝 Using clear font: {os.path.basename(font_path)}")
            return font
        except:
            continue
    
    # Fallback
    print("⚠️  Using default font")
    try:
        return ImageFont.load_default()
    except:
        class DefaultFont:
            def __init__(self):
                self.size = size
        return DefaultFont()

def add_ocr_friendly_noise(image):
    """Add noise that makes images look more like real scans"""
    # Convert to numpy array for processing
    img_array = np.array(image)
    height, width, channels = img_array.shape
    
    # Add subtle Gaussian noise (like scanner noise)
    noise = np.random.normal(0, 3, (height, width, channels))
    img_array = img_array + noise
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image
    image = Image.fromarray(img_array)
    
    # Add slight blur to simulate real document scanning
    image = image.filter(ImageFilter.GaussianBlur(0.5))
    
    # Add subtle brightness/contrast variations
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.95, 1.05))
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.95, 1.05))
    
    return image

def create_ocr_test_set():
    """Create a set of images optimized for Tesseract OCR"""
    test_cases = [
        # Simple numbers (easy for OCR)
        ("42", "ocr_42.png"),
        ("15", "ocr_15.png"),
        ("8", "ocr_8.png"),
        ("123", "ocr_123.png"),
        
        # Basic math expressions
        ("2+2=4", "ocr_2plus2.png"),
        ("3+4=7", "ocr_3plus4.png"),
        ("10-5=5", "ocr_10minus5.png"),
        ("2×3=6", "ocr_2times3.png"),
        
        # Simple algebra
        ("x+2", "ocr_xplus2.png"),
        ("y=3", "ocr_yequals3.png"),
        ("a+b", "ocr_aplusb.png"),
    ]
    
    print("🎯 Creating Tesseract-optimized test images...")
    
    for text, filename in test_cases:
        create_tesseract_friendly_image(text, filename)
    
    print(f"✅ Created {len(test_cases)} Tesseract-friendly test images!")
    return len(test_cases)

# Add the required import at the top
from PIL import ImageEnhance

if __name__ == "__main__":
    create_ocr_test_set()