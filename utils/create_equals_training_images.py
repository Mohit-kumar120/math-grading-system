# utils/create_equals_training_images.py
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import numpy as np

def create_equals_training_set():
    """Create specialized training images for equals sign recognition"""
    
    # Different equals sign styles
    equals_variations = [
        ("2+2=4", "equals_standard.png"),
        ("2+2=4", "equals_bold.png"),
        ("2+2=4", "equals_spaced.png"),
        ("x=y", "equals_variables.png"),
        ("5=5", "equals_double.png"),
        ("10=10", "equals_large.png"),
    ]
    
    print("🎯 Creating equals sign training images...")
    
    for text, filename in equals_variations:
        create_equals_image(text, filename)
    
    print("✅ Created equals sign training set!")

def create_equals_image(text, filename):
    """Create images with optimized equals signs"""
    width, height = 800, 400  # Larger for better recognition
    
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Use clear, bold font
    try:
        font = ImageFont.truetype("arial.ttf", 60)
        bold_font = ImageFont.truetype("arialbd.ttf", 60)
    except:
        font = ImageFont.load_default()
        bold_font = font
    
    # Calculate text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw with extra bold equals sign
    if '=' in text:
        parts = text.split('=')
        left_bbox = draw.textbbox((0, 0), parts[0], font=font)
        left_width = left_bbox[2] - left_bbox[0]
        
        # Draw left part
        draw.text((x, y), parts[0], fill='black', font=font)
        
        # Draw bold equals sign
        equals_x = x + left_width + 10
        draw.text((equals_x, y), "=", fill='black', font=bold_font)
        
        # Draw right part
        right_x = equals_x + 40
        draw.text((right_x, y), parts[1], fill='black', font=font)
    else:
        draw.text((x, y), text, fill='black', font=font)
    
    # Add some realistic noise
    img = add_training_noise(img)
    
    os.makedirs('training_images', exist_ok=True)
    img.save(f'training_images/{filename}', 'PNG')
    print(f"✅ Created: training_images/{filename}")

def add_training_noise(image):
    """Add realistic noise to training images"""
    # Convert to array for processing
    img_array = np.array(image)
    
    # Add slight Gaussian noise
    noise = np.random.normal(0, 5, img_array.shape)
    img_array = img_array + noise
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # Convert back to PIL
    return Image.fromarray(img_array)

if __name__ == "__main__":
    create_equals_training_set()