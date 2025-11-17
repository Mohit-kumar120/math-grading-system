# utils/create_equals_test_images.py
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import os
import random

def create_equals_test_images():
    """Create test images with clear equals signs"""
    
    test_cases = [
        ("2+2=4", "clear_equals_2plus2.png"),
        ("3+4=7", "clear_equals_3plus4.png"),
        ("5=5", "clear_equals_5equals5.png"),
        ("x=2", "clear_equals_xequals2.png"),
        ("10-5=5", "clear_equals_10minus5.png"),
    ]
    
    print("🎯 Creating equals sign test images...")
    
    for text, filename in test_cases:
        create_clear_equals_image(text, filename)
    
    print(f"✅ Created {len(test_cases)} equals sign test images!")

def create_clear_equals_image(text, filename):
    """Create images with very clear equals signs"""
    width, height = 600, 300
    
    # Clean white background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Use a clear, bold font
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()
    
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
    
    # Draw in solid black with extra spacing around equals
    draw.text((x, y), text, fill='black', font=font)
    
    # Make the equals sign even clearer by adding extra contrast
    # Save as high-quality PNG
    os.makedirs('test_images_ocr', exist_ok=True)
    img.save(f'test_images_ocr/{filename}', 'PNG')
    
    print(f"✅ Created: test_images_ocr/{filename}")

def create_double_line_equals():
    """Create images where equals sign is drawn as two explicit lines"""
    equations = [
        ("2+2=4", "double_equals_2plus2.png"),
        ("3+4=7", "double_equals_3plus4.png"),
    ]
    
    for text, filename in equations:
        create_explicit_equals_image(text, filename)

def create_explicit_equals_image(text, filename):
    """Create image with manually drawn equals sign as two lines"""
    width, height = 600, 300
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Use font for the numbers/operators
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()
    
    # Split the equation to manually draw the equals sign
    if '=' in text:
        parts = text.split('=')
        left_side = parts[0]
        right_side = parts[1]
        
        # Calculate positions
        try:
            left_bbox = draw.textbbox((0, 0), left_side, font=font)
            left_width = left_bbox[2] - left_bbox[0]
            
            right_bbox = draw.textbbox((0, 0), right_side, font=font)
            right_width = right_bbox[2] - right_bbox[0]
        except:
            left_width = len(left_side) * 30
            right_width = len(right_side) * 30
        
        total_width = left_width + 80 + right_width  # 80px for equals sign
        
        # Draw left side
        left_x = (width - total_width) // 2
        y = (height - 60) // 2
        
        draw.text((left_x, y), left_side, fill='black', font=font)
        
        # Draw equals sign as two lines
        equals_x = left_x + left_width + 20
        line_y1 = y + 15
        line_y2 = y + 35
        
        # Two thick lines for equals sign
        draw.line([(equals_x, line_y1), (equals_x + 40, line_y1)], fill='black', width=4)
        draw.line([(equals_x, line_y2), (equals_x + 40, line_y2)], fill='black', width=4)
        
        # Draw right side
        right_x = equals_x + 60
        draw.text((right_x, y), right_side, fill='black', font=font)
    
    os.makedirs('test_images_ocr', exist_ok=True)
    img.save(f'test_images_ocr/{filename}', 'PNG')
    print(f"✅ Created explicit equals: test_images_ocr/{filename}")

if __name__ == "__main__":
    create_equals_test_images()
    create_double_line_equals()