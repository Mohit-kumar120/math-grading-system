# utils/create_test_images.py
from PIL import Image, ImageDraw, ImageFont
import os

def create_test_image(text, filename):
    """Create a simple test image with text using PIL"""
    # Create a white background image (RGB mode, no alpha)
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a larger font
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        try:
            # Fallback to default font
            font = ImageFont.truetype("Arial.ttf", 40)
        except:
            try:
                # Try another common font
                font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 40)
            except:
                # Final fallback - use default font
                font = ImageFont.load_default()
                # Scale up the default font
                font.size = 40
    
    # Get text bounding box
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        # Fallback if textbbox doesn't work
        text_width = len(text) * 20
        text_height = 40
    
    # Calculate position to center text
    x = (400 - text_width) // 2
    y = (200 - text_height) // 2
    
    # Draw text
    draw.text((x, y), text, fill='black', font=font)
    
    # Save image
    os.makedirs('test_images', exist_ok=True)
    img.save(f'test_images/{filename}')
    print(f"Created test image: test_images/{filename}")

# Create some test images
if __name__ == "__main__":
    test_cases = [
        ("42", "test_42.jpg"),
        ("x+2", "test_xplus2.jpg"), 
        ("15", "test_15.jpg"),
        ("3+4", "test_3plus4.jpg"),
        ("8", "test_8.jpg"),
        ("Hello", "test_hello.jpg")  # Simple test
    ]
    
    for text, filename in test_cases:
        create_test_image(text, filename)
    
    print("Test images created successfully!")