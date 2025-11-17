# utils/create_handwritten_test_images.py
from PIL import Image, ImageDraw, ImageFont
import os
import random

def create_handwritten_style_image(text, filename):
    """Create test images that simulate handwritten style with variations"""
    # Create a slightly off-white background to simulate paper
    bg_color = (random.randint(240, 250), random.randint(240, 250), random.randint(235, 245))
    img = Image.new('RGB', (400, 200), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a handwriting-style font
    font = get_handwriting_font(36)
    
    # Get text bounding box
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        # Fallback if textbbox doesn't work
        text_width = len(text) * 20
        text_height = 40
    
    # Calculate position to center text with slight random offset
    x = (400 - text_width) // 2 + random.randint(-10, 10)
    y = (200 - text_height) // 2 + random.randint(-10, 10)
    
    # Add slight rotation to simulate natural handwriting
    rotation_angle = random.uniform(-2, 2)
    img_rotated = img.rotate(rotation_angle, expand=False, resample=Image.BICUBIC, fillcolor=bg_color)
    draw_rotated = ImageDraw.Draw(img_rotated)
    
    # Draw text with slight color variation (not pure black)
    text_color = (
        random.randint(20, 40),    # R
        random.randint(20, 40),    # G  
        random.randint(20, 40)     # B
    )
    
    # Add slight jitter to simulate imperfect handwriting
    jitter_x = random.randint(-2, 2)
    jitter_y = random.randint(-2, 2)
    
    draw_rotated.text((x + jitter_x, y + jitter_y), text, fill=text_color, font=font)
    
    # Add some noise to simulate paper texture and ink variations
    img_with_noise = add_paper_texture(img_rotated)
    
    # Save image
    os.makedirs('test_images', exist_ok=True)
    img_with_noise.save(f'test_images/{filename}', quality=95)
    print(f"✅ Created handwritten-style test image: test_images/{filename}")

def get_handwriting_font(size):
    """Try to load handwriting-style fonts, fallback to system fonts"""
    handwriting_fonts = [
        # Windows handwriting fonts
        "C:\\Windows\\Fonts\\comic.ttf",           # Comic Sans (handwritten style)
        "C:\\Windows\\Fonts\\comicbd.ttf",         # Comic Sans Bold
        "C:\\Windows\\Fonts\\cour.ttf",            # Courier (monospace, good for math)
        "C:\\Windows\\Fonts\\arial.ttf",           # Arial (clean, readable)
        "C:\\Windows\\Fonts\\times.ttf",           # Times New Roman
        # Common font paths
        "arial.ttf",
        "comic.ttf",
        # Mac fonts (if running on Mac)
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Comic Sans MS.ttf",
    ]
    
    for font_path in handwriting_fonts:
        try:
            font = ImageFont.truetype(font_path, size)
            print(f"📝 Using font: {os.path.basename(font_path)}")
            return font
        except:
            continue
    
    # Fallback to default font
    print("⚠️  Using default font (handwriting fonts not found)")
    try:
        return ImageFont.load_default()
    except:
        # Ultimate fallback
        class DefaultFont:
            def __init__(self):
                self.size = size
        return DefaultFont()

def add_paper_texture(image):
    """Add paper-like texture and noise to the image"""
    pixels = image.load()
    width, height = image.size
    
    # Add subtle paper texture
    for i in range(width):
        for j in range(height):
            if random.random() < 0.02:  # 2% noise for subtle texture
                r, g, b = pixels[i, j]
                # Slightly darken or lighten random pixels
                variation = random.randint(-8, 8)
                new_r = max(0, min(255, r + variation))
                new_g = max(0, min(255, g + variation))
                new_b = max(0, min(255, b + variation))
                pixels[i, j] = (new_r, new_g, new_b)
    
    # Add occasional ink spots (more realistic)
    for _ in range(random.randint(3, 8)):
        spot_x = random.randint(10, width - 10)
        spot_y = random.randint(10, height - 10)
        spot_size = random.randint(1, 3)
        for dx in range(-spot_size, spot_size + 1):
            for dy in range(-spot_size, spot_size + 1):
                if 0 <= spot_x + dx < width and 0 <= spot_y + dy < height:
                    if random.random() < 0.7:  # 70% chance for each pixel in spot
                        pixels[spot_x + dx, spot_y + dy] = (
                            random.randint(10, 30),  # Dark ink color
                            random.randint(10, 30),
                            random.randint(10, 30)
                        )
    
    return image

def create_varied_handwriting_image(text, filename, style_variation="normal"):
    """Create images with different handwriting styles"""
    # Different styles for variation - FIXED: Added "normal" style
    styles = {
        "normal": {
            "bg_color": (248, 246, 240),
            "text_color": (30, 30, 30),
            "rotation": (-1.5, 1.5),
            "jitter": (-2, 2),
            "font_size": 35
        },
        "neat": {
            "bg_color": (250, 250, 245),
            "text_color": (30, 30, 30),
            "rotation": (-1, 1),
            "jitter": (0, 1),
            "font_size": 34
        },
        "messy": {
            "bg_color": (245, 242, 235),
            "text_color": (40, 35, 30),
            "rotation": (-3, 3),
            "jitter": (-3, 3),
            "font_size": 32
        },
        "pencil": {
            "bg_color": (248, 246, 240),
            "text_color": (60, 55, 50),  # Gray for pencil
            "rotation": (-1.5, 1.5),
            "jitter": (-2, 2),
            "font_size": 33
        }
    }
    
    style = styles.get(style_variation, styles["normal"])
    
    # Create image with specific style
    bg_color = style["bg_color"]
    img = Image.new('RGB', (400, 200), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    font = get_handwriting_font(style["font_size"])
    
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        text_width = len(text) * 20
        text_height = 40
    
    x = (400 - text_width) // 2 + random.randint(*style["jitter"])
    y = (200 - text_height) // 2 + random.randint(*style["jitter"])
    
    rotation_angle = random.uniform(*style["rotation"])
    img_rotated = img.rotate(rotation_angle, expand=False, resample=Image.BICUBIC, fillcolor=bg_color)
    draw_rotated = ImageDraw.Draw(img_rotated)
    
    # Add color variation
    base_color = style["text_color"]
    text_color = (
        base_color[0] + random.randint(-5, 5),
        base_color[1] + random.randint(-5, 5),
        base_color[2] + random.randint(-5, 5)
    )
    
    draw_rotated.text((x, y), text, fill=text_color, font=font)
    
    # Add appropriate texture based on style
    if style_variation == "messy":
        img_with_texture = add_messy_texture(img_rotated)
    elif style_variation == "pencil":
        img_with_texture = add_pencil_texture(img_rotated)
    else:
        img_with_texture = add_paper_texture(img_rotated)
    
    os.makedirs('test_images', exist_ok=True)
    img_with_texture.save(f'test_images/{filename}', quality=95)
    print(f"✅ Created {style_variation} style image: test_images/{filename}")

def add_messy_texture(image):
    """Add more aggressive noise for messy handwriting"""
    pixels = image.load()
    width, height = image.size
    
    # More noise for messy style
    for i in range(width):
        for j in range(height):
            if random.random() < 0.04:  # 4% noise
                r, g, b = pixels[i, j]
                variation = random.randint(-12, 12)
                new_r = max(0, min(255, r + variation))
                new_g = max(0, min(255, g + variation))
                new_b = max(0, min(255, b + variation))
                pixels[i, j] = (new_r, new_g, new_b)
    
    # More ink spots
    for _ in range(random.randint(5, 12)):
        spot_x = random.randint(5, width - 5)
        spot_y = random.randint(5, height - 5)
        spot_size = random.randint(1, 4)
        for dx in range(-spot_size, spot_size + 1):
            for dy in range(-spot_size, spot_size + 1):
                if 0 <= spot_x + dx < width and 0 <= spot_y + dy < height:
                    if random.random() < 0.6:
                        pixels[spot_x + dx, spot_y + dy] = (
                            random.randint(15, 40),
                            random.randint(15, 40),
                            random.randint(15, 40)
                        )
    
    return image

def add_pencil_texture(image):
    """Add pencil-like texture (lighter, more subtle)"""
    pixels = image.load()
    width, height = image.size
    
    # Subtle noise for pencil
    for i in range(width):
        for j in range(height):
            if random.random() < 0.015:  # 1.5% subtle noise
                r, g, b = pixels[i, j]
                variation = random.randint(-5, 5)
                new_r = max(0, min(255, r + variation))
                new_g = max(0, min(255, g + variation))
                new_b = max(0, min(255, b + variation))
                pixels[i, j] = (new_r, new_g, new_b)
    
    # Pencil smudges
    for _ in range(random.randint(2, 5)):
        smudge_x = random.randint(20, width - 20)
        smudge_y = random.randint(20, height - 20)
        smudge_size = random.randint(8, 15)
        for dx in range(-smudge_size, smudge_size + 1):
            for dy in range(-smudge_size, smudge_size + 1):
                distance = (dx**2 + dy**2)**0.5
                if distance < smudge_size and 0 <= smudge_x + dx < width and 0 <= smudge_y + dy < height:
                    if random.random() < 0.3:
                        r, g, b = pixels[smudge_x + dx, smudge_y + dy]
                        # Light gray smudge
                        pixels[smudge_x + dx, smudge_y + dy] = (
                            max(0, r - random.randint(5, 15)),
                            max(0, g - random.randint(5, 15)),
                            max(0, b - random.randint(5, 15))
                        )
    
    return image

def create_comprehensive_test_suite():
    """Create a comprehensive set of test images for various math problems"""
    
    # Basic test cases covering different math concepts
    test_cases = [
        # Basic arithmetic
        ("42", "basic_42.jpg"),
        ("15", "basic_15.jpg"),
        ("8", "basic_8.jpg"),
        ("3.14", "basic_pi_approx.jpg"),
        ("-5", "basic_negative.jpg"),
        
        # Simple expressions
        ("x+2", "expr_x_plus_2.jpg"),
        ("2x+1", "expr_2x_plus_1.jpg"),
        ("y-3", "expr_y_minus_3.jpg"),
        ("x÷2", "expr_x_div_2.jpg"),
        
        # Equations
        ("2+2=4", "eq_2plus2.jpg"),
        ("3+4=7", "eq_3plus4.jpg"),
        ("x+1=3", "eq_x_plus_1.jpg"),
        ("y=2x+1", "eq_y_equals.jpg"),
        ("2×3=6", "eq_multiplication.jpg"),
        ("10÷2=5", "eq_division.jpg"),
        
        # With parentheses
        ("(x+1)", "paren_simple.jpg"),
        ("(x+1)^2", "paren_squared.jpg"),
        ("2(x+1)", "paren_with_coeff.jpg"),
        
        # Multi-digit numbers
        ("123", "multidigit_123.jpg"),
        ("45+67", "multidigit_addition.jpg"),
        ("100-25", "multidigit_subtraction.jpg"),
    ]
    
    # Create variations for key test cases
    style_variations = [
        ("normal", "normal_"),
        ("neat", "neat_"),
        ("messy", "messy_"), 
        ("pencil", "pencil_")
    ]
    
    print("🎨 Creating comprehensive test image suite...")
    
    # Create basic test images
    for text, filename in test_cases:
        create_handwritten_style_image(text, filename)
    
    # Create style variations for important test cases
    key_cases = [
        ("42", "number"),
        ("x+2", "expression"),
        ("2+2=4", "equation"),
        ("(x+1)^2", "parentheses"),
    ]
    
    for text, category in key_cases:
        for style, prefix in style_variations:
            filename = f"{prefix}{category}.jpg"
            create_varied_handwriting_image(text, filename, style)
    
    total_count = len(test_cases) + len(key_cases) * len(style_variations)
    print(f"✅ Created {total_count} test images!")
    
    return total_count

def create_simple_test_set():
    """Create a simple set of test images for quick testing"""
    simple_test_cases = [
        ("42", "test_42.jpg"),
        ("x+2", "test_xplus2.jpg"), 
        ("15", "test_15.jpg"),
        ("3+4=7", "test_3plus4.jpg"),
        ("8", "test_8.jpg"),
        ("2×3=6", "test_multiplication.jpg"),
        ("y=2x+1", "test_algebra.jpg"),
        ("(x+1)^2", "test_parentheses.jpg"),
        ("10÷2=5", "test_division.jpg"),
    ]
    
    print("🎨 Creating simple test image set...")
    
    for text, filename in simple_test_cases:
        create_handwritten_style_image(text, filename)
    
    print(f"✅ Created {len(simple_test_cases)} simple test images!")
    return len(simple_test_cases)

# Create some test images
if __name__ == "__main__":
    print("🖋️  Math Grading System - Test Image Generator")
    print("=" * 50)
    
    # Let user choose which set to create
    print("\nChoose test image set:")
    print("1. Simple set (9 images - recommended)")
    print("2. Comprehensive set (25+ images)")
    print("3. Custom single image")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        count = create_simple_test_set()
    elif choice == "2":
        count = create_comprehensive_test_suite()
    elif choice == "3":
        text = input("Enter the math text: ").strip()
        filename = input("Enter filename (e.g., 'custom_test.jpg'): ").strip()
        if not filename:
            filename = "custom_test.jpg"
        create_handwritten_style_image(text, filename)
        count = 1
    else:
        print("⚠️  Invalid choice, creating simple set...")
        count = create_simple_test_set()
    
    print(f"\n🎉 Successfully created {count} test images in 'test_images' folder!")
    print("📁 You can now use these images to test your math grading system.")
    print("\n💡 Tip: Use the 'Test Images' tab in the web app to easily test with these images.")