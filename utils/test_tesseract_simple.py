# utils/test_tesseract_simple.py
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import os

def test_tesseract_simple():
    print("🧪 Simple Tesseract Test")
    print("=" * 40)
    
    # Set paths
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
    
    # Create a dead simple test image
    img = Image.new('RGB', (200, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    # Use Arial font, large size
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Draw clear black text
    draw.text((50, 30), "42", fill='black', font=font)
    
    # Save it
    img.save('simple_test.png')
    
    print("📁 Created test image: simple_test.png")
    
    # Try OCR with different configs
    configs = [
        '--psm 8',      # Single word
        '--psm 7',      # Single line  
        '--psm 6',      # Block
        '--psm 10',     # Single character
    ]
    
    for config in configs:
        try:
            text = pytesseract.image_to_string(img, config=config).strip()
            print(f"🔧 {config}: '{text}'")
            
            if "42" in text:
                print("✅ SUCCESS! Tesseract is working!")
                return True
                
        except Exception as e:
            print(f"❌ {config} failed: {e}")
    
    print("❌ Tesseract test failed")
    return False

if __name__ == "__main__":
    test_tesseract_simple()