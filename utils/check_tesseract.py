# utils/check_tesseract.py
import shutil
import os

def check_tesseract():
    print("🔍 Checking Tesseract OCR configuration...")
    
    # Method 1: Check if tesseract is in PATH
    tesseract_path = shutil.which('tesseract')
    if tesseract_path:
        print(f"✅ Tesseract found in PATH: {tesseract_path}")
    else:
        print("❌ Tesseract not found in PATH")
    
    # Method 2: Check common installation directories
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\ProgramData\chocolatey\lib\tesseract\tools\tesseract.exe",
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            print(f"✅ Tesseract found at: {path}")
            tesseract_path = path
            break
    else:
        print("❌ Tesseract not found in common locations")
    
    # Method 3: Try to import pytesseract and check configuration
    try:
        import pytesseract
        print("✅ pytesseract is installed")
        
        # Check if pytesseract can find tesseract
        try:
            tesseract_version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract version: {tesseract_version}")
        except Exception as e:
            print(f"❌ pytesseract cannot find tesseract: {e}")
            
    except ImportError:
        print("❌ pytesseract is not installed")
    
    return tesseract_path

if __name__ == "__main__":
    check_tesseract()