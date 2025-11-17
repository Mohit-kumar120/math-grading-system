# utils/download_tessdata_fix.py
import os
import urllib.request
import sys

def download_tessdata():
    """Download the essential Tesseract language data files"""
    tessdata_dir = r"C:\Program Files\Tesseract-OCR\tessdata"
    
    # Create tessdata directory if it doesn't exist
    os.makedirs(tessdata_dir, exist_ok=True)
    
    # Essential files for math OCR
    files_to_download = {
        "eng.traineddata": "https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata",
        "osd.traineddata": "https://github.com/tesseract-ocr/tessdata/raw/main/osd.traineddata",
    }
    
    print("📥 Downloading essential Tesseract language data files...")
    print(f"📁 Destination: {tessdata_dir}")
    
    for filename, url in files_to_download.items():
        filepath = os.path.join(tessdata_dir, filename)
        
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
            print(f"✅ {filename} already exists ({file_size:.1f} MB)")
            
            # Check if file is reasonable size (eng.traineddata should be ~20MB)
            if filename == "eng.traineddata" and file_size < 10:
                print(f"⚠️  {filename} seems too small, re-downloading...")
                os.remove(filepath)
            else:
                continue
        
        try:
            print(f"⬇️  Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            
            # Verify download
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                print(f"✅ Downloaded {filename} ({file_size:.1f} MB)")
            else:
                print(f"❌ Download failed for {filename}")
                
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
    
    print("\n🎉 Language data download completed!")
    
    # Verify the files
    verify_tessdata()

def verify_tessdata():
    """Verify that tessdata files are properly installed"""
    tessdata_dir = r"C:\Program Files\Tesseract-OCR\tessdata"
    essential_files = ["eng.traineddata", "osd.traineddata"]
    
    print("\n🔍 Verifying installation...")
    
    all_good = True
    for filename in essential_files:
        filepath = os.path.join(tessdata_dir, filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✅ {filename}: {file_size:.1f} MB")
            
            # Check for reasonable file sizes
            if filename == "eng.traineddata" and file_size < 10:
                print(f"❌ {filename} is too small (should be ~20MB)")
                all_good = False
        else:
            print(f"❌ {filename}: Missing")
            all_good = False
    
    if all_good:
        print("\n🎉 All essential files are properly installed!")
    else:
        print("\n❌ Some files are missing or corrupted.")
        print("💡 Try running this script again or reinstall Tesseract.")

if __name__ == "__main__":
    download_tessdata()