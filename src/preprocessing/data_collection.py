# src/preprocessing/data_collection.py
import cv2
import os
import numpy as np
from PIL import Image

class DataCollector:
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.tiff']
    
    def validate_image(self, image_path):
        """Validate uploaded image format and quality"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, "Cannot read image"
            
            # Check image dimensions
            height, width = img.shape[:2]
            if height < 100 or width < 100:
                return False, "Image too small"
                
            return True, "Valid image"
        except Exception as e:
            return False, str(e)
    
    def standardize_image(self, image_path, output_path):
        """Standardize image format and size"""
        img = cv2.imread(image_path)
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard dimensions
        img = cv2.resize(img, (800, 600))
        
        # Enhance contrast
        img = cv2.equalizeHist(img)
        
        cv2.imwrite(output_path, img)
        return img