# src/preprocessing/image_processing.py
import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self):
        self.kernel = np.ones((2,2), np.uint8)
    
    def preprocess_image(self, image):
        """Main preprocessing pipeline"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Noise removal
        denoised = cv2.medianBlur(gray, 5)
        
        # Thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, 
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean image
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, self.kernel)
        
        return cleaned
    
    def segment_characters(self, preprocessed_image):
        """Segment individual characters/digits"""
        # Find contours
        contours, _ = cv2.findContours(preprocessed_image, 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        characters = []
        bounding_boxes = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out very small contours (noise)
            if w > 10 and h > 15 and w < 100 and h < 100:
                # Extract character
                char_img = preprocessed_image[y:y+h, x:x+w]
                
                # Resize to standard size
                char_img = cv2.resize(char_img, (28, 28))
                
                characters.append(char_img)
                bounding_boxes.append((x, y, w, h))
        
        return characters, bounding_boxes