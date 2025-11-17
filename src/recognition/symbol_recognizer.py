# src/recognition/symbol_recognizer.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
from PIL import Image
import os

class SymbolRecognizer:
    """
    Recognizes mathematical symbols using CNN
    Supports: +, -, ×, ÷, =, (, ), [, ], {, }
    """
    
    def __init__(self):
        self.model = None
        self.img_height = 32
        self.img_width = 32
        self.symbol_classes = {
            0: '+',    # Addition
            1: '-',    # Subtraction/Minus
            2: '×',    # Multiplication
            3: '÷',    # Division
            4: '=',    # Equals
            5: '(',    # Left parenthesis
            6: ')',    # Right parenthesis
            7: '[',    # Left bracket
            8: ']',    # Right bracket
            9: '{',    # Left brace
            10: '}',   # Right brace
            11: 'x',   # Variable x
            12: 'y',   # Variable y
        }
        self.class_names = list(self.symbol_classes.values())
    
    def build_model(self):
        """Build CNN model for symbol recognition"""
        self.model = keras.Sequential([
            # First convolutional layer
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(self.img_height, self.img_width, 1)),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional layer
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional layer
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Fully connected layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.symbol_classes), activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Symbol recognition model built for {len(self.symbol_classes)} classes")
        return self.model
    
    def preprocess_symbol(self, symbol_image):
        """Preprocess individual symbol image for model input"""
        # Ensure image is grayscale
        if len(symbol_image.shape) == 3:
            symbol_image = cv2.cvtColor(symbol_image, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        symbol_image = cv2.resize(symbol_image, (self.img_width, self.img_height))
        
        # Normalize pixel values
        symbol_image = symbol_image.astype('float32') / 255.0
        
        # Add channel dimension
        symbol_image = symbol_image.reshape(self.img_height, self.img_width, 1)
        
        return symbol_image
    
    def predict_symbol(self, symbol_image):
        """Predict symbol from image"""
        try:
            # Preprocess the image
            processed_image = self.preprocess_symbol(symbol_image)
            
            # Add batch dimension
            processed_image = np.expand_dims(processed_image, axis=0)
            
            # Make prediction
            prediction = self.model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            symbol = self.symbol_classes.get(predicted_class, '?')
            
            return symbol, confidence
            
        except Exception as e:
            print(f"❌ Symbol prediction error: {e}")
            return '?', 0.0
    
    def train(self, x_train, y_train, x_val, y_val, epochs=50):
        """Train the symbol recognition model"""
        print(f"🎯 Training symbol recognizer on {len(x_train)} samples...")
        
        # Add early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("✅ Symbol recognition model training completed")
        return history
    
    def save_model(self, filepath='models/symbol_recognizer.h5'):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"💾 Symbol model saved to: {filepath}")
    
    def load_model(self, filepath='models/symbol_recognizer.h5'):
        """Load pre-trained model"""
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            print(f"📂 Symbol model loaded from: {filepath}")
            return True
        else:
            print(f"❌ Symbol model not found at: {filepath}")
            return False
    
    def evaluate_model(self, x_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            print("❌ Model not loaded or built")
            return
        
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"📊 Symbol Recognition - Test Accuracy: {test_accuracy:.2%}")
        return test_accuracy


class SymbolSegmenter:
    """
    Segments mathematical expressions into individual symbols
    """
    
    def __init__(self):
        self.min_symbol_width = 10
        self.min_symbol_height = 15
        self.max_symbol_width = 80
        self.max_symbol_height = 80
    
    def segment_expression(self, preprocessed_image):
        """
        Segment a mathematical expression image into individual symbols
        Returns list of (symbol_image, bounding_box) tuples
        """
        # Find contours (connected components)
        contours, _ = cv2.findContours(
            preprocessed_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        symbols = []
        bounding_boxes = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out noise and very large/small regions
            if (w >= self.min_symbol_width and h >= self.min_symbol_height and
                w <= self.max_symbol_width and h <= self.max_symbol_height):
                
                # Extract symbol with some padding
                padding = 2
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(preprocessed_image.shape[1], x + w + padding)
                y_end = min(preprocessed_image.shape[0], y + h + padding)
                
                symbol_img = preprocessed_image[y_start:y_end, x_start:x_end]
                
                symbols.append(symbol_img)
                bounding_boxes.append((x, y, w, h))
        
        # Sort symbols by x-coordinate (left to right)
        sorted_symbols = sorted(zip(symbols, bounding_boxes), 
                               key=lambda item: item[1][0])
        
        if sorted_symbols:
            symbols, bounding_boxes = zip(*sorted_symbols)
            return list(symbols), list(bounding_boxes)
        else:
            return [], []
    
    def separate_digits_from_symbols(self, symbols, bounding_boxes):
        """
        Separate digit images from symbol images based on aspect ratio and features
        """
        digits = []
        digit_boxes = []
        math_symbols = []
        symbol_boxes = []
        
        for symbol, bbox in zip(symbols, bounding_boxes):
            x, y, w, h = bbox
            aspect_ratio = w / h
            
            # Digits tend to be more square, symbols more rectangular
            if 0.5 <= aspect_ratio <= 1.2 and h > 20:
                digits.append(symbol)
                digit_boxes.append(bbox)
            else:
                math_symbols.append(symbol)
                symbol_boxes.append(bbox)
        
        return digits, digit_boxes, math_symbols, symbol_boxes


class AdvancedSymbolProcessor:
    """
    Advanced symbol processing with context awareness
    """
    
    def __init__(self):
        self.symbol_recognizer = SymbolRecognizer()
        self.segmenter = SymbolSegmenter()
        
        # Common symbol confusions to handle
        self.common_confusions = {
            '1': ['l', 'I', '|'],      # 1 confused with l, I, |
            '0': ['O', 'o'],           # 0 confused with O, o
            '5': ['S', 's'],           # 5 confused with S
            '2': ['Z', 'z'],           # 2 confused with Z
            '8': ['B'],                # 8 confused with B
            '-': ['=', '_'],           # minus confused with equals
            '×': ['x', 'X'],           # multiplication confused with x
        }
    
    def process_math_expression(self, image_path):
        """
        Complete pipeline: segment expression and recognize symbols
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            return "Error: Could not load image"
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to binary
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Segment expression
        symbols, bounding_boxes = self.segmenter.segment_expression(binary)
        
        if not symbols:
            return "No symbols detected"
        
        # Recognize each symbol
        recognized_expression = []
        confidence_scores = []
        
        for i, symbol_img in enumerate(symbols):
            # Convert to positive image for display/recognition
            symbol_positive = cv2.bitwise_not(symbol_img)
            
            # Recognize symbol
            symbol, confidence = self.symbol_recognizer.predict_symbol(symbol_positive)
            recognized_expression.append(symbol)
            confidence_scores.append(confidence)
            
            print(f"🔍 Symbol {i+1}: '{symbol}' (confidence: {confidence:.2f})")
        
        # Reconstruct expression
        final_expression = ''.join(recognized_expression)
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        return {
            'expression': final_expression,
            'confidence': avg_confidence,
            'symbol_details': list(zip(recognized_expression, confidence_scores)),
            'symbol_count': len(symbols)
        }
    
    def correct_common_errors(self, recognized_text, context=None):
        """
        Correct common OCR/symbol recognition errors using context
        """
        corrected = recognized_text
        
        # Handle equals sign confusion
        if '−' in corrected or '–' in corrected or '—' in corrected:
            corrected = corrected.replace('−', '=').replace('–', '=').replace('—', '=')
        
        # Handle multiplication symbol confusion
        if any(char in corrected for char in ['x', 'X']) and '*' not in corrected:
            # If we see 'x' in algebraic context, it might be multiplication
            corrected = corrected.replace('x', '×').replace('X', '×')
        
        # Fix common number-symbol confusions
        confusion_map = {
            'l': '1', 'I': '1', '|': '1',
            'O': '0', 'o': '0',
            'S': '5', 's': '5',
            'Z': '2', 'z': '2',
            'B': '8'
        }
        
        for wrong, correct in confusion_map.items():
            corrected = corrected.replace(wrong, correct)
        
        # Context-based corrections for equations
        if context == 'equation' and '=' not in corrected:
            # Try to infer equals position
            if '+' in corrected and corrected.count('+') == 1:
                parts = corrected.split('+')
                if len(parts) == 2 and parts[1].isdigit():
                    # Common pattern: "2+24" should be "2+2=4"
                    if len(parts[1]) > 1:
                        corrected = parts[0] + '+' + parts[1][0] + '=' + parts[1][1:]
        
        return corrected


# Utility functions for training data generation
def create_symbol_training_data():
    """Create training data for mathematical symbols"""
    # This would typically generate synthetic training images
    # or load from a dataset like HASYv2 or CROHME
    
    print("📚 Creating symbol training data...")
    # Implementation would go here
    pass


def test_symbol_recognition():
    """Test the symbol recognition system"""
    processor = AdvancedSymbolProcessor()
    
    # Build or load model
    processor.symbol_recognizer.build_model()
    
    # Test with a sample (you would need actual training data)
    print("🧪 Symbol recognition system ready")
    print(f"Supported symbols: {processor.symbol_recognizer.class_names}")


if __name__ == "__main__":
    test_symbol_recognition()