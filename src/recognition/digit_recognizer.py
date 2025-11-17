# src/recognition/digit_recognizer.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class DigitRecognizer:
    def __init__(self):
        self.model = None
        self.img_height = 28
        self.img_width = 28
        
    def build_model(self):
        """Build CNN model for digit recognition"""
        self.model = keras.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', 
                         input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')  # 0-9 digits
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, x_train, y_train, x_val, y_val, epochs=20):
        """Train the digit recognition model"""
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=32
        )
        
        return history
    
    def predict_digit(self, image):
        """Predict digit from image"""
        # Preprocess for model input
        image = image.reshape(1, 28, 28, 1)
        image = image.astype('float32') / 255.0
        
        prediction = self.model.predict(image, verbose=0)
        return np.argmax(prediction), np.max(prediction)