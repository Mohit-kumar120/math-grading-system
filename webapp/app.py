# webapp/app.py
import streamlit as st
import numpy as np
from PIL import Image
import os
import sys
import tempfile

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.recognition.pipeline import MathRecognitionPipeline
    pipeline_available = True
except ImportError as e:
    st.error(f"Import error: {e}")
    pipeline_available = False

class GradingWebApp:
    def __init__(self):
        self.pipeline_ready = pipeline_available
        if self.pipeline_ready:
            try:
                self.pipeline = MathRecognitionPipeline()
            except Exception as e:
                st.error(f"Pipeline initialization error: {e}")
                self.pipeline_ready = False
        self.setup_ui()
    
    def setup_ui(self):
        """Setup Streamlit user interface"""
        st.set_page_config(page_title="Math Grading System", layout="wide")
        
        st.title("📝 Instant Grading for Handwritten Math Answers")
        st.markdown("Upload scanned handwritten math answers for automatic grading")
        
        if not self.pipeline_ready:
            st.warning("""
            ⚠️ **Demo Mode**: Some advanced features are not available. 
            The system will use simplified text recognition for demonstration.
            """)
        
        # Create tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["Grade Answer", "Test Images", "About & Setup"])
        
        with tab1:
            self.single_problem_interface()
        
        with tab2:
            self.test_images_interface()
        
        with tab3:
            self.about_interface()
    
    def single_problem_interface(self):
        """Interface for grading single problems"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Student Answer")
            uploaded_file = st.file_uploader("Choose an image...", 
                                           type=['jpg', 'jpeg', 'png'],
                                           key="file_uploader")
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                
                # Convert RGBA to RGB if necessary
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                
                st.image(image, caption='Uploaded Answer', width='stretch')
                
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name)
                    self.uploaded_file_path = tmp_file.name
                
                # Show basic image info
                with st.expander("Image Information"):
                    st.write(f"Format: {image.format}")
                    st.write(f"Size: {image.size} pixels")
                    st.write(f"Mode: {image.mode}")
        
        with col2:
            st.subheader("Grading Setup")
            correct_answer = st.text_input("Enter correct answer (e.g., 'x+2' or '42' or '2+2=4'):", 
                                         value="42")
            
            problem_type = st.selectbox("Problem Type", 
                                      ["Auto-detect", "Numerical", "Expression", "Equation"])
            
            if st.button("🎯 Grade Answer", type="primary") and uploaded_file and correct_answer:
                if not self.pipeline_ready:
                    # Demo mode - simulate grading
                    self.demo_grading(correct_answer)
                else:
                    with st.spinner("Analyzing handwriting and grading..."):
                        try:
                            # Convert problem type for the pipeline
                            prob_type = "auto"
                            if problem_type == "Numerical":
                                prob_type = "numerical"
                            elif problem_type == "Expression":
                                prob_type = "expression"
                            elif problem_type == "Equation":
                                prob_type = "equation"
                            
                            result = self.pipeline.grade_handwritten_answer(
                                self.uploaded_file_path, correct_answer
                            )
                            self.display_results(result)
                        except Exception as e:
                            st.error(f"Error during grading: {e}")
                            st.info("Trying demo mode instead...")
                            self.demo_grading(correct_answer)
    
    def demo_grading(self, correct_answer):
        """Demo grading when pipeline is not available"""
        import random
        # More realistic demo responses
        demo_responses = [
            ("42", 0.8, "numerical"),
            ("x+2", 0.7, "expression"),
            ("15", 0.9, "numerical"), 
            ("3+4=7", 0.6, "equation"),
            ("2x+1", 0.5, "expression")
        ]
        
        recognized_text, confidence, prob_type = random.choice(demo_responses)
        
        # Simple comparison for demo
        is_correct = recognized_text == correct_answer
        score = 1.0 if is_correct else 0.0
        
        result = {
            'score': score,
            'is_correct': is_correct,
            'feedback': "Demo: This is a simulated result. Install dependencies for full functionality.",
            'student_answer': recognized_text,
            'correct_answer': correct_answer,
            'recognition_confidence': confidence,
            'recognized_text': recognized_text,
            'problem_type': prob_type
        }
        
        self.display_results(result)
        
        st.info("""
        💡 **This is a demo result**. For full functionality:
        - Install OpenCV: `pip install opencv-python`
        - Install pytesseract: `pip install pytesseract`
        - Install Tesseract OCR on your system
        """)
    
    def display_results(self, result):
        """Display grading results"""
        st.subheader("📊 Grading Results")
        
        # Main result card
        if result['is_correct']:
            st.success("🎉 ✅ Correct Answer!")
        else:
            st.error("❌ Incorrect Answer")
        
        # Metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Score", f"{result['score']:.1f}/1.0")
        
        with col2:
            confidence_color = "green" if result['recognition_confidence'] > 0.7 else "orange"
            st.metric("Recognition Confidence", 
                     f"{result['recognition_confidence']:.1%}")
        
        with col3:
            st.metric("Problem Type", result.get('problem_type', 'Unknown'))
        
        # Detailed information
        with st.expander("Detailed Analysis"):
            st.write(f"**Recognized Text:** `{result['recognized_text']}`")
            st.write(f"**Expected Answer:** `{result['correct_answer']}`")
            st.write(f"**Feedback:** {result['feedback']}")
            
            if not self.pipeline_ready:
                st.warning("🔧 **Demo Mode**: Results are simulated for demonstration.")
    
    def test_images_interface(self):
        """Interface for testing with sample images"""
        st.header("🧪 Test with Sample Images")
        st.markdown("Try the system with these pre-generated test images:")
        
        # Check if test images exist
        test_folders = ['test_images', 'test_images_ocr', 'training_images']
        available_images = []
        
        for folder in test_folders:
            if os.path.exists(folder):
                images = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
                available_images.extend([(folder, f) for f in images])
        
        if not available_images:
            st.error("No test images found. Please run the test image generators first.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_folder_image = st.selectbox("Choose a test image:", 
                                               [f"{folder}/{img}" for folder, img in available_images])
            if selected_folder_image:
                folder, image_name = selected_folder_image.split('/')
                image_path = os.path.join(folder, image_name)
                try:
                    image = Image.open(image_path)
                    st.image(image, caption=f"Test Image: {image_name}", width='stretch')
                except Exception as e:
                    st.error(f"Could not load image: {e}")
        
        with col2:
            # Suggest correct answer based on filename
            suggested_answer = image_name.replace('test_', '').replace('ocr_', '').replace('clear_equals_', '').replace('double_equals_', '').replace('equals_', '').replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
            correct_answer = st.text_input("Correct answer for this test:", 
                                         value=suggested_answer,
                                         key="test_answer")
            
            if st.button("Grade Test Image", type="secondary"):
                if self.pipeline_ready:
                    with st.spinner("Grading test image..."):
                        result = self.pipeline.grade_handwritten_answer(
                            image_path, correct_answer
                        )
                        self.display_results(result)
                else:
                    self.demo_grading(correct_answer)
    
    def about_interface(self):
        """About and setup instructions section"""
        st.header("ℹ️ About & Setup Instructions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("About This Project")
            st.markdown("""
            **Instant Grading for Handwritten Math Answers**
            
            This system automatically grades handwritten mathematical answers using computer vision.
            
            **Current Features:**
            - Image upload and preprocessing
            - Text recognition (digits and math symbols)
            - Mathematical expression comparison
            - Support for equations and numerical answers
            - Instant grading with confidence scores
            
            **Current Mode:** {'Full Version' if self.pipeline_ready else 'Demo Version'}
            """)
        
        with col2:
            st.subheader("Setup Instructions")
            st.markdown("""
            **For full functionality:**
            
            ```bash
            # Install OpenCV for image processing
            pip install opencv-python
            
            # Install pytesseract for OCR
            pip install pytesseract
            
            # Install Tesseract OCR engine
            # Windows: Download from UB-Mannheim/tesseract on GitHub
            # Mac: brew install tesseract  
            # Linux: sudo apt-get install tesseract-ocr
            ```
            """)
        
        st.markdown("---")
        st.subheader("Supported Math Formats")
        st.markdown("""
        - **Numerical**: `42`, `15.5`, `-3.14`
        - **Expressions**: `x+2`, `2*x+1`, `(x+1)^2`
        - **Equations**: `2+2=4`, `x+1=3`, `y=2x+1`
        - **Basic operations**: `+`, `-`, `*`, `/`
        """)

def main():
    app = GradingWebApp()

if __name__ == "__main__":
    main()