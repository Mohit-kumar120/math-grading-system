# webapp/simple_app.py
import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import cv2

st.set_page_config(page_title="Math Grading Demo", layout="wide")

st.title("📝 Math Grading System - Demo Version")
st.markdown("""
This is a demo version of the handwritten math grading system.
For the full version, please install all dependencies and set up the models.
""")

# Demo grading function
def demo_grade_handwritten_answer(image_path, correct_answer):
    """Demo grading function - in real implementation, this would use ML models"""
    # This is a simplified demo - in practice, you'd use the full pipeline
    return {
        'score': 1.0,  # Demo always returns correct
        'is_correct': True,
        'feedback': "Demo: Answer correct",
        'student_answer': "42",  # Demo recognized text
        'correct_answer': correct_answer,
        'recognition_confidence': 0.85,
        'recognized_text': "42"
    }

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Student Answer")
    uploaded_file = st.file_uploader("Choose an image...", 
                                   type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Answer', use_column_width=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image.save(tmp_file.name)
            uploaded_file_path = tmp_file.name

with col2:
    st.subheader("Correct Answer")
    correct_answer = st.text_input("Enter correct answer:", "42")
    
    if st.button("Grade Answer (Demo)") and uploaded_file:
        with st.spinner("Grading answer..."):
            result = demo_grade_handwritten_answer(uploaded_file_path, correct_answer)
        
        st.subheader("Demo Results")
        st.success("✅ Correct! (Demo)")
        st.metric("Score", "1.0/1.0")
        st.write(f"Recognized: `{result['recognized_text']}`")
        st.info("Note: This is a demo. Full version would analyze handwriting.")

st.markdown("---")
st.info("To use the full version, please ensure all dependencies are installed and models are trained.")