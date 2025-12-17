"""
Streamlit deployment interface for EMNIST Alphabet Classification.
"""

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# Model and class configuration
MODEL_PATH = '../saved_models/vgg19/final_emnist_vgg.h5'
CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e',
               'f', 'g', 'h', 'n', 'q', 'r', 't']


@st.cache_resource
def load_model(model_path=MODEL_PATH):
    """Load and cache the trained model."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_image(image):
    """
    Preprocess the drawn image for model prediction.
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        numpy array: preprocessed image (1, 28, 28, 1)
    """
    if image is None:
        return None
    
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Handle RGBA format
    if len(image.shape) == 3 and image.shape[2] == 4:
        alpha_channel = image[:, :, 3]
        rgb_channels = image[:, :, :3]
        
        white_background = np.ones_like(rgb_channels, dtype=np.uint8) * 255
        alpha_factor = alpha_channel[:, :, np.newaxis] / 255.0
        image = rgb_channels * alpha_factor + white_background * (1 - alpha_factor)
        image = image.astype(np.uint8)
    
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to 28x28
    image = cv2.resize(image, (28, 28))
    
    # Invert if background is white
    if np.mean(image) > 127:
        image = 255 - image
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    # Reshape for model
    image = image.reshape(1, 28, 28, 1)
    
    return image


def predict_character(image, model):
    """
    Predict the character from the drawn image.
    
    Args:
        image: numpy array or PIL Image
        model: loaded Keras model
    
    Returns:
        tuple: (predicted_class, confidence, all_predictions)
    """
    if image is None or model is None:
        return None, None, None
    
    # Preprocess image
    processed_image = preprocess_image(image)
    
    if processed_image is None:
        return None, None, None
    
    # Make prediction
    prediction = model.predict(processed_image, verbose=0)[0]
    
    # Get top prediction
    predicted_idx = np.argmax(prediction)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = prediction[predicted_idx]
    
    # Get top 5 predictions
    top_5_idx = np.argsort(prediction)[-5:][::-1]
    top_5_predictions = [(CLASS_NAMES[i], prediction[i]) for i in top_5_idx]
    
    return predicted_class, confidence, top_5_predictions


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="EMNIST Classifier",
        page_icon="🔠",
        layout="wide"
    )
    
    st.title("🔠 EMNIST Alphabet Classifier")
    st.markdown("""
    Draw any digit (0-9) or letter (A-Z, a-z) and see the model's predictions in real-time.
    
    **Supported Characters:** 47 classes (digits + uppercase + some lowercase letters)  
    **Model:** VGG-19 architecture trained on EMNIST Balanced dataset
    """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check the model path.")
        st.stop()
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Draw Here")
        
        # Canvas for drawing
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=20,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        if st.button("Clear Canvas"):
            st.rerun()
    
    with col2:
        st.subheader("Predictions")
        
        if canvas_result.image_data is not None:
            # Check if something is drawn
            if np.sum(canvas_result.image_data[:, :, 3]) > 0:
                # Make prediction
                predicted_class, confidence, top_5 = predict_character(
                    canvas_result.image_data, 
                    model
                )
                
                if predicted_class is not None:
                    # Display top prediction
                    st.markdown(f"### Predicted: **{predicted_class}**")
                    st.progress(float(confidence))
                    st.markdown(f"Confidence: **{confidence:.2%}**")
                    
                    # Display top 5 predictions
                    st.markdown("#### Top 5 Predictions:")
                    for i, (char, conf) in enumerate(top_5, 1):
                        st.markdown(f"{i}. **{char}** - {conf:.2%}")
            else:
                st.info("Draw something on the canvas!")
        else:
            st.info("Waiting for input...")
    
    # Sidebar with additional info
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app uses a deep learning model to classify handwritten characters.
        
        **Dataset:** EMNIST Balanced  
        **Classes:** 47 (digits 0-9 + letters)  
        **Model:** Custom VGG-19  
        **Accuracy:** ~90% on test set
        
        ---
        
        **How to use:**
        1. Draw a character on the canvas
        2. View real-time predictions
        3. Click "Clear Canvas" to try again
        """)
        
        st.header("Model Info")
        st.json({
            "Architecture": "VGG-19",
            "Input Size": "28x28x1",
            "Classes": len(CLASS_NAMES),
            "Parameters": "~2M"
        })


if __name__ == "__main__":
    main()
