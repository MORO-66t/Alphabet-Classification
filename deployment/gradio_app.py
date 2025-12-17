"""
Gradio deployment interface for EMNIST Alphabet Classification.
"""

import gradio as gr
import tensorflow as tf
import cv2
import numpy as np


# Model and class configuration
MODEL_PATH = '../saved_models/vgg19/final_emnist_vgg.h5'
CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e',
               'f', 'g', 'h', 'n', 'q', 'r', 't']


def load_model(model_path=MODEL_PATH):
    """Load the trained model."""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        return None


def preprocess_image(image):
    """
    Preprocess the drawn image for model prediction.
    
    Args:
        image: numpy array from Gradio Sketchpad
    
    Returns:
        numpy array: preprocessed image (1, 28, 28, 1)
    """
    if image is None:
        return None
    
    # Handle dictionary format (from Sketchpad)
    if isinstance(image, dict):
        image = image['composite']
    
    image = np.array(image)
    
    # Handle RGBA format (remove alpha channel)
    if len(image.shape) == 3 and image.shape[2] == 4:
        alpha_channel = image[:, :, 3]
        rgb_channels = image[:, :, :3]
        
        # Create white background
        white_background = np.ones_like(rgb_channels, dtype=np.uint8) * 255
        
        # Blend with alpha
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
        image: numpy array from Gradio Sketchpad
        model: loaded Keras model
    
    Returns:
        dict: {class_name: confidence} for all classes
    """
    if image is None or model is None:
        return None
    
    # Preprocess image
    processed_image = preprocess_image(image)
    
    if processed_image is None:
        return None
    
    # Make prediction
    prediction = model.predict(processed_image, verbose=0)[0]
    
    # Create confidence dictionary
    confidences = {CLASS_NAMES[i]: float(prediction[i]) for i in range(len(CLASS_NAMES))}
    
    return confidences


def create_gradio_interface(model):
    """
    Create and return Gradio interface.
    
    Args:
        model: loaded Keras model
    
    Returns:
        gr.Interface: Gradio interface object
    """
    def predict_fn(image):
        return predict_character(image, model)
    
    interface = gr.Interface(
        fn=predict_fn,
        inputs=gr.Sketchpad(
            label="Draw your character here (digit or letter)", 
            type="numpy",
            image_mode="RGB"
        ),
        outputs=gr.Label(
            num_top_classes=5,
            label="Predictions (Top 5)"
        ),
        live=True,
        title="🔠 EMNIST Alphabet Classifier",
        description="""
        Draw any digit (0-9) or letter (A-Z, a-z) and see the model's predictions in real-time.
        
        **Supported Characters:** 47 classes (digits + uppercase + some lowercase letters)
        
        **Model:** VGG-19 architecture trained on EMNIST Balanced dataset
        """,
        examples=None,
        theme="default"
    )
    
    return interface


def main():
    """Main function to launch Gradio app."""
    # Load model
    model = load_model()
    
    if model is None:
        print("❌ Failed to load model. Please check the model path.")
        return
    
    # Create interface
    interface = create_gradio_interface(model)
    
    # Launch
    print("\n🚀 Launching Gradio interface...")
    interface.launch(
        share=True,
        debug=True,
        server_name="0.0.0.0",
        server_port=7860
    )


if __name__ == "__main__":
    main()
