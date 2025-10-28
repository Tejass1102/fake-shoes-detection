import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ------------------------------
# Load the trained model
# ------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("shoe_model_mobilenet_v2_final.keras")
    return model

model = load_model()

# ------------------------------
# Streamlit Page Setup
# ------------------------------
st.set_page_config(page_title="Fake Shoe Detector", layout="centered")
st.title("👟 Fake Shoe Detection App")
st.write("Upload an image of a shoe to predict whether it is **Real** or **Fake**.")

# ------------------------------
# Image Upload + Prediction Section
# ------------------------------
uploaded_file = st.file_uploader("Choose a shoe image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Step 1: Load and display
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Shoe Image", use_container_width=True)

    # Step 2: Show model input shape (for debugging)
    st.write("Model Input Shape:", model.input_shape)

    # Step 3: Resize to match model input
    target_size = model.input_shape[1:3]
    img = img.resize(target_size)

    # Step 4: Convert and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Step 5: Predict
    prediction = model.predict(img_array)
    score = float(prediction[0][0])

    # Step 6: Interpret result
    if score > 0.6:
        st.success(f"✅ Real Shoe ({score*100:.2f}%)")
    else:
        st.error(f"❌ Fake Shoe ({(1-score)*100:.2f}%)")
