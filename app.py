
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="ExplainDR",
    page_icon="👁️",
    layout="wide"
)

# -----------------------------
# Model path
# -----------------------------
MODEL_PATH = "diabetic_retinopathy_model.keras"

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------
# Grad-CAM model
# -----------------------------
rescaling_layer = model.layers[0]
base_model = model.layers[1]

grad_input = tf.keras.Input(shape=(224, 224, 3))

x = rescaling_layer(grad_input)
feature_map = base_model(x, training=False)

x = model.layers[2](feature_map)
x = model.layers[3](x)
prediction = model.layers[4](x)

grad_model = tf.keras.Model(
    inputs=grad_input,
    outputs=[feature_map, prediction]
)

# -----------------------------
# Image quality
# -----------------------------
def check_image_quality(image):
    image_array = np.array(image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return sharpness

# -----------------------------
# Prediction
# -----------------------------
def predict_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(
        img_array,
        verbose=0
    )[0][0]

    if prediction >= 0.5:
        result = "DR"
        confidence = prediction * 100
    else:
        result = "No DR"
        confidence = (1 - prediction) * 100

    return result, confidence, prediction

# -----------------------------
# Grad-CAM
# -----------------------------
def generate_gradcam(image):

    img = image.resize((224, 224))
    original = np.array(img).astype("uint8")

    img_array = np.expand_dims(
        original.astype("float32"),
        axis=0
    )

    with tf.GradientTape() as tape:

        feature_maps, predictions = grad_model(
            img_array
        )

        dr_score = predictions[:, 0]

    gradients = tape.gradient(
        dr_score,
        feature_maps
    )

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    cam = tf.reduce_sum(
        feature_maps *
        weights[:, tf.newaxis, tf.newaxis, :],
        axis=-1
    )

    cam = tf.maximum(cam, 0)

    cam = cam / (
        tf.reduce_max(cam) + 1e-8
    )

    heatmap = cam[0].numpy()

    heatmap = cv2.resize(
        heatmap,
        (224, 224)
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return original, heatmap, overlay

# -----------------------------
# UI
# -----------------------------
st.title("👁️ ExplainDR")

st.write(
    "Explainable AI based diabetic retinopathy "
    "screening support system."
)

uploaded_file = st.file_uploader(
    "Upload a retinal fundus image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("🖼️ Uploaded Retina Image")

    st.image(
        image,
        width=500
    )

    # -----------------------------
    # Image quality
    # -----------------------------
    sharpness = check_image_quality(image)

    st.subheader("📷 Image Quality")

    if sharpness < 50:
        st.warning(
            f"Image may be blurry. "
            f"Sharpness score: {sharpness:.1f}"
        )
    else:
        st.success(
            f"Image quality acceptable. "
            f"Sharpness score: {sharpness:.1f}"
        )

    # -----------------------------
    # Prediction
    # -----------------------------
    result, confidence, dr_probability = predict_image(
        image
    )

    st.subheader("🔬 AI Screening Result")

    if result == "DR":
        st.error(
            f"Result: {result}"
        )
    else:
        st.success(
            f"Result: {result}"
        )

    st.metric(
        "Model Confidence",
        f"{confidence:.1f}%"
    )

    # -----------------------------
    # Summary
    # -----------------------------
    st.subheader("📋 Screening Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Result",
            result
        )

    with col2:
        st.metric(
            "Confidence",
            f"{confidence:.1f}%"
        )

    with col3:
        st.metric(
            "Sharpness",
            f"{sharpness:.1f}"
        )

    # -----------------------------
    # Explanation
    # -----------------------------
    st.subheader(
        "🩺 What does this result mean?"
    )

    if result == "DR":

        st.write(
            "The AI model detected image patterns "
            "associated with diabetic retinopathy. "
            "This result is intended as screening "
            "support and should be reviewed by a "
            "qualified eye-care professional."
        )

    else:

        st.write(
            "The AI model did not detect strong "
            "image patterns associated with "
            "diabetic retinopathy. This is a "
            "screening result and not a medical "
            "diagnosis."
        )

    # -----------------------------
    # Grad-CAM
    # -----------------------------
    st.subheader(
        "🔍 Grad-CAM Explanation"
    )

    original, heatmap, overlay = generate_gradcam(
        image
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            original,
            caption="Original Image"
        )

    with col2:
        st.image(
            heatmap,
            caption="Grad-CAM Heatmap"
        )

    with col3:
        st.image(
            overlay,
            caption="Grad-CAM Overlay"
        )

    st.caption(
        "Red/yellow highlighted regions indicate "
        "areas that contributed more strongly to "
        "the model's DR score. The heatmap is an "
        "AI explanation and is not proof of a "
        "lesion or a medical diagnosis."
    )

    # -----------------------------
    # Model performance
    # -----------------------------
    st.subheader(
        "📊 Model Performance"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Accuracy", "95.82%")
    c2.metric("Precision", "96.63%")
    c3.metric("Sensitivity", "95.03%")
    c4.metric("Specificity", "96.63%")
    c5.metric("F1 Score", "95.82%")

    st.caption(
        "Metrics shown are from the experimental "
        "validation split used during development."
    )

    # -----------------------------
    # Download report
    # -----------------------------
    report = f"""
ExplainDR - Screening Report
=============================

Result: {result}
Confidence: {confidence:.1f}%
DR Probability: {dr_probability * 100:.1f}%
Image Sharpness: {sharpness:.1f}

Model Performance
-----------------
Accuracy: 95.82%
Precision: 96.63%
Sensitivity: 95.03%
Specificity: 96.63%
F1 Score: 95.82%

Note:
This report is generated by an AI screening
prototype for educational and screening support.
It is not a medical diagnosis.
"""

    st.subheader(
        "📄 Download Screening Report"
    )

    st.download_button(
        label="Download Report",
        data=report,
        file_name="ExplainDR_screening_report.txt",
        mime="text/plain"
    )
