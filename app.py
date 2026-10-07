import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Cloud YOLO Object Detection",
    page_icon="🔍",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    font-size: 16px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">Cloud-Enabled Intelligent Object Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Intelligent image analysis powered by YOLO</div>',
    unsafe_allow_html=True
)

st.divider()

# Sidebar
with st.sidebar:
    st.header("About the Project")
    st.write(
        "This application uses a pre-trained YOLO model "
        "to identify objects in uploaded images."
    )

    st.subheader("Features")
    st.write("📤 Image upload")
    st.write("🎯 Object detection")
    st.write("📦 Object counting")
    st.write("📊 Confidence scores")

    st.info("Upload an image to get started.")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# Image uploader
st.subheader("Upload Your Image")
confidence_threshold = st.slider(
    "Detection Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.1
)
uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, or PNG image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detection Results")

        if st.button("🔍 Detect Objects", type="primary"):

            with st.spinner("Analyzing image..."):
                results = model(
                    np.array(image),
                    conf=confidence_threshold
                )
                result = results[0]
                detected_image = result.plot()

            st.image(
                detected_image,
                channels="BGR",
                use_container_width=True
            )

            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:

                object_names = []
                confidence_scores = []

                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    object_names.append(result.names[class_id])
                    confidence_scores.append(round(confidence * 100, 2))

                object_counts = pd.Series(object_names).value_counts()

                st.divider()
                st.subheader("Detection Summary")

                metric1, metric2 = st.columns(2)

                metric1.metric(
                    "Total Objects",
                    len(object_names)
                )

                metric2.metric(
                    "Object Categories",
                    len(object_counts)
                )

                st.subheader("Object Counts")

                count_df = object_counts.rename_axis(
                    "Object"
                ).reset_index(name="Count")

                st.dataframe(
                    count_df,
                    hide_index=True,
                    use_container_width=True
                )

                st.subheader("Confidence Scores")

                confidence_df = pd.DataFrame({
                    "Object": object_names,
                    "Confidence (%)": confidence_scores
                })

                st.dataframe(
                    confidence_df,
                    hide_index=True,
                    use_container_width=True
                )

            else:
                st.warning("No objects were detected in this image.")

else:
    st.info("Please upload an image to begin object detection.")

st.divider()

st.caption(
    "Cloud Computing Mini Project | YOLO-Based Object Detection"
)