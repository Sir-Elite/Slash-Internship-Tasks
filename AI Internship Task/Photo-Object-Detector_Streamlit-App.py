import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # Replace with your model path

def detect_objects(uploaded_image):
    """Detects objects in the uploaded image and returns a list of object names."""
    if uploaded_image is not None:
            image_bytes = uploaded_image.read()

        # try:
            # Open image using Pillow (PIL Fork)
            img = Image.open(io.BytesIO(image_bytes))

            # Resize image to a suitable size for the model (640x640)
            img = img.resize((640, 640), Image.BICUBIC)

            # Convert image to RGB format (if necessary)
            img = img.convert('RGB') if img.mode != 'RGB' else img

            # Convert image to a byte array for the model
            image_bytes = io.BytesIO()
            img.save(image_bytes, format='JPEG')
            image_bytes = image_bytes.getvalue()

            results = model(image_bytes)  # Perform object detection

            detected_objects = []
            for result in results.pandas().xyxy[0]:  # Extract object information from results
                confidence = result["confidence"]
                if confidence > 0.5:  # Set a minimum confidence threshold (optional)
                    name = result["name"]
                    detected_objects.append(name)

            return detected_objects
        # except TypeError:
        #     st.error("Unsupported image format. Please upload a jpg or png image.")
        #     return []
    else:
        return []

st.write("""
# Photo-Object-Detector-App

This app detects the objects in the photo you upload.

Format Supported: ".jpg", ".png"
""")

uploaded_image = st.file_uploader("Choose an image:", type="jpg")


if uploaded_image is not None:
    # Display uploaded image (optional)
    st.image(uploaded_image)

    if st.button("Analyse Image"):
        detected_objects = detect_objects(uploaded_image)

        if detected_objects:
            st.write("The Objects detected in the image are:")
            for obj in detected_objects:
                st.write(f"- {obj}")
        else:
            st.write("No objects detected with confidence above the threshold.")
    else:
        st.write("Please upload an image and click the Analyse Image button.")