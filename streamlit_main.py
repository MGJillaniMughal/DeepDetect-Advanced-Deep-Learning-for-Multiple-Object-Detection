import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

from transformers import DetrFeatureExtractor, DetrForObjectDetection


# Initialize the object detection pipeline
# Initialize another model and feature extractor
feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101')

# Initialize the object detection pipeline
object_detector = pipeline("object-detection", model = model, feature_extractor = feature_extractor)

# Function to draw bounding boxes
def draw_bounding_box(im, bounding_boxes, scale_factor=0.8):
    for bounding_box in bounding_boxes:
        box = bounding_box["box"]
        xmin, ymin, xmax, ymax = [int(coord * scale_factor) for coord in [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]]

        # Draw the actual bounding box
        im_with_rectangle = ImageDraw.Draw(im)  
        im_with_rectangle.rounded_rectangle((xmin, ymin, xmax, ymax), outline="red", width=5, radius=10)

        # Draw the label
        im_with_rectangle.text((xmin + 35, ymin - 25), bounding_box["label"], fill="white", stroke_fill="red", font=ImageFont.truetype("arial.ttf", 40))

    return im

# Streamlit page configuration
st.title("Object Detection App")
st.write("This app uses a machine learning model to detect objects in images.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file).convert("RGB")

    # Perform object detection
    bounding_boxes = object_detector(image)

    # Draw bounding boxes
    image_with_boxes = draw_bounding_box(image, bounding_boxes)

    # Display the image
    st.image(image_with_boxes, caption="Processed Image", use_column_width=True)
