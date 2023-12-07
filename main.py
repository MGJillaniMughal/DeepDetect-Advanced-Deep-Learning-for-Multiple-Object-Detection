import cv2
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Load font
font = ImageFont.truetype("arial.ttf", 40)

# Initialize the object detection pipeline
object_detector = pipeline("object-detection")

# Draw bounding box definition
def draw_bounding_box(im, score, label, xmin, ymin, xmax, ymax, index, num_boxes):
    """ Draw a bounding box. """
    # Draw the actual bounding box
    im_with_rectangle = ImageDraw.Draw(im)  
    im_with_rectangle.rounded_rectangle((xmin, ymin, xmax, ymax), outline="red", width=5, radius=10)

    # Draw the label
    im_with_rectangle.text((xmin + 35, ymin - 25), label, fill="white", stroke_fill="red", font=font)

    # Return the intermediate result
    return im

# Function to process video from webcam
def process_webcam():
    # Start capturing from webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to PIL Image
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform object detection
        bounding_boxes = object_detector(im)

        # Draw bounding box for each result
        for bounding_box in bounding_boxes:
            # Get actual box
            box = bounding_box["box"]
            im = draw_bounding_box(im, bounding_box["score"], bounding_box["label"], box["xmin"], box["ymin"], box["xmax"], box["ymax"], 0, len(bounding_boxes))

        # Convert back to OpenCV format
        processed_frame = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

        # Display the frame
        cv2.imshow('Object Detection', processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything
    cap.release()
    cv2.destroyAllWindows()

# Process video from the webcam
process_webcam()
