import cv2
import numpy as np
import pytesseract
from keras.src.saving.saving_api import load_model  # Ensure the correct import for Keras models
from ultralytics import YOLO

# Configure Tesseract executable path (update it to the location of your tesseract executable)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # For Windows

# Load YOLOv8 models
object_model = YOLO("yolov8s.pt")  # Replace with the path to your YOLOv8 weights file for general objects
logo_model = YOLO("best (4).pt")  # Replace with the path to your YOLOv8 weights file for logos

# Load class names (directly from the models if available)
object_class_names = object_model.names
logo_class_names = logo_model.names

# Load freshness detection model
fresh_model = load_model('rottenvsfresh.h5')

# Function to classify fresh/rotten
def print_fresh(res):
    threshold_fresh = 0.10  # set according to standards
    threshold_medium = 0.35  # set according to standards
    if res < threshold_fresh:
        return "FRESH"
    elif threshold_fresh < res < threshold_medium:
        return "MEDIUM FRESH"
    else:
        return "NOT FRESH"

# Preprocess image for freshness detection
def pre_proc_img(image):
    img = cv2.resize(image, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Evaluate freshness
def evaluate_rotten_vs_fresh(image):
    prediction = fresh_model.predict(pre_proc_img(image))
    return prediction[0][0]

# Draw bounding boxes and labels for objects and logos
def draw_boxes(frame, results, class_names, color):
    h, w, _ = frame.shape
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        confidences = result.boxes.conf.cpu().numpy()  # Get confidences
        class_ids = result.boxes.cls.cpu().numpy()  # Get class IDs
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            if confidence > 0.5:  # Adjust confidence threshold as needed
                x1, y1, x2, y2 = map(int, box)
                label = f'{class_names[int(class_id)]}: {confidence:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Freshness detection for specific classes
                if class_names[int(class_id)] in ['apple', 'banana', 'orange']:  # Add classes as needed
                    crop_img = frame[y1:y2, x1:x2]
                    freshness_score = evaluate_rotten_vs_fresh(crop_img)
                    freshness_label = print_fresh(freshness_score)
                    cv2.putText(frame, f"{freshness_label}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
    return frame

# Perform OCR
def perform_ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray).strip()
    return text

# Overlay detected text on the frame
def overlay_text(frame, text):
    # Add a background rectangle for the text
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.rectangle(frame, (10, 30 - text_height - baseline), (10 + text_width, 30 + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return frame

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        object_results = object_model(frame)

        # Draw bounding boxes and labels for objects
        frame = draw_boxes(frame, object_results, object_class_names, (0, 255, 0))

        # Perform logo detection
        logo_results = logo_model(frame)

        # Draw bounding boxes and labels for logos
        frame = draw_boxes(frame, logo_results, logo_class_names, (255, 0, 0))

        # Perform OCR
        text = perform_ocr(frame)

        # Overlay the detected text on the frame
        frame = overlay_text(frame, text)

        # Print detected text to the terminal
        if text:
            print("Detected text:", text)

        # Show the frame with detected objects, logos, and OCR text
        cv2.imshow('YOLOv8 Object, Logo Detection, Freshness Evaluation and OCR', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
