import cv2
import pytesseract
from ultralytics import YOLO

# Configure Tesseract executable path (update it to the location of your tesseract executable)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # For Windows

# Load YOLOv8 model
model = YOLO('yolov8s.pt')  # Use yolov8n.pt or any other YOLOv8 model you prefer
# Load class names
def load_class_names(names_file):
    with open(names_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Path to the class names file
class_names_file = 'coco.names'
class_names = load_class_names(class_names_file)

def detect_objects(frame):
    # Perform object detection
    results = model(frame)
    return results

def draw_boxes(frame, results):
    h, w, _ = frame.shape
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        confidences = result.boxes.conf.cpu().numpy()  # Get confidences
        class_ids = result.boxes.cls.cpu().numpy()  # Get class IDs

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            if confidence > 0.5:  # Threshold for displaying detected objects
                x1, y1, x2, y2 = map(int, box)
                label = f'{class_names[int(class_id)]}: {confidence:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def perform_ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hocr_data = pytesseract.image_to_boxes(gray)
    text = pytesseract.image_to_string(gray).strip()
    return hocr_data, text

def draw_text_boxes(frame, hocr_data):
    h, w, _ = frame.shape
    imgH = h
    for line in hocr_data.splitlines():
        parts = line.split(' ')
        if len(parts) == 6:  # Ensure that there are enough parts
            x, y, w, h = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            cv2.rectangle(frame, (x, imgH - y), (w, imgH - h), (0, 0, 255), 2)
            cv2.putText(frame, parts[0], (x, imgH - y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    return frame

def overlay_text(frame, text):
    # Add a background rectangle for the text
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.rectangle(frame, (10, 30 - text_height - baseline), (10 + text_width, 30 + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return frame

def main():
    # Capture video from webcam
    cap = cv2.VideoCapture(0)

    #  cap setting
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform object detection
        results = detect_objects(frame)
        
        # Draw bounding boxes and labels for objects
        frame = draw_boxes(frame, results)
        
        # Perform OCR
        hocr_data, text = perform_ocr(frame)
        
        # Draw bounding boxes around detected text characters
        frame = draw_text_boxes(frame, hocr_data)
        
        # Overlay the detected text on the frame
        frame = overlay_text(frame, text)
        
        # Print detected text to the terminal
        if text:
            print("Detected text:", text)
        
        # Show the frame with detected objects and OCR text
        cv2.imshow('YOLOv8 Object Detection and OCR', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
