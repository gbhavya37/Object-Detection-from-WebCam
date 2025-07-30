import cv2
from ultralytics import YOLO
import ctypes

# Load your preferred YOLOv8 model (medium here, you can keep yolov8m.pt)
model = YOLO('yolov8m.pt')

# Open webcam
cap = cv2.VideoCapture(0)

# Set webcam resolution lower for better speed (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get screen size (Windows)
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

# Create named window with normal flag for fullscreen support
window_name = 'Improved YOLOv8 Object Detection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally to fix left-right mirror issue
    frame = cv2.flip(frame, 1)

    # Resize frame for faster model inference (keep aspect ratio)
    max_dim = 640
    h, w = frame.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        frame_resized = frame.copy()

    # Detect objects on the resized frame
    results = model(frame_resized)[0]

    # Draw bounding boxes on original frame
    for box in results.boxes:
        if box.conf >= 0.5:
            # Get box coordinates and map them back to original frame size
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if scale < 1:
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)

            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f'{model.names[cls]} {conf:.2f}'

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Resize the frame to match screen resolution for fullscreen display
    frame_display = cv2.resize(frame, (screen_width, screen_height))

    # Show frame
    cv2.imshow(window_name, frame_display)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
