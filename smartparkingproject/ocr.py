from ultralytics import YOLO
import cv2
import numpy as np
from paddleocr import PaddleOCR
import pytesseract
from pymongo import MongoClient
from datetime import datetime
import requests
import time
import re
import os
import sys
import queue
import multiprocessing

# === CONFIGURATION ===
# External webcam setup
ESP32_IP = "http://192.168.5.132"
TESSERACT_PATH = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"  # Use .exe extension
COOLDOWN_SECS = 10
HD_WIDTH = 1920   # Full HD width
HD_HEIGHT = 1080  # Full HD height

# Camera settings
CAMERA_SETTINGS = {
    "width": HD_WIDTH,
    "height": HD_HEIGHT,
    "fps": 30
}

# Window settings
DISPLAY_WIDTH = 1920  # Width of the combined display window
DISPLAY_HEIGHT = 1080  # Height of the combined display window

# Number of camera indices to check
MAX_CAMERA_INDEX = 3  # Only check first 3 indices (0, 1, 2)

# Tesseract setup - Improved path handling
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    print(f"✅ Tesseract configured at {TESSERACT_PATH}")
else:
    print(f"⚠ Tesseract path not found at {TESSERACT_PATH}. Looking for tesseract in PATH...")
    # Try to find tesseract in standard locations
    alternative_paths = [
        r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
        r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
        "tesseract"  # Let the system find it in PATH
    ]
    
    for path in alternative_paths:
        try:
            if os.path.exists(path) or "tesseract" == path:
                pytesseract.pytesseract.tesseract_cmd = path
                test_result = pytesseract.get_tesseract_version()
                print(f"✅ Found Tesseract at {path}, version: {test_result}")
                break
        except Exception:
            continue
    else:
        print("❌ Tesseract not found. OCR functionality will be limited.")

tess_config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"

# Global variables for tracking last detections
last_seen_entry = {}
last_seen_exit = {}

# MongoDB setup
try:
    client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
    client.server_info()  # Will raise an exception if cannot connect
    db = client["detect1234"]
    collection = db["photos12"]
    print("✅ MongoDB connected")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    print("⚠ Running without database. Detection will work but no data will be saved.")
    db = None
    collection = None

# Load YOLO model
try:
    model = YOLO("yolov8n.pt")
    print("✅ YOLOv8 model loaded")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    sys.exit(1)

# PaddleOCR setup
try:
    ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
    print("✅ PaddleOCR initialized")
except Exception as e:
    print(f"❌ Failed to initialize PaddleOCR: {e}")
    sys.exit(1)

# Camera class to manage all camera operations
class Camera:
    def __init__(self, index, name):  # <-- Fixed constructor name
        self.index = index
        self.name = name
        self.cap = None
        self.frame = None
        self.frame_with_detections = None
        self.connected = False
        self.resolution_set = False
        self.last_frame_time = 0
        self.fps = 0
        self.frame_queue = queue.Queue(maxsize=2)  # Small queue to avoid memory buildup
        self.running = False

    def connect(self):
        """Connect to the camera and set properties"""
        # Release any existing capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        print(f"Attempting to connect to {self.name} camera (index: {self.index})...")
        
        # Try different backends to ensure connection
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                self.cap = cv2.VideoCapture(self.index, backend)
                if self.cap.isOpened():
                    # Short pause to let the camera initialize properly
                    time.sleep(0.5)
                    
                    # Verify we can actually get a frame
                    ret, test_frame = self.cap.read()
                    if not ret or test_frame is None:
                        print(f"⚠ Camera {self.index} opened but couldn't read frame, trying different backend")
                        self.cap.release()
                        continue
                        
                    # Try to set resolution
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SETTINGS["width"])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SETTINGS["height"])
                    self.cap.set(cv2.CAP_PROP_FPS, CAMERA_SETTINGS["fps"])
                    
                    # Check if resolution was actually set
                    actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    
                    print(f"📷 {self.name.capitalize()} camera connected (index: {self.index}, backend: {backend})")
                    print(f"   Resolution: {actual_width}x{actual_height} @ {actual_fps} FPS")
                    
                    self.connected = True
                    
                    # Initialize the frame with a successful read
                    self.frame = test_frame
                    self.frame_with_detections = test_frame.copy()
                    
                    return True
            except Exception as e:
                print(f"❌ Error connecting to camera {self.index}: {e}")
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
        
        print(f"❌ Failed to connect to {self.name} camera (index: {self.index})")
        return False

    def read_frame(self):
        """Read a frame from the camera with error handling"""
        if not self.connected or self.cap is None:
            return False
        
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print(f"⚠ Frame grab failed for {self.name} camera, attempting to reconnect...")
                self.connected = False
                self.cap.release()
                self.cap = None
                time.sleep(1)
                return self.connect()  # Try to reconnect immediately
            
            # Calculate FPS
            current_time = time.time()
            time_diff = current_time - self.last_frame_time
            if time_diff > 0:
                self.fps = 1 / time_diff
            self.last_frame_time = current_time
            
            self.frame = frame
            self.frame_with_detections = frame.copy()
            return True
        except Exception as e:
            print(f"❌ Error reading frame from {self.name} camera: {e}")
            self.connected = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False

    def release(self):
        """Release the camera resources"""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.connected = False

# Preprocess
def preprocess_plate(plate_img):
    """Preprocess the plate image for better OCR results"""
    # Original grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter to reduce noise while preserving edges
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Adaptive thresholding for better handling of lighting conditions
    adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
    
    # Otsu's thresholding
    _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)
    
    return {
        "gray": gray, 
        "blur": blur, 
        "adaptive_thresh": adaptive_thresh,
        "otsu_thresh": otsu_thresh,
        "morph": morph
    }

# OCR functions
def ocr_paddle(img_dict):
    """Use PaddleOCR to recognize text"""
    best_text = ""
    best_confidence = 0
    
    for img_type, img in img_dict.items():
        try:
            result = ocr.ocr(img, cls=True)
            if result and result[0]:
                for item in result[0]:
                    text = item[1][0].strip().upper()
                    confidence = float(item[1][1])
                    if confidence > best_confidence and re.match(r"^[A-Z0-9\-]{4,12}$", text):
                        best_text = text
                        best_confidence = confidence
        except Exception as e:
            print(f"PaddleOCR error on {img_type}: {e}")
    
    return best_text

def ocr_tesseract(img_dict):
    """Use Tesseract to recognize text"""
    best_text = ""
    
    for img_type, img in img_dict.items():
        try:
            text = pytesseract.image_to_string(img, config=tess_config)
            cleaned_text = re.sub(r"[^A-Z0-9\-]", "", text.upper())
            if re.match(r"^[A-Z0-9\-]{4,12}$", cleaned_text) and len(cleaned_text) > len(best_text):
                best_text = cleaned_text
        except Exception as e:
            print(f"Tesseract error on {img_type}: {e}")
    
    return best_text

def ensemble_ocr(plate_img):
    """Combine results from multiple OCR engines for better accuracy"""
    if plate_img.shape[0] < 20 or plate_img.shape[1] < 20:
        return ""
        
    processed = preprocess_plate(plate_img)
    
    # Get results from both OCR engines
    paddle_result = ocr_paddle(processed)
    
    # Only use Tesseract if it's properly installed
    tesseract_result = ""
    if pytesseract.pytesseract.tesseract_cmd and os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        tesseract_result = ocr_tesseract(processed)
    
    # Simple ensemble logic
    if paddle_result == tesseract_result and paddle_result:
        return paddle_result  # Both agree
    elif paddle_result and not tesseract_result:
        return paddle_result
    elif tesseract_result and not paddle_result:
        return tesseract_result
    elif paddle_result and tesseract_result:
        # Choose the one that better matches license plate pattern
        if re.match(r"^[A-Z]{2,3}[0-9]{3,4}$", paddle_result):
            return paddle_result
        if re.match(r"^[A-Z]{2,3}[0-9]{3,4}$", tesseract_result):
            return tesseract_result
        # Default to the longer one as it might have more information
        return paddle_result if len(paddle_result) >= len(tesseract_result) else tesseract_result
    
    return ""

# Vehicle detection and processing logic
def process_frame(camera, cam_type):
    """Process a frame to detect vehicles and license plates"""
    if camera.frame is None:
        return
    
    # Create a copy of the frame for detection drawing
    frame = camera.frame.copy()
    detection_frame = frame.copy()
    
    # Convert to RGB for YOLO (it expects RGB, OpenCV uses BGR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run YOLO detection
    results = model(rgb, conf=0.35)[0]  # Slightly higher confidence threshold
    now = time.time()
    
    detected_count = 0
    
    for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls_id)]
        conf_val = float(conf)
        
        # Filter for vehicle classes
        if label not in ["car", "motorcycle", "truck", "bus"]:
            continue
            
        detected_count += 1
        
        # Extract the vehicle image
        vehicle_img = frame[y1:y2, x1:x2]
        
        # Try to find license plate
        plate = ensemble_ocr(vehicle_img)
        
        # Skip if no valid plate detected
        if not plate or not re.match(r"^[A-Z0-9\-]{4,12}$", plate):
            # Draw box for vehicle without readable plate
            cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange
            cv2.putText(detection_frame, f"{label} {conf_val:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            continue
            
        # Apply cooldown to prevent duplicate detections
        if cam_type == "entry":
            if plate in last_seen_entry and now - last_seen_entry[plate] < COOLDOWN_SECS:
                continue
            last_seen_entry[plate] = now
        else:  # exit
            if plate in last_seen_exit and now - last_seen_exit[plate] < COOLDOWN_SECS:
                continue
            last_seen_exit[plate] = now
        
        # Record to database if connected
        if collection is not None:
            timestamp = datetime.now()
            date_str = timestamp.strftime("%Y-%m-%d")
            day_str = timestamp.strftime("%A")
            
            if cam_type == "entry":
                existing = collection.find_one({"plate_number": plate, "exit_time": None})
                if not existing:
                    collection.insert_one({
                        "plate_number": plate,
                        "vehicle_type": label,
                        "entry_time": timestamp,
                        "entry_date": date_str,
                        "entry_day": day_str,
                        "exit_time": None,
                        "exit_date": None,
                        "exit_day": None,
                        "slot": None,
                        "confidence": conf_val
                    })
                    print(f"✅ Entry logged: {plate} at {timestamp}")
                    
                    # Try to open gate
                    try:
                        res = requests.get(f"{ESP32_IP}/open_gate", timeout=3)
                        if res.status_code == 200:
                            print("🟢 Gate Opened")
                    except Exception as e:
                        print(f"⚠ Failed to reach ESP32: {e}")
                    
                    # Try to get parking slot
                    try:
                        slot_resp = requests.get(f"{ESP32_IP}/get_slots", timeout=3)
                        data = slot_resp.json()
                        slot = data.get("occupied_slots", [])[-1] if data.get("occupied_slots") else "Unknown"
                        collection.update_one(
                            {"plate_number": plate, "exit_time": None},
                            {"$set": {"slot": slot}}
                        )
                        print(f"🅿 Slot Assigned: {slot}")
                    except Exception as e:
                        print(f"⚠ Slot fetch failed: {e}")
            else:  # exit
                updated = collection.update_one(
                    {"plate_number": plate, "exit_time": None},
                    {"$set": {
                        "exit_time": timestamp,
                        "exit_date": date_str,
                        "exit_day": day_str
                    }}
                )
                if updated.modified_count:
                    print(f"🔴 Exit logged: {plate} at {timestamp}")
        
        # Draw bounding box and text on the frame
        cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green
        cv2.putText(detection_frame, f"{label}: {plate} ({conf_val:.2f})", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
    
    # Add info text
    camera_info = f"{cam_type.upper()} | FPS: {camera.fps:.1f} | Detected: {detected_count}"
    cv2.putText(detection_frame, camera_info, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Update the camera's detection frame
    camera.frame_with_detections = detection_frame

# Function to check available cameras
def check_available_cameras():
    """Detect available cameras and return their indexes"""
    available_cameras = []
    
    print("Checking available cameras...")
    for i in range(MAX_CAMERA_INDEX):  # Check only the first 3 indexes
        print(f"Testing camera index {i}...")
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    # Wait a bit for camera initialization
                    time.sleep(0.5)
                    
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Get camera info
                        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        available_cameras.append({
                            "index": i,
                            "width": width,
                            "height": height, 
                            "fps": fps,
                            "backend": backend
                        })
                        
                        print(f"✅ Camera {i} is available: {width}x{height} @ {fps}fps (backend: {backend})")
                        
                        # Show a frame from this camera
                        small_frame = cv2.resize(frame, (640, 360))
                        cv2.imshow(f"Camera {i}", small_frame)
                        cv2.waitKey(1000)
                        cv2.destroyWindow(f"Camera {i}")
                        
                        # Break once we've found a working backend for this camera
                        break
                    else:
                        print(f"⚠ Camera {i} opened but couldn't read frame with backend {backend}")
                else:
                    print(f"⚠ Could not open camera {i} with backend {backend}")
            except Exception as e:
                print(f"❌ Error testing camera {i} with backend {backend}: {e}")
            finally:
                if 'cap' in locals() and cap is not None:
                    cap.release()
    
    return available_cameras

def main():
    """Main function to run the vehicle detection system with a non-threaded approach"""
    # Check available cameras
    available_cameras = check_available_cameras()
    
    if not available_cameras:
        print("❌ No cameras detected!")
        return
    
    print(f"Found {len(available_cameras)} camera(s)")
    
    # Select cameras based on available ones
    if len(available_cameras) >= 2:
        # Use the first two cameras
        entry_camera = Camera(available_cameras[1]["index"], "entry")
        exit_camera = Camera(available_cameras[0]["index"], "exit")
    else:
        # Use the same camera for both (demo mode)
        print("⚠ Only one camera available. Using it for both entry and exit (demo mode)")
        if len(available_cameras) == 0:
            print("❌ No available cameras to use.")
            return
        entry_camera = Camera(available_cameras[0]["index"], "entry")
        exit_camera = entry_camera  # Use the same camera object for both
    
    # Connect to cameras
    if not entry_camera.connect():
        print("❌ Failed to connect to entry camera")
        return
    
    if entry_camera != exit_camera and not exit_camera.connect():  # Only connect exit if it's a different camera
        print("❌ Failed to connect to exit camera")
        entry_camera.release()
        return
    
    print("✅ Cameras connected")
    print("📊 Display window initialized. Press 'q' to quit.")
    
    # Create a named window for the combined display
    cv2.namedWindow("Vehicle Detection System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vehicle Detection System", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    # Initialize frame counters for processing optimization
    frame_counter = 0
    process_every = 3  # Process every Nth frame for better performance
    
    try:
        while True:
            frame_counter += 1
            
            # Read entry camera frame
            if not entry_camera.read_frame():
                print("⚠ Failed to read from entry camera, attempting to reconnect...")
                if not entry_camera.connect():
                    time.sleep(1)  # Wait before retrying
                    continue
            
            # Read exit camera frame if it's a different camera
            if entry_camera != exit_camera:
                if not exit_camera.read_frame():
                    print("⚠ Failed to read from exit camera, attempting to reconnect...")
                    if not exit_camera.connect():
                        time.sleep(1)  # Wait before retrying
                        continue
            
            # Only process every Nth frame to improve performance
            if frame_counter % process_every == 0:
                # Process entry frame
                process_frame(entry_camera, "entry")
                
                # Process exit frame if it's a different camera
                if entry_camera != exit_camera:
                    process_frame(exit_camera, "exit")
            
            # Create combined display
            if entry_camera.frame_with_detections is not None:
                # For single-camera mode
                if entry_camera == exit_camera:
                    # Just use the same frame for both sides
                    entry_resized = cv2.resize(entry_camera.frame_with_detections, 
                                            (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT))
                    exit_resized = entry_resized.copy()
                    
                    # Modify the label on exit side
                    cv2.putText(exit_resized, "EXIT (DEMO - SAME CAMERA)", (20, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                else:
                    # Resize frames to fit the display (each takes half the width)
                    entry_resized = cv2.resize(entry_camera.frame_with_detections, 
                                            (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT))
                    exit_resized = cv2.resize(exit_camera.frame_with_detections, 
                                            (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT))
                
                # Add labels to the frames
                cv2.putText(entry_resized, "ENTRY", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
                
                if entry_camera != exit_camera:
                    cv2.putText(exit_resized, "EXIT", (20, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
                
                # Combine frames horizontally
                combined_frame = np.hstack((entry_resized, exit_resized))
                
                # Add date/time and system status
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(combined_frame, current_time, (DISPLAY_WIDTH - 400, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display the combined frame
                cv2.imshow("Vehicle Detection System", combined_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('r'):
                print("Reconnecting cameras...")
                entry_camera.connect()
                if entry_camera != exit_camera:
                    exit_camera.connect()
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Clean up
        entry_camera.release()
        if entry_camera != exit_camera:
            exit_camera.release()
        cv2.destroyAllWindows()
        print("System shutdown complete")

if __name__ == "__main__":
    # Set process priority higher to improve performance
    try:
        import psutil
        process = psutil.Process(os.getpid())
        process.nice(psutil.HIGH_PRIORITY_CLASS)
        print("✅ Set process to high priority")
    except ImportError:
        print("⚠ psutil not available, running with default priority")
    except Exception as e:
        print(f"⚠ Could not set process priority: {e}")
    
    # Run the main function
    main()