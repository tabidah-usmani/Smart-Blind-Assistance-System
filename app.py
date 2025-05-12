import os
import cv2
import torch
import numpy as np
import pygame
import time
import winsound
from flask import Flask, render_template, Response, request, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import base64
import logging
from typing import List, Union, Optional
from typing import Generator, Any


# Handle deprecation warnings
import warnings
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast\(args...\)` is deprecated")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__, template_folder='D:\dip\22i-2025_Amna_Javaid_22i-2070_Tabidah_Usmani_22i-2060_Tasmiya_Asad_22i-1998_Ziyan_Murtaza\templates')
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

try:
    import pyttsx3
    TTS_AVAILABLE = True
    logging.info("pyttsx3 loaded successfully.")
except ImportError:
    logging.warning("pyttsx3 not installed. Falling back to print-only messages.")
    TTS_AVAILABLE = False

current_status = {
    "camera_status": "Connecting...",
    "detection_feedback": "Waiting for detection...",
    "distance": "-"
}

class BlindAssistanceFeedback:
    def __init__(self):
        global TTS_AVAILABLE  # Add this line to access the global variable
        pygame.mixer.init()
        logging.info("pygame.mixer initialized successfully.")
        
        # Alert thresholds (adjusted for normalized depth)
        self.VERY_CLOSE_THRESHOLD = 0.2
        self.CLOSE_THRESHOLD = 0.4
        self.ALERT_COOLDOWN = 2.0
        self.last_alert_time = 0
        
        # Initialize TTS if available
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
                logging.info("TTS engine initialized successfully.")
            except Exception as e:
                logging.error(f"Failed to initialize TTS engine: {e}")
                TTS_AVAILABLE = False
        
        # Load alert sounds
        try:
            self.alert_very_close = pygame.mixer.Sound('sounds/alert_very_close.wav')
            self.alert_close = pygame.mixer.Sound('sounds/alert_close.wav')
            self.alert_moderate = pygame.mixer.Sound('sounds/alert_moderate.wav')
            self.alert_left = pygame.mixer.Sound('sounds/left.wav')
            self.alert_right = pygame.mixer.Sound('sounds/right.wav')
            logging.info("All sound files loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load sound files: {e}")
            self.alert_very_close = None
            self.alert_close = None
            self.alert_moderate = None
            self.alert_left = None
            self.alert_right = None

    def get_obstacle_position(self, detections: Union[List[List[float]], np.ndarray], frame_width: int) -> Optional[float]:
        """Estimate obstacle horizontal position (0=left, 1=right)"""
        try:
            # Check if detections is empty or None
            if detections is None or len(detections) == 0:
                return None
            
            # Convert detections to list if it's a numpy array
            if isinstance(detections, np.ndarray):
                detections = detections.tolist()
            
            # Use the largest detection (by area) to determine position
            max_area = 0
            center_x = None
            
            for detection in detections:
                # Ensure detection has at least [x1, y1, x2, y2]
                if len(detection) >= 4:
                    x1, y1, x2, y2 = detection[:4]
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        center_x = (x1 + x2) / 2
            
            return center_x / frame_width if center_x is not None else None
        
        except Exception as e:
            logging.error(f"Error in get_obstacle_position: {e}")
            return None

    def provide_feedback(self, distance_normalized: float, 
                        detections: Union[List[List[float]], np.ndarray, None], 
                        frame_width: int, 
                        message_prefix: str = "Object") -> Optional[str]:
        """Provide audio and TTS feedback based on distance and detections"""
        current_time = time.time()
        if current_time - self.last_alert_time < self.ALERT_COOLDOWN:
            return None
        
        try:
            # Handle empty/None detections
            if detections is None:
                detections = []
            elif isinstance(detections, np.ndarray):
                detections = detections.tolist() if detections.size > 0 else []
            
            obstacle_pos = self.get_obstacle_position(detections, frame_width)
            distance_meters = distance_normalized * 10  # Convert to meters
            
            # Determine the most confident detection for the message
            max_conf = 0
            object_label = "Object"
            
            for detection in detections:
                if len(detection) >= 6:  # [x1, y1, x2, y2, conf, cls]
                    conf = detection[4]
                    cls = int(detection[5])
                    if conf > max_conf:
                        max_conf = conf
                        object_label = yolo_model.names[cls]
            
            # Generate appropriate message based on distance
            if distance_normalized < self.VERY_CLOSE_THRESHOLD:
                self.play_alert(self.alert_very_close, 1000, 300)
                message = f"{object_label} very close at {distance_meters:.1f} meters"
            elif distance_normalized < self.CLOSE_THRESHOLD:
                self.play_alert(self.alert_close, 800, 200)
                message = f"{object_label} close at {distance_meters:.1f} meters"
            else:
                self.play_alert(self.alert_moderate, 600, 150)
                message = f"{object_label} detected at {distance_meters:.1f} meters"
            
            # Add position information if available
            if obstacle_pos is not None:
                if obstacle_pos < 0.4:
                    message += " on your left"
                    self.play_alert(self.alert_left, 500, 100)
                elif obstacle_pos > 0.6:
                    message += " on your right"
                    self.play_alert(self.alert_right, 500, 100)
            
            logging.info(message)
            
            # Use TTS if available
            if TTS_AVAILABLE:
                try:
                    self.tts_engine.say(message)
                    self.tts_engine.runAndWait()
                    logging.info(f"TTS played: {message}")

                except Exception as e:
                    logging.error(f"TTS playback failed: {e}")
            
            self.last_alert_time = current_time
            return message
        
        except Exception as e:
            logging.error(f"Error in provide_feedback: {e}")
            return None

    def play_alert(self, sound: Optional[pygame.mixer.Sound], freq: Optional[int] = None, duration: Optional[int] = None) -> None:
        """Play audio alert or fallback to system beep"""
        try:
            if sound is not None:
                sound.play()
            elif freq is not None and duration is not None:
                winsound.Beep(freq, duration)
        except Exception as e:
            logging.error(f"Error playing alert: {e}")

    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            pygame.mixer.quit()
            if TTS_AVAILABLE:
                self.tts_engine.stop()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

def generate_tone(filename: str, frequency: int, duration: float = 0.3, sample_rate: int = 44100, volume: float = 0.5) -> None:
    try:
        from scipy.io.wavfile import write
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * frequency * t) * volume
        tone = np.int16(tone * 32767)
        write(filename, sample_rate, tone)
        logging.info(f"Saved: {filename}")
    except Exception as e:
        logging.error(f"Error generating tone: {e}")

# Create sounds directory if it doesn't exist
os.makedirs("sounds", exist_ok=True)

# Generate default alert sounds if they don't exist
for tone in [
    ("sounds/alert_very_close.wav", 1000, 0.4),
    ("sounds/alert_close.wav", 800, 0.3),
    ("sounds/alert_moderate.wav", 600, 0.2),
    ("sounds/left.wav", 500, 0.15),
    ("sounds/right.wav", 700, 0.15)
]:
    if not os.path.exists(tone[0]):
        generate_tone(tone[0], tone[1], tone[2])

# Load MiDaS model for depth estimation
try:
    midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
    midas.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    midas.to(device)
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
    transform = midas_transforms.small_transform
    logging.info("MiDaS model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load MiDaS model: {e}")
    raise

# Load YOLOv5 model for object detection
try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo_model.eval()
    yolo_model.to(device)
    logging.info("YOLOv5 model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load YOLOv5 model: {e}")
    raise

# Initialize feedback system
feedback = BlindAssistanceFeedback()

def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(img_rgb: np.ndarray) -> tuple:
    """Process a single image for depth and object detection"""
    try:
        logging.info(f"Processing image with shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
        
        # Edge detection (Canny)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Depth estimation (MiDaS)
        input_batch = transform(img_rgb).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        depth_range = np.max(depth_map) - np.min(depth_map)
        
        if depth_range == 0:
            depth_map_normalized = np.zeros_like(depth_map)
            avg_depth = 0.0
        else:
            depth_map_normalized = (depth_map - np.min(depth_map)) / depth_range
            threshold_value = np.percentile(depth_map_normalized, 30)
            close_pixels = depth_map_normalized[depth_map_normalized < threshold_value]
            avg_depth = np.mean(close_pixels) if close_pixels.size > 0 else 0.0
        
        # Object detection (YOLOv5)
        results = yolo_model(img_rgb)
        detections = results.xyxy[0].cpu().numpy()
        detections_list = detections.tolist() if detections.size > 0 else []
        logging.info(f"Detections: {detections_list}")
        
        # Generate object detection image with bounding boxes
        img_with_boxes = img_rgb.copy()  # Copy to avoid modifying original
        for detection in detections_list:
            if len(detection) >= 6:  # [x1, y1, x2, y2, conf, cls]
                x1, y1, x2, y2 = map(int, detection[:4])
                conf = detection[4]
                cls = int(detection[5])
                if conf > 0.5:  # Confidence threshold
                    label = f"{yolo_model.names[cls]} {conf:.2f}"
                    # Convert RGB to BGR for OpenCV drawing
                    img_with_boxes_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(img_with_boxes_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_with_boxes_bgr, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    img_with_boxes = cv2.cvtColor(img_with_boxes_bgr, cv2.COLOR_BGR2RGB)
        
        # Colorized depth map
        depth_colorized = cv2.applyColorMap((depth_map_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Binary thresholded depth map
        _, depth_binary = cv2.threshold((depth_map_normalized * 255).astype(np.uint8), 
                                       int(threshold_value * 255), 255, cv2.THRESH_BINARY_INV)
        
        # Get feedback message
        message = feedback.provide_feedback(avg_depth, detections_list, img_rgb.shape[1], "Image")
        
        # Encode images
        _, img_buffer = cv2.imencode('.jpg', img_rgb)
        _, edges_buffer = cv2.imencode('.jpg', edges)
        _, depth_buffer = cv2.imencode('.jpg', (depth_map_normalized * 255).astype(np.uint8))
        _, boxes_buffer = cv2.imencode('.jpg', img_with_boxes)
        _, depth_color_buffer = cv2.imencode('.jpg', depth_colorized)
        _, depth_binary_buffer = cv2.imencode('.jpg', depth_binary)
        
        return (
            base64.b64encode(img_buffer).decode('utf-8'),      # Original image
            base64.b64encode(edges_buffer).decode('utf-8'),    # Edge detection
            base64.b64encode(depth_buffer).decode('utf-8'),    # Grayscale depth
            base64.b64encode(boxes_buffer).decode('utf-8'),    # Object detection with boxes
            base64.b64encode(depth_color_buffer).decode('utf-8'),  # Colorized depth
            base64.b64encode(depth_binary_buffer).decode('utf-8'), # Binary depth
            message,
            avg_depth * 10  # Distance in meters
        )
    except Exception as e:
        logging.error(f"Error in process_image: {e}")
        return None, None, None, None, None, None, f"Error: {str(e)}", 0.0

@app.route('/')
def index() -> str:
    """Render main page"""
    return render_template('index.html')

@app.route('/live_feed')
def live_feed() -> str:
    """Render live feed page"""
    return render_template('live_feed.html')

@app.route('/status')
def status():
    return jsonify(current_status)

@app.route('/video_feed')
def video_feed() -> Response:
    """Video streaming route"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames() -> Generator[bytes, None, None]:
    """Generate video frames with object detection and depth estimation"""
    cap = cv2.VideoCapture(1)  # Open laptop camera
    if not cap.isOpened():
        current_status['camera_status'] = "Camera not available"
        logging.error("Could not open camera.")
        yield (f"data:image/jpeg;base64,|"
               f"Error: Could not open camera.|"
               f"0.0\n").encode('utf-8')
        return
    
    current_status['camera_status'] = "Connected"
    
    try:
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture frame.")
                yield (f"data:image/jpeg;base64,|"
                       f"Error: Failed to capture frame.|"
                       f"0.0\n").encode('utf-8')
                continue
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Object detection with YOLOv5
            results = yolo_model(img_rgb)
            detections = results.xyxy[0].cpu().numpy()
            detections_list = detections.tolist() if detections.size > 0 else []
            
            # Draw bounding boxes and labels
            for detection in detections_list:
                if len(detection) >= 6:  # [x1, y1, x2, y2, conf, cls]
                    x1, y1, x2, y2 = map(int, detection[:4])
                    conf = detection[4]
                    cls = int(detection[5])
                    if conf > 0.5:  # Confidence threshold
                        label = f"{yolo_model.names[cls]} {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Depth estimation
            input_batch = transform(img_rgb).to(device)
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode='bicubic',
                    align_corners=False,
                ).squeeze()
            
            depth_map = prediction.cpu().numpy()
            depth_range = np.max(depth_map) - np.min(depth_map)
            
            if depth_range == 0:
                depth_map_normalized = np.zeros_like(depth_map)
                avg_depth = 0.0
            else:
                depth_map_normalized = (depth_map - np.min(depth_map)) / depth_range
                threshold_value = np.percentile(depth_map_normalized, 30)
                close_pixels = depth_map_normalized[depth_map_normalized < threshold_value]
                avg_depth = np.mean(close_pixels) if close_pixels.size > 0 else 0.0
            
            # Convert distance to meters
            distance_meters = avg_depth * 10

            current_status['distance'] = f"{distance_meters:.2f} meters"
            
            # Provide feedback
            message = feedback.provide_feedback(avg_depth, detections_list, frame.shape[1])

            current_status['detection_feedback'] = message if message else "No object detected"
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                logging.error("Failed to encode frame.")
                continue
            
            frame_data = base64.b64encode(buffer).decode('utf-8')
            
            # Yield the frame in the multipart response format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')

            # Small delay to reduce CPU usage
            time.sleep(0.05)
            
    except Exception as e:
        logging.error(f"Error in frame generation: {e}")
        yield (f"data:image/jpeg;base64,|"
               f"Error: {str(e)}|"
               f"0.0\n").encode('utf-8')
    finally:
        cap.release()
        logging.info("Camera released")



@app.route('/coco_dataset', methods=['GET', 'POST'])
def coco_dataset() -> str:
    """Handle COCO dataset image processing"""
    img_base64 = None
    edges_base64 = None
    depth_base64 = None
    boxes_base64 = None
    depth_color_base64 = None
    depth_binary_base64 = None
    message = None
    avg_depth = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
        else:
            file = request.files['file']
            if file.filename == '':
                flash('No file selected')
            elif not allowed_file(file.filename):
                flash('Invalid file type. Please upload a PNG, JPG, or JPEG')
            else:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                try:
                    file.save(file_path)
                    img = cv2.imread(file_path)
                    if img is None:
                        flash('Error loading image. Please ensure the file is a valid image')
                    else:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        (img_base64, edges_base64, depth_base64, boxes_base64, 
                         depth_color_base64, depth_binary_base64, message, avg_depth) = process_image(img_rgb)
                        if img_base64 is None:
                            flash(message or 'Error processing image')
                        else:
                            message = message or "Image processed successfully"

                except Exception as e:
                    logging.error(f"Error in coco_dataset POST: {e}")
                    flash(f'Error processing image: {str(e)}')

                finally:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logging.error(f"Error removing uploaded file: {e}")

    return render_template('coco_dataset.html',
                           img_base64=img_base64,
                           edges_base64=edges_base64,
                           depth_base64=depth_base64,
                           boxes_base64=boxes_base64,
                           depth_color_base64=depth_color_base64,
                           depth_binary_base64=depth_binary_base64,
                           message=message,
                           avg_depth=avg_depth)

@app.route('/shutdown')
def shutdown() -> str:
    """Clean shutdown route"""
    feedback.cleanup()
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        feedback.cleanup()