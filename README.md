# Overview

The Blind Assistance System is a web-based application designed to assist visually impaired users by providing real-time feedback on nearby objects using depth estimation and object detection. The application integrates MiDaS for depth estimation and YOLOv5 for object detection, offering auditory and visual feedback for navigation.

The system processes live video input from the user's camera, detects obstacles, estimates their distance, and provides audio alerts and feedback messages to help the user avoid potential hazards. It also supports uploading images for processing and displays detailed information about the detected objects.

# **Features**

Real-Time Object Detection: Uses YOLOv5 to detect objects in the camera feed and provide bounding boxes around them.

Depth Estimation: Utilizes MiDaS for depth estimation to measure the distance to detected objects and obstacles.

Auditory Feedback: Provides feedback using pygame sound alerts (with optional Text-to-Speech support via pyttsx3) based on the proximity of detected obstacles.

Web Interface: Built using Flask, the application serves a web interface with live video streaming and image upload functionality.

Alert System: Sends feedback about the detected obstacles' position (left, right, or straight ahead) and distance.

Error Handling: Alerts users about issues such as missing files or camera connection problems.

# Project Structure

   ├── app.py # Main Flask backend
   
   ├── templates/ (index.html, live_feed.html, coco_dataset.html)
   
   ├── uploads/ # COCO dataset

   https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
   
   ├── sounds/ # Pre-generated alert tone WAV files
   
   └── static/ # Optional for CSS/JS assets

# **Installation**
1. Clone the Repository
   
   https://github.com/tabidah-usmani/Smart-Blind-Assistance-System.git
   
   cd Smart-Blind-Assistance-System
   
3. Install Dependencies
   
    Flask: For web server and routing.

    OpenCV: For video capture and image processing.
    
    PyTorch: For running YOLOv5 and MiDaS models.
    
    pygame: For sound alerts.
    
    pyttsx3: For text-to-speech feedback (optional).

5. Usage
   
   * Start the application
   `   python app.py
   `

     Visit http://127.0.0.1:5000/ in your browser to access the main interface.

   * Live Feed
   
      Navigate to /live_feed to see the live camera feed with object detection and depth estimation.

   * Upload Images for Processing

      Navigate to /coco_dataset to upload an image file (PNG, JPG, or JPEG).


   


   
