# 🦯 Smart Blind Assistance System

### 👁️‍🗨️ Real-Time Object Detection & Depth Estimation for the Visually Impaired

A web-based assistive technology application that helps visually impaired users navigate their surroundings safely. By combining **real-time object detection** 🎯 and **monocular depth estimation** 📏, the system identifies obstacles in a live camera feed, estimates how far away they are, and delivers instant **audio and visual alerts** 🔊 to guide the user around hazards — all through a lightweight, browser-accessible interface.

---

## 📚 Table of Contents

- [🌟 Overview](#-overview)
- [✨ Features](#-features)
- [🧠 How It Works](#-how-it-works)
- [🏗️ System Architecture](#️-system-architecture)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [🌐 Routes & Endpoints](#-routes--endpoints)
- [🔊 Alert System Details](#-alert-system-details)
- [📦 Dataset](#-dataset)
- [🐞 Troubleshooting & Error Handling](#-troubleshooting--error-handling)
- [🎯 Use Cases](#-use-cases)
- [🚧 Limitations](#-limitations)
- [🔮 Future Improvements](#-future-improvements)
- [🤝 Contributing](#-contributing)
- [👩‍💻 Credits](#-credits)
- [📄 License](#-license)

---

## 🌟 Overview

Navigating unfamiliar or cluttered environments is one of the biggest daily challenges faced by people who are blind or visually impaired. The **Smart Blind Assistance System** was built to ease that challenge using accessible, low-cost technology: a regular webcam, a browser, and a pair of deep learning models running underneath a simple Flask web app.

The system continuously:
1. 🎥 Captures live video from the user's camera (or accepts an uploaded image)
2. 🔍 Detects objects and obstacles in the frame using **YOLOv5**
3. 📐 Estimates the relative distance to each detected object using **MiDaS** depth estimation
4. 🧭 Determines whether an obstacle is to the **left**, **right**, or **straight ahead**
5. 🔔 Triggers an **audio alert** (and optional spoken feedback) when an obstacle is dangerously close
6. 🖥️ Displays bounding boxes, labels, and distance information on a live web feed

The goal is to give users an extra "sense" of their surroundings — turning raw camera input into simple, actionable, real-time guidance. 🧑‍🦯➡️🏞️

---

## ✨ Features

- 🎯 **Real-Time Object Detection** — Powered by YOLOv5, the system detects and draws bounding boxes around obstacles, people, furniture, vehicles, and other objects in the live camera feed, labeling each with its class name and confidence score.
- 📏 **Depth Estimation** — MiDaS estimates a relative depth map for every frame, allowing the system to approximate how close or far each detected object is — critical for prioritizing which obstacles matter most right now.
- 🔊 **Auditory Feedback** — Distinct alert tones (played via `pygame`) warn the user when an obstacle is within a risky distance, with optional **Text-to-Speech** narration via `pyttsx3` (e.g., *"Obstacle ahead, chair, 1.5 meters"*).
- 🧭 **Directional Awareness** — Feedback isn't just "something's there" — it tells the user whether the obstacle is on their **left**, **right**, or **directly ahead**, so they know which way to adjust.
- 🌐 **Web Interface** — Built entirely with Flask, so there's no app to install — just open a browser and go. Includes:
  - A **home page** with an overview/entry point
  - A **live feed page** streaming annotated video in real time
  - An **image upload page** for testing detection/depth estimation on static images (e.g., from the COCO dataset)
- 🚨 **Proximity-Based Alert System** — Alerts scale with urgency: the closer the obstacle, the more insistent the feedback, helping users react appropriately to imminent vs. distant hazards.
- 🛡️ **Robust Error Handling** — Friendly alerts for common failure points: missing model files, camera not detected/connected, unsupported image formats, and more, so the app fails gracefully instead of silently breaking.
- 🖼️ **Static Image Testing Mode** — Beyond live video, users (or developers) can upload PNG/JPG/JPEG images to see how the detection + depth pipeline performs on demand, which is also great for demoing and debugging.

---

## 🧠 How It Works

At a high level, each video frame (or uploaded image) flows through two parallel deep learning models before being combined into a single, human-friendly alert:

1. **YOLOv5 (You Only Look Once, v5)** 🎯 — a fast, single-pass convolutional neural network that detects and localizes multiple objects in an image simultaneously, returning bounding boxes, class labels, and confidence scores.
2. **MiDaS (Mixed Dataset Training for Monocular Depth Estimation)** 📏 — a model that predicts a dense relative depth map from a *single* 2D image (no stereo camera or LiDAR needed), estimating which pixels are near and which are far.

The system then **fuses** these two outputs: for every bounding box YOLOv5 finds, it samples the corresponding region of the MiDaS depth map to estimate that object's distance from the camera. Combined with the object's horizontal position in the frame, this produces a compact piece of guidance like:

> 🔊 *"Person ahead, 2 meters — move right"*

This fused signal is what gets converted into sound alerts and/or speech.

---

## 🏗️ System Architecture

```
                     ┌─────────────────────────┐
                     │      Camera /  Upload   │
                     └────────────┬────────────┘
                                  │
                                  v
                     ┌─────────────────────────┐
                     │      Flask Backend      │
                     │        (app.py)         │
                     └────────────┬────────────┘
                        ┌─────────┴───────────┐
                        v                     v
            ┌────────────────────┐  ┌────────────────────┐
            │       YOLOv5       │  │      MiDaS         │
            │  Object Detection  │  │  Depth Estimation  │
            └──────────┬─────────┘  └──────────┬─────────┘
                       └────────────┬──────────┘
                                    v
                     ┌─────────────────────────┐
                     │   Fusion & Alert Logic  │
                     │ (position + distance)   │
                     └────────────┬────────────┘
                        ┌─────────┴───────────┐
                        v                     v
            ┌────────────────────┐  ┌─────────────────────────┐
            │   Audio Alerts     │  │   Annotated Web Feed    │
            │ (pygame / pyttsx3) │  │ (bounding boxes, labels)│
            └────────────────────┘  └─────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| 🌐 Web Framework | **Flask** | Serves the web interface, routes, and live video stream |
| 🎯 Object Detection | **YOLOv5** (PyTorch) | Detects and classifies obstacles in each frame |
| 📏 Depth Estimation | **MiDaS** (PyTorch) | Estimates relative distance of detected objects |
| 🎥 Computer Vision | **OpenCV** | Captures camera frames, image processing, drawing bounding boxes |
| 🔊 Audio Alerts | **pygame** | Plays pre-generated alert tone WAV files |
| 🗣️ Text-to-Speech | **pyttsx3** *(optional)* | Converts alert messages into spoken feedback |
| 🧮 Deep Learning Backend | **PyTorch** | Runs both YOLOv5 and MiDaS models |
| 🖼️ Frontend | HTML/CSS (Flask templates) | Renders the home, live feed, and upload pages |

---

## 📁 Project Structure

```
Smart-Blind-Assistance-System/
│
├── app.py                     # 🧠 Main Flask backend — routes, model loading, detection & depth logic
│
├── templates/                 # 🖼️ HTML templates rendered by Flask
│   ├── index.html              #     🏠 Home page / entry point
│   ├── live_feed.html          #     🎥 Live camera feed with real-time detection overlay
│   └── coco_dataset.html       #     📤 Image upload & processing page
│
├── uploads/                   # 📦 Uploaded images / COCO dataset samples
│   └── (COCO 2017 dataset — https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)
│
├── sounds/                    # 🔔 Pre-generated alert tone WAV files
│
└── static/                    # 🎨 Static assets
    └── background.jpg          #     Background image for the web UI
```

---

## ⚙️ Installation

### ✅ Prerequisites

- 🐍 Python 3.8+
- 📷 A working webcam (for live feed mode)
- 🖥️ A machine with a GPU is recommended for smoother real-time inference, but CPU will also work (slower)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/tabidah-usmani/Smart-Blind-Assistance-System.git
cd Smart-Blind-Assistance-System
```

### 2️⃣ Create a Virtual Environment *(recommended)*

```bash
python -m venv venv
source venv/bin/activate      # 🐧 macOS/Linux
venv\Scripts\activate         # 🪟 Windows
```

### 3️⃣ Install Dependencies

The core dependencies used by this project are:

```bash
pip install flask opencv-python torch torchvision pygame pyttsx3
```

| Package | Why it's needed |
|---|---|
| 🌐 `flask` | Web server and routing |
| 🎥 `opencv-python` | Video capture and image processing |
| 🔥 `torch` / `torchvision` | Runs the YOLOv5 and MiDaS deep learning models |
| 🔊 `pygame` | Plays sound alerts |
| 🗣️ `pyttsx3` | Optional text-to-speech feedback |

> 💡 Tip: Consider freezing these into a `requirements.txt` (`pip freeze > requirements.txt`) so future installs are one command: `pip install -r requirements.txt`.

### 4️⃣ Download Model Weights

YOLOv5 and MiDaS weights are typically fetched automatically via `torch.hub` on first run — make sure you have an internet connection the first time you launch the app so the models can download. 🌐⬇️

### 5️⃣ Prepare the Dataset (optional, for image upload testing)

Download the [COCO 2017 dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) 📦 and place sample images in the `uploads/` folder if you'd like to test the image-upload detection pipeline with real-world data.

---

## 🚀 Usage

### ▶️ Start the Application

```bash
python app.py
```

Then open your browser and visit:

```
http://127.0.0.1:5000/
```

🎉 You should now see the home page of the Smart Blind Assistance System!

### 🎥 Live Feed Mode

Navigate to:

```
http://127.0.0.1:5000/live_feed
```

This activates your webcam and streams a **live, annotated video feed** — showing bounding boxes 🟩, object labels 🏷️, and estimated distances 📏 in real time, along with audio alerts 🔊 as obstacles approach.

### 📤 Image Upload Mode

Navigate to:

```
http://127.0.0.1:5000/coco_dataset
```

Here you can upload a static image (`.png`, `.jpg`, or `.jpeg`) and see the detection + depth estimation pipeline run on it — great for testing, demos, or exploring how the system performs on the COCO dataset. 🖼️🔍

---

## 🌐 Routes & Endpoints

| Route | Description |
|---|---|
| `/` | 🏠 Home page — entry point to the application |
| `/live_feed` | 🎥 Streams the live camera feed with real-time object detection and depth overlays |
| `/coco_dataset` | 📤 Upload interface for processing static PNG/JPG/JPEG images |

---

## 🔊 Alert System Details

The alert system is designed to be **simple, fast, and unambiguous** — critical for a real-time assistive tool:

- 📍 **Position Detection**: The system determines whether an obstacle lies in the **left**, **center**, or **right** portion of the frame based on its bounding box location.
- 📏 **Distance Estimation**: Using the MiDaS depth map, the system approximates how close the object is.
- 🔔 **Tone Alerts**: Pre-generated WAV files in the `sounds/` directory are played via `pygame` when an obstacle crosses a proximity threshold — different tones or repetition rates can signal different urgency levels.
- 🗣️ **Optional Spoken Feedback**: When `pyttsx3` is enabled, the system can literally *say* what's ahead (e.g., *"Chair on your left, 1 meter"*), which is especially helpful for first-time or lower-vision users who benefit from explicit verbal cues.

---

## 📦 Dataset

This project uses the **[COCO 2017 Dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)** 📦 — a large-scale, richly annotated object detection dataset — for testing the image upload and detection pipeline. COCO's broad category coverage (people, furniture, vehicles, animals, everyday objects, etc.) makes it a strong benchmark for validating detection quality in the kinds of scenes a visually impaired user might actually encounter. 🏙️🛋️🚗

---

## 🐞 Troubleshooting & Error Handling

The app includes handling for several common issues:

| Issue | What Happens |
|---|---|
| 📷 Camera not connected/detected | User is alerted that the camera feed could not be initialized |
| 📁 Missing model files | App alerts the user rather than crashing silently |
| 🖼️ Unsupported upload format | Upload is rejected with a message indicating accepted formats (PNG/JPG/JPEG) |
| 🔌 General runtime errors | Errors are surfaced with user-facing feedback instead of failing silently |

---

## 🎯 Use Cases

- 🧑‍🦯 Assisting visually impaired individuals with everyday indoor/outdoor navigation
- 🏫 Educational demonstrations of combining object detection + depth estimation
- 🧪 A research/prototyping base for accessibility-focused computer vision tools
- 🏠 Indoor obstacle awareness (furniture, doorways, pets, people)
- 🚶 Basic outdoor hazard awareness (poles, curbs, vehicles, other pedestrians)

---

## 🚧 Limitations

- 📏 MiDaS provides **relative**, not absolute, depth — real-world distance accuracy may vary by scene and lighting.
- 💡 Performance can degrade in poor lighting, heavy motion blur, or very cluttered scenes.
- ⚡ Real-time performance depends heavily on hardware — CPU-only machines may experience noticeable lag.
- 🎥 Currently relies on a single monocular camera; no stereo or LiDAR-based distance validation.
- 🌐 Requires a browser and local network access — not yet packaged as a standalone mobile app.

---

## 🔮 Future Improvements

- 📱 Package as a native or progressive mobile app for true on-the-go use
- 🧭 Add GPS-based outdoor navigation and route guidance
- 🗣️ Expand voice feedback with more natural, context-aware descriptions
- 🎚️ Adaptive alert sensitivity based on user walking speed
- 🧠 Explore lighter-weight model variants (e.g., YOLOv5n, smaller MiDaS variants) for better performance on low-power/embedded devices
- 🌍 Multi-language support for spoken alerts
- 📊 On-device logging/analytics to help users and caregivers understand common hazard patterns

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! 🙌 Feel free to:

1. 🍴 Fork the repository
2. 🌿 Create a new branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes
4. 📤 Push to the branch
5. 🔁 Open a Pull Request

---

## 👩‍💻 Credits

Developed by **Tabidah Usmani**, **Amna Javaid**, **Tasmiya Asad**, **Ziyan Murtaza** ✨

