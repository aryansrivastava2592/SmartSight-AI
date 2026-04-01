# 🚗 SmartSight-AI: Real-Time Road Hazard Detection

SmartSight-AI is a computer vision application built to detect road hazards (potholes, debris, rocks) in real-time. Powered by a custom-trained YOLOv8 model and wrapped in a sleek Streamlit dashboard, this tool aims to improve road safety by providing drivers with visual and audio alerts.

## ✨ Features
* **📡 Live Detection:** Uses OpenCV and YOLOv8 to identify hazards through a webcam or camera feed in real-time.
* **🔊 Voice Alerts:** Integrated `pyttsx3` text-to-speech warns users of upcoming hazards instantly.
* **📊 Hazard Analytics:** An interactive dashboard using `Plotly` and `SQLite` to track and visualize hazard detections over time.
* **🕒 Detection History:** Automatically captures and stores snapshot images of high-confidence hazards for review.
* **⚙️ Admin Panel:** Monitor system health, view safety scores, and manage the underlying SQL database directly from the UI.

## 🛠️ Tech Stack
* **Deep Learning:** YOLOv8 (`ultralytics`)
* **Computer Vision:** OpenCV (`cv2`)
* **Frontend UI:** Streamlit
* **Database:** SQLite
* **Data Visualization:** Pandas, Plotly Express
* **Audio Alerts:** `pyttsx3`

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/aryansrivastava2592/SmartSight-AI.git](https://github.com/aryansrivastava2592/SmartSight-AI.git)
cd SmartSight-AI
