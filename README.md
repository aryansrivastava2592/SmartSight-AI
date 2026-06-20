# 🚗 SmartSight-AI: Real-Time Road Hazard Detection

SmartSight-AI is an intelligent computer vision system designed to identify road hazards such as potholes, debris, and rocks in real time. Powered by a custom-trained YOLOv8 model and delivered through an interactive Streamlit dashboard, the application helps improve road safety by providing instant visual and voice alerts.

## ✨ Key Features

* **📡 Real-Time Hazard Detection:** Detects road hazards using live camera feeds with OpenCV and a custom YOLOv8 model.
* **🔊 Instant Voice Alerts:** Uses text-to-speech to notify drivers about detected hazards in real time.
* **📊 Interactive Analytics Dashboard:** Visualizes detection trends and statistics using Plotly and SQLite.
* **🕒 Detection History:** Automatically stores high-confidence detections along with snapshot images for later review.
* **⚙️ Admin Panel:** Monitor system performance, review safety metrics, and manage stored data directly from the dashboard.

## 🛠️ Technology Stack

| Category        | Technologies           |
| --------------- | ---------------------- |
| Deep Learning   | YOLOv8 (Ultralytics)   |
| Computer Vision | OpenCV                 |
| Frontend        | Streamlit              |
| Database        | SQLite                 |
| Data Analysis   | Pandas, Plotly Express |
| Audio Alerts    | pyttsx3                |

## 📋 Prerequisites

Before installing the project, ensure you have:

* Python 3.9 or later
* Git
* A webcam or compatible camera device

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/aryansrivastava2592/SmartSight-AI.git
cd SmartSight-AI
```

### 2. Create and Activate a Virtual Environment

**macOS/Linux**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Application

```bash
streamlit run app.py
```

Open the local Streamlit URL displayed in your terminal to access the dashboard.

## 📷 Supported Hazard Types

* Potholes
* Road debris
* Rocks and obstacles

## 📈 Future Enhancements

* GPS-based hazard mapping
* Cloud synchronization for shared hazard reports
* Mobile application integration
* Support for multiple camera sources
* Advanced driver assistance features

## 🤝 Contributing

Contributions are welcome. Feel free to open an issue, suggest improvements, or submit a pull request.

## 📄 License

This project is licensed under the MIT License.
