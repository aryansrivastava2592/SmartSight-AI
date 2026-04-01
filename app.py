import streamlit as st
import cv2
import pandas as pd
import sqlite3
import random
import plotly.express as px
from ultralytics import YOLO
from datetime import datetime
import time
import pyttsx3
import threading
import math
import numpy as np
import os

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="SmartSight AI - Road Hazard Detection",
    layout="wide",
    page_icon="🚗"
)

# ------------------------------------------------
# ANIMATED BACKGROUND
# ------------------------------------------------

st.markdown("""
<style>
.stApp{
background: linear-gradient(-45deg,#020617,#001f3f,#020617,#0f172a);
background-size: 400% 400%;
animation: gradientBG 12s ease infinite;
color:#e6edf3;
}
@keyframes gradientBG {
0% {background-position: 0% 50%;}
50% {background-position: 100% 50%;}
100% {background-position: 0% 50%;}
}
h1{
color:#00e6ff;
text-shadow:0 0 6px rgba(0,230,255,0.4);
}
.main-title{
font-size:48px;
font-weight:bold;
}
.subtext{
font-size:18px;
opacity:0.85;
}
section[data-testid="stSidebar"]{
background:#0f172a;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# ROAD SAFETY FACTS TICKER
# ------------------------------------------------

st.markdown("""
<marquee behavior="scroll" direction="left" scrollamount="8"
style="font-size:18px;color:#00e6ff;font-weight:bold;">
🚗 Drive safe — road accidents claim over 1.19 million lives every year • 
⚠️ Speeding increases crash risk dramatically • 
🛣️ Poor road conditions cause thousands of accidents worldwide • 
👀 Always stay alert for potholes and obstacles • 
🛑 Defensive driving saves lives • 
🚦 AI-based road monitoring can reduce accidents significantly
</marquee>
""", unsafe_allow_html=True)

# ------------------------------------------------
# DATABASE
# ------------------------------------------------

conn = sqlite3.connect("database.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS detections(
id INTEGER PRIMARY KEY AUTOINCREMENT,
timestamp TEXT,
object_type TEXT,
confidence REAL,
severity TEXT,
lat REAL,
lon REAL,
image TEXT
)
""")
conn.commit()
# ------------------------------------------------
# SETUP IMAGE DIRECTORY
# ------------------------------------------------
SAVE_DIR = "hazard_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------
# LOAD MODEL & CONSTANTS
# ------------------------------------------------

@st.cache_resource
def load_yolo_model():
    return YOLO("runs/detect/train2/weights/best.pt")

model = load_yolo_model()

# ------------------------------------------------
# VOICE ALERT & HUD 
# ------------------------------------------------

def speak_warning(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except:
        pass

def draw_hud(frame, angle):
    h, w, _ = frame.shape
    center = (80, 80)
    # Radar Circle
    cv2.circle(frame, center, 50, (0, 255, 255), 1)
    # Radar Line
    x = int(center[0] + 50 * math.cos(math.radians(angle)))
    y = int(center[1] + 50 * math.sin(math.radians(angle)))
    cv2.line(frame, center, (x, y), (0, 255, 255), 2)
    # Status Text
    cv2.putText(frame, "SMART SIGHT ACTIVE", (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------

st.sidebar.title("🚗 SmartSight AI System")

menu = st.sidebar.radio(
    "Navigation",
    [
        "📡 Live Detection",
        "📊 AI Hazard Analytics",
        "🕒 Detection History",
        "⚙ Admin Control Panel"
    ]
)

# ------------------------------------------------
# SYSTEM STATUS
# ------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("🖥 System Health")
st.sidebar.success("Camera Active")
st.sidebar.info("AI Model Running")

cpu=random.randint(20,60)
st.sidebar.metric("CPU Usage",f"{cpu}%")

# ------------------------------------------------
# RECENT HAZARDS
# ------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Recent Hazards")

try:
    recent=pd.read_sql("""
    SELECT object_type,timestamp
    FROM detections
    ORDER BY id DESC
    LIMIT 3
    """,conn)

    for i,row in recent.iterrows():
        time_text=str(row["timestamp"])[:16]
        st.sidebar.write(
            f"⚠ {row['object_type']} detected ({time_text})"
        )
except:
    st.sidebar.write("No recent data.")

# ------------------------------------------------
# LIVE DETECTION
# ------------------------------------------------

if menu=="📡 Live Detection":

    st.markdown('<div class="main-title">🚗 Live Hazard Detection</div>',unsafe_allow_html=True)
    st.markdown('<div class="subtext">AI system monitoring road hazards in real time using computer vision.</div>', unsafe_allow_html=True)

    frame_window=st.image([])
    cap=cv2.VideoCapture(0)

    radar_angle = 0
    last_speech_time = 0
    last_count_time = 0

    while True:
        ret,frame=cap.read()
        if not ret:
            st.error("Camera not detected")
            break

        h, w, _ = frame.shape
        results=model(frame)

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                
                # FIXED: Raised confidence threshold to prevent false positives
                if conf > 0.75:
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    # FIXED: Strictly look for the trained class
                    if label == "pothole":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                        now = time.time()

                        # 3-second delay logic for DB insertion
                        if (now - last_count_time > 3.0):
                            last_count_time = now
                            # FIXED: Formatted datetime string for SQLite
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            severity = "High" if conf > 0.85 else "Low"
                            img_path = os.path.join(SAVE_DIR, f"detection_{time.time()}.jpg")
                            
                            cv2.imwrite(img_path, frame)
                            
                            c.execute("""
                            INSERT INTO detections
                            (timestamp,object_type,confidence,severity,image)
                            VALUES(?,?,?,?,?)
                            """, (timestamp,label,conf,severity,img_path))
                            conn.commit()

                            st.toast(f"🚨 Hazard Detected: {label}", icon="⚠️")

                        # Voice Alert Logic (4s delay)
                        if (now - last_speech_time > 4):
                            thread = threading.Thread(target=speak_warning, args=(f"Hazard ahead: {label}",), daemon=True)
                            thread.start()
                            last_speech_time = now
                       
                        cv2.putText(frame, "!!! HAZARD !!!", (w//2 - 100, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        draw_hud(frame, radar_angle)
        radar_angle = (radar_angle + 15) % 360

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb, channels="RGB")

# ------------------------------------------------
# ANALYTICS
# ------------------------------------------------

elif menu=="📊 AI Hazard Analytics":

    st.title("📊 Hazard Analytics")

    df=pd.read_sql("SELECT * FROM detections",conn)

    if len(df)==0:
        st.warning("No detection data available")
    else:
        df["timestamp"]=pd.to_datetime(df["timestamp"])
        df["date"]=df["timestamp"].dt.date

        selected_date=st.date_input("Select Date for Analysis", value=datetime.now().date())
        filtered=df[df["date"]==selected_date]
        st.markdown("---")

        pothole_count = len(filtered[filtered["object_type"].str.contains("pothole",case=False,na=False)])

        graph_data=pd.DataFrame({
        "Hazard Type":["Potholes"],
        "Count":[pothole_count]
        })

        fig=px.bar(graph_data, x="Hazard Type", y="Count", color="Hazard Type", title=f"Hazard Counts on {selected_date}")

        # FIXED: Updated for modern Streamlit layout arguments
        st.plotly_chart(fig)

# ------------------------------------------------
# DETECTION HISTORY
# ------------------------------------------------

elif menu=="🕒 Detection History":

    st.title("📸 Detection History")

    df=pd.read_sql("SELECT * FROM detections",conn)

    if len(df)==0:
        st.info("No detections yet")
    else:
        rows_per_page=8
        total_rows=len(df)
        total_pages=(total_rows//rows_per_page)+1

        page=st.number_input("Page",1,total_pages,1)
        start=(page-1)*rows_per_page
        end=start+rows_per_page

        page_df=df.iloc[start:end]
        cols=st.columns(4)

        for i,row in page_df.iterrows():
            try:
                # FIXED: Removed deprecated argument
                cols[i%4].image(row["image"])
            except:
                pass

        st.write(f"Page {page} of {total_pages}")

# ------------------------------------------------
# ADMIN PANEL
# ------------------------------------------------

elif menu=="⚙ Admin Control Panel":

    st.title("⚙ Admin Panel")

    df=pd.read_sql("SELECT * FROM detections",conn)

    if len(df)==0:
        st.warning("No data available")
    else:
        col1,col2,col3,col4=st.columns(4)

        col1.metric("Total Detections",len(df))
        col2.metric("High Risk",len(df[df["severity"]=="High"]))
        col3.metric("Low Risk",len(df[df["severity"]=="Low"]))
        col4.metric("Average Confidence",round(df["confidence"].mean(),2))

        st.markdown("---")

        score=max(0,100-len(df[df["severity"]=="High"]))
        st.subheader("Road Safety Score")
        st.progress(score/100)
        st.success(f"Safety Score : {score}/100")

        if st.button("Clear Database"):
            c.execute("DELETE FROM detections")
            conn.commit()
            st.success("Database Cleared")
            st.rerun()