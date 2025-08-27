# app.py
#python -m streamlit run app.py
import os
import cv2
import time
import glob
import smtplib
import numpy as np
import pandas as pd
import streamlit as st
import face_recognition
from datetime import datetime
from email.message import EmailMessage

# == CONFIG ==
IMAGES_DIR = "faculty_db"              # Changed to match your folder name
LOG_FILE = "attendance_log.csv"        # CSV log file
DETECTION_SECONDS = 10                 # Timer length for each run
SENDER_EMAIL = "alertsystemb@gmail.com"
SENDER_APP_PASSWORD = "swmh yeve bxkf rkja"   # Gmail App Password
DEFAULT_RECEIVER = "rociga1188@aravites.com"

# Ensure folders/files exist
os.makedirs(IMAGES_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "status", "name", "confidence"]).to_csv(LOG_FILE, index=False)

# == EMAIL ==
def send_email(subject: str, body: str, receiver_email: str = DEFAULT_RECEIVER) -> bool:
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = receiver_email
        msg.set_content(body)
        
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"❌ Email failed: {e}")
        return False

# == ENCODINGS CACHE ==
@st.cache_resource(show_spinner=False)
def load_known_faces(images_dir: str):
    encodings = []
    names = []
    bad_files = []
    
    for path in sorted(glob.glob(os.path.join(images_dir, "*"))):
        if not path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
            
        name = os.path.splitext(os.path.basename(path))[0]
        
        # Load as RGB (face_recognition expects RGB)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            bad_files.append(os.path.basename(path))
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_encodings(img_rgb, model="small")  # 128d embeddings
        
        if len(faces) == 0:
            bad_files.append(os.path.basename(path))
            continue
        elif len(faces) > 1:
            st.warning(f"⚠ Multiple faces detected in {os.path.basename(path)}. Using the first one.")
            
        encodings.append(faces[0])
        names.append(name)
    
    return encodings, names, bad_files

def refresh_encodings_cache():
    load_known_faces.clear()

# == MATCHING ==
def best_match_name(known_encs, known_names, face_encoding, threshold=0.5):
    """
    Returns (matched_bool, name, confidence_float)
    confidence ~ inverse of distance (1 - normalized_distance)
    """
    if not known_encs:
        return False, None, 0.0
    
    distances = face_recognition.face_distance(known_encs, face_encoding)
    idx = np.argmin(distances)
    best_dist = distances[idx]
    
    if best_dist <= threshold:
        # Confidence heuristic: map [0, threshold] -> [1.0, ~0.0]
        confidence = float(max(0.0, 1.0 - (best_dist / threshold)))
        return True, known_names[idx], confidence
    
    return False, None, 0.0

# == UI ==
st.set_page_config(page_title="Faculty Attendance", page_icon="📷", layout="centered")
st.title("📷 Faculty Attendance — Timed Webcam Check")

# == FACULTY MANAGEMENT SECTION ==
st.subheader("👥 Faculty Database Management")

# Upload new faculty images
with st.expander("➕ Add New Faculty Members", expanded=True):
    st.markdown("""
    *Instructions:*
    1. Upload clear photos of faculty members (one person per photo)
    2. The filename should be the person's name (e.g., 'John_Smith.jpg')
    3. Supported formats: JPG, JPEG, PNG
    """)
    
    up_files = st.file_uploader(
        "Upload Faculty Images", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        help="Select one or more images. Each image should contain only one person's face."
    )
    
    if up_files:
        success_count = 0
        error_count = 0
        
        for f in up_files:
            try:
                save_path = os.path.join(IMAGES_DIR, f.name)
                with open(save_path, "wb") as out:
                    out.write(f.getbuffer())
                success_count += 1
            except Exception as e:
                st.error(f"Failed to save {f.name}: {e}")
                error_count += 1
        
        if success_count > 0:
            refresh_encodings_cache()
            st.success(f"✅ Successfully uploaded {success_count} image(s) to faculty database!")
            if error_count > 0:
                st.warning(f"⚠ {error_count} file(s) failed to upload.")
            
            # Show what was uploaded
            st.info("*Uploaded files:* " + ", ".join([f.name for f in up_files]))

# Display current faculty database
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Current Faculty Database")
    known_encodings, known_names, bad_files = load_known_faces(IMAGES_DIR)
    
    if known_names:
        st.success(f"{len(known_names)} faculty members loaded:")
        # Display in a nice format
        for i, name in enumerate(known_names, 1):
            st.write(f"{i}. {name.replace('_', ' ')}")
    else:
        st.warning("⚠ No faculty members found in database. Please upload some images first.")

with col2:
    st.subheader("🗂 Database Stats")
    total_files = len(glob.glob(os.path.join(IMAGES_DIR, "*")))
    st.metric("Total Files", total_files)
    st.metric("Valid Faces", len(known_names))
    st.metric("Invalid Files", len(bad_files))

# Show problematic files if any
if bad_files:
    with st.expander("⚠ Files with Issues"):
        st.warning("The following files couldn't be processed (no face detected or file corrupt):")
        for file in bad_files:
            st.write(f"• {file}")

st.divider()

# == ATTENDANCE CHECKING SECTION ==
st.subheader("🎯 Attendance Detection")

# Settings
col1, col2 = st.columns(2)
with col1:
    receiver = st.text_input("Alert recipient email", value=DEFAULT_RECEIVER)
with col2:
    threshold = st.slider("Match threshold (lower = stricter)", 0.3, 0.8, 0.5, 0.01)

st.divider()

# Session state for restarts/runs
if "run_token" not in st.session_state:
    st.session_state.run_token = 0

# Control buttons
col1, col2 = st.columns(2)
with col1:
    start_btn = st.button(f"▶ Start {DETECTION_SECONDS}s Check", type="primary")
with col2:
    restart_btn = st.button("🔄 Restart Check", type="secondary")

if restart_btn:
    st.session_state.run_token += 1
    st.rerun()

if start_btn:
    if not known_encodings:
        st.error(f"❌ No faculty members in database. Please upload faculty images first.")
    else:
        st.success("📹 Webcam starting… keep your face in view.")
        
        frame_holder = st.empty()
        info_holder = st.empty()
        progress = st.progress(0)
        countdown = st.empty()
        
        # Try both default and DirectShow on Windows
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
        if not cap.isOpened():
            # Fallback
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not access webcam. Please check your camera permissions.")
        else:
            start_t = time.time()
            matched = False
            matched_name = None
            matched_conf = 0.0
            run_id = st.session_state.run_token  # freeze for this loop
            
            while time.time() - start_t < DETECTION_SECONDS:
                # Stop if user pressed "Restart" (rerun increments run_token)
                if run_id != st.session_state.run_token:
                    break
                
                ok, frame = cap.read()
                if not ok:
                    info_holder.error("⚠ Could not read from webcam.")
                    break
                
                # Smaller/faster processing frame
                small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                
                # HOG (CPU) detection
                locations = face_recognition.face_locations(rgb_small, model="hog")
                encs = face_recognition.face_encodings(rgb_small, locations, model="small")
                
                # Try to match any face
                for enc in encs:
                    ok_match, name, conf = best_match_name(known_encodings, known_names, enc, threshold=threshold)
                    if ok_match:
                        matched = True
                        matched_name = name
                        matched_conf = conf
                        break
                
                # Draw boxes + labels on full-size frame for display
                for (top, right, bottom, left) in locations:
                    # Scale back up
                    top *= 2; right *= 2; bottom *= 2; left *= 2
                    color = (0, 255, 0) if matched else (0, 255, 255)
                    label = f"{matched_name.replace('_', ' ')} ({matched_conf:.2f})" if matched else "Scanning…"
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, label, (left, max(30, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Countdown + progress
                elapsed = time.time() - start_t
                remaining = max(0, DETECTION_SECONDS - int(elapsed))
                countdown.markdown(f"⏳ *Time remaining:* {remaining}s")
                progress.progress(min(1.0, elapsed / DETECTION_SECONDS))
                frame_holder.image(frame, channels="BGR")
                
                if matched:
                    break
            
            cap.release()
            
            # Result + email + log
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if matched:
                display_name = matched_name.replace('_', ' ')
                st.success(f"✅ *Faculty PRESENT:* {display_name}  |  *Confidence:* {matched_conf:.2f}")
                body = f"✅ Faculty '{display_name}' detected at {ts} with confidence {matched_conf:.2f}."
                if send_email("✅ Faculty Detected", body, receiver):
                    st.info("📧 Presence notification sent.")
                row = {"timestamp": ts, "status": "Present", "name": display_name, "confidence": f"{matched_conf:.4f}"}
            else:
                st.error("❌ *Faculty ABSENT* within time window.")
                body = f"❌ No matching faculty detected during the {DETECTION_SECONDS}s window at {ts}."
                if send_email("❌ Faculty Not Detected", body, receiver):
                    st.info("📧 Absence notification sent.")
                row = {"timestamp": ts, "status": "Absent", "name": "Unknown", "confidence": "0.0000"}
            
            # Append to CSV
            try:
                df = pd.read_csv(LOG_FILE)
                df.loc[len(df)] = row
                df.to_csv(LOG_FILE, index=False)
            except Exception as e:
                st.error(f"Failed to save to log: {e}")

st.divider()

# == LOG SECTION ==
st.subheader("📜 Attendance History")
try:
    log_df = pd.read_csv(LOG_FILE)
    if len(log_df) > 0:
        # Show recent entries first
        log_df = log_df.sort_values('timestamp', ascending=False)
        st.dataframe(log_df, use_container_width=True)
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Checks", len(log_df))
        with col2:
            present_count = len(log_df[log_df['status'] == 'Present'])
            st.metric("Present Count", present_count)
        with col3:
            if len(log_df) > 0:
                attendance_rate = (present_count / len(log_df)) * 100
                st.metric("Attendance Rate", f"{attendance_rate:.1f}%")
    else:
        st.info("📭 No attendance records yet. Run some checks to see history here.")
except Exception as e:
    st.error(f"Could not load attendance log: {e}")