import streamlit as st
import requests
import sounddevice as sd
import numpy as np
import wave
import tempfile
import time
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000/predict-gender/"
st.set_page_config(page_title="Real-Time Gender Detector", page_icon="🎤", layout="centered")
st.markdown(
    """
    <h1 style="text-align: center; color: #4A90E2;">🎤 Real-Time Gender Detection</h1>
    <p style="text-align: center; color: #666;">
        Speak into the microphone, and let AI predict your gender in real-time! 
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)
col1, col2 = st.columns(2)
start_button = col1.button("Start Streaming", use_container_width=True)
stop_button = col2.button("Stop Streaming", use_container_width=True)

output_text = st.empty()  
graph_placeholder = st.empty()  

SAMPLE_RATE = 16000  
DURATION = 2 
CHANNELS = 1
is_streaming = False

def record_audio():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16)
        sd.wait()  

        with wave.open(temp_wav.name, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(recording.tobytes())

        return temp_wav.name, recording

def send_audio_to_api(audio_path):
    with open(audio_path, "rb") as audio_file:
        files = {"file": audio_file}
        response = requests.post(API_URL, files=files)
        return response.json()

def plot_waveform(audio_data):
    plt.figure(figsize=(5, 2))
    plt.plot(audio_data, color="royalblue")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    plt.xticks([])
    plt.yticks([])
    
    graph_placeholder.pyplot(plt)  

if start_button:
    is_streaming = True
    with st.status("🎙️ **Listening...** Speak into the mic!", expanded=True):
        while is_streaming:
            audio_file_path, audio_data = record_audio()
            graph_placeholder.empty()  
            plot_waveform(audio_data)  
            result = send_audio_to_api(audio_file_path)
            gender = result.get("gender", "Unknown")
            confidence = result.get("confidence", "N/A")
            output_text.markdown(
                f"""
                <div style="text-align: center; font-size: 24px; font-weight: bold; color: #4A90E2;">
                    Detected: {gender} 🎭
                </div>
                <div style="text-align: center; font-size: 18px; color: #888;">
                    Confidence: {confidence}
                </div>
                """,
                unsafe_allow_html=True,
            )

            time.sleep(2)  

if stop_button:
    is_streaming = False
    output_text.markdown(
        "<div style='text-align: center; color: #D9534F; font-size: 20px;'>⛔ Streaming Stopped.</div>",
        unsafe_allow_html=True,
    )
