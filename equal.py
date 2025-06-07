import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter
import io
import librosa
import matplotlib.pyplot as plt

# --- Audio loading function ---
def load_audio(file):
    y, sr = librosa.load(file, sr=None, mono=True)
    return y, sr

# --- Bandpass filter function ---
def bandpass_filter(data, lowcut, highcut, fs, numtaps=101):
    taps = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)
    return lfilter(taps, 1.0, data)

# --- Apply equalizer with adjustable gains ---
def apply_equalizer(data, fs, gains):
    bands = [(60, 250), (250, 4000), (4000, 10000)]  # Bass, Mid, Treble
    processed = np.zeros_like(data)
    for (low, high), gain in zip(bands, gains):
        filtered = bandpass_filter(data, low, high, fs)
        processed += filtered * gain
    return processed

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🎚️ Digital Music Equalizer")

uploaded_file = st.file_uploader("Upload audio file (WAV)", type=["wav"])

if uploaded_file is not None:
    data, fs = load_audio(uploaded_file)
    st.audio(uploaded_file)

    st.markdown("---")
    st.subheader("🎛️ Graphic Equalizer")

    # Use columns to align sliders horizontally (but sliders are horizontal)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Bass**<br><sub>60–250 Hz</sub>", unsafe_allow_html=True)
        bass = st.slider("Bass Gain", 0.0, 2.0, 1.0, 0.1, key="bass")

    with col2:
        st.markdown("**Midrange**<br><sub>250–4k Hz</sub>", unsafe_allow_html=True)
        mid = st.slider("Mid Gain", 0.0, 2.0, 1.0, 0.1, key="mid")

    with col3:
        st.markdown("**Treble**<br><sub>4k–10k Hz</sub>", unsafe_allow_html=True)
        treble = st.slider("Treble Gain", 0.0, 2.0, 1.0, 0.1, key="treble")

    output = apply_equalizer(data, fs, [bass, mid, treble])

    # Save to buffer and playback
    buf = io.BytesIO()
    sf.write(buf, output, fs, format='WAV')
    st.audio(buf, format='audio/wav')
    st.download_button("🎵 Download Processed Audio", buf.getvalue(), file_name="equalized_output.wav")

    st.markdown("---")
    st.subheader("📈 Processed Audio Waveform")
    fig, ax = plt.subplots(figsize=(10, 3))
    time = np.linspace(0, len(output) / fs, num=len(output))
    ax.plot(time, output, linewidth=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Processed Audio Waveform")
    st.pyplot(fig)

    # Optional: mimic preset panel (static)
    with st.sidebar:
        st.markdown("## 🎚️ Preset Menu")
        st.text_input("Name:")
        st.button("Save current settings to preset")
        st.markdown("### Presets")
        st.write("• Audio Technica ATH-M50x\n• Monoprice Quarts\n• Sony MDR-EX1000")
        st.button("Load selected preset")
