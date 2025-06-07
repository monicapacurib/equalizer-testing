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
st.title("🎧 Digital Music Equalizer")

uploaded_file = st.file_uploader("Upload audio file (WAV)", type=["wav"])

if uploaded_file is not None:
    data, fs = load_audio(uploaded_file)
    st.audio(uploaded_file)

    st.subheader("🎚️ Adjust Frequency Bands")
    bass = st.slider("Bass (60–250 Hz)", 0.0, 2.0, 1.0, 0.1)
    mid = st.slider("Midrange (250 Hz – 4 kHz)", 0.0, 2.0, 1.0, 0.1)
    treble = st.slider("Treble (4–10 kHz)", 0.0, 2.0, 1.0, 0.1)

    output = apply_equalizer(data, fs, [bass, mid, treble])

    # Save to buffer and playback
    buf = io.BytesIO()
    sf.write(buf, output, fs, format='WAV')
    st.audio(buf, format='audio/wav')
    st.download_button("Download Processed Audio", buf.getvalue(), file_name="equalized_output.wav")

    # --- Stylized waveform visualization ---
    st.markdown("---")
    st.subheader("📈 Processed Audio Waveform")

    plt.style.use("seaborn-v0_8-darkgrid")  # Optional style

    fig, ax = plt.subplots(figsize=(10, 4))
    time = np.linspace(0, len(output) / fs, num=len(output))
    
    ax.plot(time, output, color="#00C6A9", linewidth=1.0, linestyle="-")
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")
    
    ax.set_xlabel("Time [s]", fontsize=12, color="white")
    ax.set_ylabel("Amplitude", fontsize=12, color="white")
    ax.set_title("🎵 Stylized Audio Waveform", fontsize=14, color="white")

    ax.tick_params(colors='white')
    ax.grid(True, linestyle='--', alpha=0.4)

    st.pyplot(fig)
