import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter
import io
import librosa
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

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

    st.subheader("Adjust Frequency Bands")
    bass = st.slider("Bass (60–250 Hz)", 0.0, 2.0, 1.0, 0.1)
    mid = st.slider("Midrange (250 Hz – 4 kHz)", 0.0, 2.0, 1.0, 0.1)
    treble = st.slider("Treble (4–10 kHz)", 0.0, 2.0, 1.0, 0.1)

    output = apply_equalizer(data, fs, [bass, mid, treble])

    # Save to buffer and playback
    buf = io.BytesIO()
    sf.write(buf, output, fs, format='WAV')
    st.audio(buf, format='audio/wav')
    st.download_button("Download Processed Audio", buf.getvalue(), file_name="equalized_output.wav")

    # --- Waveform visualization with white background and black waveform ---
    st.markdown("---")
    st.subheader("📈 Waveform Visualization")

    fig, ax = plt.subplots(figsize=(10, 4))
    time = np.linspace(0, len(output) / fs, num=len(output))

    # Plot waveform in black
    ax.plot(time, output, color="black", linewidth=1.0)

    # Set white background
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Set black bold text
    font_props = {
        "fontsize": 12,
        "color": "black",
        "fontweight": "bold"
    }
    shadow = [pe.withStroke(linewidth=3, foreground="white")]

    ax.set_title("Processed Audio Waveform", path_effects=shadow, **font_props)
    ax.set_xlabel("Time [s]", path_effects=shadow, **font_props)
    ax.set_ylabel("Amplitude", path_effects=shadow, **font_props)

    # Ticks and grid styling
    ax.tick_params(colors="black")
    ax.grid(True, linestyle="--", color="gray", alpha=0.3)

    st.pyplot(fig)
