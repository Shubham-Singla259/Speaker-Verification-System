import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from scipy.spatial.distance import cosine

# Load trained model
model = tf.keras.models.load_model("speaker_verification_model.h5")

# Set Streamlit Page Configuration
st.set_page_config(page_title="🎤 Speaker Classification", layout="wide", page_icon="🎙️")

# Custom CSS for Styling
st.markdown(
    """
    <style>
        .stApp {background: linear-gradient(to right, #e3f2fd, #ffffff);}
        .title {text-align: center; font-size: 2.5rem; font-weight: bold; color: #2E86C1;}
        .subtitle {text-align: center; font-size: 1.2rem; color: #555;}
        .result {font-size: 1.5rem; text-align: center; padding: 12px; border-radius: 8px;}
        .success {background-color: #d4edda; color: #155724;}
        .fail {background-color: #f8d7da; color: #721c24;}
        .metric-box {padding: 12px; background-color: #f1f1f1; border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True
)

# Title Section
st.markdown("<h1 class='title'>🎤 Real-Time Speaker Classification</h1>", unsafe_allow_html=True)
st.markdown("<h5 class='subtitle'>Upload a reference audio file and classify new recordings</h5>", unsafe_allow_html=True)
st.write("---")

# Function to extract MFCC features
def extract_mfcc(audio_data, sr=22050, n_mfcc=40, max_pad_length=200):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    
    # Padding MFCC to ensure uniform shape
    if mfcc.shape[1] < max_pad_length:
        pad_width = max_pad_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_length]
    
    return mfcc

# Layout: Two Columns
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📌 Upload Target Speaker's Reference Audio")
    target_audio = st.file_uploader("Upload a reference audio file", type=["wav"], key="target_audio")
    
    if target_audio is not None:
        y_target, sr_target = librosa.load(target_audio, sr=22050)
        st.audio(target_audio, format='audio/wav')
        target_mfcc = extract_mfcc(y_target, sr=sr_target)

        # Normalize and reshape for model input
        scaler = StandardScaler()
        target_mfcc = scaler.fit_transform(target_mfcc)
        target_mfcc = target_mfcc.reshape(1, target_mfcc.shape[0], target_mfcc.shape[1], 1)

        st.success("✅ Reference audio processed successfully!")

    st.subheader("🎤 Upload Audio for Classification")
    test_audio = st.file_uploader("Upload an audio file to classify", type=["wav"], key="test_audio")

with col2:
    st.subheader("📊 Classification Results")

    if target_audio is not None and test_audio is not None:
        y_test, sr_test = librosa.load(test_audio, sr=22050)
        st.audio(test_audio, format='audio/wav')

        test_mfcc = extract_mfcc(y_test, sr=sr_test)
        test_mfcc = scaler.transform(test_mfcc)
        test_mfcc = test_mfcc.reshape(1, test_mfcc.shape[0], test_mfcc.shape[1], 1)

        # Model Predictions
        test_prediction = model.predict(test_mfcc)
        target_prediction = model.predict(target_mfcc)

        # Compute Similarity Score
        similarity_score = 1 - cosine(target_prediction.flatten(), test_prediction.flatten())
        similarity_percentage = similarity_score * 100

        # Classification Decision
        threshold = 70
        is_target_speaker = similarity_percentage >= threshold
        result_text = "✅ Matched: This is the Target Speaker!" if is_target_speaker else "❌ Not Matched: This is a Non-Target Speaker."
        result_class = "success" if is_target_speaker else "fail"
        
        # Display Result
        st.markdown(f"<div class='result {result_class}'>{result_text}</div>", unsafe_allow_html=True)
        st.markdown(f"<h4>🔹 Similarity Score: {similarity_percentage:.2f}%</h4>", unsafe_allow_html=True)

        # Evaluation Metrics
        y_true = [1]
        y_pred = [1 if is_target_speaker else 0]
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        st.write("---")
        st.subheader("📈 Model Evaluation Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='metric-box'><strong>✔ Accuracy:</strong> {acc*100:.2f}%</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-box'><strong>✔ F1-Score:</strong> {f1*100:.2f}%</div>", unsafe_allow_html=True)
        
        st.write("---")

        # 📊 MFCC Feature Visualization
        st.subheader("📊 MFCC Features")
        fig, ax = plt.subplots(figsize=(6, 3))
        librosa.display.specshow(test_mfcc[0, :, :, 0], x_axis='time', cmap='coolwarm')
        ax.set(title="MFCC Features", xlabel="Time", ylabel="MFCC Coefficients")
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        st.pyplot(fig)

