# 🎤 Speaker Verification System using Deep Learning

![Result Image](banner.png)

## 📌 Overview  
This project implements a **Speaker Verification System** using **MFCC feature extraction**, **CNN-LSTM model**, and **Streamlit-based UI** for real-time speaker verification. The system classifies speakers based on their unique voice characteristics and predicts whether a new audio sample belongs to a known speaker.

---

## 🔥 Features  
✅ **Extract MFCC features** from audio files  
✅ **Train a CNN-LSTM model** for speaker classification  
✅ **Evaluate model performance** using accuracy and F1-score  
✅ **Streamlit-based UI** for easy speaker verification  
✅ **Similarity score-based speaker identification** using cosine similarity  

---

## 🫠 Technologies Used  
- **Python** 🐍  
- **TensorFlow/Keras** 🔬  
- **Librosa** 🎵  
- **Streamlit** 🖥️  
- **Scikit-Learn** 📊  
- **Matplotlib** 📈  

---

## 💂️ Project Structure  
```bash
📺 Speaker-Classification
├── 📁 dataset                        # Dataset folder containing audio samples
├── 📁 models                         # Trained models (saved as .h5 files)
├── 📁 streamlit_app                   # Streamlit UI implementation
├── 📝 README.md                      # Project documentation
├── 📄 train.py                        # Model training script
├── 📄 predict.py                      # Speaker prediction script
├── 📄 app.py                          # Streamlit application
└── 📄 requirements.txt                # Required dependencies
```

---

## 📌 Workflow  
Below is a **flowchart** explaining the speaker classification workflow:

```mermaid
graph TD;
    A[Input Audio File] -->|Extract MFCC Features| B(Feature Extraction)
    B -->|Preprocessing & Normalization| C(Train CNN-LSTM Model)
    C -->|Save Trained Model| D(Model Storage)
    D -->|Load Model| E(Streamlit UI)
    E -->|Upload Test Audio| F(Extract MFCC & Normalize)
    F -->|Predict Speaker| G(Compute Similarity Score)
    G -->|Display Result| H(Identify Speaker)
```

---

## 🔧 Installation  

### 1⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/speaker-classification.git
cd speaker-classification
```

### 2⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3⃣ Run Model Training  
```bash
Deep_Learning_Model_Code.ipynb
```

### 4⃣ Start Streamlit App  
```bash
streamlit run app.py
```

---

## 🎯 Model Architecture  
The model consists of:  
- **Conv1D Layers** for extracting spatial features from MFCC  
- **BatchNormalization & MaxPooling** for feature refinement  
- **LSTM Layers** to capture time-series dependencies  
- **Dense Layers** with Softmax activation for classification  

---

## 📊 Results  
Test 1:  
![Result Image](output.JPG)  

Test 2:  
![Result Image](output_2.JPG)  

---

## 🎼 MFCC Feature Visualization  
![MFCC Features](MFCC.png)  

The **Mel-Frequency Cepstral Coefficients (MFCCs)** are extracted from audio signals to capture essential spectral features for speaker verification. The above heatmap visualizes the MFCC features of a speech sample.

### 🔹 Key Aspects:  
- **X-axis (Time):** Represents the duration of the audio signal in seconds.  
- **Y-axis (MFCC Coefficients):** Different MFCC values representing the frequency characteristics of speech.  
- **Color Intensity:** The heatmap colors indicate energy levels in **decibels (dB)**:  
  - **Red shades** → Higher energy levels  
  - **Blue shades** → Lower energy levels  

### 📌 Why MFCC?  
MFCCs mimic human auditory perception by emphasizing perceptually relevant frequency components, making them highly effective for speaker classification and verification.  

---

## 🚀 Future Scope  
- **Real-time Deployment:** Optimize the model for real-time applications, enabling instant speaker verification.  
- **Multi-Language Support:** Expand the dataset to include multiple languages and accents for better generalization.  
- **Noise Robustness:** Improve noise-handling capabilities by incorporating noise reduction techniques.  
- **Edge Computing Compatibility:** Deploy on edge devices like Raspberry Pi or mobile platforms for offline verification.  
- **Integration with Voice Assistants:** Enhance smart home assistants and security systems with speaker verification.  
- **Federated Learning:** Train models on decentralized user data while maintaining privacy and security.  
- **GANs for Data Augmentation:** Use Generative Adversarial Networks (GANs) to synthesize diverse speaker voices and enhance dataset diversity.  

---

## 📩 Contribution  
🚀 Contributions are welcome! Feel free to fork and submit PRs.  

---

## 📞 Contact  
📧 **Email:** shubhamsingla259@gmail.com  

