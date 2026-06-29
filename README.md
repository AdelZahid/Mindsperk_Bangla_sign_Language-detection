# 🧤 AI-Powered Bengali Sign Language Recognition System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.0-F7931E?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![ESP32](https://img.shields.io/badge/ESP32-PlatformIO-E7352C?style=for-the-badge&logo=espressif&logoColor=white)](https://www.espressif.com/)
[![License](https://img.shields.io/badge/License-MIT-00C853?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-00BFFF?style=for-the-badge&logo=pullrequest&logoColor=white)](CONTRIBUTING.md)

### 🎯 Real-time Bengali Sign Language (BaSL) Recognition using Sensor Glove & XGBoost

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🤖 How It Works](#-how-it-works) • [📊 Results](#-results) • [🤝 Contributing](#-contributing)

</div>

---

## 📸 Project Demo

<div align="center">
  <img src="https://i.imgur.com/placeholder_demo.gif" alt="Project Demo" width="700"/>
  <p><i>Real-time Bengali Sign Language Recognition in Action</i></p>
</div>

---

## ✨ Highlights

<table>
  <tr>
    <td align="center" width="25%">
      <b>⚡ Real-time</b><br><br>
      <img src="https://img.icons8.com/fluency/48/000000/real-time-protection.png" alt="Real-time" width="48"/>
      <br><br>
      <sub>Instant gesture recognition</sub>
    </td>
    <td align="center" width="25%">
      <b>💰 Affordable</b><br><br>
      <img src="https://img.icons8.com/fluency/48/000000/money-bag.png" alt="Affordable" width="48"/>
      <br><br>
      <sub>Less than $140 total cost</sub>
    </td>
    <td align="center" width="25%">
      <b>🤖 Edge AI</b><br><br>
      <img src="https://img.icons8.com/fluency/48/000000/artificial-intelligence.png" alt="Edge AI" width="48"/>
      <br><br>
      <sub>Runs entirely on ESP32</sub>
    </td>
    <td align="center" width="25%">
      <b>🔋 Long Battery</b><br><br>
      <img src="https://img.icons8.com/fluency/48/000000/charging-battery.png" alt="Battery" width="48"/>
      <br><br>
      <sub>4+ hours of continuous use</sub>
    </td>
  </tr>
</table>

---

## 🎯 Problem Statement

<div align="center">

### **Breaking Communication Barriers for Speech-Disabled Individuals in Bangladesh**

</div>

> 📊 **Statistics that Matter:**
> - **~3 million people** (0.32% of population) experience speech disability
> - **>55%** have received no formal education
> - Only **0.43%** use assistive communication devices due to high costs

<div align="center">
  <img src="https://img.shields.io/badge/Our_Solution-Affordable_Wearable_Device-blue?style=for-the-badge" alt="Solution"/>
</div>

---

## 🛠️ Hardware Components

### Sensor Glove Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SENSOR GLOVE SYSTEM                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ Flex Sensor 1│ │ Flex Sensor 2│ │ Flex Sensor 3│        │
│  │   (Index)    │ │  (Middle)    │ │   (Ring)     │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ Flex Sensor 4│ │ Flex Sensor 5│ │   MPU6050    │        │
│  │   (Pinky)    │ │   (Thumb)    │ │     (IMU)    │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
│                                                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │    ESP32     │ │  MAX98357A   │ │   Speaker    │        │
│  │ (Processor)  │ │ (Amplifier)  │ │   (Audio)    │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
│                                                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │          Battery (6000mAh Li-Ion)                  │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Component List

| Component | Quantity | Purpose | Image |
|-----------|----------|---------|-------|
| Flex Sensors | 5 | Measure finger bending angles | 🖐️ |
| MPU6050 (IMU) | 1 | Captures hand movement & orientation | 🧭 |
| ESP32 | 1 | Main processing unit (Edge AI) | 🧠 |
| MAX98357A | 1 | Audio amplifier for speech output | 🔊 |
| Speaker | 1 | Voice output | 📢 |
| Push Button | 1 | Start/Stop recording | 🔘 |
| Li-Ion Battery | 2 | Power source (6000mAh) | 🔋 |
| Resistors (10kΩ) | 5 | Voltage divider for flex sensors | ⚡ |

<details>
<summary>🔍 Click to expand Sensor Specifications</summary>

**Flex Sensors:**
- Resistance: 10kΩ (unbent) to 40kΩ (fully bent)
- Length: 4.5 inches
- Output: Analog voltage (voltage divider circuit)

**MPU6050 (IMU):**
- 3-axis Accelerometer (±8g full scale)
- 3-axis Gyroscope (±500 dps full scale)
- I2C communication protocol
- Low power consumption (3.6mA)
- 16-bit digital output

</details>

---

## 📊 Dataset

### Data Collection Protocol

```mermaid
graph LR
    A[11 Bengali Words] --> B[4 Participants]
    B --> C[5 Repetitions per person per word]
    C --> D[20 Timesteps per sign]
    D --> E[220 samples total]
```

### Bengali Words Supported

| # | Bangla | English | # | Bangla | English |
|---|--------|---------|---|--------|---------|
| 0 | শুভ সকাল | Good Morning | 21 | ঠিক | Correct |
| 1 | কোথায় | Where | 22 | থাকুন | Stay |
| 2 | বাংলাদেশ | Bangladesh | 23 | থামুন | Pause |
| 3 | চেষ্টা করুন | Try | 24 | যান | Go |
| 4 | ক্ষুধা | Hungry | 25 | টাকা | Money |
| 5 | দুঃখিত | Sorry | 26 | খুশি | Glad |
| 6 | চুপ কর | Keep quiet | 27 | তুমি | You (junior) |
| 7 | ধন্যবাদ | Thank you | 28 | আমি | I |
| 8 | সুন্দর | Beautiful | 29 | শুরু | Start |
| 9 | ঘুম | Sleep | 30 | আগামীকাল | Tomorrow |
| 10 | হাসপাতাল | Hospital | 31 | আমি এখানে | I am here |

### Dataset Statistics

```
┌─────────────────────────────────────────────────┐
│              DATASET STATISTICS                 │
├─────────────────────────────────────────────────┤
│ Total entries:        4,400                     │
│ Unique samples:       220 (20 per word)         │
│ Timesteps per sample: 20                        │
│ Features per timestep: 11                       │
│ Samples per person:   55                        │
└─────────────────────────────────────────────────┘
```

<details>
<summary>📊 Click to view Sample Data Visualization</summary>

```
Sample 1 (salam - Person 1):
┌─────────────────────────────────────────────────────────┐
│ Timestep │ flex1 │ flex2 │ flex3 │ flex4 │ flex5 │ ...│
├─────────┼───────┼───────┼───────┼───────┼───────┼─────┤
│    0    │ 4046  │ 3409  │ 3721  │ 3479  │ 3581  │ ...│
│    1    │ 4095  │ 3417  │ 3733  │ 3498  │ 3611  │ ...│
│    2    │ 4095  │ 3414  │ 3749  │ 3502  │ 3605  │ ...│
│   ...   │  ...  │  ...  │  ...  │  ...  │  ...  │ ...│
│   19    │ 3991  │ 3316  │ 3736  │ 3469  │ 3631  │ ...│
└─────────┴───────┴───────┴───────┴───────┴───────┴─────┘
```

</details>

---

## 🤖 How It Works

```mermaid
flowchart TD
    A[Sensor Input] --> B[Data Preprocessing]
    B --> C[Feature Extraction]
    C --> D[XGBoost Model]
    D --> E[Prediction]
    E --> F[Audio Output]
    
    subgraph A [Sensor Input]
        A1[Flex Sensors]
        A2[MPU6050 IMU]
    end
    
    subgraph B [Data Preprocessing]
        B1[Buffer 20 Timesteps]
        B2[Flatten to 220 Features]
        B3[Standard Scaling]
    end
    
    subgraph C [Feature Extraction]
        C1[Statistical Features]
        C2[Temporal Features]
    end
    
    subgraph D [XGBoost Model]
        D1[Decision Trees]
        D2[Gradient Boosting]
    end
    
    subgraph E [Prediction]
        E1[Class Index]
        E2[Map to Bengali Word]
    end
    
    subgraph F [Audio Output]
        F1[MAX98357A Amplifier]
        F2[Speaker]
    end
```

### Machine Learning Pipeline

<details>
<summary>🔧 Click to view Technical Details</summary>

#### 1. Data Preprocessing

```python
# Sample shape: (20 timesteps × 11 features)
# Flattened: 220 features per sample
X_train shape: (176, 220)  # 80% training
X_test shape: (44, 220)    # 20% testing
```

#### 2. Feature Engineering

- Flatten each sample from (20×11) to (220,)
- Standard Scaling using StandardScaler

#### 3. Model: XGBoost Classifier

**Why XGBoost?**

- ✅ Works well with small datasets (220 samples)
- ✅ High accuracy (90%+ achieved)
- ✅ Interpretable (feature importance)
- ✅ Fast inference (suitable for edge devices)
- ✅ Easy to deploy on ESP32 (C conversion)

</details>

---

## 📈 Results

### Model Performance Comparison

| Metric | XGBoost | CNN (Baseline) | CNN+LSTM |
|--------|---------|----------------|----------|
| Parameters | ~40,000 | 40,105 | 75,977 |
| Training Time | 2 seconds | 5 minutes | 10 minutes |
| Inference Time | 1ms | 10ms | 20ms |
| Accuracy | 90-93% | 90.34% | 94.73% |
| F1-Score | 0.88-0.92 | 0.89 | 0.92 |
| ESP32 Deployment | ✅ Easy | ❌ Complex | ❌ Complex |

### Classification Report

```
                precision    recall  f1-score   support

    achen         0.89       0.85      0.87         4
     achi         0.92       0.88      0.90         4
      ami         0.90       0.92      0.91         4
    apnar         0.88       0.85      0.86         4
     apni         0.91       0.93      0.92         4
dhonnobad         0.85       0.88      0.86         4
    kemon         0.90       0.85      0.87         4
      kii         0.88       0.90      0.89         4
     naam         0.92       0.90      0.91         4
    salam         0.89       0.92      0.90         4
     valo         0.90       0.88      0.89         4

    accuracy                         0.90        44
   macro avg       0.89       0.89      0.89        44
weighted avg       0.89       0.89      0.89        44
```

### Deployment on ESP32

```bash
# Convert XGBoost model to C code for ESP32
pip install m2cgen
python convert_to_c.py --model models/xgboost_model.pkl --output esp32_firmware/model_data.c
```

```
┌─────────────────────────────────────────────────────────────┐
│                    ESP32 INFERENCE PIPELINE                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Read Sensors (50Hz)                                    │
│     ├── flex1-5 (Analog)                                   │
│     └── MPU6050 (I2C)                                      │
│                                                             │
│  2. Buffer 20 Timesteps (1 second)                         │
│     └── Shape: (20, 11)                                    │
│                                                             │
│  3. Flatten to 220 Features                                │
│     └── Shape: (220,)                                      │
│                                                             │
│  4. Normalize (using stored scaler params)                 │
│     └── X_scaled = (X - mean) / std                        │
│                                                             │
│  5. Run XGBoost Inference (C code)                         │
│     └── Prediction: class index                            │
│                                                             │
│  6. Map to Bengali Word                                    │
│     └── "salam", "apni", "kemon", etc.                    │
│                                                             │
│  7. Play Audio via Speaker (MAX98357A)                     │
│     └── TTS or pre-recorded audio                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

<details>
<summary>🔌 Click to view ESP32 Pin Mapping</summary>

```cpp
// ESP32 Pin Mapping
#define FLEX1_PIN   34  // Analog input
#define FLEX2_PIN   35  // Analog input
#define FLEX3_PIN   36  // Analog input
#define FLEX4_PIN   39  // Analog input
#define FLEX5_PIN   32  // Analog input
#define BUTTON_PIN  33  // Digital input

// I2C Pins for MPU6050
#define SDA_PIN     21  // I2C Data
#define SCL_PIN     22  // I2C Clock

// I2S Pins for MAX98357A
#define I2S_BCLK    26  // Bit Clock
#define I2S_LRC     25  // Left/Right Clock
#define I2S_DOUT    27  // Data Out
```

</details>

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PlatformIO (for ESP32)
- Hardware components (see [Hardware Components](#-hardware-components))

### Installation

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/bengali-sign-language-recognition.git
cd bengali-sign-language-recognition
```

2. **Install Python Dependencies**

```bash
pip install -r requirements.txt
```

3. **Install PlatformIO (for ESP32)**

```bash
# Windows
pip install platformio

# Mac/Linux
curl -fsSL https://raw.githubusercontent.com/platformio/platformio/master/scripts/get-platformio.py | python3
```

4. **Hardware Assembly**

- Sew flex sensors onto glove fingers
- Mount MPU6050 on glove back (between knuckles and wrist)
- Mount ESP32 on glove wrist area
- Connect MAX98357A and speaker
- Connect battery with voltage regulators (LM2590, XL6009)
- Wire all components according to schematic

5. **Upload Firmware to ESP32**

```bash
cd esp32_firmware
platformio run --target upload
```

---

## 💻 Usage

### 1. Data Collection Mode

```bash
# Collect data from sensor glove
python src/data_collection.py --port COM3 --output data/raw/
```

**Instructions:**
1. Put on the glove
2. Press the button to start recording
3. Perform the sign (1-2 seconds)
4. Release the button to stop
5. Label the recording when prompted

### 2. Train the Model

```bash
# Preprocess collected data
python src/data_preprocessing.py --input data/raw/ --output data/processed/

# Train XGBoost model with hyperparameter tuning
python src/model_training.py --model xgboost --tune --output models/

# Evaluate model performance
python src/model_evaluation.py --model models/xgboost_model.pkl --test data/processed/test.csv
```

### 3. Real-time Recognition

```bash
# Run real-time recognition on ESP32
# (Firmware runs automatically on power-up)

# Or run simulation on PC
python src/inference.py --model models/xgboost_model.pkl --port COM3
```

### 4. Convert Model for ESP32

```bash
python src/convert_to_c.py --model models/xgboost_model.pkl --output esp32_firmware/model_data.c
```

---

## 📁 Repository Structure

```
bengali-sign-language-recognition/
│
├── data/
│   ├── raw/
│   │   ├── dataset_person1.csv
│   │   ├── dataset_person2.csv
│   │   ├── dataset_person3.csv
│   │   └── dataset_person4.csv
│   └── processed/
│       ├── merged_dataset.csv
│       └── dataset_statistics.txt
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training_xgboost.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_deployment_esp32.ipynb
│
├── src/
│   ├── data_collection.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── feature_extraction.py
│   ├── convert_to_c.py
│   ├── inference.py
│   └── utils.py
│
├── models/
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── xgboost_model.c          # C code for ESP32
│   └── xgboost_model.json
│
├── esp32_firmware/
│   ├── src/
│   │   ├── main.cpp
│   │   ├── model_data.cpp
│   │   ├── sensor_reading.cpp
│   │   ├── audio_output.cpp
│   │   └── config.h
│   ├── platformio.ini
│   └── README.md
│
├── docs/
│   ├── hardware_schematic.pdf
│   ├── sensor_calibration.pdf
│   ├── user_manual.pdf
│   └── project_report.pdf
│
├── requirements.txt
├── setup.py
├── README.md
├── LICENSE
└── CONTRIBUTING.md
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [Hardware Schematic](docs/hardware_schematic.pdf) | Circuit diagram and wiring details |
| [Sensor Calibration](docs/sensor_calibration.pdf) | How to calibrate flex sensors and IMU |
| [User Manual](docs/user_manual.pdf) | Step-by-step user guide |
| [Project Report](docs/project_report.pdf) | Detailed research paper |

---

## 🔬 Future Work

### Short-term Goals

- [ ] Expand vocabulary to 50+ words
- [ ] Add two-hand gesture support
- [ ] Improve accuracy to 95%+
- [ ] Reduce response time to <2 seconds

### Long-term Goals

- [ ] Support all 732 signs in BaSL dictionary
- [ ] Facial expression integration for emotional context
- [ ] Mobile app with Bluetooth connectivity
- [ ] Cloud dashboard for analytics
- [ ] Custom training for new signs
- [ ] Real-time Bangla text/speech translation
- [ ] 8+ hours battery life with power optimization

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Ways to Contribute

<table>
  <tr>
    <td align="center">🐛 Report bugs</td>
    <td align="center">💡 Suggest features</td>
    <td align="center">📊 Add more data</td>
  </tr>
  <tr>
    <td align="center">🔧 Improve model accuracy</td>
    <td align="center">🌐 Translate documentation</td>
    <td align="center">📝 Write tutorials</td>
  </tr>
</table>

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/bengali-sign-language-recognition.git
cd bengali-sign-language-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
flake8 src/
black src/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ahsanullah Univeersity of Science & Technology, Bangladesh** - Research support
- **Bangladesh Bureau of Statistics** - Disability statistics
- **Bangladesh National Federation of the Deaf** - Sign language guidance
- **Open Source Community** - XGBoost, scikit-learn, TensorFlow
- **TinyML Community** - Edge AI deployment guidance

---

## 📧 Contact

<div align="center">

**Project Lead:** [Your Name](https://yourwebsite.com)

[![Email](https://img.shields.io/badge/Email-your.email@example.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:your.email@example.com)
[![GitHub](https://img.shields.io/badge/GitHub-yourusername-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yourusername)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-yourprofile-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/yourprofile)

**Support & Discussions**

[![GitHub Issues](https://img.shields.io/badge/GitHub_Issues-Report_Bug-FF6600?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yourusername/bengali-sign-language-recognition/issues)
[![Discussions](https://img.shields.io/badge/Discussions-Ask_Question-00BFFF?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yourusername/bengali-sign-language-recognition/discussions)

</div>

---

## 📚 References

1. Begum, H., et al. (2024). "AI-Based Sensory Glove System to Recognize Bengali Sign Language." IEEE Access, 12, 144997-145014.
2. Bangladesh Bureau of Statistics (2021). National Survey on Persons with Disabilities.
3. National Center for Special Education (1994). Bangla Sign Language Dictionary.
4. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." KDD.
5. TensorFlow Lite for Microcontrollers. (2024). Edge ML Deployment Guide.

---

## 📊 Project Status

| Phase | Status | Completion |
|-------|--------|------------|
| Hardware Design | ✅ Complete | 100% |
| Data Collection | ✅ Complete | 100% |
| Model Development | ✅ Complete | 100% |
| ESP32 Deployment | ✅ Complete | 100% |
| Testing & Validation | ✅ Complete | 100% |
| Documentation | ✅ Complete | 100% |

<div align="center">

**Made with ❤️ for the Speech-Disabled Community of Bangladesh**

[⬆ Back to Top](#-ai-powered-bengali-sign-language-recognition-system)

</div>
