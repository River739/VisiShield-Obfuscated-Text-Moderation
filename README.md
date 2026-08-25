# 🛡️ VisiShield: Dual-Stage Hybrid AI Content Moderation Engine

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20Interface-orange.svg)](https://gradio.app/)

VisiShield is a production-grade, multi-tier content moderation system designed to catch obfuscated hate speech, leetspeak, spaced-out text bypasses, and acronym slang in real time. 

Instead of relying solely on brittle text matching, VisiShield combines a fast-path **Text Engine (Stage 1)** with a custom **Convolutional Neural Network (Stage 2 - SilhouetteNet)** that evaluates words based on their visual typography and spatial silhouettes.

---

## 🏗️ System Architecture
[ Incoming Raw Comment ]
│
▼
[ Preprocessor ]  ───► Cleans formatting & fixes spaced-out bypasses ("F u c k" ➔ "Fuck")
│
▼
[ Stage 1 Filter ]
├───► Safe Word / Stop Word Dictionary? ───► Natively [ APPROVED ] (0% GPU)
├───► Direct Banned Match / Slang?     ───► Instantly [ REJECTED ]
└───► Levenshtein / Obfuscated?       ───► Escalate to [ SUSPICIOUS ]
│
▼
[ Stage 2 Filter ]
├───► Normalizes & strips suffixes ("fucking" ➔ "fuck")
├───► Renders text to image canvas & applies Gaussian Blur
└───► PyTorch CNN (SilhouetteNet) evaluates visual weight
│
▼
[ Final System Verdict ] ───► HARD REJECT / Hard Drop

---

## 🚀 Key Features
* **Hybrid Multi-Tier Pipeline:** Saves server resources by using dictionary lookups and Levenshtein distance for clean text, only escalating ambiguous shapes to the neural network.
* **Computer Vision Fallback (SilhouetteNet):** Custom PyTorch CNN trained on rendered font images with simulated Gaussian blur to catch visual leetspeak (`sh1t`, `b!tch`, `f*ck`).
* **Advanced Text Preprocessing:** Handles uppercase casing traps, suffix stemming (`-ing`, `-ed`), and complex space-obfuscation patterns (`F u- C..k`).
* **Interactive Web App:** Includes a fully functional Gradio frontend for live testing and audit logging.

---

## ⚙️ Installation & Running Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/River739/VisiShield-Content-Moderation.git](https://github.com/River739/VisiShield-Content-Moderation.git)
   cd VisiShield-Content-Moderation

1. Install dependencies:
Bash
pip install -r requirements.txt

2. Launch the Web App:
Bash
python app.py