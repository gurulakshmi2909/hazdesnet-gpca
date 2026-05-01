# 🌫 GPCA-Augmented HazDesNet for Haze Density Estimation

## 📌 Overview

This project implements a deep learning model for estimating haze density from images. It integrates **HazDesNet architecture** with **Gaussian Process Channel Attention (GPCA)** to improve feature representation and prediction accuracy.

---

## 🧠 Model Architecture

Input Image → CNN Layers → Feature Map → GPCA Attention → Dense Layers → Haze Density + Uncertainty

---

## ⚙️ Features

* Deep learning-based haze estimation
* Gaussian Process-inspired attention mechanism
* Predicts both **mean (density)** and **uncertainty**
* Lightweight and efficient model

---

## 📊 Dataset

* Custom labeled haze dataset
* Images resized to 128×128
* Labels normalized between 0–1

---

## 🏋️ Training

* Loss Function: Gaussian Negative Log-Likelihood
* Optimizer: Adam
* Epochs: 50
* Batch Size: 16

---

## 📈 Results

* Evaluation Loss: -4.43
* Stable training and good generalization
* Provides uncertainty-aware predictions

---

## 🖼 Sample Output

Predicted Haze Density: 0.79
Uncertainty: 0.992

---

## 🚀 How to Run

```bash
python train.py
python eval.py
python predict.py
```

---

## 📦 Requirements

* Python 3.x
* TensorFlow
* OpenCV
* NumPy
* Pandas

---

## 🔮 Future Improvements

* Real-time haze detection
* Larger dataset training
* Mobile application integration
* Advanced architectures

---

## 👨‍💻 Author

Guru Lakshmi A

