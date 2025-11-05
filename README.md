# DataScienceCourse8_Project
**Deep Learning Project — Image Classification using Convolutional Neural Networks (CNNs)**  
*Internshala Data Science PGC — Course 8 Final Project by Utkarsh Anand*

---

## 🧩 Overview
This project focuses on implementing a **Convolutional Neural Network (CNN)** from scratch using **TensorFlow and Keras** to classify images from the **CIFAR-10 dataset**.  
It covers the full pipeline from **data preparation → CNN architecture → model training → evaluation → optimizer experiments.**

---

## 🧠 Project Objectives
1. Understand and implement CNN architecture for image classification.  
2. Explore model evaluation metrics and confusion matrices.  
3. Experiment with optimizers like **Adam**, **SGD**, and **RMSprop** to improve accuracy.  
4. Compare model performances and visualize results.

---

## 🧰 Dataset
**Dataset:** CIFAR-10 (60,000 color images, 10 classes)  
**Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck  
**Train/Test Split:** 80% train, 20% test  

---

## 🧪 Tasks Summary

### 🧩 Task 1 — Data Exploration and Preparation
- Loaded CIFAR-10 dataset from `tensorflow.keras.datasets`.  
- Normalized pixel values from `[0,255]` → `[0,1]`.  
- Displayed 5 sample images with class labels.  
- Verified balanced class distribution.

```python
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

✅ **Dataset shape:**  
- Training: (50,000, 32, 32, 3)  
- Testing: (10,000, 32, 32, 3)

---

### ⚙️ Task 2 — Build and Train CNN Model
**Model Architecture**
```
Conv2D(32) → ReLU → MaxPooling → Dropout(0.25)
Conv2D(64) → ReLU → MaxPooling → Dropout(0.25)
Flatten → Dense(128, ReLU) → Dropout(0.5)
Dense(10, Softmax)
```

**Model Summary:**
| Layer Type | Output Shape | Params |
|-------------|--------------|--------|
| Conv2D | (30, 30, 32) | 896 |
| MaxPooling2D | (15, 15, 32) | 0 |
| Conv2D | (13, 13, 64) | 18,496 |
| Flatten | (2304) | 0 |
| Dense | (128) | 295,040 |
| Dense | (10) | 1,290 |

**Total Parameters:** 315,722  
**Optimizer:** Adam  
**Loss Function:** Categorical Crossentropy  
**Epochs:** 10  
**Batch Size:** 64  

✅ **Training Accuracy:** 63%  
✅ **Validation Accuracy:** 67%  
➡️ *Mild underfitting observed; more epochs or deeper layers could improve performance.*

---

### 🧾 Task 3 — Evaluate Model
- Evaluated on 12,000 test images.  
- Generated confusion matrix and classification report.  

**Test Accuracy:** `66.55%`

**Sample Classification Report:**
| Class | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|----------|
| Airplane | 0.74 | 0.71 | 0.72 | 1200 |
| Automobile | 0.80 | 0.81 | 0.80 | 1200 |
| Cat | 0.51 | 0.40 | 0.45 | 1200 |
| Frog | 0.76 | 0.75 | 0.76 | 1200 |
| ... | ... | ... | ... | ... |
| **Overall Accuracy** | **0.67** |   |   | **12000** |

✅ **Best classified:** Automobile, Ship  
⚠️ **Most confusion:** Cat ↔ Dog  

---

### 🧠 Task 4 — Experimentation (Optimizer Comparison)
Trained the same CNN using **Adam**, **SGD**, and **RMSprop** for comparison.

| Optimizer | Test Accuracy (%) |
|------------|-------------------|
| Adam | 66.55 |
| SGD | 47.42 |
| RMSprop | 66.35 |

✅ **Best Optimizer:** *RMSprop* (stable convergence & slightly better generalization).  

---

## 📊 Visualizations
- **Training vs Validation Accuracy/Loss curves**  
- **Confusion Matrix** with Seaborn heatmap  
- **Sample Correct & Incorrect Predictions** (5 each)  

---

## 🧰 Tools & Libraries
| Category | Tools |
|-----------|-------|
| Language | Python 3 |
| Framework | TensorFlow / Keras |
| Libraries | numpy, matplotlib, seaborn, sklearn |
| Environment | Google Colab |
| Model | CNN (2 Conv blocks + Dense) |

---

## 📂 Files Included
```
Deep Learning Assignment.docx.pdf                  # Problem statement
Copy of DeepLearningAssignment.ipynb - Colab.pdf   # Notebook output (PDF)
Copy_of_DeepLearningAssignment.ipynb               # Jupyter Notebook
copy_of_deeplearningassignment.py                  # Python script version
```

---

## 🧭 How to Review
1. Open `.ipynb` in Jupyter or Google Colab.  
2. View `.pdf` for explanations, plots, and outputs.  
3. Run `.py` for a clean executable script version.  

---

## 📚 Learning Outcomes
✔ Understanding of CNN architecture design  
✔ Model evaluation using confusion matrix  
✔ Effects of different optimizers (Adam, SGD, RMSprop)  
✔ Training visualization and performance comparison  

---

## 👤 Author
**Utkarsh Anand**  
Data Science PGC — Course 8 Final Project  
Internshala Placement Guarantee Program
