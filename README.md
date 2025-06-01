# Image Classification using Deep Learning and Machine Learning

**🚫 This notebook is not intended for public, academic or production use. It is currently part of an ongoing research project. Please do not use or distribute it.**
[License](./LICENSE.txt)

## 📚 Overview

This project explores an image classification pipeline using a combination of **DL** based feature extraction, Dimensionality reduction using **PCA** and classical **ML** classifiers for classifying. This is a project where this hybrid pipeline is used to classify rice grain images.

### The key stages in the workflow include:

- Image preprocessing using Keras and `ImageDataGenerator`.
  - This includes Data Augmentation and Image Normalization.
- Feature extraction with MobileNet V2 (a pretrained DL model based on **Inverted Residual Architecture**).
- Dimensionality reduction via PCA.
- Classification using multiple ML models (SVM, KNN, Logistic Regression, Decision Trees, Random Forests, XGBoost, LightGBM, CatBoost).
- Evaluation through confusion matrix, classification reports, and accuracy metrics.

### Reason for choosing hybrid approch:

- Well suited for low resource devices.
- MobileNet V2 Deep Learning model is good for mobile and embedded devices.
- It is lighter model compared to VGG or any Deep-CNN models.
- Its **Depthwise Seperable Convolution, Inverted Residual and Linear Bottlenecks** Layers are more lightweighted and advantagious than standard CNN based workflow.

## 📁 Dataset

- It is collected from : <a href='https://www.muratkoklu.com/datasets/'>Mural Koklu Dataset</a>.
- A working dataset is created in CSV format (`Image Path Class.csv`), which maps image file paths to their respective class labels.
- Sample classes include: `Arborio`, `Basmati`, `Ipsala`, `Jasmine`, and `Karacadag`.

## 🔧 Models Used

- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Each model is trained and evaluated using features extracted from the MobileNet V2 model and reduced via PCA.

## Workflow:

![How the project as well as the research proceded](.images/procedure.png)

## MobileNet V2 architecture

![MobileNet V2 Model's Architecture](.images/mv2_architecture.png)
![Inverted Residual Architecture](.images/inv_res_block.png)

## 📊 Evaluation Metrics

- Accuracy Score
- Classification Report
- Confusion Matrix (with visualizations)

## 🧪 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
