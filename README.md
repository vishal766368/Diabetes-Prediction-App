# 🩺 Diabetes Prediction System
**AI-Powered Risk Assessment Tool**

This repository features a machine learning application built with **Streamlit** that predicts the probability of diabetes using the **SVM (Support Vector Machine)** algorithm.

## 🚀 Live Application
[**Click here to launch the Streamlit App**](https://diabetes-prediction-by-krishna.streamlit.app/)

## 🖥️ Application Preview
*If the images below do not load, you can view them directly in the repository files.*

1. **Dashboard Overview:** 
   * Displays the model type (SVM), Accuracy (~78%), and Dataset size (768 samples).
  
     <img width="1917" height="1038" alt="Screenshot 2026-02-05 141202" src="https://github.com/user-attachments/assets/a356ea47-e6a3-48bb-b4be-ee4035ffeaaf" />


2. **Prediction Results:**
   * Shows the risk level (e.g., "LOW RISK - Not Diabetic") and a probability gauge (e.g., 12% Risk).
  
     <img width="1917" height="1037" alt="Screenshot 2026-02-05 141244" src="https://github.com/user-attachments/assets/2182e82f-0cfb-4e97-9541-63c4342f8810" />


3. **Risk Analysis & Recommendations:**
   * Provides identified risk factors like High Blood Pressure or Genetic Predisposition, along with health recommendations.
  
     <img width="1917" height="1038" alt="Screenshot 2026-02-05 141258" src="https://github.com/user-attachments/assets/de820750-f14b-4b4a-be50-4fcf98bf18af" />


## 📊 Model Performance
- **Algorithm:** Support Vector Machine (SVM)
- **Accuracy:** ~78%
- **Dataset Size:** 768 samples
- **Outputs:** The model provides a **Probability Breakdown** (e.g., 88% Non-Diabetic vs 12.0% Diabetic) and a visual **Risk Level Gauge**.

## 📁 Project Structure
The repository is composed of the following core files:
* `app.py`: The Streamlit web application interface.
* `Diabetes Prediction.ipynb`: Detailed Jupyter Notebook containing data exploration and model training (98.5% of total code).
* `diabetes.csv`: The clinical dataset used for training.
* `diabetes_model.pkl`: The saved SVM model.
* `scaler.pkl`: Scaling tool for data normalization.
* `requirements.txt`: Necessary dependencies.

## 🧠 What I Learned
- **Algorithm Comparison:** I learned that while Random Forest is powerful, **SVM provided the best accuracy (78%)** for this specific clinical dataset.
- **Feature Impact:** Identifying how factors like **Glucose**, **Blood Pressure**, and **BMI** directly influence the risk percentage.
- **Actionable AI:** I implemented a system that doesn't just give a "Yes/No" but provides **Recommendations**—such as regular health check-ups and monitoring BMI—based on the results.

## ⚠️ Medical Disclaimer
This tool is for **educational purposes only**. It should NOT replace professional medical advice. Always consult healthcare professionals for medical concerns.
