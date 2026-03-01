# Breast-Cancer-Prediction
A machine learning–powered web application built using Streamlit that predicts whether a breast tumor is benign or malignant based on diagnostic tumor measurements. The system uses a Support Vector Machine (SVM) model trained on the Breast Cancer Wisconsin dataset to deliver accurate predictions along with confidence scores.

📌 Project Description

Breast cancer is one of the most common cancers worldwide, and early detection significantly improves survival rates.
This project demonstrates how machine learning can assist in healthcare by providing a simple, interactive, and data-driven prediction system.

Users can input tumor features through a clean web interface, and the application instantly returns:

Cancer prediction (Benign / Malignant)

Confidence score

Model accuracy

Confusion matrix visualization


⚙️ Technologies Used

Python

Streamlit – Web application framework

NumPy – Numerical computations

Pandas – Data handling and analysis

Scikit-learn – Machine learning model & preprocessing

Matplotlib & Seaborn – Data visualization

🤖 Machine Learning Model

Algorithm: Support Vector Machine (SVM)

Kernel: Linear

Feature Scaling: StandardScaler

Dataset: Breast Cancer Wisconsin Dataset (sklearn)

Train–Test Split: 80% training, 20% testing

🧠 Key Features

User-friendly sidebar for feature input

Real-time cancer prediction

Confidence score for predictions

Model accuracy displayed on test data

Confusion matrix heatmap for evaluation

Input validation to prevent unrealistic predictions

📊 Model Performance

Evaluation Metrics Used:

Accuracy Score

Confusion Matrix

Provides transparency into model reliability and performance

🚀 How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/your-username/your-repository-name.git
2️⃣ Navigate to the project folder
cd your-repository-name
3️⃣ Install required dependencies
pip install -r requirements.txt
4️⃣ Run the Streamlit app
streamlit run breast_cancer_prediction.py
📂 Project Structure
📁 Breast-Cancer-Prediction-System
 ├── breast_cancer_prediction.py
 ├── README.md

🎯 Applications

Academic and educational learning

Demonstration of ML in healthcare

Streamlit + ML portfolio project

Beginner-friendly ML deployment example

This application is not a medical diagnostic tool and should not be used for real-world clinical decisions. Always consult a certified medical professional for medical advice.

