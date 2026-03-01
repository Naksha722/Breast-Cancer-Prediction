import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


best_model = SVC(kernel='linear', probability=True, random_state=42)
best_model.fit(X_train_scaled, y_train)
accuracy = accuracy_score(y_test, best_model.predict(X_test_scaled))
cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))


st.title("🩺 Breast Cancer Prediction App")
st.write("Enter tumor features in the sidebar and click Predict.")

st.sidebar.header("Tumor Feature Inputs")
feature_names = X.columns[:10]  # Only first 10 for simplicity
input_data = []

# Sidebar inputs
for feature in feature_names:
    val = st.sidebar.number_input(f"{feature}", min_value=0.0, max_value=100.0, value=0.0)
    input_data.append(val)

input_data = np.array(input_data).reshape(1, -1)


if st.button("Predict"):

    # Validate input
    if np.all(input_data == 0):
        st.warning("⚠️ Please enter realistic values — all zeros are not valid tumor data.")
    else:
        # Fill remaining features with dataset mean
        missing_features = np.mean(X, axis=0)[10:]
        input_full = np.concatenate([input_data[0], missing_features])
        input_scaled = scaler.transform(input_full.reshape(1, -1))

        # Prediction
        prediction = best_model.predict(input_scaled)[0]
        probability = best_model.predict_proba(input_scaled)[0][prediction]

        # Display result
        st.subheader("🔍 Prediction Result")
        if prediction == 0:
            st.error(f"Prediction: Malignant (Cancer Detected) | Confidence: {probability*100:.2f}%")
        else:
            st.success(f"Prediction: Benign (No Cancer) | Confidence: {probability*100:.2f}%")

        # Model info
        st.subheader("📊 Model Performance")
        st.write(f"Accuracy on Test Data: {accuracy*100:.2f}%")

        # Confusion Matrix
        st.subheader("📈 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False,
                    xticklabels=data.target_names, yticklabels=data.target_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
