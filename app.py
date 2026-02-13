import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Heart Disease Prediction App")
st.caption("Developed by Ganesh G – 2025AA05882")
st.markdown("---")

# Upload CSV
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Dataset Preview:")
    st.write(data.head())

    # Check if target column exists
    if "target" not in data.columns:
        st.error("Uploaded file must contain 'target' column.")
    else:
        X = data.drop("target", axis=1)
        y = data["target"]

        # List models
        model_files = [f for f in os.listdir("models") if f.endswith(".pkl")]
        model_name = st.selectbox("Select Model", model_files)

        if st.button("Run Prediction"):
            model_path = os.path.join("models", model_name)
            model = joblib.load(model_path)

            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1]

            acc = accuracy_score(y, preds)
            auc = roc_auc_score(y, probs)
            prec = precision_score(y, preds)
            rec = recall_score(y, preds)
            f1 = f1_score(y, preds)
            mcc = matthews_corrcoef(y, preds)

            st.subheader("Evaluation Metrics")
            st.write(f"Accuracy: {acc:.3f}")
            st.write(f"AUC: {auc:.3f}")
            st.write(f"Precision: {prec:.3f}")
            st.write(f"Recall: {rec:.3f}")
            st.write(f"F1 Score: {f1:.3f}")
            st.write(f"MCC: {mcc:.3f}")

            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y, preds)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
