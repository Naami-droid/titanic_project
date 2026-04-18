import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
import sys

# Ensure the src directory is in the path so we can import train
sys.path.append(os.path.dirname(__file__))
from train import load_data, preprocess_data

# 1. Memory Cache for Data Loading and Preprocessing
@st.cache_data
def load_and_clean_data(filepath):
    raw_df = load_data(filepath)
    clean_df = preprocess_data(raw_df.copy())
    return raw_df, clean_df

def train_model_streamlit(X_train, y_train, model_type='logistic_regression'):
    if model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(max_depth=5, random_state=42)
    else:
        raise ValueError("Invalid model type specified.")
    model.fit(X_train, y_train)
    return model

def evaluate_model_streamlit(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions, output_dict=True)
    return acc, conf_matrix, class_report, predictions

def display_metrics(acc, conf_matrix, class_report, title):
    st.subheader(f"📊 {title} Metrics")
    st.metric("Accuracy Score", f"{acc:.4f}")
    
    m_col1, m_col2 = st.columns([1, 1])
    
    with m_col1:
        st.write("**Classification Report:**")
        report_df = pd.DataFrame(class_report).transpose()
        st.dataframe(report_df.style.format(precision=2).background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))

    with m_col2:
        st.write("**Confusion Matrix:**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Died', 'Survived'], 
                    yticklabels=['Died', 'Survived'], ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)

def main():
    st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")
    st.title("🚢 Titanic Survival Analysis & Prediction")

    st.sidebar.header("Configuration")
    app_mode = st.sidebar.radio("Choose App Mode", ["Single Model View", "Model Comparison"])

    # Define the data path
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Titanic-Dataset.csv"))

    try:
        df, processed_df = load_and_clean_data(data_path)
        X = processed_df.drop('Survived', axis=1)
        y = processed_df['Survived'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    if app_mode == "Single Model View":
        model_choice = st.sidebar.selectbox("Select Model", ("Logistic Regression", "Random Forest"))
        model_type = 'logistic_regression' if model_choice == "Logistic Regression" else 'random_forest'
        
        st.header(f"1. {model_choice} Performance")
        with st.spinner(f"Training {model_choice}..."):
            model = train_model_streamlit(X_train, y_train, model_type=model_type)
            acc, conf_matrix, class_report, _ = evaluate_model_streamlit(model, X_test, y_test)
            display_metrics(acc, conf_matrix, class_report, model_choice)

    else:
        st.header("1. Model Comparison: Logistic Regression vs Random Forest")
        
        # Train both models
        with st.spinner("Training both models for comparison..."):
            lr_model = train_model_streamlit(X_train, y_train, model_type='logistic_regression')
            lr_acc, lr_conf, lr_report, _ = evaluate_model_streamlit(lr_model, X_test, y_test)
            
            rf_model = train_model_streamlit(X_train, y_train, model_type='random_forest')
            rf_acc, rf_conf, rf_report, _ = evaluate_model_streamlit(rf_model, X_test, y_test)

        # Log accuracies for verification
        st.write(f"Logistic Regression Accuracy: {lr_acc:.4f}")
        st.write(f"Random Forest Accuracy: {rf_acc:.4f}")

        # Comparison Table
        st.subheader("🏆 Accuracy Comparison")
        comparison_df = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest"],
            "Accuracy": [lr_acc, rf_acc]
        })
        st.table(comparison_df.style.highlight_max(axis=0, subset=['Accuracy'], color='lightgreen'))

        # Detailed Metrics side-by-side or stacked
        tab1, tab2 = st.tabs(["Logistic Regression", "Random Forest"])
        with tab1:
            display_metrics(lr_acc, lr_conf, lr_report, "Logistic Regression")
        with tab2:
            display_metrics(rf_acc, rf_conf, rf_report, "Random Forest")

    st.header("2. Data Preview")
    with st.expander("Show Raw and Processed Data"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original Data (Head):**")
            st.dataframe(df.head())
        with col2:
            st.write("**Processed Data (Head):**")
            st.dataframe(processed_df.head())

if __name__ == "__main__":
    main()
