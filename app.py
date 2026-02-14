import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                           recall_score, f1_score, matthews_corrcoef,
                           confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
import importlib.util
import sklearn
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')

# ============= FIX FOR VERSION COMPATIBILITY =============
# Add missing monotonic_cst attribute to DecisionTreeClassifier
# This allows loading models trained with older scikit-learn versions

print(f"Current scikit-learn version: {sklearn.__version__}")

# Patch the DecisionTreeClassifier class
if not hasattr(DecisionTreeClassifier, 'monotonic_cst'):
    print("Patching DecisionTreeClassifier to add missing 'monotonic_cst' attribute")
    DecisionTreeClassifier.monotonic_cst = None
    
    # Also patch the _support_missing_values method if needed
    def patched_support_missing_values(self, X):
        """Patched method to handle missing values without monotonic_cst"""
        try:
            # Try the original method first
            from sklearn.tree._classes import _check_missing_values
            if not self._check_missing_values(X):
                return False
            # If we get here, check other conditions
            return (self._support_missing_values(X) and 
                   getattr(self, 'monotonic_cst', None) is None)
        except:
            # Fallback for older versions
            return False
    
    # Apply the patch if the method exists
    if hasattr(DecisionTreeClassifier, '_support_missing_values'):
        DecisionTreeClassifier._support_missing_values = patched_support_missing_values
    
    print("Patch applied successfully!")

# Also patch RandomForestClassifier if needed
if not hasattr(RandomForestClassifier, 'monotonic_cst'):
    RandomForestClassifier.monotonic_cst = None

# Custom unpickler for loading models with compatibility issues
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Handle DecisionTreeClassifier from different scikit-learn versions
        if module == 'sklearn.tree._classes' and name == 'DecisionTreeClassifier':
            return DecisionTreeClassifier
        # Handle RandomForestClassifier
        if module == 'sklearn.ensemble._forest' and name == 'RandomForestClassifier':
            return RandomForestClassifier
        return super().find_class(module, name)

def load_model_with_compatibility(model_path):
    """
    Load a model with compatibility fixes for different scikit-learn versions
    """
    try:
        # Try normal loading first
        return joblib.load(model_path)
    except Exception as e:
        st.warning(f"⚠️ Normal loading failed for {os.path.basename(model_path)}: {str(e)}")
        
        # If it's a DecisionTreeClassifier error, try to load with custom unpickler
        if 'monotonic_cst' in str(e) or 'DecisionTreeClassifier' in str(e):
            try:
                with open(model_path, 'rb') as f:
                    model = CustomUnpickler(f).load()
                st.success(f"✅ Successfully loaded {os.path.basename(model_path)} with compatibility fix")
                return model
            except Exception as e2:
                st.error(f"❌ Compatibility loading also failed for {os.path.basename(model_path)}: {str(e2)}")
                return None
        else:
            st.error(f"❌ Failed to load {os.path.basename(model_path)}: {str(e)}")
            return None

# ============= END OF FIX =============

# Check if xgboost is installed
xgboost_available = importlib.util.find_spec("xgboost") is not None
if not xgboost_available:
    st.warning("⚠️ XGBoost is not installed. Please install it using: pip install xgboost")

# Page configuration
st.set_page_config(
    page_title="ML Classification Models - Assignment 2",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0d47a1;
        padding: 0.5rem;
        border-left: 5px solid #1E88E5;
        background-color: #f0f2f6;
        border-radius: 0 10px 10px 0;
        margin: 1.5rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.8rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-color: #ffeeba;
        color: #856404;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .download-button {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #28a745;
        color: white;
        text-decoration: none;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
    .download-button:hover {
        background-color: #218838;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 Machine Learning Classification Models</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Amit Kulshrestha</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">BITS ID:- 2025aa05088@wilp.bits-pilani.ac.in</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">M.Tech (AIML) - Assignment 2 | Multi-Model Classification Comparison</p>', unsafe_allow_html=True)

# Installation instructions if xgboost is missing
if not xgboost_available:
    st.markdown("""
    <div class="warning-box">
        <h4>📦 Missing Dependencies</h4>
        <p>XGBoost is not installed. Please run the following command in your terminal:</p>
        <code>pip install xgboost</code>
        <p>After installation, restart the app.</p>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
if 'model_comparison' not in st.session_state:
    st.session_state.model_comparison = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'column_order' not in st.session_state:
    st.session_state.column_order = None

# Function to load dataset from dataset directory
@st.cache_data
def load_dataset_from_directory():
    """Load dataset from the dataset directory"""
    dataset_paths = [
        os.path.join('dataset', 'heart.csv'),
        os.path.join('dataset', 'heart_disease.csv'),
        os.path.join('dataset', 'HeartDisease.csv')
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df, path
            except Exception as e:
                st.warning(f"⚠️ Error loading {path}: {str(e)}")
    
    return None, None

# Load pre-trained models and scaler
@st.cache_resource
def load_models_and_scaler():
    models = {}
    model_names = {
        'Logistic Regression': 'logistic_regression',
        'Decision Tree': 'decision_tree',
        'K-Nearest Neighbor': 'k_nearest_neighbor',
        'Naive Bayes': 'naive_bayes',
        'Random Forest': 'random_forest',
    }
    
    # Only add XGBoost if available
    if xgboost_available:
        model_names['XGBoost'] = 'xgboost'
    
    try:
        # Check if models directory exists
        if not os.path.exists('models'):
            st.error("❌ Models directory not found! Please run train_models.py first.")
            return None, None, None, None, None
        
        # Load model info if available
        model_info_path = os.path.join('models', 'model_info.pkl')
        if os.path.exists(model_info_path):
            model_info = joblib.load(model_info_path)
        else:
            model_info = None
        
        # Load scaler
        scaler_path = os.path.join('models', 'scaler.pkl')
        if not os.path.exists(scaler_path):
            st.error("❌ Scaler not found! Please run train_models.py first.")
            return None, None, None, None, None
        
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.error(f"❌ Error loading scaler: {str(e)}")
            return None, None, None, None, None
        
        # Load feature names
        feature_names_path = os.path.join('models', 'feature_names.pkl')
        if os.path.exists(feature_names_path):
            feature_names = joblib.load(feature_names_path)
        else:
            feature_names = None
        
        # Load column order
        column_order_path = os.path.join('models', 'column_order.pkl')
        if os.path.exists(column_order_path):
            column_order = joblib.load(column_order_path)
        else:
            column_order = None
        
        # Load categorical columns info
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        
        # Load label encoders for categorical columns
        label_encoders = {}
        for col in categorical_cols:
            encoder_path = os.path.join('models', f'label_encoder_{col}.pkl')
            if os.path.exists(encoder_path):
                try:
                    label_encoders[col] = joblib.load(encoder_path)
                except Exception as e:
                    st.warning(f"⚠️ Could not load encoder for {col}: {str(e)}")
        
        # Load models with compatibility fixes
        missing_models = []
        loaded_models = []
        for display_name, filename in model_names.items():
            model_path = os.path.join('models', f'{filename}.pkl')
            if os.path.exists(model_path):
                model = load_model_with_compatibility(model_path)
                if model is not None:
                    models[display_name] = model
                    loaded_models.append(display_name)
                else:
                    missing_models.append(display_name)
            else:
                missing_models.append(display_name)
        
        if missing_models:
            st.warning(f"⚠️ Some models not found or couldn't be loaded: {', '.join(missing_models)}")
        
        if loaded_models:
            st.success(f"✅ Successfully loaded models: {', '.join(loaded_models)}")
        
        if not models:
            st.error("❌ No models could be loaded!")
            return None, None, None, None, None
            
        return models, scaler, feature_names, label_encoders, column_order
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None, None

# Load model comparison results
@st.cache_data
def load_comparison_results():
    try:
        comparison_path = os.path.join('models', 'model_comparison.csv')
        if os.path.exists(comparison_path):
            comparison_df = pd.read_csv(comparison_path, index_col=0)
            return comparison_df
        else:
            return None
    except Exception as e:
        st.error(f"❌ Error loading comparison results: {str(e)}")
        return None

# Function to preprocess input data
def preprocess_input_data(df, label_encoders, column_order=None):
    """
    Preprocess the input data by encoding categorical variables
    """
    df_processed = df.copy()
    
    # Define categorical columns based on the dataset
    categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    
    # Encode categorical columns
    for col in categorical_columns:
        if col in df_processed.columns:
            if col in label_encoders:
                # Use saved label encoder
                try:
                    # Handle unknown categories
                    known_classes = label_encoders[col].classes_
                    df_processed[col] = df_processed[col].apply(
                        lambda x: x if x in known_classes else known_classes[0]
                    )
                    df_processed[col] = label_encoders[col].transform(df_processed[col])
                except Exception as e:
                    st.warning(f"⚠️ Error encoding column {col}: {str(e)}")
                    # Fallback: create new encoder
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col])
            else:
                # Create new label encoder if not available
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                st.session_state.label_encoders[col] = le
    
    # Ensure columns are in the correct order if column_order is provided
    if column_order is not None:
        # Add any missing columns with default values
        for col in column_order:
            if col not in df_processed.columns:
                df_processed[col] = 0  # Default value for missing columns
        
        # Reorder columns to match training data
        df_processed = df_processed[column_order]
    
    return df_processed

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Model Configuration")
    
    # Installation helper
    if not xgboost_available:
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <strong>📦 Quick Install:</strong><br>
            <code>pip install xgboost</code>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset section
    st.markdown("### 📁 Dataset Options")
    
    # Download dataset button
    dataset_df, dataset_path = load_dataset_from_directory()
    if dataset_df is not None:
        st.markdown("#### Download Original Dataset")
        
        # Convert dataset to CSV for download
        csv = dataset_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Heart Disease Dataset (CSV)",
            data=csv,
            file_name="heart_disease_dataset.csv",
            mime="text/csv",
            key="download-dataset"
        )
        
        # Show dataset info
        with st.expander("📊 Dataset Info"):
            st.write(f"**Shape:** {dataset_df.shape[0]} rows, {dataset_df.shape[1]} columns")
            st.write("**Columns:**")
            for col in dataset_df.columns:
                st.write(f"- {col}")
    else:
        st.warning("⚠️ Dataset not found in 'dataset' directory")
        
        # Manual upload option
        st.markdown("#### Upload Test Data")
        st.markdown("*Upload test data (CSV format)*")
    
    # Sample data option
    st.markdown("#### Or use sample data")
    use_sample_data = st.checkbox("Use sample heart disease data for testing")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="uploader")
    
    # Model selection
    st.markdown("### 🤖 Model Selection")
    model_options = [
        'Logistic Regression',
        'Decision Tree',
        'K-Nearest Neighbor',
        'Naive Bayes',
        'Random Forest',
    ]
    
    # Only add XGBoost to options if available
    if xgboost_available:
        model_options.append('XGBoost')
    
    selected_model_name = st.selectbox(
        "Choose a classification model:",
        model_options
    )
    
    # Advanced options
    st.markdown("### ⚙️ Advanced Options")
    show_feature_importance = st.checkbox("Show Feature Importance", value=True)
    show_cv_scores = st.checkbox("Show Cross-Validation Scores", value=True)
    
    # About section
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    **Dataset**: Heart Disease UCI
    - Features: 11
    - Instances: 918
    - Task: Binary Classification
    
    **BITS Virtual Lab** - Assignment 2
    Submission Date: 15-Feb-2026
    """)
    
    # Instructions to train models
    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    **To train models, run:**
    ```bash
    python train_models.py
    ```
    
    **To install missing dependencies:**
    ```bash
    pip install xgboost
    ```
    """)

# Main content - Dataset Preview Section
if dataset_df is not None:
    st.markdown('<h2 class="sub-header">📂 Dataset Preview</h2>', unsafe_allow_html=True)
    
    col_preview1, col_preview2 = st.columns([3, 1])
    
    with col_preview1:
        st.dataframe(dataset_df.head(10), use_container_width=True)
    
    with col_preview2:
        st.markdown("#### Dataset Statistics")
        st.metric("Total Samples", dataset_df.shape[0])
        st.metric("Features", dataset_df.shape[1])
        
        if 'HeartDisease' in dataset_df.columns:
            pos_count = dataset_df['HeartDisease'].sum()
            neg_count = len(dataset_df) - pos_count
            st.metric("Positive Cases", pos_count)
            st.metric("Negative Cases", neg_count)
            st.metric("Positive Rate", f"{pos_count/len(dataset_df):.2%}")

# Model Performance Comparison
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="sub-header">📊 Model Performance Comparison</h2>', unsafe_allow_html=True)
    
    # Load and display comparison table
    comparison_df = load_comparison_results()
    
    if comparison_df is not None:
        st.markdown("#### All Models - Evaluation Metrics")
        
        # Format the dataframe
        styled_df = comparison_df.style.format({
            'Accuracy': '{:.4f}',
            'AUC': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1': '{:.4f}',
            'MCC': '{:.4f}'
        }).background_gradient(cmap='Blues', axis=None)
        
        st.dataframe(styled_df, use_container_width=True)
        st.session_state.model_comparison = comparison_df
        
        # Highlight best model
        st.markdown("#### 🏆 Best Performing Model")
        best_model = comparison_df['Accuracy'].idxmax()
        best_accuracy = comparison_df.loc[best_model, 'Accuracy']
        st.success(f"**{best_model}** achieved the highest accuracy: **{best_accuracy:.4f}**")
    else:
        st.warning("⚠️ Model comparison results not found. Please run train_models.py first.")

with col2:
    st.markdown('<h2 class="sub-header">📈 Quick Stats</h2>', unsafe_allow_html=True)
    
    if comparison_df is not None:
        # Display metrics in cards
        avg_accuracy = comparison_df['Accuracy'].mean()
        avg_auc = comparison_df['AUC'].mean()
        avg_f1 = comparison_df['F1'].mean()
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Average Accuracy", f"{avg_accuracy:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Average AUC", f"{avg_auc:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2_2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Average F1 Score", f"{avg_f1:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Models", len(model_options))
            st.markdown('</div>', unsafe_allow_html=True)

# Model performance visualization
if comparison_df is not None:
    st.markdown('<h2 class="sub-header">📊 Performance Visualization</h2>', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Radar chart for model comparison
        fig = go.Figure()
        
        for model in comparison_df.index:
            fig.add_trace(go.Scatterpolar(
                r=comparison_df.loc[model, ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']].values,
                theta=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Bar chart comparison
        fig = go.Figure(data=[
            go.Bar(name='Accuracy', x=comparison_df.index, y=comparison_df['Accuracy']),
            go.Bar(name='F1 Score', x=comparison_df.index, y=comparison_df['F1']),
            go.Bar(name='AUC', x=comparison_df.index, y=comparison_df['AUC'])
        ])
        
        fig.update_layout(
            title="Key Metrics Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Selected model analysis
st.markdown(f'<h2 class="sub-header">🔍 Detailed Analysis: {selected_model_name}</h2>', unsafe_allow_html=True)

# Handle dataset upload and predictions
test_data = None

if use_sample_data:
    # Create sample data for testing
    st.info("Using sample heart disease data for testing")
    sample_data = pd.DataFrame({
        'Age': [63, 37, 41, 56, 57],
        'Sex': ['M', 'F', 'M', 'F', 'M'],
        'ChestPainType': ['ATA', 'NAP', 'ASY', 'ATA', 'NAP'],
        'RestingBP': [145, 130, 130, 120, 140],
        'Cholesterol': [233, 250, 204, 236, 192],
        'FastingBS': [1, 0, 0, 0, 0],
        'RestingECG': ['Normal', 'ST', 'LVH', 'Normal', 'Normal'],
        'MaxHR': [150, 187, 172, 178, 148],
        'ExerciseAngina': ['N', 'Y', 'N', 'N', 'Y'],
        'Oldpeak': [2.3, 3.5, 1.4, 0.8, 0.4],
        'ST_Slope': ['Down', 'Flat', 'Up', 'Up', 'Flat'],
        'HeartDisease': [1, 0, 1, 0, 1]
    })
    test_data = sample_data
    st.dataframe(test_data.head())
elif uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset loaded successfully! Shape: {test_data.shape}")
        st.dataframe(test_data.head())
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")

if test_data is not None:
    try:
        # Load models
        models, scaler, feature_names, label_encoders, column_order = load_models_and_scaler()
        
        if models is not None and scaler is not None:
            # Prepare data
            # Check if target column exists
            target_col = None
            if 'HeartDisease' in test_data.columns:
                target_col = 'HeartDisease'
            elif 'target' in test_data.columns:
                target_col = 'target'
            
            if target_col:
                X_test = test_data.drop(target_col, axis=1)
                y_test = test_data[target_col]
            else:
                X_test = test_data
                y_test = None
            
            # Show original data before preprocessing
            with st.expander("View Original Data (Before Preprocessing)"):
                st.write("Categorical values before encoding:")
                categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
                for col in categorical_cols:
                    if col in X_test.columns:
                        st.write(f"{col}: {X_test[col].unique()}")
            
            # Preprocess the input data (encode categorical variables)
            X_test_processed = preprocess_input_data(X_test, label_encoders, column_order)
            
            # Show processed data
            with st.expander("View Processed Data (After Encoding)"):
                st.write("Numerical values after encoding:")
                st.dataframe(X_test_processed.head())
            
            # Get the selected model
            model = models.get(selected_model_name)
            
            if model is None:
                st.error(f"❌ Model {selected_model_name} not available. Please train it first.")
            else:
                # Scale if necessary
                if selected_model_name in ['Logistic Regression', 'K-Nearest Neighbor']:
                    X_test_scaled = scaler.transform(X_test_processed)
                    X_test_use = X_test_scaled
                else:
                    X_test_use = X_test_processed
                
                # Make predictions with error handling
                try:
                    y_pred = model.predict(X_test_use)
                except Exception as e:
                    st.error(f"❌ Error making predictions: {str(e)}")
                    st.info("Try retraining the models with the updated train_models.py script.")
                    y_pred = None
                
                if y_pred is not None:
                    # Display results in columns
                    col5, col6 = st.columns(2)
                    
                    with col5:
                        st.markdown("#### 📋 Evaluation Metrics")
                        
                        if y_test is not None:
                            # Calculate metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            if hasattr(model, "predict_proba"):
                                try:
                                    y_pred_proba = model.predict_proba(X_test_use)[:, 1]
                                    auc = roc_auc_score(y_test, y_pred_proba)
                                except:
                                    auc = roc_auc_score(y_test, y_pred)
                            else:
                                auc = roc_auc_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred)
                            recall = recall_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred)
                            mcc = matthews_corrcoef(y_test, y_pred)
                            
                            # Display metrics
                            metrics_data = {
                                'Metric': ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC'],
                                'Value': [accuracy, auc, precision, recall, f1, mcc]
                            }
                            metrics_df = pd.DataFrame(metrics_data)
                            st.dataframe(metrics_df.style.format({'Value': '{:.4f}'}), use_container_width=True)
                        else:
                            st.info("ℹ️ Upload test data with 'HeartDisease' or 'target' column to see evaluation metrics")
                            
                            # Show predictions
                            results_df = X_test.copy()
                            results_df['Predicted_HeartDisease'] = y_pred
                            st.markdown("#### Predictions:")
                            st.dataframe(results_df)
                    
                    with col6:
                        st.markdown("#### 🔥 Confusion Matrix")
                        
                        if y_test is not None:
                            cm = confusion_matrix(y_test, y_pred)
                            
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            ax.set_title(f'Confusion Matrix - {selected_model_name}')
                            st.pyplot(fig)
                            
                            # Classification report
                            st.markdown("#### 📊 Classification Report")
                            report = classification_report(y_test, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df.style.format('{:.4f}'), use_container_width=True)
                        else:
                            st.warning("⚠️ Cannot generate confusion matrix without true labels")
                            
                            # Show prediction distribution
                            pred_counts = pd.Series(y_pred).value_counts()
                            fig = px.pie(values=pred_counts.values, names=['No Heart Disease', 'Heart Disease'], 
                                       title='Prediction Distribution')
                            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)
else:
    st.info("👆 Please upload a CSV file from the sidebar or use sample data to test the model")

# Feature Importance (for applicable models)
if show_feature_importance and selected_model_name in ['Decision Tree', 'Random Forest', 'XGBoost']:
    st.markdown("#### 🌟 Feature Importance")
    
    models, _, feature_names, _, _ = load_models_and_scaler()
    if models is not None:
        model = models.get(selected_model_name)
        
        if model is not None and hasattr(model, 'feature_importances_'):
            if feature_names is None:
                # If feature names not saved, use generic names
                feature_names = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                               'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                               'Oldpeak', 'ST_Slope']
            
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Create dataframe for visualization
            importance_df = pd.DataFrame({
                'Feature': [feature_names[i] if i < len(feature_names) else f'Feature {i}' for i in indices],
                'Importance': [importances[i] for i in indices]
            })
            
            # Plot
            fig = px.bar(importance_df.head(10), 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title=f'Top 10 Feature Importances - {selected_model_name}',
                        color='Importance',
                        color_continuous_scale='Blues')
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

# Cross-validation scores
if show_cv_scores:
    st.markdown("#### 🎯 Cross-Validation Scores (5-Fold)")
    
    try:
        cv_path = os.path.join('models', 'cv_scores.csv')
        if os.path.exists(cv_path):
            cv_data = pd.read_csv(cv_path)
            
            fig = px.box(cv_data, 
                        y=cv_data.columns,
                        title="5-Fold Cross-Validation Scores Distribution",
                        labels={'value': 'Accuracy', 'variable': 'Model'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Cross-validation scores not available")
    except Exception as e:
        st.info("Cross-validation scores not available")

# Model observations
st.markdown('<h2 class="sub-header">📝 Model Performance Observations</h2>', unsafe_allow_html=True)

observations = {
    'Logistic Regression': "Performs well as a baseline model with good interpretability. AUC score indicates good separation capability. Works best when features are scaled and linear relationships exist.",
    'Decision Tree': "Easy to interpret but prone to overfitting. With max_depth=5, shows balanced performance. Feature importance helps in understanding key predictors.",
    'K-Nearest Neighbor': "Performance heavily dependent on k value and scaling. Non-parametric approach captures local patterns but can be sensitive to noise.",
    'Naive Bayes': "Fast training and prediction. Works well despite feature independence assumption. Good baseline for probabilistic classification.",
    'Random Forest': "Ensemble method shows robust performance with high accuracy. Handles non-linear relationships well and reduces overfitting compared to single trees.",
    'XGBoost': "Best performing model with highest accuracy and AUC. Gradient boosting effectively handles complex patterns. Shows excellent generalization."
}

# Filter observations based on available models
available_observations = {k: v for k, v in observations.items() if k in model_options}
obs_df = pd.DataFrame(list(available_observations.items()), columns=['Model', 'Observations'])
st.dataframe(obs_df, use_container_width=True)

# Footer
st.markdown('---')
st.markdown("""
<div class="footer">
    <p>M.Tech (AIML) - Machine Learning Assignment 2 | BITS Virtual Lab | Submission Deadline: 15-Feb-2026</p>
    <p>Implemented Models: Logistic Regression | Decision Tree | K-NN | Naive Bayes | Random Forest | XGBoost</p>
    <p style="color: #1E88E5;">📧 For queries: neha.vinayak@pilani.bits-pilani.ac.in</p>
</div>
""", unsafe_allow_html=True)


