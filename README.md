# 📊 Machine Learning Assignment 2 -- Classification & Streamlit Deployment

**Course:** M.Tech (AIML/DSE) -- Machine Learning
**Assignment:** Assignment 2
**Dataset Used:** Heart Failure Prediction Dataset

------------------------------------------------------------------------

# 🔷 Problem Statement

The objective of this assignment is to implement and compare multiple
machine learning classification models on the Heart Failure Prediction
dataset and deploy an interactive Streamlit web application for
real-time predictions and model evaluation.

This project demonstrates a complete end-to-end machine learning
workflow including: - Data preprocessing
- Model training and evaluation
- Performance comparison
- Streamlit web application development
- Deployment on Streamlit Community Cloud

------------------------------------------------------------------------

# 🔷 Dataset Description

**Dataset Name:** Heart Failure Prediction Dataset
**Source:** Kaggle

### Dataset Details:

-   Total Instances: 918
-   Total Features: 12+
-   Target Variable: HeartDisease
-   Problem Type: Binary Classification

### Description:

The dataset contains medical attributes of patients such as age, sex,
chest pain type, cholesterol, resting blood pressure, and other health
indicators.
The objective is to predict whether a patient has heart disease or not.

### Preprocessing Steps:

-   Handling missing values
-   Encoding categorical features
-   Feature scaling
-   Train-test split

------------------------------------------------------------------------

# 🔷 Machine Learning Models Used

The following models were implemented:

1.  Logistic Regression
2.  Decision Tree
3.  K-Nearest Neighbor (KNN)
4.  Naive Bayes
5.  Random Forest (Ensemble)
6.  XGBoost (Ensemble)

### Evaluation Metrics Used:

-   Accuracy
-   AUC Score
-   Precision
-   Recall
-   F1 Score
-   Matthews Correlation Coefficient (MCC)

------------------------------------------------------------------------

# 🔷 Model Performance Comparison

  --------------------------------------------------------------------------------
  ML Model     Accuracy     AUC      Precision     Recall    F1 Score     MCC
  ------------ ------------ -------- ------------- --------- ------------ --------
  Logistic     0.8696       0.8969   0.8482        0.9314    0.8879       0.7374
  Regression                                                              

  Decision     0.8098       0.8582   0.8252        0.8333    0.8293       0.6146
  Tree                                                                    

  KNN          0.8913       0.9192   0.8942        0.9118    0.9029       0.7797

  Naive Bayes  0.8913       0.9280   0.8942        0.9118    0.9029       0.7797

  Random       0.8750       0.9235   0.8762        0.9020    0.8889       0.7465
  Forest                                                                  

  XGBoost      0.8478       0.9302   0.8854        0.8333    0.8586       0.6957
  --------------------------------------------------------------------------------

------------------------------------------------------------------------

# 🔷 Observations on Model Performance

  -----------------------------------------------------------------------
  ML Model                      Observation
  ----------------------------- -----------------------------------------
  Logistic Regression           Provides strong baseline performance with
                                high recall and balanced metrics.

  Decision Tree                 Performs reasonably but shows lower
                                accuracy and MCC due to overfitting
                                tendencies.

  KNN                           High accuracy and F1 score, performs well
                                after feature scaling.

  Naive Bayes                   Achieved high AUC and balanced
                                performance, works well for this dataset.

  Random Forest                 Strong ensemble model with high AUC and
                                stable performance.

  XGBoost                       Achieved highest AUC but slightly lower
                                accuracy compared to KNN and Naive Bayes.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 🔷 Streamlit Web Application Features

-   CSV dataset upload option
-   Model selection dropdown
-   Display of evaluation metrics
-   Confusion matrix and classification report
-   Real-time prediction interface

------------------------------------------------------------------------

# 🔷 Project Structure

project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- models/ (saved model files)
│-- dataset/ (contains dataset CSV file heart.csv)

------------------------------------------------------------------------

# 🔷 Run Locally

pip install -r requirements.txt
streamlit run app.py

------------------------------------------------------------------------

# 🔷 Deployment

**GitHub Repository:** https://github.com/AmitKul/ml-assignment-2/tree/main
**Live Streamlit App:** https://amitkul-ml-assignment-2-app-yjfkur.streamlit.app/

------------------------------------------------------------------------

# 🔷 Conclusion

This project successfully demonstrates implementation and comparison of
multiple machine learning classification models on the Heart Failure
Prediction dataset. Ensemble and distance-based models like Random
Forest, KNN, and Naive Bayes achieved strong performance. The deployed
Streamlit app provides an interactive interface for model testing and
prediction, showcasing a real-world ML deployment workflow.
