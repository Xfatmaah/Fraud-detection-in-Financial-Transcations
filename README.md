# Fraud-detection-in-Financial-Transcations

## 1. Project Objective
The goal of this project is to develop a machine learning system that can accurately detect fraudulent financial transactions. By analyzing transaction patterns and behaviors, the system will classify transactions as either fraudulent or legitimate. This solution aims to assist financial institutions in reducing the impact of fraud, minimizing financial losses, and improving security measures.

Key objectives include:
Identifying patterns in financial transactions that indicate fraud.
Developing both machine learning and natural language processing (NLP) models for prediction.
Building a user-friendly web application for real-time fraud detection.

## 2. Tools and Technologies
- Python: Primary language for data processing, model building, and backend development.
- Machine Learning Libraries: Scikit-learn, TensorFlow/PyTorch, SMOTE(Synthetic Minority Over-sampling Technique).
- NLP Libraries: NLTK for natural language processing tasks, tokenization, and embedding of action sequences.
- Data Processing : Pandas & NumPy.
- Visualization : Matplotlib & Seaborn.
- Model Management : MLflow for tracking experiments, managing models, and versioning.
- Deployment & Web Frameworks: Flask, HTML/CSS & JavaScript.

## 3. Project Breakdown
* Phase 1: Data Exploration & Preprocessing
    Data Understanding : Explore the structure and characteristics of the dataset. 
    Data Cleaning : Handle missing values, outliers, and ensure data consistency.
    Feature Engineering : Create new features such as padded_action_sequences from your action sequences dataset.
    Label Balancing : Use SMOTE to address the class imbalance problem, given the rarity of fraudulent transactions.
  
* Phase 2: Model Development
    Model Selection:
      Decision Tree Model - been selected from ( Logistic Regression, Decision Tree, Random Forest).
      LSTM Model (for sequence-based fraud detection) - been selected from ( LSTM, GRU, Fully Connected Neural Network).
   
    Training & Validation: Split the dataset into training, validation, and test sets.Train models using various techniques and evaluate their performance using metrics like       precision, recall, F1 score, and accuracy.
  
* Phase 3: Feature Integration (NLP-based)
    NLP Preprocessing: Tokenize and embed action sequences.
    Feature Embedding: Integrate embedded action sequences into the LSTM model for better predictions.
  
* Phase 4: Model Evaluation & Comparison
    Evaluation Metrics: Compare the performance of different models using confusion matrices and metrics like AUC-ROC, precision-recall curves.
  
* Phase 5: Web Application Development
    Backend Development (Flask):
      Create routes for the ML model and NLP model predictions.
      Accept inputs like step, amount, type, and for NLP-based, action sequences.
    Frontend Development: Build a simple interface where users can input transaction details and get results in real-time.



Resources :
Data Link: https://www.kaggle.com/datasets/ealaxi/paysim1

Research Link : https://www.researchgate.net/publication/372466905_Fraud_detection_with_natural_language_processing
