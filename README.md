# Laptop Price Prediction - README

This repository contains a machine learning project for predicting laptop prices based on their specifications. It includes data preprocessing, model training, and deployment through Flask and Streamlit.

---

## Repository Overview

### 1. **Data File**
- `laptop_price.csv`: The dataset used to train and test the machine learning models. It contains laptop specifications and their corresponding prices.

### 2. **Notebooks and Scripts**
- `LP_jupyterNB.ipynb`: Jupyter notebook with exploratory data analysis (EDA), data preprocessing, and model training. Various regression algorithms were tested to find the best-performing model.
- `lp_flask_dep.py`: Python script for deploying the machine learning model as a Flask web application. It includes dropdown-based input forms for user interaction and predicts laptop prices in real-time.
- `lp_streamlit_dep.py`: Python script for deploying the model using Streamlit. It provides an interactive and user-friendly interface for laptop price prediction.

---

## Machine Learning Models
The following models were tested:
- **Linear Regression**
- **Ridge Regression** (Best model: Achieved ~91.2% accuracy)
- **Lasso Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **AdaBoost Regressor**
- **XGBoost Regressor**

Evaluation metrics such as **R² score** and **Mean Squared Error (MSE)** were used to compare the models. Ridge Regression provided the best results on the validation set.

---

## Key Features
- Predicts laptop prices based on key specifications like **brand, screen size, RAM, CPU, GPU, and storage**.
- Supports both categorical and numerical inputs.
- Interactive UI for easy user engagement.
- Deployed using two distinct methods for flexibility and accessibility.

---
