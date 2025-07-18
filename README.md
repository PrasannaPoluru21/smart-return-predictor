# Smart Return Predictor
A machine learning system to predict product returns in an e-commerce setting, complete with MLOps features like model tracking, deployment, and monitoring.

### Project Overview
This project simulates a real-world ML workflow:
- Predict whether an order will be returned (using historical e-commerce data)
- Track experiments using MLflow
- Serve model via REST API (Flask/FastAPI)
- Automate training and deployment via GitHub Actions
- Simulate monitoring for production-readiness
### Dataset

Used the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce):
- Multiple CSVs (orders, products, reviews, payments, etc.)
- Merged key tables to engineer features like delivery time, payment method, etc.
### Problem Statement
Can we predict if an order will be returned based on delivery speed, product type, customer behavior, and other features?

### Tech Stack

| Component      | Tools Used                                   |
|----------------|-----------------------------------------------|
| Language       | Python                                        |
| Data Wrangling | Pandas, NumPy                                 |
| Modeling       | Scikit-learn                                  |
| Tracking       | MLflow                                        |
| API Serving    | Flask or FastAPI                              |
| CI/CD          | GitHub Actions                                |
| Deployment     | Localhost (Docker optional)                   |
| Monitoring     | MLflow UI, logs, CSV-based tracking           |

### Project Structure
smart-return-predictor/
├── data/               ← Raw datasets              
├── notebooks/          ← EDA, experiments
├── src/                ← Scripts (train.py, utils.py)
├── app/                ← API code (app.py)
├── model/              ← Saved models
├── mlruns/             ← MLflow experiment logs
├── requirements.txt
├── .gitignore
├── README.md

### Key Features
- Exploratory Data Analysis with visualizations
- ML Model to predict returns
- MLflow for tracking model runs
- REST API to serve predictions
- CI/CD pipeline using GitHub Actions
- Simulated monitoring of model metrics

### Future Enhancements
- Add model drift detection
- Store prediction logs in cloud storage (e.g., S3 or Azure Blob)
- Deploy using Docker/Kubernetes
- Automate retraining with new data

### Acknowledgements
- Dataset by Olist on Kaggle
- Inspiration from real-world MLOps and product analytics use cases

### Contact
**Prasanna Poluru**  
[LinkedIn](https://www.linkedin.com/in/prasanna-poluru/) | prasanna.poluru.data@gmail.com



**Smart Return Predictor** is a machine learning project built to predict whether a product will be returned based on customer and order details. It covers the complete lifecycle of a data science project, including data preprocessing, model selection, deployment, and CI/CD automation.

---

## Project Overview

This project solves a binary classification problem using a dataset containing features such as delivery delay, payment value, product category, review score, and payment type. The main goal is to predict the likelihood of a product return.

Key aspects of the project include:

- Data cleaning, encoding, and scaling
- Class imbalance handling
- Comparison of multiple models
- MLflow experiment tracking
- Local deployment using Flask
- Docker containerization
- GitHub Actions for CI/CD automation

---

## Model Development and Tracking

### Steps Completed:

- Performed data preprocessing including one-hot encoding and handling missing values
- Handled class imbalance using `class_weight='balanced'`
- Trained and evaluated four models: Logistic Regression, Random Forest, LightGBM, and XGBoost
- Chose XGBoost based on overall performance
- Tuned classification threshold for better recall on the minority class
- Logged metrics, parameters, confusion matrix, and model artifacts using MLflow
- Saved model and scaler as pickle files

---

## Deployment and MLOps

- Developed a Flask application (`app.py`) to serve predictions
- Used HTML templates to collect user input and display results
- Containerized the application using Docker
- Deployed the container locally and accessed it at `http://localhost:5001`
- Set up a GitHub Actions workflow (`train.yaml`) to automatically run `train.py` and retrain the model on every push

---

## Directory Structure

```
smart-return-predictor/
├── app.py
├── train.py
├── Dockerfile
├── requirements.txt
├── models/
│   ├── model.pkl
│   └── scaler.pkl
├── templates/
│   └── index.html
├── .github/
│   └── workflows/
│       └── train.yaml
├── mlruns/
├── mlartifacts/
├── data/
└── notebooks/
```

---

## Running the Project

### 1. Train the Model

```bash
python train.py
```

### 2. Run the Flask App

```bash
python app.py
# or via Docker
docker build -t smart-return-app .
docker run -p 5001:5000 smart-return-app
```

### 3. Use the App

Visit `http://localhost:5001` and input feature values to get return predictions.

---

## Performance Summary

- **ROC AUC Score**: 0.927
- **Confusion Matrix after threshold tuning**:
  ```
  [[22047   972]
   [   61    47]]
  ```

---

## Future Improvements

- Add monitoring for prediction drift and logs
- Visualize model metrics and input trends using Streamlit or similar tools
- Version models and track performance over time
- Enable cloud deployment for scalability

---

## Author

**Prasanna Poluru**  
M.S. in Data Science, University of Colorado Boulder  
[LinkedIn](https://www.linkedin.com/in/prasannapoluru) | [GitHub](https://github.com/PrasannaPoluru21)
"""
