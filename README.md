**Smart Return Predictor** is a machine learning project built to predict whether a product will be returned based on customer and order details, in an e-commerce setting. It covers the complete lifecycle of a data science project, including data preprocessing, model selection, deployment, complete with MLOps features like model tracking, deployment, and monitoring.

## Problem Statement
Can we predict if an order will be returned based on delivery speed, product type, customer behavior, and other features?

## Dataset
Used the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce):
- Multiple CSVs (orders, products, reviews, payments, etc.)
- Merged key tables to engineer features like delivery time, payment method, etc.

## Project Overview
This project solves a binary classification problem using a dataset containing features such as delivery delay, payment value, product category, review score, and payment type. The main goal is to predict the likelihood of a product return.

Key aspects of the project include:
- Data cleaning, encoding, and scaling
- Class imbalance handling
- Comparison of multiple models
- MLflow experiment tracking
- Local deployment using Flask
- Docker containerization
  
This project simulates a real-world ML workflow:
- Predict whether an order will be returned (using historical e-commerce data)
- Track experiments using MLflow
- Serve model via REST API (Flask/FastAPI)
- Automate training and deployment via GitHub Actions
- Simulate monitoring for production-readiness
  
## Tech Stack

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

## Model Development and Tracking

## Steps Completed:
- Performed data preprocessing including one-hot encoding and handling missing values
- Handled class imbalance using `class_weight='balanced'`
- Trained and evaluated four models: Logistic Regression, Random Forest, LightGBM, and XGBoost
- Chose XGBoost based on overall performance
- Tuned classification threshold for better recall on the minority class
- Logged metrics, parameters, confusion matrix, and model artifacts using MLflow
- Saved model and scaler as pickle files

 ## Key Features
- Exploratory Data Analysis with visualizations
- ML Model to predict returns
- MLflow for tracking model runs
- REST API to serve predictions
  
## Deployment and MLOps

- Developed a Flask application (`app.py`) to serve predictions
- Used HTML templates to collect user input and display results
- Containerized the application using Docker
- Deployed the container locally and accessed it at `http://localhost:5001`
- Set up a GitHub Actions workflow (`train.yaml`) to automatically run `train.py` and retrain the model on every push

## Running the Project
# 1. Train the Model

```bash
python train.py
```

# 2. Run the Flask App

```bash
python app.py
# or via Docker
docker build -t smart-return-app .
docker run -p 5001:5000 smart-return-app
```

# 3. Use the App

Visit `http://localhost:5001` and input feature values to get return predictions.

## Future Improvements

- Add monitoring for prediction drift and logs
- Visualize model metrics and input trends using Streamlit or similar tools
- Version models and track performance over time
- Enable cloud deployment for scalability

## Acknowledgements
- Dataset by Olist on Kaggle
- Inspiration from real-world MLOps and product analytics use cases

## Contact
**Prasanna Poluru**  
[LinkedIn](https://www.linkedin.com/in/prasanna-poluru/) | prasanna.poluru.data@gmail.com
