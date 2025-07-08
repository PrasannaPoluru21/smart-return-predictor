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
