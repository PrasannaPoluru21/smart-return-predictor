#imports and installations
import pandas as pd
from sklearn.model_selection import train_test_split #for splitting the data
from sklearn.preprocessing import StandardScaler #for standardisation
from sklearn.linear_model import LogisticRegression #for baseline model
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score #for metrics
# pip install xgboost lightgbm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mlflow import register_model

#Importing data
X = pd.read_csv("../data/X.csv") 
y = pd.read_csv("../data/y.csv")

#train_test split, Split X and y into train/test
#Splitting X and y into train and test sets(80/20), while preserving the class imbalance(ratio of returned, non-retuned) 
#with stratify as class is imbalanced
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#we are converting boolean to int as its friendly for future pipeline building
X_train[['customer_state_SP','product_category_bed_bath_table']] = X_train[['customer_state_SP','product_category_bed_bath_table']].astype(int)
X_test[['customer_state_SP','product_category_bed_bath_table']] = X_test[['customer_state_SP','product_category_bed_bath_table']].astype(int)

#Standardising numerical columns
num_cols = ['delivery_delay', 'payment_value', 'price', 'review_score']

# Fit and transform on train, transform on test
scaler= StandardScaler()
X_train[num_cols]=scaler.fit_transform(X_train[num_cols])
X_test[num_cols]=scaler.transform(X_test[num_cols])

X_train.to_csv("../data/X_train.csv", index=False)
X_test.to_csv("../data/X_test.csv", index=False)

y_train.to_csv("../data/y_train.csv", index=False)
y_test.to_csv("../data/y_test.csv", index=False)

#Train Baseline Model
params={'class_weight':'balanced', 'max_iter':1000, 'random_state':42}
lr= LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred= lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:, 1]

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

#Tune the classification threshold (Right now, predict() uses default threshold = 0.5
# But maybe predicting return only if probability > 0.9 improves precision.)
# Set threshold high (e.g., 0.85 to 0.95)
threshold = 0.9
y_pred_thresh = (y_prob > threshold).astype(int)

# Re-evaluate
print(confusion_matrix(y_test, y_pred_thresh))
print(classification_report(y_test, y_pred_thresh))

#trying to find the best threshold to tune
threshold=[0,0.5,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
for num in threshold:
    y_pred_thresh = (y_prob > num).astype(int)
    # Re-evaluate
    print('I am for this ', num )
    print(confusion_matrix(y_test, y_pred_thresh))
    print(classification_report(y_test, y_pred_thresh))

threshold = 0.65
y_pred_final = (y_prob > threshold).astype(int)

#Extracting final metrics
precision = precision_score(y_test, y_pred_final)
recall = recall_score(y_test, y_pred_final)
f1 = f1_score(y_test, y_pred_final)
roc_auc_lr = roc_auc_score(y_test, y_prob)
report_lr =classification_report(y_test, y_pred_final, output_dict=True)
print('precision :',precision,'\n','recall :',recall,'\n','f1_score :',f1,'\n','roc-auc:',roc_auc_lr,'\n')

mlflow.set_tracking_uri('http://127.0.0.1:8080')
mlflow.set_experiment('smart_return_predictor_v2')

with mlflow.start_run(run_name='lr_return_classifier'):
    mlflow.log_params(params)
    mlflow.log_metrics({ "accuracy":report_lr['accuracy'],
            "precision_class_0": report_lr['0']['precision'],
            "precision_class_1": report_lr['1']['precision'],
            "recall_class_0": report_lr['0']['recall'],
            "recall_class_1": report_lr['1']['recall'],
            "f1_score_macro": report_lr['macro avg']['f1-score'],
            "roc_auc_lr": roc_auc_lr})

    mlflow.sklearn.log_model(lr,artifact_path="model")

#Trying RF, XGBoost, LGB models
models = {
    "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "LightGBM": lgb.LGBMClassifier(scale_pos_weight=100, random_state=42)
    "XGBoost": xgb.XGBClassifier(scale_pos_weight=100, use_label_encoder=False, eval_metric='logloss', random_state=42),

}
trained_models={}
threshold = 0.65

#Added scale_pos_weight as classes are highly imbalanced
for name, model in models.items():
    print(f"\n Training {name}")
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    cm=confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n Results for {name} at threshold = {threshold}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

    trained_models[name]={
        "model" : model,
        "report": report,
        "roc_auc": roc_auc,
        "cm" : cm        
    }

#MLFlow logginfg for RF, XGBoost, LGB Models 
mlflow.set_tracking_uri('http://127.0.0.1:8080')
mlflow.set_experiment('smart_return_predictor_v2')
for name, data in trained_models.items():
    model=data['model']
    report=data['report']
    roc_auc=data['roc_auc']
    cm=data['cm']
    
    with mlflow.start_run(run_name=f"{name}_return_classifier"):
        mlflow.log_param("threshold", threshold)
        mlflow.log_params(model.get_params())

        mlflow.log_metrics({
            "accuracy": report['accuracy'],
            "precision_class_0": report['0']['precision'],
            "precision_class_1": report['1']['precision'],
            "recall_class_0": report['0']['recall'],
            "recall_class_1": report['1']['recall'],
            "f1_score_macro": report['macro avg']['f1-score'],
            "roc_auc": roc_auc
        })

        # Confusion Matrix plot
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix_{name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
    
        # Save confusion matrix
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/confusion_matrix_{name}.png")
        mlflow.log_artifact(f"outputs/confusion_matrix_{name}.png")
        mlflow.sklearn.log_model(model,artifact_path="model")
        print(f"Logged {name} model and metrics to MLflow")

# #Register the model
# model_name = "XGBoost_model"
# run_id= input("Enter run ID :")
# model_uri = f"runs:/{run_id}/model"
# result = mlflow.register_model(model_uri, model_name)

import pickle
pickle.dump(scaler, open('scaler.pkl', 'wb')) #for standard scaling I can use this
pickle.dump(model, open('model.pkl', 'wb'))#for modeling I can use this