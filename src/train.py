import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

final_df = pd.read_csv("../data/final_df.csv")

#splitting features
X = final_df.drop(columns=['is_returned'])
y = final_df['is_returned']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
final_df = pd.read_csv("../data/final_df.csv")

#splitting features
X = final_df.drop(columns=['is_returned'])
y = final_df['is_returned']

#Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Start MLflow run
with mlflow.start_run():
    # Train model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", acc)
    print(report)

    # Log metrics and model
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    # Save scaler
    joblib.dump(scaler, "model/scaler.pkl")
    mlflow.log_artifact("model/scaler.pkl")

    # Save classification report
    with open("model/classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("model/classification_report.txt")

print('train.py successfully executes!')