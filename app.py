from flask import Flask, request, jsonify
import mlflow
import pandas as pd

app = Flask(__name__)

# Set MLflow tracking server URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load model from MLflow run
model_uri = "runs:/67d6ef1f3fad47c1b126649fe6f1f3d9/model"
loaded_model = mlflow.sklearn.load_model(model_uri)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_df = pd.DataFrame([data])
    prediction = loaded_model.predict(input_df)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)