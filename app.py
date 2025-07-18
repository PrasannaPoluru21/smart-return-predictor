#creating web application with flask
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

#import ridge regressor and standard scaler pickle
with open('models/scaler.pkl', 'rb') as f:
    standard_scaler = pickle.load(f)

with open('models/model.pkl', 'rb') as fq:
    model = pickle.load(fq)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
     if request.method=='POST':
        delivery_delay = int(request.form.get('delivery_delay'))
        payment_value = float(request.form.get('payment_value'))
        price= float(request.form.get('price'))
        # customer_state_SP= request.form.get('customer_state_SP')
        # product_category_bed_bath_table= request.form.get('product_category_bed_bath_table')
        review_score= float(request.form.get('review_score'))
        # payment_type_not_defined= request.form.get('payment_type_not_defined')
        customer_state_SP = int(request.form.get('customer_state_SP'))
        product_category_bed_bath_table = int(request.form.get('product_category_bed_bath_table'))
        # payment_type_not_defined = int(request.form.get('payment_type_not_defined'))

        new_data = np.array([[delivery_delay, payment_value,price, review_score,customer_state_SP, product_category_bed_bath_table]])
        result= model.predict(new_data)[0]

        return render_template('home.html', results=result)

     else:
         return render_template('home.html')

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# from flask import Flask, request, jsonify
# import mlflow
# import pandas as pd
# import requests

# app = Flask(__name__)

# # Set MLflow tracking server URI
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# # Load model from MLflow run
# model_uri = "runs:/67d6ef1f3fad47c1b126649fe6f1f3d9/model"
# loaded_model = mlflow.sklearn.load_model(model_uri)

# @app.route('/predict', methods=['POST'])
# def predict():
#     sample = {
#          "delivery_delay": -5,
#            "payment_value": 100.0,
#              "price": 75.0,
#              "customer_state_SP": True,
#              "product_category_bed_bath_table": False,
#              "review_score": 5.0,
#              "payment_type_not_defined": False
#             }
#     data = sample
#     input_df = pd.DataFrame([data])
#     prediction = loaded_model.predict(input_df)[0]
#     return jsonify({'prediction': int(prediction)})
   
#     res = requests.post("http://localhost:5000/predict", json=sample)
#     print(res.json())

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)