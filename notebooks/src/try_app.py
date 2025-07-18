import requests

sample = {
    "delivery_delay": -5,
    "payment_value": 100.0,
    "price": 75.0,
    "customer_state_SP": True,
    "product_category_bed_bath_table": False,
    "review_score": 5.0,
    "payment_type_not_defined": False
}

res = requests.post("http://localhost:5001/predict", json=sample)
print(res.json())
