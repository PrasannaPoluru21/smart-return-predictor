import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
customers_df = pd.read_csv('../data/olist_customers_dataset.csv')
# print(customers_df.shape)
# customers_df.head()

geolocation_df = pd.read_csv('../data/olist_geolocation_dataset.csv')
order_items_df = pd.read_csv('../data/olist_order_items_dataset.csv')
order_payments_df = pd.read_csv('../data/olist_order_payments_dataset.csv')
order_reviews_df = pd.read_csv('../data/olist_order_reviews_dataset.csv')
orders_df = pd.read_csv('../data/olist_orders_dataset.csv')
products_df = pd.read_csv('../data/olist_products_dataset.csv')
sellers_df = pd.read_csv('../data/olist_sellers_dataset.csv')
product_translation_df = pd.read_csv('../data/product_category_name_translation.csv')

print(customers_df.columns)
print(geolocation_df.columns)
print(order_items_df.columns)
print(order_payments_df.columns)
print(order_reviews_df.columns)
print(orders_df.columns)
print(products_df.columns)
print(sellers_df.columns)
print(product_translation_df.columns)

updated_orders_payments_df = pd.merge(orders_df, order_payments_df, on='order_id', how='left')
updated_products_order_items_df=pd.merge(order_items_df, products_df, on='product_id', how='left' )

# Take average lat/lng for each zip code prefix
geo_grouped = geolocation_df.groupby('geolocation_zip_code_prefix').agg({
    'geolocation_lat': 'mean',
    'geolocation_lng': 'mean'
}).reset_index()

# Merge with sellers
sellers_geolocation_df = pd.merge(sellers_df, geo_grouped, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')

updated_sellers_geolocation_products_order_items_df= pd.merge(sellers_geolocation_df, updated_products_order_items_df, on='seller_id', how='right')

updated_orders_payment_reviews_df = pd.merge(updated_orders_payments_df, order_reviews_df, on='order_id', how='left')

updated_orders_payment_reviews_customers_df=pd.merge(updated_orders_payment_reviews_df, customers_df, on='customer_id', how='left')

combined_df=pd.merge(updated_orders_payment_reviews_customers_df, updated_sellers_geolocation_products_order_items_df, on='order_id', how='left')

combined_df['order_status'].value_counts()

#Adding target Variable
combined_df['is_returned'] = combined_df['order_status'].apply(lambda x: 1 if x == 'canceled' else 0)

#for reference
# df.drop('column_name', axis=1, inplace=True)
# df.drop(['col1', 'col2'], axis=1, inplace=True)
# df.rename(columns={'old_name': 'new_name'}, inplace=True)
# df.rename(columns={'old1': 'new1', 'old2': 'new2'}, inplace=True)

combined_df=combined_df[['customer_city', 'customer_state','seller_city', 'seller_state','price',
'product_category_name','product_weight_g','product_length_cm', 'product_height_cm', 'product_width_cm', 'product_photos_qty','freight_value', 'order_approved_at','order_delivered_customer_date', 'order_estimated_delivery_date', 'review_score','payment_value','payment_type', 'payment_installments','is_returned']]

final_df=pd.merge(combined_df,product_translation_df, on='product_category_name', how='left')

final_df.rename(columns={'product_category_name_english': 'product_category'}, inplace=True)

final_df=final_df[['customer_city', 'customer_state','seller_city', 'seller_state','price',
'product_category','product_weight_g','product_length_cm', 'product_height_cm', 'product_width_cm', 'product_photos_qty','freight_value', 'order_approved_at','order_delivered_customer_date', 'order_estimated_delivery_date', 'review_score','payment_value','payment_type', 'payment_installments','is_returned']]

#Converting Dates to datetime here
final_df['order_approved_at'] = pd.to_datetime(final_df['order_approved_at'])
final_df['order_delivered_customer_date'] = pd.to_datetime(final_df['order_delivered_customer_date'])
final_df['order_estimated_delivery_date'] = pd.to_datetime(final_df['order_estimated_delivery_date'])

#To understand delivery delay
final_df['delivery_delay'] = (
    final_df['order_delivered_customer_date'] - final_df['order_estimated_delivery_date']).dt.days.fillna(0)

#According to product volume
final_df['product_volume'] = final_df['product_length_cm'] * final_df['product_height_cm'] * final_df['product_width_cm']

#Dropping the used rows
final_df = final_df.drop(columns=[
    'order_approved_at', 'order_delivered_customer_date', 'order_estimated_delivery_date'
])

# final_df = pd.get_dummies(final_df, columns=['payment_type'])
final_df = pd.get_dummies(final_df, columns=['payment_type', 'product_category', 'customer_state', 'seller_state'], drop_first=True)

final_df = final_df.dropna()
final_df = final_df.reset_index(drop=True)

#Train-Test Split
from sklearn.model_selection import train_test_split

final_df[['is_returned']].head()

final_df.columns

non_numeric = final_df.select_dtypes(include='object').columns
print(non_numeric)

final_df=final_df.drop(columns=['customer_city', 'seller_city'])

#setting the gonna split data with feature and target columns
X = final_df.drop(columns=['is_returned'])
y=  final_df['is_returned']

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# used stratify here to maintain even class distribution as there might be less no of returned items

final_df.to_csv("/Users/lakshmiprasannapoluru/Desktop/smart-return-predictor/data/final_df.csv", index=False)

print('Execution done!')